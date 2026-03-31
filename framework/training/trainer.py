"""
Full pre-training loop.

Features:
  • Mixed-precision (bf16 / fp16) via torch.amp
  • Gradient accumulation
  • Cosine LR schedule with warm-up
  • Gradient clipping
  • Checkpoint saving (best + periodic)
  • Rich progress bar and loss logging
  • Evaluation on val set after every eval_steps
  • Final test-set evaluation
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table
from rich import box

from framework.config import TrainingConfig, ModelConfig
from framework.models import TransformerLM

console = Console()


# ---------------------------------------------------------------------------
# LR Scheduler
# ---------------------------------------------------------------------------


def _cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine decay with linear warm-up."""
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


def _linear_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(step: int) -> float:
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        return max(0.0, float(num_training_steps - step) / max(1, num_training_steps - num_warmup_steps))
    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# PreTrainer
# ---------------------------------------------------------------------------


class PreTrainer:
    """Trains a TransformerLM from scratch."""

    def __init__(
        self,
        model: TransformerLM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        cfg: TrainingConfig,
        output_dir: str = "outputs",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Mixed precision
        self.use_amp = cfg.mixed_precision in ("fp16", "bf16") and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float16
        self.scaler = GradScaler(enabled=(cfg.mixed_precision == "fp16"))

        # Gradient checkpointing
        if cfg.gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # Optimiser
        self.optimizer = self._build_optimizer()

        # Steps
        steps_per_epoch = len(train_loader) // cfg.gradient_accumulation_steps
        if cfg.max_steps > 0:
            self.total_steps = cfg.max_steps
            self.num_epochs = math.ceil(self.total_steps / max(1, steps_per_epoch))
        else:
            self.num_epochs = cfg.num_train_epochs
            self.total_steps = steps_per_epoch * self.num_epochs

        num_warmup = max(1, int(self.total_steps * cfg.warmup_ratio))

        # Scheduler
        if cfg.lr_scheduler == "cosine":
            self.scheduler = _cosine_schedule_with_warmup(self.optimizer, num_warmup, self.total_steps)
        else:
            self.scheduler = _linear_schedule_with_warmup(self.optimizer, num_warmup, self.total_steps)

        # State
        self.global_step = 0
        self.best_val_loss = float("inf")
        self._log: Dict = {"train_loss": [], "val_loss": [], "lr": []}

        logger.info(f"PreTrainer ready — device:{self.device} | steps:{self.total_steps} | epochs:{self.num_epochs}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict:
        """Run the full training loop. Returns training log dict."""
        start = time.time()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        with progress:
            epoch_task = progress.add_task("Epochs", total=self.num_epochs)
            step_task = progress.add_task("Steps", total=self.total_steps)

            for epoch in range(1, self.num_epochs + 1):
                if self.cfg.max_steps > 0 and self.global_step >= self.cfg.max_steps:
                    break

                self._train_epoch(epoch, progress, step_task)

                val_loss = self._evaluate(self.val_loader, "Validation")
                self._log["val_loss"].append({"step": self.global_step, "loss": val_loss})

                self._maybe_save_checkpoint(val_loss)
                progress.advance(epoch_task)

        # Final test evaluation
        test_loss = self._evaluate(self.test_loader, "Test")
        elapsed = (time.time() - start) / 3600

        self._print_summary(test_loss, elapsed)
        return {"train_log": self._log, "test_loss": test_loss, "elapsed_hours": elapsed}

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int, progress: Progress, task_id) -> None:
        self.model.train()
        accum_loss = 0.0
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            if self.cfg.max_steps > 0 and self.global_step >= self.cfg.max_steps:
                break

            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                out = self.model(input_ids, labels=labels)
                loss = out.loss / self.cfg.gradient_accumulation_steps

            self.scaler.scale(loss).backward()
            accum_loss += loss.item()

            if (step + 1) % self.cfg.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                lr = self.scheduler.get_last_lr()[0]
                train_loss = accum_loss * self.cfg.gradient_accumulation_steps
                self._log["train_loss"].append({"step": self.global_step, "loss": train_loss})
                self._log["lr"].append(lr)
                accum_loss = 0.0

                if self.global_step % self.cfg.logging_steps == 0:
                    progress.print(
                        f"  step={self.global_step:>6}  epoch={epoch}  "
                        f"loss={train_loss:.4f}  lr={lr:.2e}"
                    )

                if self.global_step % self.cfg.eval_steps == 0:
                    val_loss = self._evaluate(self.val_loader, "Val (mid-epoch)")
                    self._log["val_loss"].append({"step": self.global_step, "loss": val_loss})
                    self._maybe_save_checkpoint(val_loss)
                    self.model.train()

                if self.global_step % self.cfg.save_steps == 0:
                    self._save_checkpoint(f"checkpoint-{self.global_step}")

                progress.advance(task_id)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, label: str = "Eval") -> float:
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                out = self.model(input_ids, labels=labels)

            total_loss += out.loss.item()
            total_batches += 1

        avg_loss = total_loss / max(1, total_batches)
        perplexity = math.exp(min(avg_loss, 20))
        logger.info(f"  [{label}] loss={avg_loss:.4f}  perplexity={perplexity:.2f}")
        return avg_loss

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _maybe_save_checkpoint(self, val_loss: float) -> None:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self._save_checkpoint("best_model")
            logger.info(f"  New best val_loss={val_loss:.4f} → saved best_model")

    def _save_checkpoint(self, name: str) -> None:
        ckpt_dir = self.output_dir / name
        ckpt_dir.mkdir(exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "model_config": self.model.cfg,
            },
            ckpt_dir / "checkpoint.pt",
        )
        # Also save model config for inference
        import json
        import dataclasses
        with open(ckpt_dir / "model_config.json", "w") as f:
            json.dump(dataclasses.asdict(self.model.cfg), f, indent=2)

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        model: TransformerLM,
        cfg: TrainingConfig,
        train_loader=None,
        val_loader=None,
        test_loader=None,
    ) -> "PreTrainer":
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        trainer = cls(model, train_loader, val_loader, test_loader, cfg)
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        trainer.global_step = ckpt["global_step"]
        trainer.best_val_loss = ckpt["best_val_loss"]
        return trainer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> AdamW:
        # Separate weight decay groups
        decay = {n for n, p in self.model.named_parameters()
                 if p.requires_grad and p.ndim >= 2}
        no_decay = {n for n, p in self.model.named_parameters()
                    if p.requires_grad and p.ndim < 2}
        params = [
            {"params": [p for n, p in self.model.named_parameters() if n in decay],
             "weight_decay": self.cfg.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if n in no_decay],
             "weight_decay": 0.0},
        ]
        return AdamW(
            params,
            lr=self.cfg.learning_rate,
            betas=(self.cfg.beta1, self.cfg.beta2),
            eps=self.cfg.epsilon,
            fused=torch.cuda.is_available(),
        )

    def _enable_gradient_checkpointing(self) -> None:
        for block in self.model.layers:
            block.gradient_checkpointing = True

    def _print_summary(self, test_loss: float, elapsed: float) -> None:
        table = Table(title="Training Summary", box=box.ROUNDED)
        table.add_column("Metric", style="cyan bold")
        table.add_column("Value", style="yellow bold", justify="right")
        table.add_row("Total steps", str(self.global_step))
        table.add_row("Best val loss", f"{self.best_val_loss:.4f}")
        table.add_row("Test loss", f"{test_loss:.4f}")
        table.add_row("Test perplexity", f"{math.exp(min(test_loss, 20)):.2f}")
        table.add_row("Training time", f"{elapsed:.2f} hours")
        console.print(table)
