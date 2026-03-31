"""
LoRA / QLoRA fine-tuning trainer.

Uses PEFT for LoRA injection and bitsandbytes for 4-bit quantisation (QLoRA).
Can fine-tune either:
  • A FastFail TransformerLM checkpoint
  • Any HuggingFace CausalLM (e.g., LLaMA, Mistral) for transfer learning
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
    TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)

from framework.config import TrainingConfig, LoRAConfig

console = Console()


class LoRATrainer:
    """
    Wraps a model (FastFail or HuggingFace) with LoRA adapters and trains
    only the adapter weights.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        train_cfg: TrainingConfig,
        lora_cfg: LoRAConfig,
        output_dir: str = "outputs/lora",
        use_qlora: bool = False,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.train_cfg = train_cfg
        self.lora_cfg = lora_cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_qlora = use_qlora

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Apply LoRA / QLoRA
        self.model = self._apply_lora(model)
        self.model.to(self.device)

        self._log_trainable_params()

        # Mixed precision
        self.use_amp = train_cfg.mixed_precision in ("fp16", "bf16") and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if train_cfg.mixed_precision == "bf16" else torch.float16
        self.scaler = GradScaler(enabled=(train_cfg.mixed_precision == "fp16"))

        # Optimiser & scheduler
        self.optimizer = self._build_optimizer()
        steps_per_epoch = len(train_loader) // train_cfg.gradient_accumulation_steps
        self.total_steps = steps_per_epoch * train_cfg.num_train_epochs
        num_warmup = max(1, int(self.total_steps * train_cfg.warmup_ratio))
        self.scheduler = self._build_scheduler(num_warmup)

        self.global_step = 0
        self.best_val_loss = float("inf")
        self._log: Dict = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def train(self) -> Dict:
        """Run LoRA fine-tuning."""
        start = time.time()

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(), TaskProgressColumn(), MofNCompleteColumn(),
            TimeElapsedColumn(), TimeRemainingColumn(),
            console=console,
        )

        with progress:
            step_task = progress.add_task(
                f"LoRA{'(Q)' if self.use_qlora else ''} Fine-tuning",
                total=self.total_steps,
            )
            for epoch in range(1, self.train_cfg.num_train_epochs + 1):
                self._train_epoch(epoch, progress, step_task)
                val_loss = self._evaluate(self.val_loader, "Validation")
                self._log["val_loss"].append({"step": self.global_step, "loss": val_loss})
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_adapter("best_adapter")

        test_loss = self._evaluate(self.test_loader, "Test")
        elapsed = (time.time() - start) / 3600
        logger.info(
            f"LoRA training done — test_loss={test_loss:.4f}  time={elapsed:.2f}h"
        )
        return {"train_log": self._log, "test_loss": test_loss, "elapsed_hours": elapsed}

    def save_merged(self, path: str) -> None:
        """Merge LoRA weights into the base model and save full weights."""
        try:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                merged = self.model.merge_and_unload()
                torch.save(merged.state_dict(), path)
                logger.info(f"Merged model saved → {path}")
            else:
                torch.save(self.model.state_dict(), path)
        except ImportError:
            torch.save(self.model.state_dict(), path)

    # ------------------------------------------------------------------
    # LoRA / QLoRA setup
    # ------------------------------------------------------------------

    def _apply_lora(self, model: nn.Module) -> nn.Module:
        try:
            from peft import (
                LoraConfig as PeftLoraConfig,
                get_peft_model,
                TaskType,
                prepare_model_for_kbit_training,
            )
        except ImportError:
            logger.warning("PEFT not installed — falling back to full fine-tune.")
            return model

        if self.use_qlora:
            model = self._quantize_model(model)
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=self.train_cfg.gradient_checkpointing,
            )

        peft_cfg = PeftLoraConfig(
            r=self.lora_cfg.r,
            lora_alpha=self.lora_cfg.lora_alpha,
            lora_dropout=self.lora_cfg.lora_dropout,
            bias=self.lora_cfg.bias,
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.lora_cfg.target_modules,
        )

        # get_peft_model works with both HuggingFace models and custom nn.Module
        try:
            model = get_peft_model(model, peft_cfg)
        except Exception as e:
            logger.warning(f"PEFT wrapping failed ({e}); applying manual LoRA.")
            model = self._apply_manual_lora(model)

        return model

    def _quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply 4-bit bitsandbytes quantisation for QLoRA."""
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
            import torch

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.lora_cfg.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=self.lora_cfg.bnb_4bit_use_double_quant,
            )
            # Replace Linear layers with bnb 4-bit equivalents
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    parent = dict(model.named_modules())[".".join(name.split(".")[:-1])] if "." in name else model
                    attr = name.split(".")[-1]
                    new_layer = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        quant_type=self.lora_cfg.bnb_4bit_quant_type,
                        compute_dtype=torch.bfloat16,
                    )
                    setattr(parent, attr, new_layer)
        except ImportError:
            logger.warning("bitsandbytes not available; QLoRA will use standard precision.")
        return model

    def _apply_manual_lora(self, model: nn.Module) -> nn.Module:
        """
        Lightweight manual LoRA injection for custom models that don't
        support the PEFT API.  Wraps nn.Linear layers that match target names.
        """
        targets = set(self.lora_cfg.target_modules)
        r = self.lora_cfg.r
        alpha = self.lora_cfg.lora_alpha
        dropout = self.lora_cfg.lora_dropout

        for name, module in list(model.named_modules()):
            short_name = name.split(".")[-1]
            if short_name in targets and isinstance(module, nn.Linear):
                lora = _ManualLoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                parent_name = ".".join(name.split(".")[:-1])
                parent = model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)
                setattr(parent, short_name, lora)

        # Freeze all except LoRA params
        for n, p in model.named_parameters():
            p.requires_grad = "lora_" in n

        return model

    # ------------------------------------------------------------------
    # Training / eval loop
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int, progress: Progress, task_id) -> None:
        self.model.train()
        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                out = self.model(input_ids, labels=labels)
                loss = out.loss / self.train_cfg.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.train_cfg.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    (p for p in self.model.parameters() if p.requires_grad),
                    self.train_cfg.max_grad_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.train_cfg.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    progress.print(
                        f"  step={self.global_step}  epoch={epoch}  "
                        f"loss={loss.item() * self.train_cfg.gradient_accumulation_steps:.4f}  lr={lr:.2e}"
                    )

                progress.advance(task_id)

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader, label: str = "Eval") -> float:
        self.model.eval()
        total, count = 0.0, 0
        for batch in loader:
            ids = batch["input_ids"].to(self.device)
            lbls = batch["labels"].to(self.device)
            with autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
                out = self.model(ids, labels=lbls)
            total += out.loss.item()
            count += 1
        avg = total / max(1, count)
        logger.info(f"  [{label}] loss={avg:.4f}  ppl={math.exp(min(avg, 20)):.2f}")
        return avg

    def _save_adapter(self, name: str) -> None:
        path = self.output_dir / name
        path.mkdir(exist_ok=True)
        try:
            self.model.save_pretrained(str(path))
        except Exception:
            torch.save(
                {n: p for n, p in self.model.named_parameters() if p.requires_grad},
                path / "adapter_weights.pt",
            )

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable,
            lr=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
            betas=(self.train_cfg.beta1, self.train_cfg.beta2),
        )

    def _build_scheduler(self, num_warmup: int):
        from torch.optim.lr_scheduler import LambdaLR

        def lr_fn(step):
            if step < num_warmup:
                return step / max(1, num_warmup)
            progress = (step - num_warmup) / max(1, self.total_steps - num_warmup)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(self.optimizer, lr_fn)

    def _log_trainable_params(self) -> None:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        pct = 100.0 * trainable / max(1, total)
        logger.info(
            f"LoRA trainable params: {trainable:,} / {total:,} ({pct:.2f}%)"
        )


# ---------------------------------------------------------------------------
# Manual LoRA Linear (fallback for custom models)
# ---------------------------------------------------------------------------


class _ManualLoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with trainable LoRA matrices A and B."""

    def __init__(self, linear: nn.Linear, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = linear
        self.base.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling
