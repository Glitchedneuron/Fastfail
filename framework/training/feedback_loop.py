"""
Feedback Loop / Continual Fine-Tuning.

Supports three input modes:
  1. FILE   — path to a JSONL / plain-text file with new examples
  2. STDIN  — interactive: user types examples in the terminal
  3. AUTO   — automatically fetch additional domain data and fine-tune

Each mode fine-tunes the existing model for a small number of epochs
using LoRA adapters (preserving base weights) to prevent catastrophic
forgetting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from loguru import logger
from rich.console import Console
from rich.prompt import Confirm, Prompt

from framework.config import FrameworkConfig, TrainingMode
from framework.corpus.base import CorpusResult
from framework.corpus.pipeline import DataPipeline

console = Console()


# ---------------------------------------------------------------------------
# FeedbackLoopTrainer
# ---------------------------------------------------------------------------


class FeedbackLoopTrainer:
    """
    Continual fine-tuning on new data or user feedback.

    The trainer:
      1. Loads new examples from the chosen source
      2. Tokenises them with the existing tokenizer
      3. Fine-tunes the model for feedback_epochs using LoRA
      4. Saves the updated adapter
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        framework_cfg: FrameworkConfig,
        output_dir: str = "outputs/feedback",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = framework_cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_from_file(self, file_path: str) -> None:
        """Fine-tune on examples from a JSONL or plain-text file."""
        texts = self._load_file(file_path)
        if not texts:
            console.print("[red]No usable examples found in file.[/red]")
            return
        console.print(f"[green]Loaded {len(texts)} examples from {file_path}[/green]")
        self._fine_tune(texts, label="file")

    def run_interactive(self) -> None:
        """Interactive REPL: user provides feedback examples in the terminal."""
        console.print(
            "\n[bold cyan]Feedback Loop — Interactive Mode[/bold cyan]\n"
            "Type or paste your training examples. "
            "Use [bold]ENTER twice[/bold] to submit each example. "
            "Type [bold]DONE[/bold] on a blank line to finish.\n"
        )
        texts: List[str] = []
        while True:
            lines: List[str] = []
            console.print("[yellow]Example (ENTER twice to submit, DONE to stop):[/yellow]")
            while True:
                line = sys.stdin.readline()
                stripped = line.strip()
                if stripped.upper() == "DONE":
                    break
                if stripped == "" and lines and lines[-1] == "":
                    lines.append("")
                    break
                lines.append(stripped)
            text = "\n".join(lines).strip()
            if text.upper() == "DONE" or (not text and Confirm.ask("Finished entering examples?")):
                break
            if text:
                texts.append(text)
                console.print(f"  [green]✓ Added example #{len(texts)}[/green]")

        if not texts:
            console.print("[yellow]No examples provided — skipping feedback loop.[/yellow]")
            return

        console.print(f"\n[bold green]{len(texts)} examples collected. Starting fine-tuning…[/bold green]\n")
        self._fine_tune(texts, label="interactive")

    def run_auto(self, extra_samples: int = 500) -> None:
        """
        Automatically fetch additional domain data and fine-tune.
        Useful for periodic model refresh.
        """
        from framework.corpus.domains import get_corpus_class

        console.print(
            f"[bold cyan]Auto feedback: fetching fresh {self.cfg.domain.value} corpus…[/bold cyan]"
        )
        corpus_cls = get_corpus_class(self.cfg.domain)
        corpus = corpus_cls(
            cache_dir=self.cfg.corpus_cache_dir,
            token_budget=extra_samples * 256,
        )
        result = corpus.get(force_refresh=True)

        if not result.texts:
            console.print("[yellow]No new data fetched.[/yellow]")
            return

        texts = result.texts[:extra_samples]
        console.print(f"[green]Fetched {len(texts)} fresh examples.[/green]")
        self._fine_tune(texts, label="auto")

    # ------------------------------------------------------------------
    # Internal fine-tuning
    # ------------------------------------------------------------------

    def _fine_tune(self, texts: List[str], label: str = "feedback") -> None:
        """Core: build loaders → LoRA fine-tune → save adapter."""
        from framework.training.lora_trainer import LoRATrainer

        # Build a tiny corpus result
        corpus = CorpusResult()
        for t in texts:
            corpus.append(t, source=f"feedback:{label}")

        if len(corpus) < 2:
            console.print("[red]Need at least 2 examples for fine-tuning.[/red]")
            return

        # Override training config for feedback (short, gentle update)
        train_cfg = self.cfg.training
        train_cfg.num_train_epochs = self.cfg.feedback_epochs
        train_cfg.max_steps = -1
        train_cfg.learning_rate = min(train_cfg.learning_rate, 5e-5)
        train_cfg.save_total_limit = 1

        pipeline = DataPipeline(train_cfg, self.cfg.model)
        train_loader, val_loader, test_loader = pipeline.build(
            corpus, self.tokenizer, seed=train_cfg.seed
        )

        use_qlora = self.cfg.training_mode == TrainingMode.QLORA
        trainer = LoRATrainer(
            self.model,
            train_loader,
            val_loader,
            test_loader,
            train_cfg,
            self.cfg.lora,
            output_dir=str(self.output_dir / label),
            use_qlora=use_qlora,
        )

        result = trainer.train()
        logger.info(
            f"Feedback fine-tune [{label}] complete — "
            f"test_loss={result.get('test_loss', 'N/A'):.4f}"
        )

        # Persist updated adapter
        adapter_path = self.output_dir / label / "final_adapter"
        adapter_path.mkdir(exist_ok=True)
        try:
            trainer.model.save_pretrained(str(adapter_path))
        except Exception:
            torch.save(
                {n: p for n, p in trainer.model.named_parameters() if p.requires_grad},
                adapter_path / "adapter_weights.pt",
            )

        console.print(
            f"\n[bold green]Feedback loop complete![/bold green] "
            f"Adapter saved → [cyan]{adapter_path}[/cyan]\n"
        )

    # ------------------------------------------------------------------
    # File loader
    # ------------------------------------------------------------------

    def _load_file(self, path: str) -> List[str]:
        p = Path(path)
        if not p.exists():
            logger.error(f"File not found: {path}")
            return []

        texts: List[str] = []

        if p.suffix == ".jsonl":
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        # Support multiple field conventions
                        text = (
                            obj.get("text")
                            or obj.get("content")
                            or obj.get("input")
                            or obj.get("output")
                            or obj.get("conversation")
                            or str(obj)
                        )
                        if text:
                            texts.append(str(text))
                    except json.JSONDecodeError:
                        texts.append(line)

        elif p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict):
                        t = item.get("text") or item.get("content") or item.get("input") or str(item)
                        texts.append(str(t))
            elif isinstance(data, dict):
                texts.append(str(data))

        else:
            # Plain text — split on double newlines (paragraphs)
            with open(p, "r", encoding="utf-8") as f:
                raw = f.read()
            texts = [t.strip() for t in raw.split("\n\n") if len(t.strip()) > 50]

        return texts
