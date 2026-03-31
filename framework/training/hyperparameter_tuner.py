"""
Hyperparameter tuning via Optuna.

Searches over:
  • Learning rate
  • Weight decay
  • Batch size + gradient accumulation
  • LoRA rank (when using LoRA/QLoRA)
  • Warmup ratio

Each trial runs a short training session (configurable steps) and reports
the validation loss to Optuna's pruner.
"""

from __future__ import annotations

import copy
import math
from typing import Callable, Dict, Optional

import torch
from loguru import logger
from rich.console import Console

from framework.config import (
    FrameworkConfig,
    HPTConfig,
    LoRAConfig,
    TrainingConfig,
    TrainingMode,
)
from framework.models import TransformerLM, build_model

console = Console()


class HyperparameterTuner:
    """
    Wraps Optuna to search for the best hyperparameters for a training job.

    Usage:
        tuner = HyperparameterTuner(cfg, corpus)
        best = tuner.run()
        print(best)
    """

    def __init__(
        self,
        framework_cfg: FrameworkConfig,
        corpus,
        tokenizer,
        steps_per_trial: int = 200,
    ):
        self.framework_cfg = framework_cfg
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.steps_per_trial = steps_per_trial
        self.hpt_cfg = framework_cfg.hpt

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Execute the Optuna study. Returns best hyperparameters dict."""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.error("Optuna not installed. Run: pip install optuna")
            raise

        # Build study
        sampler = self._build_sampler(optuna)
        pruner = self._build_pruner(optuna)

        study = optuna.create_study(
            study_name=self.hpt_cfg.study_name,
            direction=self.hpt_cfg.direction,
            storage=self.hpt_cfg.storage,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        console.print(
            f"\n[bold cyan]Starting HPT:[/bold cyan] "
            f"{self.hpt_cfg.n_trials} trials × {self.steps_per_trial} steps each\n"
        )

        study.optimize(
            self._objective,
            n_trials=self.hpt_cfg.n_trials,
            timeout=self.hpt_cfg.timeout,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        best = study.best_params
        logger.info(f"HPT complete — best params: {best}")
        logger.info(f"Best val_loss: {study.best_value:.4f}")

        self._print_results(study)
        return best

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def _objective(self, trial) -> float:
        cfg = copy.deepcopy(self.framework_cfg)

        # --- Sample hyperparameters ---
        cfg.training.learning_rate = trial.suggest_float(
            "lr", self.hpt_cfg.lr_low, self.hpt_cfg.lr_high, log=True
        )
        cfg.training.weight_decay = trial.suggest_float(
            "weight_decay", self.hpt_cfg.wd_low, self.hpt_cfg.wd_high
        )
        cfg.training.per_device_train_batch_size = trial.suggest_categorical(
            "batch_size", self.hpt_cfg.batch_sizes
        )
        cfg.training.gradient_accumulation_steps = trial.suggest_categorical(
            "grad_accum", self.hpt_cfg.grad_accum_steps
        )
        cfg.training.warmup_ratio = trial.suggest_categorical(
            "warmup_ratio", self.hpt_cfg.warmup_ratios
        )

        if cfg.training_mode in (TrainingMode.LORA, TrainingMode.QLORA):
            cfg.lora.r = trial.suggest_categorical("lora_r", self.hpt_cfg.lora_r_choices)
            cfg.lora.lora_alpha = cfg.lora.r * 2

        # Override to use only steps_per_trial steps for speed
        cfg.training.max_steps = self.steps_per_trial
        cfg.training.eval_steps = max(50, self.steps_per_trial // 4)
        cfg.training.num_train_epochs = 1

        try:
            val_loss = self._run_trial(cfg, trial)
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float("inf")

        return val_loss

    def _run_trial(self, cfg: FrameworkConfig, trial) -> float:
        """Short training run for one HPT trial."""
        from framework.corpus.pipeline import DataPipeline
        from framework.training.trainer import PreTrainer
        from framework.training.lora_trainer import LoRATrainer

        # Build data loaders (reuse tokenizer)
        pipeline = DataPipeline(cfg.training, cfg.model)
        train_loader, val_loader, test_loader = pipeline.build(
            self.corpus, self.tokenizer, seed=cfg.training.seed
        )

        # Build fresh model
        model = build_model(cfg.model_size, cfg.model)

        if cfg.training_mode == TrainingMode.PRETRAIN:
            trainer = PreTrainer(
                model, train_loader, val_loader, test_loader,
                cfg.training, output_dir=f"outputs/hpt_trial_{trial.number}"
            )
        else:
            trainer = LoRATrainer(
                model, train_loader, val_loader, test_loader,
                cfg.training, cfg.lora,
                output_dir=f"outputs/hpt_trial_{trial.number}",
                use_qlora=(cfg.training_mode == TrainingMode.QLORA),
            )

        # Monkey-patch to report intermediate values for pruning
        original_eval = trainer._evaluate

        def _pruning_eval(loader, label=""):
            loss = original_eval(loader, label)
            trial.report(loss, step=trainer.global_step)
            if trial.should_prune():
                import optuna
                raise optuna.exceptions.TrialPruned()
            return loss

        trainer._evaluate = _pruning_eval

        result = trainer.train()
        # Clean up GPU memory between trials
        del model, trainer
        torch.cuda.empty_cache()

        return result.get("test_loss", float("inf"))

    # ------------------------------------------------------------------
    # Optuna helpers
    # ------------------------------------------------------------------

    def _build_sampler(self, optuna):
        name = self.hpt_cfg.sampler.lower()
        if name == "tpe":
            return optuna.samplers.TPESampler(seed=42)
        if name == "random":
            return optuna.samplers.RandomSampler(seed=42)
        if name == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=42)
        return optuna.samplers.TPESampler(seed=42)

    def _build_pruner(self, optuna):
        name = self.hpt_cfg.pruner.lower()
        if name == "hyperband":
            return optuna.pruners.HyperbandPruner()
        if name == "median":
            return optuna.pruners.MedianPruner()
        return optuna.pruners.NopPruner()

    def _print_results(self, study) -> None:
        from rich.table import Table
        from rich import box

        t = Table(title="Top-5 HPT Trials", box=box.ROUNDED)
        t.add_column("Trial", style="cyan", justify="right")
        t.add_column("Val Loss", style="yellow", justify="right")
        t.add_column("Params", style="white")

        trials = sorted(study.trials, key=lambda x: x.value or float("inf"))[:5]
        for tr in trials:
            if tr.value is not None:
                t.add_row(
                    str(tr.number),
                    f"{tr.value:.4f}",
                    str({k: f"{v:.4g}" if isinstance(v, float) else v for k, v in tr.params.items()}),
                )
        console.print(t)
