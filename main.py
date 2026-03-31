#!/usr/bin/env python3
"""
FastFail LLM/SLM Training Framework — Main Entry Point

Usage:
    python main.py train                    # Interactive wizard
    python main.py train --domain sports --size slm --mode pretrain
    python main.py tune                     # Hyperparameter tuning
    python main.py feedback --input data.jsonl
    python main.py feedback --interactive
    python main.py feedback --auto
    python main.py eval  --model outputs/best_model
    python main.py infer --model outputs/best_model --prompt "Hello"
    python main.py export --model outputs/best_model --format safetensors
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_BANNER = """
[bold cyan]
  ███████╗ █████╗ ███████╗████████╗███████╗ █████╗ ██╗██╗
  ██╔════╝██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔══██╗██║██║
  █████╗  ███████║███████╗   ██║   █████╗  ███████║██║██║
  ██╔══╝  ██╔══██║╚════██║   ██║   ██╔══╝  ██╔══██║██║██║
  ██║     ██║  ██║███████║   ██║   ██║     ██║  ██║██║███████╗
  ╚═╝     ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝
[/bold cyan]
[bold white]  LLM/SLM Training Framework  ·  v1.0.0[/bold white]
"""


def _print_banner() -> None:
    console.print(_BANNER)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _select_domain():
    from framework.config import Domain, domain_display_name
    try:
        import questionary
        choices = [
            questionary.Choice(title=domain_display_name(d), value=d.value)
            for d in Domain
        ]
        val = questionary.select("Select domain:", choices=choices).ask()
        return Domain(val)
    except Exception:
        console.print("\n[bold]Available domains:[/bold]")
        for i, d in enumerate(Domain, 1):
            console.print(f"  {i}. {domain_display_name(d)}")
        idx = int(input("Enter number: ").strip()) - 1
        return list(Domain)[idx]


def _select_model_size():
    from framework.config import ModelSize
    try:
        import questionary
        choices = [
            questionary.Choice(title="SLM  (~125 M params, faster)", value="slm"),
            questionary.Choice(title="LLM  (~1 B params, more capable)", value="llm"),
        ]
        val = questionary.select("Select model size:", choices=choices).ask()
        return ModelSize(val)
    except Exception:
        choice = input("Model size [slm/llm]: ").strip().lower()
        return ModelSize(choice)


def _select_training_mode():
    from framework.config import TrainingMode
    try:
        import questionary
        choices = [
            questionary.Choice("Pre-train from scratch", value="pretrain"),
            questionary.Choice("LoRA fine-tune (existing checkpoint)", value="lora"),
            questionary.Choice("QLoRA fine-tune (4-bit, low VRAM)", value="qlora"),
            questionary.Choice("Feedback / continual fine-tune", value="feedback"),
        ]
        val = questionary.select("Select training mode:", choices=choices).ask()
        return TrainingMode(val)
    except Exception:
        modes = [m.value for m in TrainingMode]
        console.print(f"Modes: {modes}")
        val = input("Enter mode: ").strip().lower()
        return TrainingMode(val)


def _load_model_and_tokenizer(model_path: str, cfg=None):
    """Load a saved model checkpoint and its tokenizer."""
    import torch
    import json
    from pathlib import Path
    from framework.models import TransformerLM
    from framework.config import ModelConfig
    from transformers import PreTrainedTokenizerFast

    model_path = Path(model_path)

    # Load config
    config_file = model_path / "model_config.json"
    if config_file.exists():
        with open(config_file) as f:
            cfg_dict = json.load(f)
        import dataclasses
        model_cfg = ModelConfig(**{k: v for k, v in cfg_dict.items() if k in {f.name for f in dataclasses.fields(ModelConfig)}})
    elif cfg is not None:
        model_cfg = cfg.model
    else:
        raise FileNotFoundError(f"model_config.json not found in {model_path}")

    # Load tokenizer
    tok_dir = model_path / "tokenizer"
    if not tok_dir.exists():
        tok_dir = model_path
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))

    # Load model weights
    model = TransformerLM(model_cfg)
    ckpt_file = model_path / "checkpoint.pt"
    safetensors_file = model_path / "model.safetensors"

    if safetensors_file.exists():
        from safetensors.torch import load_file
        state = load_file(str(safetensors_file))
        model.load_state_dict(state, strict=False)
    elif ckpt_file.exists():
        ckpt = torch.load(str(ckpt_file), map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        raise FileNotFoundError(f"No model weights found in {model_path}")

    return model, tokenizer, model_cfg


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@click.group()
def cli():
    """FastFail — Domain-specific LLM/SLM training framework."""
    pass


# ── train ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--domain", type=str, default=None, help="Domain name (e.g. sports, stocks)")
@click.option("--size", type=click.Choice(["slm", "llm"]), default=None)
@click.option("--mode", type=click.Choice(["pretrain", "lora", "qlora", "feedback"]), default=None)
@click.option("--base-model", type=str, default=None, help="Path to base model for fine-tuning")
@click.option("--output-dir", type=str, default="outputs", help="Training output directory")
@click.option("--skip-infra-check", is_flag=True, default=False)
@click.option("--corpus-refresh", is_flag=True, default=False, help="Force re-fetch corpus")
@click.option("--run-hpt", is_flag=True, default=False, help="Run HPT before training")
def train(domain, size, mode, base_model, output_dir, skip_infra_check, corpus_refresh, run_hpt):
    """Train or fine-tune a model from scratch."""
    _print_banner()

    from framework.config import (
        Domain, ModelSize, TrainingMode, get_default_config, domain_display_name
    )
    from framework.infra import check_and_confirm
    from framework.corpus.domains import get_corpus_class
    from framework.corpus.pipeline import DataPipeline
    from framework.models import build_model
    from framework.training import PreTrainer, LoRATrainer, HyperparameterTuner
    from framework.evaluation import ModelEvaluator
    from framework.export import ModelExporter

    # --- Wizard for missing options ---
    if domain is None:
        domain_enum = _select_domain()
    else:
        domain_enum = Domain(domain)

    if size is None:
        size_enum = _select_model_size()
    else:
        size_enum = ModelSize(size)

    if mode is None:
        mode_enum = _select_training_mode()
    else:
        mode_enum = TrainingMode(mode)

    console.print(
        f"\n[bold green]Configuration:[/bold green]\n"
        f"  Domain  : [cyan]{domain_display_name(domain_enum)}[/cyan]\n"
        f"  Size    : [cyan]{size_enum.value.upper()}[/cyan]\n"
        f"  Mode    : [cyan]{mode_enum.value}[/cyan]\n"
    )

    cfg = get_default_config(domain_enum, size_enum)
    cfg.training_mode = mode_enum
    cfg.training.output_dir = output_dir
    if base_model:
        cfg.base_model_path = base_model

    # --- Infrastructure check ---
    if not skip_infra_check:
        if not check_and_confirm(cfg):
            sys.exit(0)

    # --- Corpus ---
    console.rule("[bold yellow]Step 1: Corpus[/bold yellow]")
    corpus_cls = get_corpus_class(domain_enum)
    corpus = corpus_cls(
        cache_dir=cfg.corpus_cache_dir,
        token_budget=cfg.max_corpus_tokens,
    ).get(force_refresh=corpus_refresh)
    console.print(
        f"  Documents : [green]{len(corpus):,}[/green]\n"
        f"  Approx tokens : [green]{corpus.total_tokens_approx:,}[/green]\n"
    )

    # --- Tokenizer ---
    console.rule("[bold yellow]Step 2: Tokenizer[/bold yellow]")
    tokenizer = DataPipeline.load_or_train_tokenizer(
        corpus.texts,
        vocab_size=cfg.model.vocab_size,
        save_path=f"{cfg.data_dir}/tokenizer",
    )
    # Sync vocab size in case tokenizer shrunk the vocabulary
    cfg.model.vocab_size = tokenizer.vocab_size

    # --- Data pipeline ---
    console.rule("[bold yellow]Step 3: Data Pipeline[/bold yellow]")
    pipeline = DataPipeline(cfg.training, cfg.model)
    train_loader, val_loader, test_loader = pipeline.build(corpus, tokenizer)

    # --- HPT (optional) ---
    if run_hpt:
        console.rule("[bold yellow]Step 3.5: Hyperparameter Tuning[/bold yellow]")
        tuner = HyperparameterTuner(cfg, corpus, tokenizer, steps_per_trial=100)
        best_params = tuner.run()
        # Apply best params to cfg
        if "lr" in best_params:
            cfg.training.learning_rate = best_params["lr"]
        if "weight_decay" in best_params:
            cfg.training.weight_decay = best_params["weight_decay"]
        if "batch_size" in best_params:
            cfg.training.per_device_train_batch_size = best_params["batch_size"]
        if "lora_r" in best_params:
            cfg.lora.r = best_params["lora_r"]
        console.print("[green]Best HPT params applied to training config.[/green]")

    # --- Model ---
    console.rule("[bold yellow]Step 4: Model[/bold yellow]")
    if mode_enum == TrainingMode.PRETRAIN or base_model is None:
        model = build_model(size_enum, cfg.model)
    else:
        model, _, _ = _load_model_and_tokenizer(base_model, cfg)
    console.print(f"  {model}")

    # --- Training ---
    console.rule("[bold yellow]Step 5: Training[/bold yellow]")
    if mode_enum == TrainingMode.PRETRAIN:
        trainer = PreTrainer(model, train_loader, val_loader, test_loader, cfg.training, output_dir)
        result = trainer.train()
    else:
        use_qlora = (mode_enum == TrainingMode.QLORA)
        trainer = LoRATrainer(
            model, train_loader, val_loader, test_loader,
            cfg.training, cfg.lora,
            output_dir=output_dir,
            use_qlora=use_qlora,
        )
        result = trainer.train()

    # --- Evaluation ---
    console.rule("[bold yellow]Step 6: Evaluation[/bold yellow]")
    evaluator = ModelEvaluator(trainer.model, tokenizer, domain_enum)
    eval_results = evaluator.full_evaluation(test_loader)

    # --- Export ---
    console.rule("[bold yellow]Step 7: Export[/bold yellow]")
    exporter = ModelExporter(
        trainer.model,
        tokenizer,
        cfg.model,
        cfg.export,
        domain_name=domain_enum.value,
    )
    export_path = exporter.export()

    console.print(
        Panel(
            f"[bold green]Training pipeline complete![/bold green]\n\n"
            f"Model saved to: [cyan]{export_path}[/cyan]\n\n"
            f"Run inference:\n"
            f"  [bold]python main.py infer --model {export_path} --prompt 'Your prompt'[/bold]",
            border_style="green",
        )
    )


# ── tune ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--domain", type=str, required=True)
@click.option("--size", type=click.Choice(["slm", "llm"]), default="slm")
@click.option("--mode", type=click.Choice(["pretrain", "lora", "qlora"]), default="pretrain")
@click.option("--trials", type=int, default=30)
@click.option("--steps-per-trial", type=int, default=200)
def tune(domain, size, mode, trials, steps_per_trial):
    """Run hyperparameter tuning only."""
    _print_banner()

    from framework.config import Domain, ModelSize, TrainingMode, get_default_config
    from framework.corpus.domains import get_corpus_class
    from framework.corpus.pipeline import DataPipeline
    from framework.training import HyperparameterTuner

    domain_enum = Domain(domain)
    size_enum = ModelSize(size)
    cfg = get_default_config(domain_enum, size_enum)
    cfg.training_mode = TrainingMode(mode)
    cfg.hpt.n_trials = trials

    corpus = get_corpus_class(domain_enum)(
        cache_dir=cfg.corpus_cache_dir,
        token_budget=cfg.max_corpus_tokens,
    ).get()

    tokenizer = DataPipeline.load_or_train_tokenizer(
        corpus.texts,
        vocab_size=cfg.model.vocab_size,
        save_path=f"{cfg.data_dir}/tokenizer",
    )
    cfg.model.vocab_size = tokenizer.vocab_size

    tuner = HyperparameterTuner(cfg, corpus, tokenizer, steps_per_trial=steps_per_trial)
    best = tuner.run()
    console.print(f"\n[bold green]Best params:[/bold green] {best}")


# ── feedback ───────────────────────────────────────────────────────────────

@cli.command()
@click.option("--model", required=True, type=str, help="Path to trained model directory")
@click.option("--input", "input_file", type=str, default=None, help="Path to JSONL/text feedback file")
@click.option("--interactive", is_flag=True, default=False, help="Enter feedback interactively")
@click.option("--auto", is_flag=True, default=False, help="Auto-fetch new domain data")
@click.option("--domain", type=str, default=None, help="Domain (required for --auto)")
@click.option("--output-dir", type=str, default="outputs/feedback")
def feedback(model, input_file, interactive, auto, domain, output_dir):
    """Fine-tune model on new data or interactive feedback."""
    _print_banner()

    from framework.config import Domain, get_default_config, ModelSize
    from framework.training.feedback_loop import FeedbackLoopTrainer

    loaded_model, tokenizer, model_cfg = _load_model_and_tokenizer(model)

    # Build a minimal cfg for feedback
    d = Domain(domain) if domain else Domain.CHATBOT
    cfg = get_default_config(d, ModelSize.SLM)
    cfg.model = model_cfg

    trainer = FeedbackLoopTrainer(loaded_model, tokenizer, cfg, output_dir=output_dir)

    if input_file:
        trainer.run_from_file(input_file)
    elif interactive:
        trainer.run_interactive()
    elif auto:
        trainer.run_auto()
    else:
        console.print("[red]Specify one of: --input FILE, --interactive, or --auto[/red]")
        sys.exit(1)


# ── eval ───────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--model", required=True, type=str)
@click.option("--domain", type=str, default="chatbot")
@click.option("--data-dir", type=str, default="data/processed")
def eval(model, domain, data_dir):
    """Evaluate a trained model on its test set."""
    _print_banner()

    import torch
    from framework.config import Domain, get_default_config, ModelSize
    from framework.corpus.domains import get_corpus_class
    from framework.corpus.pipeline import DataPipeline
    from framework.evaluation import ModelEvaluator

    domain_enum = Domain(domain)
    loaded_model, tokenizer, model_cfg = _load_model_and_tokenizer(model)
    cfg = get_default_config(domain_enum, ModelSize.SLM)
    cfg.model = model_cfg

    corpus = get_corpus_class(domain_enum)(cache_dir=cfg.corpus_cache_dir).get()
    pipeline = DataPipeline(cfg.training, model_cfg)
    _, _, test_loader = pipeline.build(corpus, tokenizer)

    evaluator = ModelEvaluator(loaded_model, tokenizer, domain_enum)
    evaluator.full_evaluation(test_loader)


# ── infer ──────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--model", required=True, type=str)
@click.option("--prompt", type=str, default=None)
@click.option("--max-tokens", type=int, default=200)
@click.option("--temperature", type=float, default=0.8)
@click.option("--interactive", is_flag=True, default=False, help="Chat-style REPL")
def infer(model, prompt, max_tokens, temperature, interactive):
    """Run inference on a trained model."""
    _print_banner()

    import torch
    loaded_model, tokenizer, _ = _load_model_and_tokenizer(model)
    loaded_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded_model.to(device)

    def _generate(text: str) -> str:
        ids = tokenizer.encode(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = loaded_model.generate(
                ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    if interactive:
        console.print("\n[bold cyan]Inference REPL[/bold cyan]  (Ctrl+C to exit)\n")
        while True:
            try:
                user_input = input("[You]: ").strip()
                if not user_input:
                    continue
                response = _generate(user_input)
                console.print(f"[bold green][Model]:[/bold green] {response}\n")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Bye.[/yellow]")
                break
    elif prompt:
        response = _generate(prompt)
        console.print(f"\n[bold green]Response:[/bold green]\n{response}\n")
    else:
        console.print("[red]Provide --prompt TEXT or use --interactive[/red]")


# ── export ─────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--model", required=True, type=str)
@click.option("--format", "fmt", type=click.Choice(["safetensors", "gguf", "onnx", "all"]), default="safetensors")
@click.option("--output-dir", type=str, default="exported_models")
@click.option("--quantize", is_flag=True, default=False)
@click.option("--quantize-bits", type=int, default=4)
@click.option("--domain", type=str, default="custom")
def export(model, fmt, output_dir, quantize, quantize_bits, domain):
    """Export a trained model to safetensors / GGUF / ONNX."""
    _print_banner()

    from framework.config import ExportConfig
    from framework.export import ModelExporter

    loaded_model, tokenizer, model_cfg = _load_model_and_tokenizer(model)

    export_cfg = ExportConfig(
        format=fmt,
        output_dir=output_dir,
        quantize=quantize,
        quantize_bits=quantize_bits,
    )

    exporter = ModelExporter(loaded_model, tokenizer, model_cfg, export_cfg, domain_name=domain)
    out = exporter.export()
    console.print(f"\n[bold green]Exported to:[/bold green] {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
