"""
Infrastructure checker.

Inspects the current host (CPU, GPU, RAM, disk) and estimates the compute
requirements (time, power, VRAM) for a given training job *before* any work
begins.  Presents a Rich-formatted warning and a Proceed / Cancel prompt.
"""

from __future__ import annotations

import math
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import psutil
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from framework.config import FrameworkConfig, ModelSize, TrainingMode

console = Console()


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_gb: float
    free_memory_gb: float
    utilization_pct: float
    is_cuda: bool = True


@dataclass
class InfraReport:
    # Hardware
    cpu_name: str = "Unknown"
    cpu_cores_physical: int = 1
    cpu_cores_logical: int = 1
    cpu_freq_ghz: float = 0.0
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    disk_total_gb: float = 0.0
    disk_free_gb: float = 0.0
    gpus: List[GPUInfo] = field(default_factory=list)

    # Requirements
    required_vram_gb: float = 0.0
    required_ram_gb: float = 0.0
    required_disk_gb: float = 0.0
    estimated_hours: float = 0.0
    estimated_peak_watts: float = 0.0
    estimated_kwh: float = 0.0

    # Warnings
    warnings: List[str] = field(default_factory=list)
    can_proceed: bool = True


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

# Rough parameter count look-up (millions)
_PARAM_M = {
    ModelSize.SLM: 125,
    ModelSize.LLM: 1_000,
}

# Bytes per param under different training modes
_BYTES_PER_PARAM = {
    TrainingMode.PRETRAIN: 18,   # fp32 params + grads + Adam states ≈ 18 B/param
    TrainingMode.LORA:    10,    # fp16 base + LoRA extras
    TrainingMode.QLORA:    6,    # 4-bit base + LoRA extras
    TrainingMode.FEEDBACK: 10,
}

# GPU TFLOPs (bf16) for common cards — used for time estimation
_GPU_TFLOPS = {
    "a100": 312,
    "h100": 989,
    "v100": 125,
    "3090": 142,
    "4090": 330,
    "3080": 119,
    "4080": 240,
    "t4":    65,
    "cpu":    1,   # very rough single-core fp32 baseline
}

# TDP (Watts) estimates
_GPU_TDP = {
    "a100": 400,
    "h100": 700,
    "v100": 300,
    "3090": 350,
    "4090": 450,
    "3080": 320,
    "4080": 320,
    "t4":   70,
    "cpu":  95,
}


def _gpu_key(name: str) -> str:
    """Map a GPU name string to a TFLOPs look-up key."""
    n = name.lower()
    for key in _GPU_TFLOPS:
        if key in n:
            return key
    return "cpu"


def _estimate_flops(params_m: int, tokens: int) -> float:
    """Approximate training FLOPs: 6 × params × tokens (Chinchilla rule)."""
    return 6.0 * (params_m * 1e6) * tokens


# ---------------------------------------------------------------------------
# InfraChecker
# ---------------------------------------------------------------------------


class InfraChecker:
    """Collects hardware info and estimates training cost."""

    def __init__(self, cfg: FrameworkConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(self) -> InfraReport:
        report = InfraReport()
        self._collect_cpu(report)
        self._collect_ram(report)
        self._collect_disk(report)
        self._collect_gpus(report)
        self._estimate_requirements(report)
        self._check_feasibility(report)
        return report

    # ------------------------------------------------------------------
    # Hardware collectors
    # ------------------------------------------------------------------

    def _collect_cpu(self, r: InfraReport) -> None:
        try:
            import cpuinfo
            info = cpuinfo.get_cpu_info()
            r.cpu_name = info.get("brand_raw", platform.processor())
        except Exception:
            r.cpu_name = platform.processor() or "Unknown CPU"

        r.cpu_cores_physical = psutil.cpu_count(logical=False) or 1
        r.cpu_cores_logical = psutil.cpu_count(logical=True) or 1
        freq = psutil.cpu_freq()
        r.cpu_freq_ghz = round((freq.max if freq else 0) / 1000, 2)

    def _collect_ram(self, r: InfraReport) -> None:
        mem = psutil.virtual_memory()
        r.ram_total_gb = round(mem.total / 1e9, 1)
        r.ram_available_gb = round(mem.available / 1e9, 1)

    def _collect_disk(self, r: InfraReport) -> None:
        usage = shutil.disk_usage(os.getcwd())
        r.disk_total_gb = round(usage.total / 1e9, 1)
        r.disk_free_gb = round(usage.free / 1e9, 1)

    def _collect_gpus(self, r: InfraReport) -> None:
        # Try torch first (most reliable VRAM info)
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    total = props.total_memory / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    free = total - reserved
                    r.gpus.append(GPUInfo(
                        index=i,
                        name=props.name,
                        total_memory_gb=round(total, 1),
                        free_memory_gb=round(free, 1),
                        utilization_pct=0.0,
                        is_cuda=True,
                    ))
                return
        except Exception:
            pass

        # Fallback: GPUtil
        try:
            import GPUtil
            for g in GPUtil.getGPUs():
                r.gpus.append(GPUInfo(
                    index=g.id,
                    name=g.name,
                    total_memory_gb=round(g.memoryTotal / 1024, 1),
                    free_memory_gb=round(g.memoryFree / 1024, 1),
                    utilization_pct=g.load * 100,
                ))
        except Exception:
            pass  # No GPU — will warn later

    # ------------------------------------------------------------------
    # Requirement estimator
    # ------------------------------------------------------------------

    def _estimate_requirements(self, r: InfraReport) -> None:
        cfg = self.cfg
        params_m = _PARAM_M[cfg.model_size]
        bpp = _BYTES_PER_PARAM[cfg.training_mode]

        # VRAM / RAM needed
        r.required_vram_gb = round((params_m * 1e6 * bpp) / 1e9, 1)
        r.required_ram_gb = round(r.required_vram_gb * 1.5, 1)

        # Disk: raw corpus + tokenised dataset + checkpoints
        corpus_gb = round(cfg.max_corpus_tokens / 1e9 * 2, 1)  # ~2 bytes/token raw
        checkpoints_gb = round(r.required_vram_gb * cfg.training.save_total_limit, 1)
        r.required_disk_gb = round(corpus_gb + checkpoints_gb + 5, 1)

        # Time estimate
        tokens = cfg.max_corpus_tokens * cfg.training.num_train_epochs
        total_flops = _estimate_flops(params_m, tokens)

        if r.gpus:
            key = _gpu_key(r.gpus[0].name)
            tflops = _GPU_TFLOPS.get(key, 50)
            num_gpus = len(r.gpus)
            # Assume 35% MFU (model flop utilisation) — realistic for single-node
            effective_tflops = tflops * num_gpus * 0.35
            r.estimated_hours = round(total_flops / (effective_tflops * 1e12 * 3600), 1)
            tdp = _GPU_TDP.get(key, 200) * num_gpus
        else:
            # CPU only — very slow
            effective_tflops = _GPU_TFLOPS["cpu"] * 0.5
            r.estimated_hours = round(total_flops / (effective_tflops * 1e12 * 3600), 1)
            tdp = 95  # CPU TDP

        r.estimated_peak_watts = tdp
        r.estimated_kwh = round(r.estimated_hours * tdp / 1000, 1)

    # ------------------------------------------------------------------
    # Feasibility checks
    # ------------------------------------------------------------------

    def _check_feasibility(self, r: InfraReport) -> None:
        # GPU VRAM
        if r.gpus:
            total_free_vram = sum(g.free_memory_gb for g in r.gpus)
            if total_free_vram < r.required_vram_gb:
                r.warnings.append(
                    f"Insufficient VRAM: need {r.required_vram_gb} GB, "
                    f"have {total_free_vram:.1f} GB free. "
                    "Consider switching to QLoRA or reducing batch size."
                )
        else:
            r.warnings.append(
                "No GPU detected — training will be extremely slow on CPU."
            )

        # System RAM
        if r.ram_available_gb < r.required_ram_gb:
            r.warnings.append(
                f"Low system RAM: need {r.required_ram_gb} GB, "
                f"have {r.ram_available_gb:.1f} GB available."
            )

        # Disk
        if r.disk_free_gb < r.required_disk_gb:
            r.warnings.append(
                f"Insufficient disk: need {r.required_disk_gb} GB, "
                f"have {r.disk_free_gb:.1f} GB free."
            )
            r.can_proceed = False   # Hard block — can't store data

        # Estimated time sanity
        if r.estimated_hours > 168:  # > 1 week
            r.warnings.append(
                f"Estimated training time is {r.estimated_hours:.0f} hours "
                "(>1 week). Consider reducing corpus size or using LoRA/QLoRA."
            )


# ---------------------------------------------------------------------------
# Rich rendering helpers
# ---------------------------------------------------------------------------


def _render_hardware_table(r: InfraReport) -> Table:
    t = Table(title="System Hardware", box=box.ROUNDED, show_lines=True)
    t.add_column("Component", style="cyan bold", no_wrap=True)
    t.add_column("Details", style="white")
    t.add_column("Status", justify="center")

    def _ok(val: float, need: float) -> str:
        return "[green]OK[/green]" if val >= need else "[red]LOW[/red]"

    t.add_row("CPU", f"{r.cpu_name} ({r.cpu_cores_physical}P/{r.cpu_cores_logical}L cores @ {r.cpu_freq_ghz} GHz)", "")
    t.add_row(
        "RAM",
        f"{r.ram_available_gb} GB free / {r.ram_total_gb} GB total",
        _ok(r.ram_available_gb, r.required_ram_gb),
    )
    t.add_row(
        "Disk",
        f"{r.disk_free_gb} GB free / {r.disk_total_gb} GB total",
        _ok(r.disk_free_gb, r.required_disk_gb),
    )

    if r.gpus:
        for g in r.gpus:
            t.add_row(
                f"GPU {g.index}",
                f"{g.name} | {g.free_memory_gb} GB free / {g.total_memory_gb} GB",
                _ok(g.free_memory_gb, r.required_vram_gb / len(r.gpus)),
            )
    else:
        t.add_row("GPU", "[yellow]None detected[/yellow]", "[yellow]WARN[/yellow]")

    return t


def _render_requirements_table(r: InfraReport) -> Table:
    t = Table(title="Training Requirements", box=box.ROUNDED, show_lines=True)
    t.add_column("Metric", style="cyan bold", no_wrap=True)
    t.add_column("Estimate", style="yellow bold", justify="right")

    t.add_row("VRAM needed", f"{r.required_vram_gb} GB")
    t.add_row("System RAM needed", f"{r.required_ram_gb} GB")
    t.add_row("Disk space needed", f"{r.required_disk_gb} GB")
    t.add_row("Estimated training time", f"~{r.estimated_hours:.1f} hours")
    t.add_row("Peak power draw", f"~{r.estimated_peak_watts} W")
    t.add_row("Estimated energy", f"~{r.estimated_kwh} kWh")

    return t


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def check_and_confirm(cfg: FrameworkConfig) -> bool:
    """
    Run the infrastructure check, display a Rich report, and ask the user
    whether to proceed.

    Returns True if the user chooses to proceed, False to cancel.
    """
    console.rule("[bold yellow]Infrastructure Check[/bold yellow]")
    console.print()

    checker = InfraChecker(cfg)
    with console.status("[bold green]Inspecting system…[/bold green]"):
        report = checker.collect()

    console.print(_render_hardware_table(report))
    console.print()
    console.print(_render_requirements_table(report))
    console.print()

    # Warnings panel
    if report.warnings:
        warn_text = "\n".join(f"[yellow]⚠[/yellow]  {w}" for w in report.warnings)
        console.print(Panel(warn_text, title="[bold red]Warnings[/bold red]", border_style="red"))
        console.print()

    # Hard-block
    if not report.can_proceed:
        console.print(
            Panel(
                "[bold red]Training CANNOT proceed — critical resource shortage detected.\n"
                "Free up disk space and re-run.[/bold red]",
                border_style="red",
            )
        )
        return False

    # Summary line
    hours_str = f"[bold yellow]{report.estimated_hours:.1f} hours[/bold yellow]"
    power_str = f"[bold yellow]{report.estimated_kwh:.1f} kWh[/bold yellow]"
    console.print(
        Panel(
            f"Estimated training will take {hours_str} and consume {power_str} of energy.",
            title="[bold cyan]Proceed?[/bold cyan]",
            border_style="cyan",
        )
    )

    # Proceed / Cancel prompt
    try:
        import questionary
        answer = questionary.confirm(
            "Do you want to start training?",
            default=False,
        ).ask()
        if answer is None:
            answer = False
    except (ImportError, Exception):
        # Fallback to plain input
        choice = input("\n  [P]roceed  /  [C]ancel  > ").strip().lower()
        answer = choice.startswith("p")

    if answer:
        console.print("\n[bold green]Starting training…[/bold green]\n")
    else:
        console.print("\n[bold red]Training cancelled.[/bold red]\n")

    return bool(answer)
