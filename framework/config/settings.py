"""
Central configuration for the FastFail LLM/SLM Training Framework.
All hyper-parameters, domain presets, and export options live here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Domain(str, Enum):
    LANGUAGE_LEARNING = "language_learning"
    SPORTS = "sports"
    STOCKS = "stocks"
    RUNBOOKS = "runbooks"
    CHATBOT = "chatbot"
    PSYCHOTHERAPY = "psychotherapy"


class ModelSize(str, Enum):
    SLM = "slm"       # ~125 M parameters
    LLM = "llm"       # ~1 B parameters


class TrainingMode(str, Enum):
    PRETRAIN = "pretrain"          # Full training from scratch
    LORA = "lora"                  # LoRA fine-tune on existing checkpoint
    QLORA = "qlora"                # 4-bit QLoRA fine-tune
    FEEDBACK = "feedback"          # Feedback / continual fine-tune


class ExportFormat(str, Enum):
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    ONNX = "onnx"
    ALL = "all"


# ---------------------------------------------------------------------------
# Model architecture configs
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    # Architecture
    vocab_size: int = 32_000
    max_seq_len: int = 2048
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 12          # GQA; equals num_heads for vanilla MHA
    intermediate_size: int = 3072   # FFN hidden dim
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0    # RoPE base frequency
    tie_embeddings: bool = True

    # Size presets
    @classmethod
    def slm(cls) -> "ModelConfig":
        """~125 M parameter GPT-style SLM."""
        return cls(
            vocab_size=32_000,
            max_seq_len=2048,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            num_kv_heads=12,
            intermediate_size=3072,
            dropout=0.1,
        )

    @classmethod
    def llm(cls) -> "ModelConfig":
        """~1 B parameter LLaMA-style LLM."""
        return cls(
            vocab_size=32_000,
            max_seq_len=4096,
            hidden_size=2048,
            num_layers=22,
            num_heads=16,
            num_kv_heads=8,          # GQA
            intermediate_size=5632,
            dropout=0.0,
            rope_theta=500_000.0,
        )


# ---------------------------------------------------------------------------
# Training configs
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    # General
    output_dir: str = "outputs"
    seed: int = 42
    mixed_precision: str = "bf16"   # "no" | "fp16" | "bf16"
    gradient_checkpointing: bool = True

    # Optimiser
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Schedule
    lr_scheduler: str = "cosine"    # "cosine" | "linear" | "constant"
    warmup_ratio: float = 0.03

    # Batch / steps
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 3
    max_steps: int = -1             # -1 means use epochs

    # Evaluation / saving
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 50
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # DataLoader
    num_workers: int = 4
    train_split: float = 0.85
    val_split: float = 0.10
    test_split: float = 0.05        # must sum to 1.0


@dataclass
class LoRAConfig:
    r: int = 16                     # LoRA rank
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"              # "none" | "all" | "lora_only"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )
    # QLoRA extras
    load_in_4bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True


# ---------------------------------------------------------------------------
# Hyperparameter tuning config
# ---------------------------------------------------------------------------


@dataclass
class HPTConfig:
    n_trials: int = 30
    timeout: Optional[int] = 3600   # seconds; None = no limit
    direction: str = "minimize"     # minimise eval_loss
    sampler: str = "tpe"            # "tpe" | "random" | "cmaes"
    pruner: str = "hyperband"       # "hyperband" | "median" | "none"
    storage: Optional[str] = None   # Optuna DB URL; None = in-memory
    study_name: str = "fastfail_hpt"

    # Search space bounds
    lr_low: float = 1e-5
    lr_high: float = 1e-3
    wd_low: float = 0.0
    wd_high: float = 0.3
    batch_sizes: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    grad_accum_steps: List[int] = field(default_factory=lambda: [4, 8, 16])
    lora_r_choices: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    warmup_ratios: List[float] = field(default_factory=lambda: [0.01, 0.03, 0.05])


# ---------------------------------------------------------------------------
# Export config
# ---------------------------------------------------------------------------


@dataclass
class ExportConfig:
    format: str = "safetensors"     # ExportFormat values
    output_dir: str = "exported_models"
    quantize: bool = False          # Post-training quantization
    quantize_bits: int = 4          # 4 or 8
    push_to_hub: bool = False
    hub_repo_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Top-level framework config
# ---------------------------------------------------------------------------


@dataclass
class FrameworkConfig:
    domain: Domain = Domain.CHATBOT
    model_size: ModelSize = ModelSize.SLM
    training_mode: TrainingMode = TrainingMode.PRETRAIN
    base_model_path: Optional[str] = None   # checkpoint for fine-tuning
    data_dir: str = "data"
    model: ModelConfig = field(default_factory=ModelConfig.slm)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    hpt: HPTConfig = field(default_factory=HPTConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # Corpus
    max_corpus_tokens: int = 50_000_000     # 50 M token budget
    corpus_cache_dir: str = "data/raw"

    # Feedback loop
    feedback_data_path: Optional[str] = None
    feedback_epochs: int = 1


# ---------------------------------------------------------------------------
# Domain-aware defaults
# ---------------------------------------------------------------------------

_DOMAIN_MODEL_OVERRIDES: dict = {
    Domain.LANGUAGE_LEARNING: {"max_seq_len": 512,  "vocab_size": 32_000},
    Domain.SPORTS:            {"max_seq_len": 1024, "vocab_size": 32_000},
    Domain.STOCKS:            {"max_seq_len": 2048, "vocab_size": 32_000},
    Domain.RUNBOOKS:          {"max_seq_len": 4096, "vocab_size": 32_000},
    Domain.CHATBOT:           {"max_seq_len": 2048, "vocab_size": 32_000},
    Domain.PSYCHOTHERAPY:     {"max_seq_len": 2048, "vocab_size": 32_000},
}

_DOMAIN_DISPLAY: dict = {
    Domain.LANGUAGE_LEARNING: "Language Learning",
    Domain.SPORTS:            "Sports",
    Domain.STOCKS:            "Stocks & Finance",
    Domain.RUNBOOKS:          "Runbooks & Documentation",
    Domain.CHATBOT:           "General Chatbot",
    Domain.PSYCHOTHERAPY:     "Psychotherapy Assistant",
}


def get_default_config(domain: Domain, model_size: ModelSize) -> FrameworkConfig:
    """Return a sensible default FrameworkConfig for the given domain+size."""
    base_model_cfg = ModelConfig.slm() if model_size == ModelSize.SLM else ModelConfig.llm()

    # Apply domain-specific overrides
    overrides = _DOMAIN_MODEL_OVERRIDES.get(domain, {})
    for k, v in overrides.items():
        setattr(base_model_cfg, k, v)

    train_cfg = TrainingConfig()
    if model_size == ModelSize.LLM:
        train_cfg.per_device_train_batch_size = 2
        train_cfg.gradient_accumulation_steps = 16
        train_cfg.learning_rate = 1e-4
        train_cfg.gradient_checkpointing = True

    return FrameworkConfig(
        domain=domain,
        model_size=model_size,
        model=base_model_cfg,
        training=train_cfg,
    )


def domain_display_name(domain: Domain) -> str:
    return _DOMAIN_DISPLAY.get(domain, domain.value)
