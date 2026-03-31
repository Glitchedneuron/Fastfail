"""
Model exporter.

Exports a trained model to one or more formats for local inference:

  • safetensors  — HuggingFace-compatible, memory-mapped, safe to load
  • GGUF         — llama.cpp quantised format for CPU inference
  • ONNX         — cross-platform, e.g. for ONNX Runtime / edge deployment

All outputs land in the configured export directory together with:
  • tokenizer/     — saved tokenizer files
  • model_config.json  — architecture config
  • README_inference.md  — quick-start instructions
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from framework.config import ExportConfig, ModelConfig, ExportFormat

console = Console()


class ModelExporter:
    """Exports a trained TransformerLM (or any nn.Module) to local formats."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        model_cfg: ModelConfig,
        export_cfg: ExportConfig,
        domain_name: str = "custom",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_cfg = model_cfg
        self.export_cfg = export_cfg
        self.domain_name = domain_name

        self.out_dir = Path(export_cfg.output_dir) / domain_name
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self) -> Path:
        """
        Export according to ExportConfig.format.
        Returns the path to the export directory.
        """
        fmt = self.export_cfg.format.lower()

        console.print(f"\n[bold cyan]Exporting model → {self.out_dir}[/bold cyan]")

        # Always save tokenizer and config
        self._save_tokenizer()
        self._save_model_config()

        if fmt in (ExportFormat.SAFETENSORS, "all"):
            self._export_safetensors()
        if fmt in (ExportFormat.GGUF, "all"):
            self._export_gguf()
        if fmt in (ExportFormat.ONNX, "all"):
            self._export_onnx()
        if fmt == ExportFormat.SAFETENSORS:
            self._export_safetensors()

        self._write_inference_readme()

        console.print(
            Panel(
                f"[bold green]Export complete![/bold green]\n"
                f"Output directory: [cyan]{self.out_dir}[/cyan]\n\n"
                f"To run inference locally:\n"
                f"  [bold]python main.py infer --model {self.out_dir}[/bold]",
                title="Export Summary",
                border_style="green",
            )
        )
        return self.out_dir

    # ------------------------------------------------------------------
    # Format-specific exporters
    # ------------------------------------------------------------------

    def _export_safetensors(self) -> None:
        """Save model weights in safetensors format."""
        try:
            from safetensors.torch import save_file

            state = self.model.state_dict()
            path = self.out_dir / "model.safetensors"
            save_file(state, str(path))
            logger.info(f"safetensors → {path} ({path.stat().st_size / 1e6:.1f} MB)")
        except ImportError:
            # Fallback to torch .pt
            path = self.out_dir / "model.pt"
            torch.save(self.model.state_dict(), path)
            logger.warning(f"safetensors not installed; saved as .pt → {path}")

    def _export_gguf(self) -> None:
        """
        Convert to GGUF format using llama.cpp's convert script.

        Steps:
          1. Remap weight keys to LLaMA HuggingFace naming convention
          2. Save remapped weights as safetensors inside an HF-format directory
          3. Write a LlamaConfig-compatible config.json
          4. Call convert_hf_to_gguf.py
          5. Optionally quantise with llama-quantize

        If llama.cpp is not available, saves a plain .pt and prints
        instructions for manual conversion.
        """
        hf_dir = self.out_dir / "hf_checkpoint"
        hf_dir.mkdir(exist_ok=True)

        # Remap weight names and save as safetensors in the HF checkpoint dir
        self._save_remapped_safetensors(hf_dir)
        self._save_tokenizer(save_dir=hf_dir)
        self._save_llama_config(save_dir=hf_dir)

        # Try using llama-cpp-python's bundled convert utilities
        gguf_path = self.out_dir / "model.gguf"

        convert_script = self._find_llama_cpp_convert()
        if convert_script:
            try:
                cmd = [
                    sys.executable, str(convert_script),
                    str(hf_dir),
                    "--outfile", str(gguf_path),
                    "--outtype", "f16",
                ]
                logger.info(f"Running GGUF conversion: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if result.returncode == 0:
                    logger.info(f"GGUF → {gguf_path}")

                    # Optional quantisation
                    if self.export_cfg.quantize:
                        self._quantize_gguf(gguf_path)
                    return
                else:
                    logger.warning(f"Conversion failed:\n{result.stderr}")
            except Exception as e:
                logger.warning(f"GGUF conversion failed: {e}")

        # Fallback: instruct user
        logger.warning(
            "llama.cpp not found. To convert manually:\n"
            "  1. Install llama.cpp: pip install llama-cpp-python\n"
            f"  2. Run: python llama.cpp/convert_hf_to_gguf.py {hf_dir} "
            f"--outfile {gguf_path}"
        )
        # Still save safetensors as the primary artifact
        logger.info("Safetensors export is available as a fallback.")

    def _quantize_gguf(self, gguf_path: Path) -> None:
        """Run llama-quantize for post-training quantisation."""
        bits = self.export_cfg.quantize_bits
        qtype = "Q4_K_M" if bits == 4 else "Q8_0"
        q_path = gguf_path.with_name(f"model_{qtype}.gguf")

        quantize_bin = shutil.which("llama-quantize") or shutil.which("quantize")
        if quantize_bin:
            try:
                subprocess.run(
                    [quantize_bin, str(gguf_path), str(q_path), qtype],
                    check=True,
                    timeout=300,
                )
                logger.info(f"Quantised GGUF ({qtype}) → {q_path}")
            except Exception as e:
                logger.warning(f"Quantisation failed: {e}")
        else:
            logger.warning("llama-quantize not in PATH; skipping quantisation.")

    def _export_onnx(self) -> None:
        """Export to ONNX for cross-platform inference."""
        try:
            onnx_path = self.out_dir / "model.onnx"
            seq_len = min(32, self.model_cfg.max_seq_len)

            self.model.eval()
            dummy_input = torch.randint(0, self.model_cfg.vocab_size, (1, seq_len))

            torch.onnx.export(
                self.model,
                (dummy_input,),
                str(onnx_path),
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq_len"},
                    "logits": {0: "batch", 1: "seq_len"},
                },
                opset_version=17,
                do_constant_folding=True,
            )
            logger.info(f"ONNX → {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

            # Verify the export
            try:
                import onnx
                onnx.checker.check_model(str(onnx_path))
                logger.info("ONNX model verification passed.")
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_tokenizer(self, save_dir: Optional[Path] = None) -> None:
        dest = save_dir or (self.out_dir / "tokenizer")
        dest.mkdir(exist_ok=True)
        try:
            self.tokenizer.save_pretrained(str(dest))
            logger.info(f"Tokenizer → {dest}")
        except Exception as e:
            logger.warning(f"Tokenizer save failed: {e}")

    def _save_model_config(self, save_dir: Optional[Path] = None) -> None:
        dest = save_dir or self.out_dir
        config_path = dest / "model_config.json"
        try:
            with open(config_path, "w") as f:
                json.dump(asdict(self.model_cfg), f, indent=2)
            logger.info(f"Config → {config_path}")
        except Exception as e:
            logger.warning(f"Config save failed: {e}")

    # ------------------------------------------------------------------
    # GGUF helpers: weight remapping + LlamaConfig generation
    # ------------------------------------------------------------------

    def _remap_weights_for_gguf(self) -> dict:
        """
        Rename internal weight keys to the LLaMA HuggingFace convention that
        ``convert_hf_to_gguf.py`` expects.

        Our internal naming (from TransformerLM):
            embed_tokens.weight
            layers.{i}.attn_norm.weight
            layers.{i}.attn.q_proj.weight
            layers.{i}.attn.k_proj.weight
            layers.{i}.attn.v_proj.weight
            layers.{i}.attn.o_proj.weight
            layers.{i}.ffn_norm.weight
            layers.{i}.ffn.gate_proj.weight
            layers.{i}.ffn.up_proj.weight
            layers.{i}.ffn.down_proj.weight
            norm.weight
            lm_head.weight

        Target HuggingFace LLaMA naming:
            model.embed_tokens.weight
            model.layers.{i}.input_layernorm.weight
            model.layers.{i}.self_attn.q_proj.weight
            model.layers.{i}.self_attn.k_proj.weight
            model.layers.{i}.self_attn.v_proj.weight
            model.layers.{i}.self_attn.o_proj.weight
            model.layers.{i}.post_attention_layernorm.weight
            model.layers.{i}.mlp.gate_proj.weight
            model.layers.{i}.mlp.up_proj.weight
            model.layers.{i}.mlp.down_proj.weight
            model.norm.weight
            lm_head.weight
        """
        import re

        state = self.model.state_dict()
        remapped: dict = {}
        skipped: list = []

        # Static single-key renames
        _static = {
            "embed_tokens.weight": "model.embed_tokens.weight",
            "norm.weight":         "model.norm.weight",
            "lm_head.weight":      "lm_head.weight",
        }

        # Per-layer rename rules: (regex pattern, replacement template)
        _layer_rules = [
            # Attention norms
            (r"^layers\.(\d+)\.attn_norm\.weight$",
             r"model.layers.\1.input_layernorm.weight"),
            # Attention projections
            (r"^layers\.(\d+)\.attn\.(q_proj|k_proj|v_proj|o_proj)\.weight$",
             r"model.layers.\1.self_attn.\2.weight"),
            # FFN norms
            (r"^layers\.(\d+)\.ffn_norm\.weight$",
             r"model.layers.\1.post_attention_layernorm.weight"),
            # FFN projections
            (r"^layers\.(\d+)\.ffn\.(gate_proj|up_proj|down_proj)\.weight$",
             r"model.layers.\1.mlp.\2.weight"),
        ]

        for old_key, tensor in state.items():
            # Skip non-persistent buffers (e.g. _freqs)
            if old_key.startswith("_"):
                skipped.append(old_key)
                continue

            # Static renames
            if old_key in _static:
                remapped[_static[old_key]] = tensor
                continue

            # Pattern-based layer renames
            matched = False
            for pattern, replacement in _layer_rules:
                new_key, n_subs = re.subn(pattern, replacement, old_key)
                if n_subs > 0:
                    remapped[new_key] = tensor
                    matched = True
                    break

            if not matched:
                logger.warning(f"GGUF remap: no rule for key '{old_key}' — keeping as-is")
                remapped[old_key] = tensor

        if skipped:
            logger.debug(f"GGUF remap: skipped buffers: {skipped}")

        logger.info(
            f"GGUF remap: {len(state)} internal keys → {len(remapped)} HF keys "
            f"({len(skipped)} buffers skipped)"
        )
        return remapped

    def _save_remapped_safetensors(self, save_dir: Path) -> None:
        """Save GGUF-ready (remapped) weights as safetensors in *save_dir*."""
        remapped = self._remap_weights_for_gguf()
        path = save_dir / "model.safetensors"
        try:
            from safetensors.torch import save_file
            save_file(remapped, str(path))
        except ImportError:
            torch.save(remapped, str(path.with_suffix(".pt")))
            logger.warning("safetensors not installed; saved remapped weights as .pt")
        logger.info(f"Remapped weights → {path}")

    def _save_llama_config(self, save_dir: Path) -> None:
        """
        Write a ``config.json`` in LlamaConfig format so that
        ``convert_hf_to_gguf.py`` can parse the model's architecture without
        manual flags.

        Maps our ModelConfig fields to the expected LLaMA HF config keys.
        """
        cfg = self.model_cfg
        llama_cfg = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            # Dimensions
            "hidden_size":            cfg.hidden_size,
            "intermediate_size":      cfg.intermediate_size,
            "num_hidden_layers":      cfg.num_layers,
            "num_attention_heads":    cfg.num_heads,
            "num_key_value_heads":    cfg.num_kv_heads,   # GQA support
            "max_position_embeddings": cfg.max_seq_len,
            "vocab_size":             cfg.vocab_size,
            # Activations / norms
            "hidden_act":             "silu",
            "rms_norm_eps":           cfg.layer_norm_eps,
            # RoPE
            "rope_theta":             cfg.rope_theta,
            "rope_scaling":           None,
            # Misc
            "tie_word_embeddings":    cfg.tie_embeddings,
            "torch_dtype":            "bfloat16",
            "transformers_version":   "4.40.0",
            # Required by some versions of convert_hf_to_gguf.py
            "bos_token_id": 2,
            "eos_token_id": 2,
            "pad_token_id": 0,
        }
        config_path = save_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(llama_cfg, f, indent=2)
        logger.info(f"LlamaConfig → {config_path}")

    def _find_llama_cpp_convert(self) -> Optional[Path]:
        """Look for llama.cpp's convert_hf_to_gguf.py in common locations."""
        candidates = [
            Path("llama.cpp/convert_hf_to_gguf.py"),
            Path("convert_hf_to_gguf.py"),
        ]
        # Also check within the installed llama-cpp-python package
        try:
            import llama_cpp
            pkg_dir = Path(llama_cpp.__file__).parent
            candidates.append(pkg_dir / "convert_hf_to_gguf.py")
        except ImportError:
            pass

        for c in candidates:
            if c.exists():
                return c
        return None

    def _write_inference_readme(self) -> None:
        readme_path = self.out_dir / "README_inference.md"
        content = f"""# FastFail Model — {self.domain_name}

## Quick-start inference

### Python (safetensors)
```python
from framework.models import TransformerLM
from framework.config import ModelConfig
from transformers import PreTrainedTokenizerFast
import torch, json

# Load config
with open("{self.out_dir}/model_config.json") as f:
    cfg_dict = json.load(f)
from framework.config import ModelConfig
import dataclasses
cfg = ModelConfig(**cfg_dict)

# Load tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained("{self.out_dir}/tokenizer")

# Load model
model = TransformerLM(cfg)
from safetensors.torch import load_file
state = load_file("{self.out_dir}/model.safetensors")
model.load_state_dict(state)
model.eval()

# Generate
prompt = "Tell me about "
ids = tokenizer.encode(prompt, return_tensors="pt")
out = model.generate(ids, max_new_tokens=200)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### llama.cpp (GGUF)
```bash
./main -m {self.out_dir}/model.gguf -p "Tell me about " -n 200
```

### ONNX Runtime
```python
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession("{self.out_dir}/model.onnx")
input_ids = np.array([[1, 2, 3, 4]])  # tokenised prompt
logits = sess.run(["logits"], {{"input_ids": input_ids}})[0]
```

## CLI inference
```bash
python main.py infer --model {self.out_dir} --prompt "Your prompt here"
```
"""
        with open(readme_path, "w") as f:
            f.write(content)
        logger.info(f"Inference README → {readme_path}")
