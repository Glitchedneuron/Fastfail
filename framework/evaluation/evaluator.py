"""
Model Evaluator.

Computes:
  • Perplexity (primary metric)
  • BLEU score (for generation quality)
  • ROUGE-L (for summarization / generation)
  • Domain-specific task metrics
  • Token generation throughput (tokens/sec)
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box

from framework.config import Domain

console = Console()


class ModelEvaluator:
    """
    Evaluates a trained model on held-out test data.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        domain: Domain,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.domain = domain
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def full_evaluation(
        self,
        test_loader: DataLoader,
        num_generation_samples: int = 10,
        use_amp: bool = True,
    ) -> Dict:
        """Run all evaluation metrics and print a Rich summary table."""
        results: Dict = {}

        # 1. Perplexity
        ppl = self.compute_perplexity(test_loader, use_amp=use_amp)
        results["perplexity"] = ppl
        results["test_loss"] = math.log(ppl)

        # 2. Generation throughput
        tps = self.compute_throughput()
        results["tokens_per_second"] = tps

        # 3. BLEU / ROUGE on a small sample
        gen_metrics = self.compute_generation_metrics(
            test_loader, n_samples=num_generation_samples, use_amp=use_amp
        )
        results.update(gen_metrics)

        # 4. Domain-specific
        domain_metrics = self.domain_specific_eval()
        results.update(domain_metrics)

        self._print_results(results)
        return results

    def compute_perplexity(self, loader: DataLoader, use_amp: bool = True) -> float:
        """Compute perplexity on the full test set."""
        amp_dtype = torch.bfloat16
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                with autocast(
                    device_type=self.device.type, dtype=amp_dtype, enabled=use_amp and self.device.type == "cuda"
                ):
                    out = self.model(input_ids, labels=labels)

                n_tokens = (labels != -100).sum().item()
                total_loss += out.loss.item() * n_tokens
                total_tokens += n_tokens

        avg_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(min(avg_loss, 20))
        logger.info(f"Perplexity: {ppl:.2f}  (loss={avg_loss:.4f})")
        return ppl

    def compute_throughput(self, prompt: str = "Hello, how are you?", n_tokens: int = 100) -> float:
        """Measure token generation throughput (tokens/sec)."""
        try:
            ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            start = time.perf_counter()
            with torch.no_grad():
                generated = self.model.generate(ids, max_new_tokens=n_tokens)
            elapsed = time.perf_counter() - start
            new_tokens = generated.shape[1] - ids.shape[1]
            tps = new_tokens / max(elapsed, 1e-6)
            logger.info(f"Generation throughput: {tps:.1f} tokens/sec")
            return tps
        except Exception as e:
            logger.warning(f"Throughput measurement failed: {e}")
            return 0.0

    def compute_generation_metrics(
        self,
        loader: DataLoader,
        n_samples: int = 10,
        use_amp: bool = True,
        max_new_tokens: int = 50,
    ) -> Dict:
        """Compute BLEU and ROUGE-L on generated vs reference completions."""
        references: List[str] = []
        hypotheses: List[str] = []
        amp_dtype = torch.bfloat16

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= n_samples:
                    break
                input_ids = batch["input_ids"][:1].to(self.device)  # one example at a time
                labels = batch["labels"][:1].to(self.device)

                # Generate
                try:
                    generated = self.model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                    )
                    gen_text = self.tokenizer.decode(
                        generated[0, input_ids.shape[1]:], skip_special_tokens=True
                    )
                except Exception:
                    gen_text = ""

                # Reference: the actual continuation (labels)
                ref_ids = labels[0][labels[0] != -100]
                ref_text = self.tokenizer.decode(ref_ids[:max_new_tokens], skip_special_tokens=True)

                if gen_text and ref_text:
                    hypotheses.append(gen_text)
                    references.append(ref_text)

        metrics = {}

        # BLEU
        try:
            import sacrebleu
            bleu = sacrebleu.corpus_bleu(hypotheses, [references])
            metrics["bleu"] = round(bleu.score, 2)
        except Exception:
            metrics["bleu"] = None

        # ROUGE
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)]
            metrics["rouge_l"] = round(sum(scores) / max(len(scores), 1), 4)
        except Exception:
            metrics["rouge_l"] = None

        return metrics

    def domain_specific_eval(self) -> Dict:
        """
        Lightweight domain-specific checks.
        Returns a dict of metric_name → value.
        """
        checks = {
            Domain.LANGUAGE_LEARNING: self._eval_language_learning,
            Domain.STOCKS: self._eval_stocks,
            Domain.RUNBOOKS: self._eval_runbooks,
        }
        fn = checks.get(self.domain)
        if fn:
            try:
                return fn()
            except Exception as e:
                logger.warning(f"Domain-specific eval failed: {e}")
        return {}

    # ------------------------------------------------------------------
    # Domain-specific evaluators
    # ------------------------------------------------------------------

    def _eval_language_learning(self) -> Dict:
        prompts = [
            ("Complete the sentence: 'She ____ to the store yesterday.'", "went"),
            ("What is the past tense of 'run'?", "ran"),
        ]
        return self._completion_accuracy(prompts, "lang_accuracy")

    def _eval_stocks(self) -> Dict:
        prompts = [
            ("What does P/E ratio stand for?", "price-to-earnings"),
            ("A company with high beta is considered:", "volatile"),
        ]
        return self._completion_accuracy(prompts, "finance_accuracy")

    def _eval_runbooks(self) -> Dict:
        prompts = [
            ("SRE stands for:", "site reliability"),
            ("A runbook section describing rollback steps is called:", "rollback"),
        ]
        return self._completion_accuracy(prompts, "runbook_accuracy")

    def _completion_accuracy(self, prompts: List, metric_name: str) -> Dict:
        correct = 0
        for prompt, expected in prompts:
            try:
                ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    out = self.model.generate(ids, max_new_tokens=20, temperature=0.1, top_p=0.9, top_k=10)
                answer = self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).lower()
                if expected.lower() in answer:
                    correct += 1
            except Exception:
                pass
        accuracy = correct / max(len(prompts), 1)
        return {metric_name: round(accuracy, 2)}

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _print_results(self, results: Dict) -> None:
        t = Table(title="Evaluation Results", box=box.ROUNDED)
        t.add_column("Metric", style="cyan bold")
        t.add_column("Value", style="yellow bold", justify="right")

        label_map = {
            "perplexity": "Perplexity",
            "test_loss": "Test Loss",
            "tokens_per_second": "Generation Speed (tok/s)",
            "bleu": "BLEU Score",
            "rouge_l": "ROUGE-L",
            "lang_accuracy": "Language Accuracy",
            "finance_accuracy": "Finance Accuracy",
            "runbook_accuracy": "Runbook Accuracy",
        }

        for key, val in results.items():
            if val is not None:
                label = label_map.get(key, key)
                t.add_row(label, f"{val:.4f}" if isinstance(val, float) else str(val))

        console.print(t)
