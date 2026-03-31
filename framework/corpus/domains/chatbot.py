"""
Corpus fetcher — General Chatbot domain.

Sources:
  • OpenAssistant / OASST1 (HF)
  • Alpaca instruction-following data
  • ShareGPT conversation pairs
  • DailyDialog (HF)
  • Blended Skill Talk (HF)
"""

from __future__ import annotations

import json
from ..base import BaseCorpus, CorpusResult
from loguru import logger


class ChatbotCorpus(BaseCorpus):
    domain_name = "chatbot"
    default_token_budget = 15_000_000

    def _fetch(self, result: CorpusResult) -> None:
        self._fetch_oasst(result)
        self._fetch_alpaca(result)
        self._fetch_daily_dialog(result)
        self._fetch_blended_skill_talk(result)
        self._fetch_ultrachat(result)

    def _fetch_oasst(self, result: CorpusResult) -> None:
        """OpenAssistant conversations — multi-turn human + AI dialogue."""
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "OpenAssistant/oasst1",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            # Build conversation threads from message tree
            threads: dict = {}
            count = 0
            for row in ds:
                mid = row.get("message_id", "")
                pid = row.get("parent_id")
                role = row.get("role", "prompter")
                text = row.get("text", "")
                threads[mid] = {"pid": pid, "role": role, "text": text}
                count += 1
                if count >= 20_000:
                    break

            # Reconstruct conversations
            convos = []
            def build_thread(mid):
                node = threads.get(mid, {})
                pid = node.get("pid")
                prefix = build_thread(pid) if pid and pid in threads else []
                role = "Human" if node.get("role") == "prompter" else "Assistant"
                return prefix + [f"{role}: {node.get('text', '')}"]

            seen_roots = set()
            for mid, node in threads.items():
                if node["pid"] is None and mid not in seen_roots:
                    seen_roots.add(mid)
                    thread = build_thread(mid)
                    if len(thread) >= 2:
                        convos.append("\n".join(thread))

            for c in convos[:5_000]:
                if not self._budget_remaining(result):
                    break
                result.append(c, source="OpenAssistant/oasst1")

            logger.info(f"  Loaded {len(convos)} OASST conversations")
        except Exception as e:
            logger.warning(f"  OASST fetch failed: {e}")

    def _fetch_alpaca(self, result: CorpusResult) -> None:
        """Stanford Alpaca instruction-following pairs."""
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "tatsu-lab/alpaca",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            count = 0
            for row in ds:
                if not self._budget_remaining(result) or count >= 10_000:
                    break
                instruction = row.get("instruction", "")
                input_text = row.get("input", "")
                output = row.get("output", "")
                if instruction and output:
                    prompt = f"Instruction: {instruction}"
                    if input_text:
                        prompt += f"\nInput: {input_text}"
                    text = f"{prompt}\nResponse: {output}"
                    result.append(text, source="tatsu-lab/alpaca")
                    count += 1
        except Exception as e:
            logger.warning(f"  Alpaca fetch failed: {e}")

    def _fetch_daily_dialog(self, result: CorpusResult) -> None:
        """DailyDialog — everyday conversational English."""
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "daily_dialog",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            count = 0
            for row in ds:
                if not self._budget_remaining(result) or count >= 10_000:
                    break
                dialog = row.get("dialog", [])
                if dialog:
                    turns = []
                    roles = ["Human", "Assistant"] * (len(dialog) // 2 + 1)
                    for role, turn in zip(roles, dialog):
                        turns.append(f"{role}: {turn}")
                    result.append("\n".join(turns), source="daily_dialog")
                    count += 1
        except Exception as e:
            logger.warning(f"  DailyDialog fetch failed: {e}")

    def _fetch_blended_skill_talk(self, result: CorpusResult) -> None:
        self._fetch_hf_dataset(
            result,
            dataset_name="blended_skill_talk",
            split="train",
            text_field="free_messages",
            max_samples=5_000,
        )

    def _fetch_ultrachat(self, result: CorpusResult) -> None:
        """UltraChat 200k — high-quality instruction fine-tuning data."""
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "HuggingFaceH4/ultrachat_200k",
                split="train_sft",
                streaming=True,
                trust_remote_code=True,
            )
            count = 0
            for row in ds:
                if not self._budget_remaining(result) or count >= 8_000:
                    break
                messages = row.get("messages", [])
                if messages:
                    turns = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
                    result.append("\n".join(turns), source="HuggingFaceH4/ultrachat_200k")
                    count += 1
        except Exception as e:
            logger.warning(f"  UltraChat fetch failed: {e}")
