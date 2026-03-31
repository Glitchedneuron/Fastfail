"""
Corpus fetcher — Language Learning domain.

Sources (all public / open-licensed):
  • Tatoeba sentence pairs (HF)
  • CC-100 multilingual corpus (HF streaming)
  • Wikipedia articles on grammar / linguistics
  • OpenSubtitles parallel corpus (HF)
  • mc4 multilingual web text
"""

from __future__ import annotations

from ..base import BaseCorpus, CorpusResult
from loguru import logger


class LanguageLearningCorpus(BaseCorpus):
    domain_name = "language_learning"
    default_token_budget = 8_000_000

    def _fetch(self, result: CorpusResult) -> None:
        self._fetch_tatoeba(result)
        self._fetch_open_subtitles(result)
        self._fetch_multilingual_wiki(result)
        self._fetch_grammar_texts(result)
        self._fetch_cc100(result)

    # ------------------------------------------------------------------

    def _fetch_tatoeba(self, result: CorpusResult) -> None:
        """Tatoeba multilingual sentence pairs."""
        self._fetch_hf_dataset(
            result,
            dataset_name="tatoeba",
            config_name="all",
            split="train",
            text_field="sourceString",
            max_samples=50_000,
        )
        # Also try individual language pairs
        for lang_pair in [("en", "fr"), ("en", "es"), ("en", "de"), ("en", "ja")]:
            try:
                from datasets import load_dataset
                config = f"{lang_pair[0]}-{lang_pair[1]}"
                ds = load_dataset("tatoeba", config, split="train", streaming=True, trust_remote_code=True)
                count = 0
                for row in ds:
                    if not self._budget_remaining(result) or count >= 10_000:
                        break
                    # Format as a translation learning example
                    trans = row.get("translation", {})
                    src = trans.get(lang_pair[0], "")
                    tgt = trans.get(lang_pair[1], "")
                    if src and tgt:
                        text = (
                            f"English: {src}\n"
                            f"{lang_pair[1].upper()}: {tgt}\n"
                            f"Translation explanation: Learn how '{src}' translates to '{tgt}' in {lang_pair[1]}."
                        )
                        result.append(text, source=f"tatoeba-{config}")
                        count += 1
            except Exception as e:
                logger.warning(f"  Tatoeba {lang_pair} failed: {e}")

    def _fetch_open_subtitles(self, result: CorpusResult) -> None:
        """OpenSubtitles — natural conversational language."""
        self._fetch_hf_dataset(
            result,
            dataset_name="open_subtitles",
            config_name="en-fr",
            split="train",
            text_field="translation",
            max_samples=20_000,
        )

    def _fetch_multilingual_wiki(self, result: CorpusResult) -> None:
        """Wikipedia articles about languages, grammar, linguistics."""
        topics = [
            "Grammar", "Linguistics", "Morphology (linguistics)", "Syntax",
            "Phonology", "Semantics", "Language acquisition", "Second language",
            "Vocabulary", "Idiom", "Metaphor", "Etymology",
        ]
        try:
            import wikipediaapi
            wiki = wikipediaapi.Wikipedia("FastFail-Bot/1.0", "en")
            for topic in topics:
                if not self._budget_remaining(result):
                    break
                page = wiki.page(topic)
                if page.exists():
                    result.append(page.text, source=f"wikipedia:{topic}")
        except Exception as e:
            logger.warning(f"  Wikipedia API fetch failed: {e}")
            # Fallback to HF wikipedia
            self._fetch_wikipedia(result, lang="en", max_articles=500)

    def _fetch_grammar_texts(self, result: CorpusResult) -> None:
        """Synthetic grammar lesson templates to seed the corpus."""
        templates = [
            ("Present Simple", "Subject + Verb (base form)", "I walk. She walks. They walk every day."),
            ("Past Simple", "Subject + Verb (past tense)", "I walked. She walked yesterday."),
            ("Present Perfect", "Subject + have/has + past participle", "I have walked. She has eaten lunch."),
            ("Conditional", "If + Simple Past, Subject + would + infinitive", "If I had money, I would travel."),
            ("Passive Voice", "Subject + to be + past participle", "The book was written by the author."),
            ("Reported Speech", "Subject + said + that + clause", "She said that she was tired."),
            ("Articles", "a / an / the — definite and indefinite", "A cat sat on the mat. The cat ran away."),
            ("Prepositions", "in / on / at / by / with / for / of / to", "She is at home. The meeting is on Monday."),
        ]
        for name, rule, example in templates:
            text = (
                f"Grammar Lesson: {name}\n\n"
                f"Rule: {rule}\n\n"
                f"Example: {example}\n\n"
                f"Practice: Use '{name}' in your daily conversation to build fluency. "
                f"Remember that consistent practice is the key to mastering any grammar rule.\n"
            )
            result.append(text, source="synthetic-grammar")

    def _fetch_cc100(self, result: CorpusResult) -> None:
        """CC-100 multilingual web text."""
        for lang in ["en", "fr", "es", "de"]:
            if not self._budget_remaining(result):
                break
            self._fetch_hf_dataset(
                result,
                dataset_name="cc100",
                config_name=lang,
                split="train",
                text_field="text",
                max_samples=5_000,
            )
