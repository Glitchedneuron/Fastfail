"""
Corpus fetcher — Sports domain.

Sources:
  • Wikipedia sports articles
  • HF sports QA datasets
  • ESPN / BBC Sport RSS feeds
  • Synthetic match commentary templates
"""

from __future__ import annotations

from ..base import BaseCorpus, CorpusResult
from loguru import logger


_SPORTS_WIKIPEDIA_TOPICS = [
    "Association football", "Basketball", "Tennis", "Cricket", "Rugby union",
    "Baseball", "Golf", "Athletics (sport)", "Swimming (sport)", "Cycling",
    "Boxing", "Mixed martial arts", "American football", "Ice hockey",
    "Volleyball", "Table tennis", "Badminton", "Rowing (sport)", "Triathlon",
    "Olympic Games", "FIFA World Cup", "NBA", "Tour de France", "Wimbledon",
    "Premier League", "Super Bowl", "World Series", "Grand Slam (tennis)",
]

_RSS_FEEDS = [
    "https://feeds.bbci.co.uk/sport/rss.xml",
    "https://rss.espn.com/rss/news",
]


class SportsCorpus(BaseCorpus):
    domain_name = "sports"
    default_token_budget = 8_000_000

    def _fetch(self, result: CorpusResult) -> None:
        self._fetch_sports_wiki(result)
        self._fetch_hf_sports_qa(result)
        self._fetch_rss(result)
        self._add_synthetic_commentary(result)

    def _fetch_sports_wiki(self, result: CorpusResult) -> None:
        try:
            import wikipediaapi
            wiki = wikipediaapi.Wikipedia("FastFail-Bot/1.0", "en")
            for topic in _SPORTS_WIKIPEDIA_TOPICS:
                if not self._budget_remaining(result):
                    break
                page = wiki.page(topic)
                if page.exists():
                    result.append(page.text, source=f"wikipedia:{topic}")
        except Exception as e:
            logger.warning(f"  Wikipedia sports fetch failed: {e}")
            self._fetch_wikipedia(result, lang="en", max_articles=200)

    def _fetch_hf_sports_qa(self, result: CorpusResult) -> None:
        """SQuAD-style sports Q&A from HF."""
        self._fetch_hf_dataset(
            result,
            dataset_name="rajpurkar/squad",
            split="train",
            text_field="context",
            max_samples=5_000,
        )

    def _fetch_rss(self, result: CorpusResult) -> None:
        try:
            import feedparser
            for url in _RSS_FEEDS:
                if not self._budget_remaining(result):
                    break
                feed = feedparser.parse(url)
                for entry in feed.entries[:200]:
                    text = entry.get("summary", "") or entry.get("description", "")
                    title = entry.get("title", "")
                    if text:
                        result.append(f"{title}\n\n{text}", source=url)
        except Exception as e:
            logger.warning(f"  RSS fetch failed: {e}")

    def _add_synthetic_commentary(self, result: CorpusResult) -> None:
        """Add templated match commentary to ensure domain coverage."""
        templates = [
            (
                "Match Report: Football",
                "The match began with high intensity as both teams competed for possession. "
                "The home side took an early lead through a well-placed header from a corner kick. "
                "Despite sustained pressure, the visiting team equalised in the second half with a long-range strike. "
                "The game finished 1-1, a fair result given the chances created on both sides."
            ),
            (
                "Basketball Game Summary",
                "In a thrilling NBA encounter, the leading scorer posted a triple-double with 28 points, 11 rebounds, "
                "and 10 assists. The team's defence held the opposition to just 89 points, securing a comfortable "
                "victory. The point guard's playmaking was instrumental in breaking down the zone defence."
            ),
            (
                "Tennis Match Analysis",
                "The first set was decided on a tiebreak after both players held serve consistently. "
                "The baseline rallies were extended, with both players displaying exceptional footwork. "
                "The serve-and-volley tactics employed in the second set proved decisive, with the winner "
                "taking the set 6-3 to claim the match in straight sets."
            ),
            (
                "Cricket Scorecard",
                "The batting side posted a commanding total of 287 runs in their allotted 50 overs. "
                "The opening partnership was worth 112 runs before the first wicket fell. "
                "The spin bowlers proved economical in the middle overs, conceding fewer than 5 runs per over. "
                "In the chase, the target proved too steep despite a valiant half-century from the number three batsman."
            ),
        ]
        for title, body in templates:
            result.append(f"# {title}\n\n{body}\n", source="synthetic-sports")
