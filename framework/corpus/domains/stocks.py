"""
Corpus fetcher — Stocks & Finance domain.

Sources:
  • financial_phrasebank (sentiment)
  • SEC filings text (HF financial_reports)
  • Wikipedia finance articles
  • yfinance news headlines
  • Synthetic financial analysis templates
"""

from __future__ import annotations

from ..base import BaseCorpus, CorpusResult
from loguru import logger


_FINANCE_WIKI_TOPICS = [
    "Stock market", "Bond (finance)", "Hedge fund", "Mutual fund",
    "Dividend", "Earnings per share", "Price-to-earnings ratio",
    "Technical analysis", "Fundamental analysis", "Portfolio (finance)",
    "Risk management", "Derivative (finance)", "Options (finance)",
    "Market capitalization", "IPO", "Nasdaq", "New York Stock Exchange",
    "Federal Reserve", "Monetary policy", "Inflation", "Interest rate",
    "Balance sheet", "Income statement", "Cash flow statement",
    "Discounted cash flow", "Valuation (finance)", "Bull market", "Bear market",
    "Algorithmic trading", "High-frequency trading", "Warren Buffett",
    "Benjamin Graham", "Efficient market hypothesis",
]


class StocksCorpus(BaseCorpus):
    domain_name = "stocks"
    default_token_budget = 10_000_000

    def _fetch(self, result: CorpusResult) -> None:
        self._fetch_financial_phrasebank(result)
        self._fetch_finance_wiki(result)
        self._fetch_financial_qa(result)
        self._fetch_yfinance_news(result)
        self._add_synthetic_analysis(result)

    def _fetch_financial_phrasebank(self, result: CorpusResult) -> None:
        self._fetch_hf_dataset(
            result,
            dataset_name="financial_phrasebank",
            config_name="sentences_allagree",
            split="train",
            text_field="sentence",
            max_samples=5_000,
        )

    def _fetch_finance_wiki(self, result: CorpusResult) -> None:
        try:
            import wikipediaapi
            wiki = wikipediaapi.Wikipedia("FastFail-Bot/1.0", "en")
            for topic in _FINANCE_WIKI_TOPICS:
                if not self._budget_remaining(result):
                    break
                page = wiki.page(topic)
                if page.exists():
                    result.append(page.text, source=f"wikipedia:{topic}")
        except Exception as e:
            logger.warning(f"  Wikipedia finance fetch failed: {e}")

    def _fetch_financial_qa(self, result: CorpusResult) -> None:
        """FinQA and other financial reasoning datasets."""
        for ds_name, cfg, field in [
            ("ibm/finqa", None, "question"),
            ("easonnie/flare-finqa", None, "input"),
            ("TheFinAI/flare-fpb", None, "input"),
        ]:
            if not self._budget_remaining(result):
                break
            self._fetch_hf_dataset(
                result,
                dataset_name=ds_name,
                config_name=cfg,
                split="train",
                text_field=field,
                max_samples=3_000,
            )

    def _fetch_yfinance_news(self, result: CorpusResult) -> None:
        """Pull recent news from yfinance for major tickers."""
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B"]
        try:
            import yfinance as yf
            for ticker_sym in tickers:
                if not self._budget_remaining(result):
                    break
                try:
                    tk = yf.Ticker(ticker_sym)
                    news = tk.news or []
                    for item in news[:20]:
                        title = item.get("title", "")
                        summary = item.get("summary", "") or item.get("description", "")
                        if title:
                            text = f"Stock News ({ticker_sym}): {title}\n\n{summary}"
                            result.append(text, source=f"yfinance:{ticker_sym}")
                except Exception:
                    pass
        except ImportError:
            logger.warning("  yfinance not installed; skipping live news.")

    def _add_synthetic_analysis(self, result: CorpusResult) -> None:
        analyses = [
            (
                "Q3 2024 Earnings Analysis",
                "The company reported revenue of $24.3 billion, beating consensus estimates by 3.2%. "
                "Operating margins expanded by 150 basis points year-over-year to 28.4%, driven by "
                "cost efficiencies in the supply chain. Free cash flow of $5.8 billion exceeded guidance. "
                "EPS of $2.14 on a diluted basis compared to the $2.05 analyst expectation. "
                "Management raised full-year guidance citing strong demand in cloud services.",
                "Recommendation: BUY | Target Price: $185 | Risk Rating: Medium"
            ),
            (
                "Technical Analysis Report",
                "The stock is trading above its 50-day and 200-day moving averages, forming a golden cross "
                "pattern that typically signals bullish momentum. The RSI at 58 is approaching overbought "
                "territory but has not yet breached the 70 threshold. Volume has been above average for "
                "three consecutive sessions, confirming the breakout from the consolidation range. "
                "Key support lies at the $142 level; resistance at $158 from the prior high.",
                "Signal: BULLISH | Stop Loss: $138 | Take Profit: $162"
            ),
            (
                "Portfolio Risk Assessment",
                "Beta of 1.24 indicates the stock moves 24% more than the overall market. Sharpe ratio "
                "of 1.8 suggests attractive risk-adjusted returns. Maximum drawdown over the past 12 months "
                "was 18.3%, occurring during the broad market sell-off in Q4. Correlation to S&P 500 is 0.76, "
                "providing moderate diversification benefit. VaR (95%, 1-day) is estimated at $12,400 per "
                "$100,000 invested.",
                "Portfolio Weight Suggestion: 4-6% of equity allocation"
            ),
        ]
        for title, body, conclusion in analyses:
            text = f"## {title}\n\n{body}\n\n**{conclusion}**\n"
            result.append(text, source="synthetic-finance")
