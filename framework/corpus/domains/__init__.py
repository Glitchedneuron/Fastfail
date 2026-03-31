from .language_learning import LanguageLearningCorpus
from .sports import SportsCorpus
from .stocks import StocksCorpus
from .runbooks import RunbooksCorpus
from .chatbot import ChatbotCorpus
from .psychotherapy import PsychotherapyCorpus

from framework.config import Domain

DOMAIN_CORPUS_MAP = {
    Domain.LANGUAGE_LEARNING: LanguageLearningCorpus,
    Domain.SPORTS: SportsCorpus,
    Domain.STOCKS: StocksCorpus,
    Domain.RUNBOOKS: RunbooksCorpus,
    Domain.CHATBOT: ChatbotCorpus,
    Domain.PSYCHOTHERAPY: PsychotherapyCorpus,
}


def get_corpus_class(domain: Domain):
    cls = DOMAIN_CORPUS_MAP.get(domain)
    if cls is None:
        raise ValueError(f"No corpus class registered for domain: {domain}")
    return cls


__all__ = [
    "LanguageLearningCorpus",
    "SportsCorpus",
    "StocksCorpus",
    "RunbooksCorpus",
    "ChatbotCorpus",
    "PsychotherapyCorpus",
    "DOMAIN_CORPUS_MAP",
    "get_corpus_class",
]
