"""
Corpus fetcher — Psychotherapy Assistant domain.

Sources:
  • Empathetic Dialogues (HF — Facebook Research)
  • Mental health conversational datasets (HF)
  • PsychologyWikipedia articles
  • Synthetic CBT / ACT session templates
  • Emotional support conversation data

IMPORTANT — ETHICAL NOTE:
  This corpus is intended exclusively for building *supportive assistant*
  tools that complement (not replace) licensed mental-health professionals.
  All synthetic templates follow evidence-based, safe-messaging guidelines.
"""

from __future__ import annotations

from ..base import BaseCorpus, CorpusResult
from loguru import logger


_PSYCH_WIKI_TOPICS = [
    "Cognitive behavioral therapy", "Acceptance and commitment therapy",
    "Dialectical behavior therapy", "Psychodynamic psychotherapy",
    "Motivational interviewing", "Mindfulness-based cognitive therapy",
    "Exposure therapy", "Trauma-focused cognitive behavioral therapy",
    "Depression (mood)", "Anxiety disorder", "Post-traumatic stress disorder",
    "Burnout (psychology)", "Grief", "Resilience (psychology)",
    "Active listening", "Empathy", "Emotional intelligence",
    "Self-compassion", "Rumination (psychology)", "Cognitive distortion",
    "Psychological resilience", "Coping (psychology)", "Stress management",
]

_SYNTHETIC_SESSIONS = [
    {
        "title": "CBT Session — Addressing Negative Automatic Thoughts",
        "dialogue": [
            ("Therapist", "It sounds like you've been having a really tough week. Can you tell me more about what's been going through your mind?"),
            ("Client", "I keep thinking that everyone at work thinks I'm incompetent. Every time I make a small mistake, I just spiral."),
            ("Therapist", "I hear you. That sounds exhausting. Let's look at that thought together — 'everyone thinks I'm incompetent'. What evidence do you have that supports that thought?"),
            ("Client", "Well, my manager did frown when I sent that report late."),
            ("Therapist", "Okay. And is there any evidence that contradicts it? Has your manager ever given you positive feedback?"),
            ("Client", "Yes, actually. She praised my presentation last month and said I'm one of the most thorough analysts on the team."),
            ("Therapist", "So we have one piece of evidence for and several pieces against. This is what we call a cognitive distortion — specifically, catastrophising and mind-reading. The thought feels true, but the evidence doesn't support it fully. How does it feel to examine it this way?"),
            ("Client", "A bit lighter, actually. Like maybe I'm not as hopeless as I thought."),
            ("Therapist", "That's a great insight. Let's work on a more balanced thought that acknowledges the mistake without the catastrophising. Something like: 'I made an error, but my overall track record shows I am competent.'"),
        ],
    },
    {
        "title": "Mindfulness-Based Stress Reduction Introduction",
        "dialogue": [
            ("Therapist", "Today I'd like to introduce you to a technique called mindfulness-based breathing. It's very simple, and it can help interrupt the stress response. Are you open to trying it?"),
            ("Client", "I'm sceptical, but sure."),
            ("Therapist", "That's perfectly fine — just observe your own experience with curiosity. Let's begin by sitting comfortably. Notice the weight of your body in the chair. Now, bring your attention to your breath — the natural rise and fall of your chest or belly. You don't need to change anything."),
            ("Client", "My mind keeps wandering to my to-do list."),
            ("Therapist", "That's completely normal. Noticing that your mind has wandered is itself a moment of mindfulness. Simply, without judgment, guide your attention back to the breath. Each time you do this, you're strengthening your ability to choose where to place your attention."),
            ("Client", "I did feel my shoulders drop a bit."),
            ("Therapist", "Excellent. That tension release was your nervous system shifting from sympathetic to parasympathetic activation — the 'rest and digest' response. Even two minutes of this daily can reduce baseline cortisol over time."),
        ],
    },
    {
        "title": "Active Listening and Validation",
        "dialogue": [
            ("Therapist", "What brings you in today?"),
            ("Client", "I just feel like no one really listens to me. Not my partner, not my friends. I feel invisible."),
            ("Therapist", "That sounds really painful — feeling unseen by the people who matter most to you. I want you to know that what you're feeling is valid, and I'm here to listen fully."),
            ("Client", "It's like I start talking and they just wait for their turn to speak."),
            ("Therapist", "That kind of half-listening can feel isolating. You're expressing yourself, but you're not truly being heard. Have you been able to tell the people close to you how this makes you feel?"),
            ("Client", "I've tried, but I don't know how to say it without sounding needy."),
            ("Therapist", "Asking to be heard is a fundamental human need, not neediness. We can work on assertive communication skills — expressing your needs clearly and calmly, using 'I' statements rather than 'you' accusations. Would that be helpful?"),
            ("Client", "Yes, I'd like that."),
        ],
    },
]


class PsychotherapyCorpus(BaseCorpus):
    domain_name = "psychotherapy"
    default_token_budget = 8_000_000

    def _fetch(self, result: CorpusResult) -> None:
        result.append(
            "IMPORTANT DISCLAIMER: This model is a supplementary assistant tool. "
            "It does not replace licensed mental-health professionals. If you are in crisis, "
            "please contact a licensed therapist or a crisis helpline immediately.\n",
            source="disclaimer",
        )
        self._add_synthetic_sessions(result)
        self._fetch_psych_wiki(result)
        self._fetch_empathetic_dialogues(result)
        self._fetch_mental_health_qa(result)
        self._fetch_emotional_support(result)

    def _add_synthetic_sessions(self, result: CorpusResult) -> None:
        for session in _SYNTHETIC_SESSIONS:
            lines = [f"# {session['title']}\n"]
            for role, text in session["dialogue"]:
                lines.append(f"**{role}:** {text}\n")
            result.append("\n".join(lines), source="synthetic-cbt")

    def _fetch_psych_wiki(self, result: CorpusResult) -> None:
        try:
            import wikipediaapi
            wiki = wikipediaapi.Wikipedia("FastFail-Bot/1.0", "en")
            for topic in _PSYCH_WIKI_TOPICS:
                if not self._budget_remaining(result):
                    break
                page = wiki.page(topic)
                if page.exists():
                    result.append(page.text, source=f"wikipedia:{topic}")
        except Exception as e:
            logger.warning(f"  Wikipedia psych fetch failed: {e}")

    def _fetch_empathetic_dialogues(self, result: CorpusResult) -> None:
        """Facebook Research Empathetic Dialogues."""
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "empathetic_dialogues",
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            count = 0
            for row in ds:
                if not self._budget_remaining(result) or count >= 15_000:
                    break
                context = row.get("utterance", "")
                prompt = row.get("prompt", "")
                if context:
                    text = f"Situation: {prompt}\nResponse: {context}" if prompt else context
                    result.append(text, source="empathetic_dialogues")
                    count += 1
        except Exception as e:
            logger.warning(f"  Empathetic Dialogues fetch failed: {e}")

    def _fetch_mental_health_qa(self, result: CorpusResult) -> None:
        """Mental health Q&A datasets."""
        for ds_name, field in [
            ("Amod/mental_health_counseling_conversations", "Context"),
            ("helpmefindaname/me-therapy-conversations", "input"),
        ]:
            if not self._budget_remaining(result):
                break
            self._fetch_hf_dataset(
                result,
                dataset_name=ds_name,
                split="train",
                text_field=field,
                max_samples=5_000,
            )

    def _fetch_emotional_support(self, result: CorpusResult) -> None:
        """Emotional Support Conversation dataset."""
        self._fetch_hf_dataset(
            result,
            dataset_name="thu-coai/esconv",
            split="train",
            text_field="dialog",
            max_samples=3_000,
        )
