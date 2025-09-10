import re
import string

import nltk
from lingua import ConfidenceValue, Language, LanguageDetectorBuilder
from nltk.corpus import stopwords as _stopwords

_STOPWORDS_ENGLISH = set(_stopwords.words("english"))
_STOPWORDS_ITALIAN = set(_stopwords.words("italian"))
_STOPWORDS_SPANISH = set(_stopwords.words("spanish"))
ALLOWED_CHARS: set[str] = set(string.ascii_letters + string.whitespace)
FILTER_LINES: set[str] = {"ate out", "nothing"}

POS_TAGSET: str = "universal"
POS_MODIFIER_TAGS: set[str] = {"VERB", "ADJ"}

LANGUAGES = {
    Language.ENGLISH: {
        "stopwords": _STOPWORDS_ENGLISH,
        "value": "english",
    },
    Language.ITALIAN: {
        "stopwords": _STOPWORDS_ITALIAN,
        "value": "italian",
    },
    Language.SPANISH: {
        "stopwords": _STOPWORDS_SPANISH,
        "value": "spanish",
    },
}
LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_languages(*LANGUAGES).build()


def prepare(text: str) -> str:
    text = text.lower()
    text = re.sub(r"leftover", "", text)

    # XXX: Handle special case of apostrophe - want this to be a single token
    #      rather than split into two.
    text = re.sub(r"za'atar", "zaatar", text)
    return text


def clean(text: str) -> str:
    # For handling compound words in Italian/Spanish.
    # eg "all'aglio e olio" -> "all aglio e olio"
    text = text.replace("'", " ")
    return "".join(char for char in text if char in ALLOWED_CHARS)


def detect_language(text: str) -> ConfidenceValue:
    language_cv, *_ = LANGUAGE_DETECTOR.compute_language_confidence_values(text)
    return language_cv


def tag_pos(tokens: list[str]) -> list[tuple[str, str]]:
    return nltk.pos_tag(tokens, tagset=POS_TAGSET)


def remove_modifiers(tagged_tokens: list[tuple[str, str]]) -> list[str]:
    return [token for token, pos in tagged_tokens if pos not in POS_MODIFIER_TAGS]


def tokenize(text: str, language: Language | None) -> list[str]:
    tokens = text.split()

    # Remove ingredient modifiers, e.g. "red pepper" or "baked" in
    # "baked tofu", as downstream deduplication/distance calculations/etc use
    # unigram tokens.
    # Skip POS tagging for non-English languages as only English is supported.
    # NB: Don't remove stopwords until after running POS tagging, as the
    #     stopwords can help provide context to the POS tagger.
    if language is Language.ENGLISH:
        tagged_tokens = tag_pos(tokens)
        tokens = remove_modifiers(tagged_tokens)

    stopwords = LANGUAGES.get(language, {}).get("stopwords", set())
    return [token for token in tokens if token not in stopwords]
