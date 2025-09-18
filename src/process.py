import re
import string

import nltk
from lingua import ConfidenceValue, Language, LanguageDetectorBuilder
from nltk.corpus import stopwords

LANGUAGE_ENGLISH = "english"
LANGUAGE_ITALIAN = "italian"
LANGUAGE_SPANISH = "spanish"

STOPWORDS_ENGLISH = set(stopwords.words("english"))
STOPWORDS_ITALIAN = set(stopwords.words("italian"))
STOPWORDS_SPANISH = set(stopwords.words("spanish"))

ALLOWED_CHARS = set(string.ascii_letters + string.whitespace)

LANGUAGES = {
    Language.ENGLISH: {
        "stopwords": STOPWORDS_ENGLISH,
        "value": LANGUAGE_ENGLISH,
        # Only English is supported for POS tagging.
        "pos_tagger": lambda tokens: nltk.pos_tag(tokens, tagset="universal"),
        "pos_modifiers": {"VERB", "ADJ"},
    },
    Language.ITALIAN: {
        "stopwords": STOPWORDS_ITALIAN,
        "value": LANGUAGE_ITALIAN,
    },
    Language.SPANISH: {
        "stopwords": STOPWORDS_SPANISH,
        "value": LANGUAGE_SPANISH,
    },
}


class TextProcessor:
    def __init__(
        self,
        preprocess_substitutions: list[tuple[str, str]],
        skip: list[list[str]],
        languages: dict,
    ):
        self.preprocess_substitutions = preprocess_substitutions
        self.skip = skip
        self.languages = languages
        self.language_detector = LanguageDetectorBuilder.from_languages(
            *languages
        ).build()

    @staticmethod
    def preprocess(text: str, substitutions: list[tuple[str, str]]) -> str:
        if not substitutions:
            return text

        for pattern, repl in substitutions:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

        return text

    def detect_language(self, text: str) -> ConfidenceValue:
        # Detected language with highest confidence.
        cv, *_ = self.language_detector.compute_language_confidence_values(text)
        return cv

    @staticmethod
    def clean(text: str) -> str:
        # Handle contractions, especially for Italian/Spanish, e.g. "all'aglio e
        # olio" -> "all aglio e olio".
        text = text.replace("'", " ")
        return "".join(char for char in text if char in ALLOWED_CHARS)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return text.split()

    def clean_tokens(
        self, tokens: list[str], language_cv: ConfidenceValue
    ) -> list[str]:
        if language_cv.language not in self.languages:
            raise ValueError(f"Unsupported language: {language_cv.language}")

        language = self.languages[language_cv.language]

        # Remove ingredient modifiers, e.g. "red pepper" or "baked" in "baked
        # tofu", as downstream deduplication/distance calculations/etc use
        # unigram tokens.
        # NB: Don't remove stopwords until after POS tagging, as they help
        #     provide context.
        tag_pos = language.get("pos_tagger")
        pos_modifiers = language.get("pos_modifiers")
        if tag_pos is not None:
            tagged_tokens = tag_pos(tokens)
            tokens = [token for token, pos in tagged_tokens if pos not in pos_modifiers]

        stopwords = language["stopwords"]
        return [token for token in tokens if token not in stopwords]

    def process(self, text: str) -> tuple[str, list[str], str, float] | None:
        text = text.lower()

        # Language detection tool is meant to be used on raw text. Some
        # preprocessing is done first to amend specific idiosyncracies in the
        # data.
        preprocessed = self.preprocess(text, self.preprocess_substitutions)
        language_cv = self.detect_language(preprocessed)
        language = self.languages[language_cv.language]["value"]  # Serializable.

        cleaned = self.clean(preprocessed)

        tokens = self.tokenize(cleaned)

        if not tokens or tokens in self.skip:
            return None

        cleaned_tokens = self.clean_tokens(tokens, language_cv)
        return cleaned, cleaned_tokens, language, language_cv.value


def make_text_processor() -> TextProcessor:
    return TextProcessor(
        preprocess_substitutions=[(r"za'atar", "zaatar"), (r"leftover\s+", "")],
        skip=[["ate", "out"], ["nothing"]],
        languages=LANGUAGES,
    )
