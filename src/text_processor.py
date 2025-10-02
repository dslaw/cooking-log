from pathlib import Path

import spacy

from src.language_detector import LanguageDetector

SUPPORTED_LANGUAGE_CODES = ["en", "es", "it"]
# Only English is currently supported for ingredient extraction.
INGREDIENT_EXTRACTION_SUPPORTED_LANGUAGE_CODES = ["en"]

INGREDIENT_EXTRACTION_MODEL_DIR = Path("ingredient_extraction_model")


class TextProcessor:
    def __init__(
        self,
        language_detector: LanguageDetector,
        ingredient_extractor: spacy.language.Language,
    ):
        self.language_detector = language_detector
        self.ingredient_extractor = ingredient_extractor

    def detect_language(self, text: str) -> tuple[str, float]:
        return self.language_detector.detect_language(text)

    def process(self, text: str) -> tuple[list[str] | None, str, float]:
        detected_lang, detected_lang_confidence = self.detect_language(text)

        if detected_lang not in (INGREDIENT_EXTRACTION_SUPPORTED_LANGUAGE_CODES):
            return None, detected_lang, detected_lang_confidence

        doc = self.ingredient_extractor(text)
        ingredients: list[str] = [entity.lemma_.lower() for entity in doc.ents]
        return ingredients, detected_lang, detected_lang_confidence


def make_text_processor() -> TextProcessor:
    language_detector = LanguageDetector(SUPPORTED_LANGUAGE_CODES)
    ingredient_extractor = spacy.load(INGREDIENT_EXTRACTION_MODEL_DIR)
    return TextProcessor(language_detector, ingredient_extractor)
