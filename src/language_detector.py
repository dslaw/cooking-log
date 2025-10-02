from fast_langdetect import LangDetectConfig, LangDetector
from lingua import IsoCode639_1, Language, LanguageDetectorBuilder


def _language_to_code(language: Language) -> str:
    code = language.iso_code_639_1.name
    return code.lower()


def _code_to_language(code: str) -> Language:
    iso_code = IsoCode639_1.from_str(code)
    return Language.from_iso_code_639_1(iso_code)


class LanguageDetector:
    def __init__(self, language_codes: list[str]):
        self.language_codes = set(language_codes)

        config = LangDetectConfig(model="lite")
        self._detector_primary = LangDetector(config)

        languages = tuple(map(_code_to_language, self.language_codes))
        self._detector_secondary = LanguageDetectorBuilder.from_languages(
            *languages
        ).build()

    def detect_language(self, text: str) -> tuple[str, float]:
        detected_language, *_ = self._detector_primary.detect(text, k=1)
        language_code = detected_language.get("lang")

        if language_code in self.language_codes:
            return language_code, detected_language["score"]

        # Fallback to lingua, which is less accurate but has a constrained
        # prediction space such that only a supported language will be returned.
        # This helps for cases where the input text is close to a supported
        # language but varies enough that it is closer to an unsupported
        # language, for example Catalan instead of Castilian Spanish, or
        # Corsican when an Italian text contains a loanword from a dialect or
        # other Italian language (e.g. Sicilian).
        detected, *_ = self._detector_secondary.compute_language_confidence_values(text)
        language_code = _language_to_code(detected.language)
        return language_code, detected.value
