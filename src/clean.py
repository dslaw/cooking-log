import re
import string

STOPWORDS: set[str] = {
    # English.
    "and",
    "with",
    "of",
    "in",
    "leftover",
    # Italian.
    "e",
    "con",
    "coi",
    "di",
    "del",
    "dell",
    "della",
    "il",
    "la",
    "i",
    "e",
    "gli",
    "ai",
    "all",
    "agli",
    "alle",
    "al",
    "alla",
    "in",
    # Spanish.
    "y",
    "con",
    "de",
    "del",
    "el",
    "la",
    "los",
    "las",
    "en",
}
ALLOWED_CHARS: set[str] = set(string.ascii_letters + string.whitespace)
FILTER_LINES: set[str] = {"ate out", "nothing"}


def clean(text: str) -> str:
    # XXX: Handle special case.
    text = re.sub(r"za'atar", "zaatar", text, flags=re.IGNORECASE)

    # For handling compound words in Italian/Spanish.
    # eg "all'aglio e olio" -> "all aglio e olio"
    text = text.lower()
    text = text.replace("'", " ")
    return "".join(char for char in text if char in ALLOWED_CHARS)


def tokenize(text: str) -> list[str]:
    tokens = text.split()
    return [token for token in tokens if token not in STOPWORDS]
