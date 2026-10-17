"""Text normalization shared by the rule reviewer, injection scanner, and claim grounding.

Advertisers can hide words from keyword rules with zero-width characters,
look-alike letters, or digits for letters. Every rule matches on the
normalized text, never on the raw text.
"""

from __future__ import annotations

import re
import unicodedata

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)

# Cyrillic and Greek letters that render like Latin ones.
HOMOGLYPHS = str.maketrans(
    {
        "а": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "у": "y",
        "х": "x",
        "і": "i",
        "ј": "j",
        "ѕ": "s",
        "ԁ": "d",
        "ɡ": "g",
        "ο": "o",
        "ν": "v",
        "α": "a",
        "ε": "e",
        "ι": "i",
        "κ": "k",
        "τ": "t",
    }
)
LEET = str.maketrans(
    {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"}
)

_STOP = (
    "a an the and or of to in on for with by at from our your you we us is are be been it "
    "its this that these those as into than then so can will may up more most per"
)
STOPWORDS = frozenset(_STOP.split())
_WORD = re.compile(r"[a-z]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def has_hidden_characters(text: str) -> bool:
    return any(ord(ch) in ZERO_WIDTH for ch in text)


def normalize(text: str) -> str:
    """NFKC, strip zero-width characters, fold homoglyphs, lowercase, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).translate(ZERO_WIDTH).lower().translate(HOMOGLYPHS)
    return re.sub(r"\s+", " ", text).strip()


def deleet(text: str) -> str:
    """A second view for keyword rules: digits and symbols read as letters (c4s1n0 → casino).
    Numbers are lost, so claims are grounded on `normalize`, not on this."""
    return normalize(text).translate(LEET)


def words(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(normalize(text))]


def content_words(text: str) -> set[str]:
    return {w for w in words(text) if w not in STOPWORDS and len(w) > 1}


def numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(normalize(text)))


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _stem(word: str) -> str:
    """Plural / third-person -s only: enough to match "reduces" with "reduce"."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word
