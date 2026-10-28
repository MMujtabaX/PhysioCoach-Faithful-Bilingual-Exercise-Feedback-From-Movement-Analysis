"""Number handling for English and Urdu.

* extracting numbers from text (digits, Urdu/Arabic-Indic digits, number words)
* spelling numbers as words for speech (MMS voices do not read digits)
"""

from __future__ import annotations

import re

_EN_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

# Urdu cardinal numbers 0-100. Urdu numbers are irregular, so each is listed.
# (Native-speaker check recommended before final submission.)
UR_NUMBERS = [
    "صفر", "ایک", "دو", "تین", "چار", "پانچ", "چھ", "سات", "آٹھ", "نو",
    "دس", "گیارہ", "بارہ", "تیرہ", "چودہ", "پندرہ", "سولہ", "سترہ", "اٹھارہ", "انیس",
    "بیس", "اکیس", "بائیس", "تئیس", "چوبیس", "پچیس", "چھبیس", "ستائیس", "اٹھائیس", "انتیس",
    "تیس", "اکتیس", "بتیس", "تینتیس", "چونتیس", "پینتیس", "چھتیس", "سینتیس", "اڑتیس", "انتالیس",
    "چالیس", "اکتالیس", "بیالیس", "تینتالیس", "چوالیس", "پینتالیس", "چھیالیس", "سینتالیس", "اڑتالیس", "انچاس",
    "پچاس", "اکاون", "باون", "ترپن", "چون", "پچپن", "چھپن", "ستاون", "اٹھاون", "انسٹھ",
    "ساٹھ", "اکسٹھ", "باسٹھ", "تریسٹھ", "چونسٹھ", "پینسٹھ", "چھیاسٹھ", "سڑسٹھ", "اڑسٹھ", "انہتر",
    "ستر", "اکہتر", "بہتر", "تہتر", "چوہتر", "پچھتر", "چھہتر", "ستتر", "اٹھہتر", "اناسی",
    "اسی", "اکیاسی", "بیاسی", "تراسی", "چوراسی", "پچاسی", "چھیاسی", "ستاسی", "اٹھاسی", "نواسی",
    "نوے", "اکیانوے", "بانوے", "ترانوے", "چورانوے", "پچانوے", "چھیانوے", "ستانوے", "اٹھانوے", "ننانوے",
    "سو",
]
assert len(UR_NUMBERS) == 101

# Urdu number words that are also ordinary words; never parsed as numbers.
_UR_AMBIGUOUS = {"نو", "سو", "چون", "ستر", "اسی", "بہتر", "تیس", "دس"}
_UR_WORD_TO_INT = {w: i for i, w in enumerate(UR_NUMBERS) if w not in _UR_AMBIGUOUS}

_DIGIT_MAP = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_INT_RE = re.compile(r"(?<![\w.])\d+(?![\w])")


def normalize_digits(text: str) -> str:
    """Convert Urdu (Extended Arabic-Indic) and Arabic-Indic digits to ASCII."""
    return text.translate(_DIGIT_MAP)


def to_words_en(n: int) -> str:
    if n < 0:
        return "minus " + to_words_en(-n)
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _EN_TENS[tens] + ("" if ones == 0 else "-" + _EN_ONES[ones])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _EN_ONES[hundreds] + " hundred" + ("" if rest == 0 else " and " + to_words_en(rest))
    return str(n)


def to_words_ur(n: int) -> str:
    if 0 <= n <= 100:
        return UR_NUMBERS[n]
    return str(n)


_EN_WORD_TO_INT = {to_words_en(i): i for i in range(0, 101)}
_EN_WORD_TO_INT.update({w.replace("-", " "): i for w, i in list(_EN_WORD_TO_INT.items()) if "-" in w})
_EN_WORD_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(w) for w in _EN_WORD_TO_INT), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def extract_numbers_en(text: str, include_words: bool = True) -> list[int]:
    nums = [int(m) for m in _INT_RE.findall(normalize_digits(text))]
    if include_words:
        for m in _EN_WORD_RE.findall(text):
            word = m.lower()
            if word == "one":  # "one" is too often a pronoun ("that one")
                continue
            nums.append(_EN_WORD_TO_INT[word])
    return nums


def extract_numbers_ur(text: str) -> list[int]:
    text = normalize_digits(text)
    nums = [int(m) for m in _INT_RE.findall(text)]
    for token in re.findall(r"[؀-ۿ]+", text):
        if token in _UR_WORD_TO_INT and token != "ایک":  # "ek" is also "a/one"
            nums.append(_UR_WORD_TO_INT[token])
    return nums


def spell_digits(text: str, lang: str) -> str:
    """Replace digit runs with words so a TTS voice can read them."""
    speller = to_words_ur if lang == "ur" else to_words_en
    return _INT_RE.sub(lambda m: speller(int(m.group())), normalize_digits(text))


NUMBER_TOKEN = r"(\d+|[a-z]+(?:[- ][a-z]+)?)"


def parse_number_en(token: str) -> int | None:
    """'12' -> 12, 'twelve' -> 12, 'twenty-one' / 'twenty one' -> 21, anything else -> None."""
    token = token.strip().lower()
    if token.isdigit():
        return int(token)
    for candidate in (token, token.replace(" ", "-"), token.split()[-1] if token.split() else ""):
        if candidate in _EN_WORD_TO_INT:
            return _EN_WORD_TO_INT[candidate]
    return None
