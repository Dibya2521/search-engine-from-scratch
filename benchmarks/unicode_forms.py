"""Measure what each Unicode normalization form does to the tokens.

Run with ``uv run python benchmarks/unicode_forms.py``.

The choice between NFC and NFKC is a correctness decision with a real cost on
each side, so it should be made against text rather than against taste.

- **NFC** composes a letter and its accent into the single code point they
  render as. It only ever merges spellings that are already indistinguishable on
  screen, so it cannot lose a distinction anyone can see.
- **NFKC** also folds compatibility forms. It recovers tokens the ASCII-only
  pattern would otherwise drop whole, a ligature or a fullwidth word, and it
  destroys distinctions: a superscript two becomes an ordinary two, which is
  wrong wherever that difference carries meaning.

This reports two things over whatever text is available: how many sources change
under each form, and the individual tokens each one gains or loses. **If no text
with the relevant characters is available, that is what it prints**, and the
decision is then made on the conservative argument rather than on a measurement
that did not happen. Saying which of the two occurred is the point of the
script.

Every constructed case is built from explicit code points rather than written as
a literal. These cases turn entirely on which code points are present, and a
literal leaves that to whatever an editor happened to write to disk.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from search_engine.tokenizer import TOKEN_PATTERN, tokenize

ROOT = Path(__file__).resolve().parent.parent
CORPORA = (
    ROOT / "tests" / "fixtures" / "evaluation_corpus.xml",
    ROOT / "tests" / "fixtures" / "sample_corpus.xml",
)
PROSE = [*sorted((ROOT / "docs").rglob("*.md")), ROOT / "README.md"]
TOP_CHANGES = 20
HIGHEST_ASCII = 127

CONSTRUCTED = (
    ("composed accent", "caf" + chr(0xE9)),
    ("combining accent", "cafe" + chr(0x301)),
    ("fi ligature", chr(0xFB01) + "refighting"),
    ("superscript two", "x" + chr(0xB2)),
    ("fullwidth letters", chr(0xFF21) + chr(0xFF22)),
    ("roman numeral four", chr(0x2163)),
    ("mathematical bold A", chr(0x1D400)),
    ("hangul jamo", chr(0x1100) + chr(0x1161) + chr(0x11A8)),
    ("hangul syllable", chr(0xAC01)),
    ("kelvin sign", chr(0x212A)),
    ("sharp s", "stra" + chr(0xDF) + "e"),
)


def say(line: str = "") -> None:
    """Print a line and flush it, so a long run shows progress as it goes."""
    print(line, flush=True)


def unnormalized(text: str) -> list[str]:
    """Tokenize the way this engine did before normalization was added.

    Kept here rather than in the engine so the before and after can be compared
    on the same text, which is the only way to say what the change was worth.
    """
    return [match.group() for match in TOKEN_PATTERN.finditer(text.lower())]


def sources() -> list[tuple[str, str]]:
    """Return every piece of text available to measure on, named by its path."""
    return [
        (str(path.relative_to(ROOT)), path.read_text(encoding="utf-8"))
        for path in (*CORPORA, *PROSE)
        if path.exists()
    ]


def difference(before: list[str], after: list[str]) -> list[str]:
    """Return the tokens each side has that the other does not."""
    left, right = Counter(before), Counter(after)
    return sorted((left - right).elements()) + sorted((right - left).elements())


def survey(available: list[tuple[str, str]]) -> None:
    """Count how many sources change under each form, and what changed in them."""
    changed_nfc = 0
    changed_nfkc_only = 0
    nfc_changes: Counter[str] = Counter()
    nfkc_changes: Counter[str] = Counter()
    for name, text in available:
        raw, nfc, nfkc = unnormalized(text), tokenize(text), tokenize(text, "NFKC")
        if raw != nfc:
            changed_nfc += 1
            nfc_changes.update(difference(raw, nfc))
        if nfc != nfkc:
            changed_nfkc_only += 1
            nfkc_changes.update(difference(nfc, nfkc))
        if raw != nfc or nfc != nfkc:
            say(f"  {name}: {len(raw)} raw, {len(nfc)} NFC, {len(nfkc)} NFKC tokens")
    say()
    say(f"sources measured                 : {len(available)}")
    say(f"sources whose tokens change, NFC : {changed_nfc}")
    say(f"sources changing under NFKC only : {changed_nfkc_only}")
    report("NFC against no normalization", nfc_changes)
    report("NFKC beyond NFC", nfkc_changes)


def report(label: str, changes: Counter[str]) -> None:
    """Print the most common individual token changes under one form."""
    say()
    if not changes:
        say(f"{label}: no token in any source changed.")
        return
    say(f"{label}: up to {TOP_CHANGES} most common token changes")
    for token, count in changes.most_common(TOP_CHANGES):
        say(f"  {count:5}  {token}")


def constructed() -> None:
    """Show what each form does on cases built from explicit code points."""
    say()
    say("### Constructed cases")
    say()
    say("| Case | Code points | No normalization | NFC | NFKC |")
    say("| --- | --- | --- | --- | --- |")
    for name, text in CONSTRUCTED:
        points = " ".join(f"U+{ord(character):04X}" for character in text)
        say(
            f"| {name} | {points} | {unnormalized(text)} "
            f"| {tokenize(text)} | {tokenize(text, 'NFKC')} |"
        )


def coverage() -> None:
    """Say how much of the available text could exercise normalization at all."""
    total = 0
    outside_ascii = 0
    for _, text in sources():
        total += len(text)
        outside_ascii += sum(1 for c in text if ord(c) > HIGHEST_ASCII)
    say()
    say(f"characters in all sources        : {total:,}")
    say(f"characters outside ASCII         : {outside_ascii:,}")
    if outside_ascii == 0:
        say("**No source contains a character normalization could affect.**")


def main() -> None:
    """Survey the available text, then show the constructed cases."""
    available = sources()
    say("### Available text")
    say()
    survey(available)
    coverage()
    constructed()
    say()
    say("Reproduce with: uv run python benchmarks/unicode_forms.py")


if __name__ == "__main__":
    main()
