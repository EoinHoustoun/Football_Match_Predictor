"""Every text colour in the UI must clear 4.5:1 against the app background.

Eoin does not find light grey type readable — "I prefer white font on black and
black font on white, it suits my eyes much better" — so this is an accessibility
constraint, not a style preference.

The app had 78 text declarations below the floor, including `#334` at **1.56:1**
in the footer and `#556` at 2.64:1 across 65 sub-labels. They look refined in a
screenshot and disappear on a real screen.

This walks the CSS and inline styles rather than trusting review, because the
next hardcoded colour will be added by someone reaching for "a slightly dimmer
grey" and it will look fine to them.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app.py"

BACKGROUND = "#0a0e1a"
FLOOR = 7.0

# Some text is deliberately dark ink on a bright fill — the green form-W chip,
# the yellow draw chip, the percentage sitting on a filled bar. Scoring those
# against the page background reads 1.04:1 and means nothing; on the fill they
# actually sit on they are better than 10:1.
#
# Guessing from the ink's own darkness does not work: #1a1d27 is dark to the eye
# but still lighter than this near-black page. So the ground is read from the
# same rule — the `background` declared in the same CSS block, or in the same
# inline style attribute — and only falls back to the page when there is none.


def _luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(fg: str, bg: str = BACKGROUND) -> float:
    a, b = _luminance(fg), _luminance(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _text_colours() -> list[tuple[int, str]]:
    """Every `color: #hex` declaration, with the ground it is painted on."""
    text = APP.read_text()
    colour_re = re.compile(r"(?<!-)color:\s*(#[0-9a-fA-F]{3,6})\b")
    bg_re = re.compile(r"background(?:-color)?:\s*(?:[^;{}\"\']*?)?(#[0-9a-fA-F]{3,6})\b")
    found = []
    for match in colour_re.finditer(text):
        colour = match.group(1)
        if len(colour.lstrip("#")) not in (3, 6):
            continue
        # The enclosing rule: a CSS block delimited by braces, or an inline
        # style attribute delimited by quotes. Whichever boundary is nearer.
        start = max(text.rfind("{", 0, match.start()),
                    text.rfind('style="', 0, match.start()))
        end = text.find("}", match.end())
        quote_end = text.find('"', match.end())
        if quote_end != -1 and (end == -1 or quote_end < end):
            end = quote_end
        scope = text[start:end if end != -1 else match.end()]
        # `background-clip: text` paints the background THROUGH the glyphs and
        # makes `color` a fallback that is never shown. Scoring the fallback
        # against the gradient behind it measures nothing real.
        if "background-clip" in scope and "text-fill-color: transparent" in scope:
            continue
        # A rule with no fill of its own can declare the ground it is painted
        # over: `/* on: #ffd600 */`.
        declared = re.search(r"/\*\s*on:\s*(#[0-9a-fA-F]{3,6})\s*\*/", scope)
        bg = bg_re.search(scope)
        ground = (declared.group(1) if declared
                  else bg.group(1) if bg else BACKGROUND)
        lineno = text.count("\n", 0, match.start()) + 1
        found.append((lineno, colour, ground))
    return found


def test_the_scan_finds_colours_at_all():
    """A regex that silently matches nothing would make this suite vacuous."""
    assert len(_text_colours()) > 100


@pytest.mark.parametrize("floor_case", [
    ("#ffffff", True), ("#e8eaf0", True),
    ("#556", False), ("#334", False),
])
def test_the_contrast_maths_is_right(floor_case):
    colour, should_pass = floor_case
    assert (contrast(colour) >= FLOOR) is should_pass


def test_dark_ink_is_scored_against_the_fill_it_sits_on():
    """The green and yellow form chips carry near-black text on a bright fill
    and are highly readable. Scored against the page they would read 1.0:1."""
    assert contrast("#003", "#00e676") > 10
    assert contrast("#332200", "#ffd600") > 10


def test_the_scan_reads_the_ground_from_the_same_rule():
    """If the background lookup silently failed, every chip would be scored
    against the page and the suite would be measuring the wrong thing."""
    grounds = {ground for _, _, ground in _text_colours()}
    assert grounds - {BACKGROUND}, "no rule-local backgrounds were resolved"


def test_no_text_colour_falls_below_the_readable_floor():
    failures = [
        (lineno, colour, ground, round(contrast(colour, ground), 2))
        for lineno, colour, ground in _text_colours()
        if contrast(colour, ground) < FLOOR
    ]
    assert not failures, (
        "Text below 4.5:1 against the ground it sits on:\n"
        + "\n".join(f"  app.py:{ln}  {c} on {g}  {ratio}:1"
                     for ln, c, g, ratio in failures)
    )
