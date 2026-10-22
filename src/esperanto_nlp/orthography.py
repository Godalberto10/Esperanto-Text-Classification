"""Esperanto orthographic normalization.

The accented letters (ĉ, ĝ, ĥ, ĵ, ŝ, ŭ) are often typed with the "x-system"
(cx, gx, hx, jx, sx, ux) when the keyboard lacks them. Since the letter x is
not part of the Esperanto alphabet, the conversion is unambiguous for
Esperanto words (it could still affect foreign names: "Linux").
"""

import re

_X_SYSTEM = {
    "cx": "ĉ", "gx": "ĝ", "hx": "ĥ", "jx": "ĵ", "sx": "ŝ", "ux": "ŭ",
    "Cx": "Ĉ", "Gx": "Ĝ", "Hx": "Ĥ", "Jx": "Ĵ", "Sx": "Ŝ", "Ux": "Ŭ",
    "CX": "Ĉ", "GX": "Ĝ", "HX": "Ĥ", "JX": "Ĵ", "SX": "Ŝ", "UX": "Ŭ",
}
_X_RE = re.compile("|".join(_X_SYSTEM))


def from_x_system(text):
    """Converts x-system text to accented letters: 'cxiu' -> 'ĉiu'."""
    return _X_RE.sub(lambda m: _X_SYSTEM[m.group()], text)
