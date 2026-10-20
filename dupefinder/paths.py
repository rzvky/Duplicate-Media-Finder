"""Path helpers. On Windows, paths longer than ~260 characters need the \\\\?\\ prefix."""

from __future__ import annotations

import os


def lp(p) -> str:
    """Return a path string that works even when it is very long on Windows."""
    s = os.path.abspath(str(p))
    if os.name == "nt" and len(s) >= 240 and not s.startswith("\\\\?\\"):
        if s.startswith("\\\\"):  # network share \\server\share\...
            return "\\\\?\\UNC\\" + s[2:]
        return "\\\\?\\" + s
    return s
