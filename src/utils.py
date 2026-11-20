"""Small helpers shared by several scripts."""

import sys


def make_console_safe() -> None:
    """Stop printing from crashing on Windows consoles that cannot show some
    characters (e.g. Hebrew playlist names or accented artist names)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
