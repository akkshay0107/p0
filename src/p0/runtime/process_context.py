"""Shared multiprocessing context for application worker processes."""

from __future__ import annotations

import multiprocessing
import sys


def _start_method() -> str:
    """Choose the safest efficient context available for this platform."""
    if sys.platform == "linux" and "forkserver" in multiprocessing.get_all_start_methods():
        return "forkserver"
    return "spawn"


# Selecting a forkserver context does not start the server yet; it is launched
# lazily when the first process pool or worker is created.
PROCESS_CONTEXT = multiprocessing.get_context(_start_method())
