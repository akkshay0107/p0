"""Training command-line composition root."""

from __future__ import annotations

import argparse
import signal
import threading

from p0.training.composition import run_training
from p0.training.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the p0 VGC policy.")
    parser.add_argument("--config", help="Path to the YAML application configuration.")
    args = parser.parse_args(argv)
    stop = threading.Event()
    previous = {
        name: signal.signal(name, lambda *_: stop.set()) for name in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        run_training(load_config(args.config), cancel_requested=stop.is_set)
    finally:
        for name, handler in previous.items():
            signal.signal(name, handler)
    return 0
