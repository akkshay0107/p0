"""Training command-line composition root."""

from __future__ import annotations

import argparse

from p0.train.train_loop import main as run_training


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the p0 VGC policy.")
    parser.parse_args()
    run_training()
    return 0
