#!/usr/bin/env python3
"""Compatibility wrapper for the installed ``p0-export-training`` command."""

from p0.cli.export_training import collect_export_files, main

__all__ = ["collect_export_files", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
