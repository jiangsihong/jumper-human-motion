"""Open the latest mapping preview; --version 1 retains the original view."""

from __future__ import annotations
import argparse
import runpy
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--version", choices=["1", "2"], default="2")
    selected, remaining = parser.parse_known_args()
    root = Path(__file__).resolve().parent
    entry = root / f"v{selected.version}" / "preview_mapping.py"
    sys.argv = [str(entry), *remaining]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
