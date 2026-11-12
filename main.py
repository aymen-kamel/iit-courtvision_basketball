"""Shortcut: `python main.py --input VIDEO --output DIR` == `python -m courtvision run ...`."""
import sys

from courtvision.cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or argv[0] not in ("run", "inspect", "models", "-h", "--help"):
        argv = ["run", *argv]
    sys.exit(main(argv))
