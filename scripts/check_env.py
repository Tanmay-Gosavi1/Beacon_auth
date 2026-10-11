"""Thin wrapper: same as `python -m beacon check-env`."""

import sys

from beacon.__main__ import main

if __name__ == "__main__":
    sys.exit(main(["check-env", *sys.argv[1:]]))
