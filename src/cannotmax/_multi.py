"""Thin launcher for multi-instance manager as standalone exe."""

import sys

if __name__ == "__main__":
    sys.argv = ["cannotmax", "multi"]
    from cannotmax.console import main

    main()
