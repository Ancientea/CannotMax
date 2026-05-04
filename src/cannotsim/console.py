"""
Command-line interface for CannotSim battle simulator.
"""

import argparse
import sys


def sim_command(args):
    """Launch battle simulator (Tkinter GUI with drag-and-drop unit placement)."""
    from cannotsim.main_sim import main as sim_main

    sim_main()


def sim_mc_command(args):
    """Launch multi-core simulator (Tkinter GUI, Monte Carlo mode)."""
    from cannotsim.sim_mc import main as sim_mc_main

    sim_mc_main()


def simulate_command(args):
    """Run headless batch simulation from CSV data."""
    from cannotsim.simulate import main as batch_main

    batch_main()


def main():
    parser = argparse.ArgumentParser(
        prog="cannotsim",
        description="CannotSim - Arknights battle simulator",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    sim_parser = subparsers.add_parser(
        "sim", help="Launch battle simulator (Tkinter, drag-and-drop)"
    )
    sim_parser.set_defaults(func=sim_command)

    sim_mc_parser = subparsers.add_parser(
        "sim_mc", help="Launch multi-core simulator (Tkinter, Monte Carlo)"
    )
    sim_mc_parser.set_defaults(func=sim_mc_command)

    batch_parser = subparsers.add_parser(
        "simulate", help="Headless batch simulation from CSV"
    )
    batch_parser.set_defaults(func=simulate_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
