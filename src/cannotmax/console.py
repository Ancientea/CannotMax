"""
Command-line interface for CannotMax runtime (GUI + automation).
"""

import argparse
import sys
from pathlib import Path


def tools_command(args):
    """Run a development tool."""
    import subprocess
    import sys as _sys

    script = Path(__file__).parent / "tools" / f"{args.script}.py"
    if not script.exists():
        print(f"错误：找不到工具脚本 {args.script}.py")
        _sys.exit(1)
    result = subprocess.run([_sys.executable, str(script)])
    _sys.exit(result.returncode)


def multi_instance_command(args):
    """Launch multi-instance automation manager."""
    import sys as _sys

    from PyQt6.QtWidgets import QApplication

    app = QApplication(_sys.argv)
    from cannotmax.gui.multi_instance import MultiInstanceManager

    window = MultiInstanceManager()
    window.show()
    _sys.exit(app.exec())


def _ensure_admin():
    """Ensure the process is running with administrator privileges for PC mode."""
    import ctypes
    import os

    try:
        if ctypes.windll.shell32.IsUserAnAdmin():
            return
    except Exception:
        return

    if getattr(sys, "frozen", False):
        params = " ".join(sys.argv[1:])
    else:
        params = "-m cannotmax"
        if sys.argv[1:]:
            params += " " + " ".join(sys.argv[1:])

    ctypes.windll.shell32.ShellExecuteW(
        None,
        "runas",
        sys.executable,
        params,
        os.getcwd(),
        1,
    )
    sys.exit(0)


def main():
    """Main CLI entry point."""
    _ensure_admin()

    parser = argparse.ArgumentParser(
        prog="cannotmax",
        description="CannotMax - Arknights battle predictor",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    multi_parser = subparsers.add_parser("multi", help="Launch multi-instance manager")
    multi_parser.set_defaults(func=multi_instance_command)

    tools_parser = subparsers.add_parser("tools", help="Run a development tool")
    tools_parser.add_argument("script", help="Tool script name (without .py)")
    tools_parser.set_defaults(func=tools_command)

    args = parser.parse_args()

    if args.command is None:
        import sys as _sys

        from PyQt6.QtWidgets import QApplication

        app = QApplication(_sys.argv)
        from cannotmax.gui.main_window import ArknightsApp

        window = ArknightsApp()
        window.show()
        _sys.exit(app.exec())

    args.func(args)


if __name__ == "__main__":
    main()
