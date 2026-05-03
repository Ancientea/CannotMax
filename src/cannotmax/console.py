"""
Command-line interface for CannotMax training and evaluation.
"""

import argparse
import sys
from pathlib import Path


def train_command(args):
    """Train the model."""
    from cannotmax.training.trainer import main as train_main

    train_main()


def eval_command(args):
    """Evaluate the model."""
    from cannotmax.training.evaluator import main as eval_main

    eval_main()


def convert_model_command(args):
    """Convert PyTorch model to ONNX."""
    import subprocess

    script_path = Path(__file__).parent / "tools" / "convert_model.py"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=args.cwd if hasattr(args, "cwd") else None,
    )
    sys.exit(result.returncode)


def multi_instance_command(args):
    """Launch multi-instance automation manager."""
    import sys as _sys

    from PyQt6.QtWidgets import QApplication

    app = QApplication(_sys.argv)
    from cannotmax.gui.multi_instance import MultiInstanceManager

    window = MultiInstanceManager()
    window.show()
    _sys.exit(app.exec())


def _run_dev_script(category: str, script_name: str):
    """Run a development script from tools/ or pipelines/ via subprocess."""
    script_path = Path(__file__).parent / category / f"{script_name}.py"
    if not script_path.exists():
        available = [
            p.stem
            for p in (Path(__file__).parent / category).glob("*.py")
            if p.stem != "__init__"
        ]
        print(f"Unknown {category} script: {script_name}")
        print(f"Available: {', '.join(sorted(available))}")
        sys.exit(1)
    import subprocess

    result = subprocess.run([sys.executable, str(script_path)])
    sys.exit(result.returncode)


def tools_command(args):
    """Run a development tool script."""
    _run_dev_script("tools", args.script)


def pipelines_command(args):
    """Run a data pipeline script."""
    _run_dev_script("pipelines", args.script)


def _ensure_admin():
    """Ensure the process is running with administrator privileges for PC mode."""
    import ctypes
    import os

    try:
        if ctypes.windll.shell32.IsUserAnAdmin():
            return
    except Exception:
        return

    args = sys.argv[1:]
    params = "-m cannotmax"
    if args:
        params += " " + " ".join(args)
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

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.set_defaults(func=train_command)

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.set_defaults(func=eval_command)

    # Convert model command
    convert_parser = subparsers.add_parser("convert", help="Convert PyTorch to ONNX")
    convert_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input PyTorch model path (.pth)",
    )
    convert_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output ONNX model path (.onnx)",
    )
    convert_parser.set_defaults(func=convert_model_command)

    # Multi-instance command
    multi_parser = subparsers.add_parser("multi", help="Launch multi-instance manager")
    multi_parser.set_defaults(func=multi_instance_command)

    # Tools command (dev only)
    tools_parser = subparsers.add_parser("tools", help="Run a development tool script")
    tools_parser.add_argument("script", help="Tool script name (without .py)")
    tools_parser.set_defaults(func=tools_command)

    # Pipelines command (dev only)
    pipelines_parser = subparsers.add_parser(
        "pipelines", help="Run a data pipeline script"
    )
    pipelines_parser.add_argument("script", help="Pipeline script name (without .py)")
    pipelines_parser.set_defaults(func=pipelines_command)

    args = parser.parse_args()

    if args.command is None:
        # 无命令时启动 GUI
        import sys as _sys

        from PyQt6.QtWidgets import QApplication

        app = QApplication(_sys.argv)  # PyQt6 默认启用 High DPI
        from cannotmax.gui.main_window import ArknightsApp

        window = ArknightsApp()
        window.show()
        _sys.exit(app.exec())

    args.func(args)


if __name__ == "__main__":
    main()
