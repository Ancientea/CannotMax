"""
Command-line interface for CannotDeeper ML training pipeline.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def _run_dev_script(category: str, name: str, args: list[str]):
    import cannotdl
    import cannotmax

    search_dirs = [
        Path(cannotdl.__file__).parent / category,
        Path(cannotmax.__file__).parent / category,
    ]

    script = None
    for d in search_dirs:
        candidate = d / f"{name}.py"
        if candidate.exists():
            script = candidate
            break

    if script is None:
        candidates = []
        for d in search_dirs:
            if d.exists():
                candidates.extend(p.stem for p in d.glob("*.py"))
        print(f"错误: 找不到脚本 '{name}'")
        if candidates:
            print(f"可用脚本: {', '.join(sorted(set(candidates)))}")
        sys.exit(1)

    result = subprocess.run([sys.executable, str(script), *args])
    sys.exit(result.returncode)


def train_command(args):
    """Train the model."""
    from cannotdl.training.trainer import main as train_main

    train_main()


def eval_command(args):
    """Evaluate the model."""
    from cannotdl.training.evaluator import main as eval_main

    eval_main()


def convert_command(args):
    """Convert PyTorch model to ONNX."""
    import cannotdl

    script_path = Path(cannotdl.__file__).parent / "tools" / "convert_model.py"
    result = subprocess.run([sys.executable, str(script_path)])
    sys.exit(result.returncode)


def tools_command(args):
    """Run a development tool script."""
    _run_dev_script("tools", args.script, [])


def pipelines_command(args):
    """Run a data pipeline script."""
    _run_dev_script("pipelines", args.script, [])


def main():
    parser = argparse.ArgumentParser(
        prog="cannotdl",
        description="CannotDeeper - ML training pipeline for Arknights battle prediction",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.set_defaults(func=train_command)

    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    eval_parser.set_defaults(func=eval_command)

    convert_parser = subparsers.add_parser(
        "convert", help="Convert PyTorch model to ONNX"
    )
    convert_parser.add_argument(
        "--input",
        "-i",
        help="Input PyTorch model path (.pth) - currently unused, uses default path",
    )
    convert_parser.add_argument(
        "--output",
        "-o",
        help="Output ONNX model path (.onnx) - currently unused, derived from input",
    )
    convert_parser.set_defaults(func=convert_command)

    tools_parser = subparsers.add_parser("tools", help="Run a development tool script")
    tools_parser.add_argument("script", help="Tool script name (without .py)")
    tools_parser.set_defaults(func=tools_command)

    pipelines_parser = subparsers.add_parser(
        "pipelines", help="Run a data pipeline script"
    )
    pipelines_parser.add_argument("script", help="Pipeline script name (without .py)")
    pipelines_parser.set_defaults(func=pipelines_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
