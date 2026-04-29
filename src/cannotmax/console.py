"""
Command-line interface for CannotMax training and evaluation.
"""
import argparse
import sys
from pathlib import Path


def train_command(args):
    """Train the model."""
    from .training.trainer import main as train_main
    train_main()


def eval_command(args):
    """Evaluate the model."""
    from .training.evaluator import main as eval_main
    eval_main()


def convert_model_command(args):
    """Convert PyTorch model to ONNX."""
    import subprocess
    script_path = Path(__file__).parent / "cli" / "convert_model.py"
    result = subprocess.run([sys.executable, str(script_path)], cwd=args.cwd if hasattr(args, 'cwd') else None)
    sys.exit(result.returncode)


def main():
    """Main CLI entry point."""
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
        "--input", "-i",
        required=True,
        help="Input PyTorch model path (.pth)",
    )
    convert_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output ONNX model path (.onnx)",
    )
    convert_parser.set_defaults(func=convert_model_command)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
