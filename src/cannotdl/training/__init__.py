"""
Training module for Arknights battle prediction models.
"""

from .evaluator import main as eval_main
from .trainer import main as train_main

__all__ = ["train_main", "eval_main"]
