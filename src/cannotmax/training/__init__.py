"""
Training module for Arknights battle prediction models.
"""
from .trainer import main as train_main
from .evaluator import main as eval_main

__all__ = ["train_main", "eval_main"]
