"""
Entry point for running CannotMax via `python -m src.cannotmax`
"""
# 预先导入 UnitAwareTransformer 以解决 torch.load 反序列化问题
from .train import UnitAwareTransformer  # noqa: F401

from .cli import main

if __name__ == "__main__":
    main()
