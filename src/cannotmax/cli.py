"""CLI entry point for CannotMax"""
import sys
import os

# Set environment variable before importing OpenCV
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

# 预先导入 UnitAwareTransformer 以解决 torch.load 反序列化问题
from .train import UnitAwareTransformer  # noqa: F401

from PyQt6.QtWidgets import QApplication
from .gui.main_window import ArknightsApp


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    window = ArknightsApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
