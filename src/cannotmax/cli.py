"""CLI entry point for CannotMax"""
import sys
import os

# Set environment variable before importing OpenCV
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

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
