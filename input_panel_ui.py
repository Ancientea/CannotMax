from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame
from PyQt6.QtWidgets import QLineEdit, QScrollArea, QGridLayout, QSizePolicy, QGraphicsDropShadowEffect
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QPainter, QColor
import numpy as np
import logging

from config import MONSTER_COUNT, MONSTER_DATA

logger = logging.getLogger(__name__)


class InputPanelUI(QFrame):
    # Signals to communicate with the main application
    predict_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    input_changed = pyqtSignal()  # Signal emitted when any monster input changes

    def __init__(self):
        super().__init__()
        self.left_monsters: dict[str, str] = {}
        self.right_monsters: dict[str, str] = {}

        self.init_ui()
        self.load_images()  # Load images and populate the grid

    def init_ui(self):
        self.setObjectName("input_panel_id")
        self.setStyleSheet(
            """
            QWidget#input_panel_id {
                background-color: rgba(0, 0, 0, 40);
                border-radius: 15px;
                border: 5px solid #F5EA2D;
            }
            """
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # 人物显示区域
        monster_group = QWidget()
        monster_layout = QVBoxLayout(monster_group)

        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            """
            QScrollBar:horizontal {
                background: rgba(0, 0, 0, 0);
                width: 12px;  /* 宽度 */
                margin: 0px;  /* 边距 */
            }
            QScrollBar::handle:horizontal {
                background: rgba(100, 100, 100, 150);
                min-height: 20px;  /* 滑块最小高度 */
                border-radius: 8px;  /* 圆角 */
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                background: none;  /* 隐藏箭头按钮 */
            }
            QScrollBar:vertical {
                background: rgba(0, 0, 0, 0);
                width: 12px;  /* 宽度 */
                margin: 0px;  /* 边距 */
            }
            QScrollBar::handle:vertical {
                background: rgba(100, 100, 100, 150);
                min-height: 20px;  /* 滑块最小高度 */
                border-radius: 8px;  /* 圆角 */
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: none;  /* 隐藏箭头按钮 */
            }
            QScrollArea {
                background-color: rgba(0, 0, 0, 0);
                border:0px
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QScrollBar:vertical {
                background: rgba(50, 50, 50, 100);
                width: 12px;
                margin: 15px 0 15px 0;
            }
            QScrollBar::handle:vertical {
                background: rgba(100, 100, 100, 150);
                min-height: 20px;
                border-radius: 6px;
            }
        """
        )

        scroll_content = QWidget()
        self.scroll_grid = QGridLayout(scroll_content)
        self.scroll_grid.setSpacing(5)
        self.scroll_grid.setContentsMargins(5, 5, 5, 5)

        # 设置5列布局
        self.COLUMNS = 7
        self.ROW_HEIGHT = 120  # 每个单元的高度

        scroll.setWidget(scroll_content)
        monster_layout.addWidget(scroll)
        self.main_layout.addWidget(monster_group)

        result_button = QWidget()
        result_button_layout = QHBoxLayout(result_button)

        # 预测按钮 - 带样式
        self.predict_button = QPushButton("开始预测")
        self.predict_button.clicked.connect(self.predict_requested.emit)
        self.predict_button.setStyleSheet(
            """
                QPushButton {
                    background-color: #313131;
                    color: #F3F31F;
                    border-radius: 16px;
                    padding: 8px;
                    font-weight: bold;
                    min-height: 30px;
                }
                QPushButton:hover {
                    background-color: #414141;
                }
                QPushButton:pressed {
                    background-color: #212121;
                }
            """
        )
        self.predict_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset_entries)
        self.reset_button.setStyleSheet(
            """
                QPushButton {
                    background-color: #313131;
                    color: #F3F31F;
                    border-radius: 16px;
                    padding: 8px;
                    font-weight: bold;
                    min-height: 30px;
                }
                QPushButton:hover {
                    background-color: #414141;
                }
                QPushButton:pressed {
                    background-color: #212121;
                }
            """
        )

        result_button_layout.addWidget(self.predict_button)
        result_button_layout.addWidget(self.reset_button)

        self.main_layout.addWidget(result_button)

    def load_images(self):
        for i in reversed(range(self.scroll_grid.count())):
            self.scroll_grid.itemAt(i).widget().setParent(None)

        # 重新计算布局
        row = 0
        col = 0

        for i in range(1, MONSTER_COUNT + 1):
            # 容器
            monster_container = QWidget()
            monster_container.setFixedHeight(self.ROW_HEIGHT)
            shadow01 = QGraphicsDropShadowEffect()
            shadow01.setBlurRadius(5)  # 模糊半径（控制发光范围）
            shadow01.setColor(QColor(0, 0, 0, 120))  # 发光颜色
            shadow01.setOffset(3)  # 偏移量（0表示均匀四周发光）
            monster_container.setGraphicsEffect(shadow01)

            monster_container.setStyleSheet("QWidget {border-radius: 0px;}")
            container_layout = QVBoxLayout(monster_container)
            container_layout.setSpacing(2)
            container_layout.setContentsMargins(2, 2, 2, 2)

            # 人物图片
            img_label = QLabel()
            img_label.setFixedSize(60, 60)
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            try:
                pixmap = QPixmap(f"images/{MONSTER_DATA['原始名称'][i]}.png")
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(
                        60,
                        60,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    img_label.setPixmap(pixmap)
            except Exception as e:
                logger.error(f"Error loading character {i} image: {str(e)}")

            # 添加鼠标悬浮提示
            if i in MONSTER_DATA.index:
                data = MONSTER_DATA.loc[i].to_dict()
                tooltip_text = ""
                for key, value in data.items():
                    tooltip_text += f"{key}: {value}\n"
                img_label.setToolTip(tooltip_text.strip())

            # 左输入框
            left_entry = QLineEdit()
            left_entry.setFixedWidth(60)
            left_entry.setPlaceholderText("左")
            left_entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
            left_entry.textChanged.connect(self.input_changed.emit)  # Connect to signal
            self.left_monsters[str(i)] = left_entry

            # 右输入框 (放在左输入框下方)
            right_entry = QLineEdit()
            right_entry.setFixedWidth(60)
            right_entry.setPlaceholderText("右")
            right_entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
            right_entry.textChanged.connect(self.input_changed.emit)  # Connect to signal
            self.right_monsters[str(i)] = right_entry

            # 添加到容器
            container_layout.addWidget(img_label, 0, Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(left_entry, 0, Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(right_entry, 0, Qt.AlignmentFlag.AlignCenter)

            # 添加到网格布局
            self.scroll_grid.addWidget(monster_container, row, col, Qt.AlignmentFlag.AlignCenter)

            # 更新行列位置
            col += 1
            if col >= self.COLUMNS:
                col = 0
                row += 1

    def reset_entries(self):
        for entry in self.left_monsters.values():
            entry.clear()
            entry.setStyleSheet("")
        for entry in self.right_monsters.values():
            entry.clear()
            entry.setStyleSheet("")
        self.input_changed.emit()  # Emit signal after resetting
        self.reset_requested.emit()  # Emit signal for main app to handle other resets

    def get_monster_counts(self):
        """Returns current monster counts from input fields."""
        return self.left_monsters, self.right_monsters

    def set_monster_counts(self, left_counts: dict, right_counts: dict):
        """Sets monster counts in input fields (e.g., after recognition)."""
        # Clear existing values and styles
        for entry in self.left_monsters.values():
            entry.setText("")
            entry.setStyleSheet("")
        for entry in self.right_monsters.values():
            entry.setText("")
            entry.setStyleSheet("")
        for monster_id, count in left_counts.items():
            if monster_id in self.left_monsters:
                self.left_monsters[monster_id].setText(str(count))
                if count > 0:
                    self.left_monsters[monster_id].setStyleSheet("background-color: yellow;")
                else:
                    self.left_monsters[monster_id].setStyleSheet("")
        for monster_id, count in right_counts.items():
            if monster_id in self.right_monsters:
                self.right_monsters[monster_id].setText(str(count))
                if count > 0:
                    self.right_monsters[monster_id].setStyleSheet("background-color: yellow;")
                else:
                    self.right_monsters[monster_id].setStyleSheet("")
        self.input_changed.emit()
