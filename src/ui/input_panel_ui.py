import json
import re
from collections import defaultdict
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
)
from PyQt6.QtWidgets import (
    QLineEdit,
    QScrollArea,
    QGridLayout,
    QSizePolicy,
    QGraphicsDropShadowEffect,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QColor
import numpy as np
import logging

from src.core.config import MONSTER_COUNT, MONSTER_DATA, FIELD_FEATURE_COUNT
from src.core.paths import image_path, resource_path

logger = logging.getLogger(__name__)


class InputPanelUI(QFrame):
    # Signals to communicate with the main application
    predict_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    input_changed = (
        pyqtSignal()
    )  # Signal emitted when any monster input changes
    terrain_changed = pyqtSignal(list)  # New signal for terrain changes

    # Terrain display mapping
    terrain_display_mapping = {
        "altar_vertical_altar": "垂直祭坛",
        "block_parallel_block": "平行方块阻挡",
        "block_vertical_altar_shape1": "垂直祭坛形阻挡1",
        "block_vertical_altar_shape2": "垂直祭坛形阻挡2",
        "block_vertical_block_shape1": "垂直方块阻挡1",
        "block_vertical_block_shape2": "垂直方块阻挡2",
        "coil_narrow_coil": "窄型线圈装置",
        "coil_wide_coil": "宽型线圈装置",
        "crossbow_top_crossbow": "顶部弩炮",
        "fire_side_crossbow": "侧边弩炮",
        "fire_side_fire": "侧边火炮",
        "fire_top_fire": "顶部火炮",
    }

    @staticmethod
    def get_terrain_feature_columns():
        """
        从data_cleaning_with_field_recognize_gpu.py的逻辑中获取地形特征列名
        """
        try:
            # 加载类别映射
            class_map_path = resource_path(
                "tools", "battlefield_recognize", "class_to_idx.json"
            )
            with open(class_map_path, "r", encoding="utf-8") as f:
                class_to_idx = json.load(f)

            # 使用与data_cleaning_with_field_recognize_gpu.py相同的逻辑
            grouped_elements = defaultdict(list)
            for class_name in class_to_idx.keys():
                if class_name.endswith("_none"):
                    continue
                condensed_name = re.sub(r"_left_", "_", class_name)
                condensed_name = re.sub(r"_right_", "_", condensed_name)
                grouped_elements[condensed_name].append(class_name)

            # 返回排序后的特征列名
            return sorted(grouped_elements.keys())
        except Exception as e:
            logger.warning(f"无法获取地形特征列名，使用默认值: {e}")
            # 如果无法获取，返回空列表，让系统使用默认值
            return []

    def __init__(self):
        super().__init__()
        self.left_monsters: dict[str, str] = {}
        self.right_monsters: dict[str, str] = {}
        self.terrain_buttons = {}  # Initialize terrain buttons
        self.terrain_feature_columns = self.get_terrain_feature_columns()
        if not self.terrain_feature_columns:
            # If fetching fails, use default terrain features
            self.terrain_feature_columns = [
                "altar_vertical",
                "block_parallel",
                "block_vertical_altar",
                "block_vertical_block",
                "coil_narrow",
                "coil_wide",
                "crossbow_top",
                "fire_side_left",
                "fire_side_right",
                "fire_top",
            ]

        self.init_ui()
        self.load_images()  # Load images and populate the grid

    def init_ui(self):
        self.setObjectName("input_panel_id")
        self.setStyleSheet("""
            QWidget#input_panel_id {
                background-color: rgba(0, 0, 0, 40);
                border-radius: 15px;
                border: 5px solid #F5EA2D;
            }
            """)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # 人物显示区域
        monster_group = QWidget()
        monster_layout = QVBoxLayout(monster_group)

        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
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
        """)

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

        # 第四行 - 地形选择（支持多选）
        row4 = QWidget()
        row4_layout = QVBoxLayout(row4)

        # 地形选择标签
        terrain_header = QWidget()
        terrain_header_layout = QHBoxLayout(terrain_header)
        terrain_header_layout.setContentsMargins(0, 0, 0, 0)

        terrain_label = QLabel("地形选择（多选）:")
        terrain_label.setStyleSheet("color: #414141; font-weight: bold;")
        terrain_header_layout.addWidget(terrain_label)

        # 只保留清空按钮
        clear_all_btn = QPushButton("清空")
        clear_all_btn.setFixedSize(40, 20)
        clear_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #5A5A5A;
                color: #FAFAFA;
                border-radius: 4px;
                padding: 2px;
                font-size: 8px;
            }
            QPushButton:hover {
                background-color: #6A6A6A;
            }
        """)
        clear_all_btn.clicked.connect(self.clear_all_terrains)

        terrain_header_layout.addWidget(clear_all_btn)
        terrain_header_layout.addStretch()

        row4_layout.addWidget(terrain_header)

        # 创建地形选择按钮组（多选）
        self.terrain_group = QWidget()
        terrain_group_layout = QVBoxLayout(self.terrain_group)
        terrain_group_layout.setSpacing(3)

        # 分两行显示按钮，每行6个，更好地利用空间
        terrain_rows = []
        for row_idx in range(2):  # Changed from 3 to 2 rows
            row_terrain = QWidget()
            row_layout = QHBoxLayout(row_terrain)
            row_layout.setSpacing(3)
            row_layout.setContentsMargins(0, 0, 0, 0)
            terrain_rows.append((row_terrain, row_layout))

        for i, terrain_key in enumerate(self.terrain_feature_columns):
            display_name = self.terrain_display_mapping.get(
                terrain_key, terrain_key
            )

            btn = QPushButton(display_name)
            btn.setCheckable(True)
            btn.setFixedHeight(26)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #7B7B7B;
                    color: #FAFAFA;
                    border-radius: 5px;
                    padding: 0px;
                    font-size: 9px;
                    min-width: 60px;
                    max-width: 90px;
                }
                QPushButton:checked {
                    background-color: #F3F31F;
                    color: #313131;
                    border: 2px solid #7B7B7B;
                    font-weight: bold;
                    padding: 0px;
                }
                QPushButton:hover {
                    border: 2px solid #616161;
                    background-color: #c0c00c;
                    color: #313131;
                    padding: 0px;
                }
                """)
            btn.clicked.connect(
                lambda checked, k=terrain_key: self.on_terrain_multi_selected(
                    k
                )
            )
            self.terrain_buttons[terrain_key] = btn

            # 分两行显示，每行6个按钮
            row_index = i // 6  # Changed from 4 to 6 buttons per row
            if row_index < len(terrain_rows):
                terrain_rows[row_index][1].addWidget(btn)
            else:
                # If it exceeds the expected, add to the last row
                terrain_rows[-1][1].addWidget(btn)

        # Add elastic space to each row, centered
        for row_terrain, row_layout in terrain_rows:
            row_layout.addStretch()
            terrain_group_layout.addWidget(row_terrain)

        row4_layout.addWidget(self.terrain_group)

        # Add row4 to main_layout
        self.main_layout.addWidget(row4)

        result_button = QWidget()
        result_button_layout = QHBoxLayout(result_button)

        # 预测按钮 - 带样式
        self.predict_button = QPushButton("开始预测")
        self.predict_button.clicked.connect(self.predict_requested.emit)
        self.predict_button.setStyleSheet("""
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
            """)
        self.predict_button.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        self.reset_button = QPushButton("重置")
        self.reset_button.clicked.connect(self.reset_entries)
        self.reset_button.setStyleSheet("""
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
            """)

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
                pixmap = QPixmap(
                    str(image_path(MONSTER_DATA['原始名称'][i]))
                )
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
            left_entry.textChanged.connect(
                self.input_changed.emit
            )  # Connect to signal
            self.left_monsters[str(i)] = left_entry

            # 右输入框 (放在左输入框下方)
            right_entry = QLineEdit()
            right_entry.setFixedWidth(60)
            right_entry.setPlaceholderText("右")
            right_entry.setAlignment(Qt.AlignmentFlag.AlignCenter)
            right_entry.textChanged.connect(
                self.input_changed.emit
            )  # Connect to signal
            self.right_monsters[str(i)] = right_entry

            # 添加到容器
            container_layout.addWidget(
                img_label, 0, Qt.AlignmentFlag.AlignCenter
            )
            container_layout.addWidget(
                left_entry, 0, Qt.AlignmentFlag.AlignCenter
            )
            container_layout.addWidget(
                right_entry, 0, Qt.AlignmentFlag.AlignCenter
            )

            # 添加到网格布局
            self.scroll_grid.addWidget(
                monster_container, row, col, Qt.AlignmentFlag.AlignCenter
            )

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
                    self.left_monsters[monster_id].setStyleSheet(
                        "background-color: yellow;"
                    )
                else:
                    self.left_monsters[monster_id].setStyleSheet("")
        for monster_id, count in right_counts.items():
            if monster_id in self.right_monsters:
                self.right_monsters[monster_id].setText(str(count))
                if count > 0:
                    self.right_monsters[monster_id].setStyleSheet(
                        "background-color: yellow;"
                    )
                else:
                    self.right_monsters[monster_id].setStyleSheet("")
        self.input_changed.emit()

    def on_terrain_multi_selected(self, terrain_key):
        """处理地形多选事件"""
        selected_terrains = self.get_selected_terrains()
        logger.info(f"当前选择的地形: {selected_terrains}")
        self.terrain_changed.emit(selected_terrains)

    def clear_all_terrains(self):
        """清空所有地形选择"""
        for btn in self.terrain_buttons.values():
            btn.setChecked(False)
        self.terrain_changed.emit([])

    def get_selected_terrains(self):
        """获取当前选择的地形列表"""
        selected = []
        for key, btn in self.terrain_buttons.items():
            if btn.isChecked():
                selected.append(key)
        return selected

    def build_terrain_features(self, left_counts, right_counts):
        """构建包含地形的完整特征向量（支持多选地形）"""
        # Use the actual number of terrain feature columns
        num_field_features = (
            len(self.terrain_feature_columns) if FIELD_FEATURE_COUNT else 0
        )

        # Build terrain feature vector
        terrain_features = np.zeros(num_field_features)

        selected_terrains = self.get_selected_terrains()

        # Set feature value to 1 for selected terrains
        for terrain in selected_terrains:
            if terrain in self.terrain_feature_columns:
                terrain_idx = self.terrain_feature_columns.index(terrain)
                if terrain_idx < num_field_features:
                    terrain_features[terrain_idx] = 1
            else:
                logger.warning(
                    f"Terrain {terrain} not in feature columns: {self.terrain_feature_columns}"
                )

        logger.debug(f"Selected terrains: {selected_terrains}")
        logger.debug(f"Terrain feature vector: {terrain_features}")

        # Organize data according to data_cleaning_with_field_recognize_gpu.py format
        # 1L-61L (Left monster features)
        # 62L-73L (Field features L)
        # 1R-61R (Right monster features)
        # 62R-73R (Field features R, copied)

        full_features = np.concatenate(
            [
                left_counts,  # 1L-61L (Left monster features)
                terrain_features,  # 62L-73L (Field features L)
                right_counts,  # 1R-61R (Right monster features)
                terrain_features,  # 62R-73R (Field features R, copied)
            ]
        )

        return full_features
