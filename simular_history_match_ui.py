from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QGraphicsDropShadowEffect, QFrame
from PyQt6.QtGui import QPixmap, QImage, QFont, QIcon, QPainter, QColor
import numpy as np
import logging

from similar_history_match import HistoryMatch
from config import MONSTER_COUNT, MONSTER_DATA

logger = logging.getLogger(__name__)


class HistoryMatchUI(QFrame):
    def __init__(self, history_match: HistoryMatch):
        super().__init__()
        self.history_match = history_match
        self.init_ui()

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # 创建滚动区域
        self.history_scroll_area = QScrollArea()
        self.history_scroll_area.setFixedWidth(540)
        self.history_scroll_area.setWidgetResizable(True)
        self.history_scroll_area.setStyleSheet(
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
                background-color: rgba(0, 0, 0, 40);
                border-radius: 15px;
                border: 5px solid #F5EA2D;
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

        # 创建内容部件
        self.history_widget = QWidget()
        self.history_layout = QVBoxLayout(self.history_widget)
        self.history_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 设置滚动区域内容
        self.history_scroll_area.setWidget(self.history_widget)

        # 添加到主布局
        self.main_layout.addWidget(self.history_scroll_area)

    def render_similar_matches(self, left_monsters, right_monsters):
        try:
            # 获取当前输入
            cur_left = np.zeros(MONSTER_COUNT, dtype=float)
            cur_right = np.zeros(MONSTER_COUNT, dtype=float)
            for name, entry in left_monsters.items():
                v = entry.text()
                if v.isdigit():
                    cur_left[int(name) - 1] = float(v)
            for name, entry in right_monsters.items():
                v = entry.text()
                if v.isdigit():
                    cur_right[int(name) - 1] = float(v)

            self.history_match.render_similar_matches(cur_left, cur_right)
            sims = self.history_match.sims
            top_indices = self.history_match.top20_idx

            # 清空现有内容
            for i in reversed(range(self.history_layout.count())):
                self.history_layout.itemAt(i).widget().setParent(None)

            # 添加标题
            title_label = QLabel(f"错题本")
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(0)  # 模糊半径（控制发光范围）
            shadow.setColor(QColor("#313131"))  # 发光颜色
            shadow.setOffset(2)  # 偏移量（0表示均匀四周发光）
            title_label.setGraphicsEffect(shadow)

            title_label.setStyleSheet(
                """
                    QWidget {
                        border-radius: 0px;
                        font-size: 24px;
                        font-weight: bold;
                        color: white;
                    }
                """
            )
            self.history_layout.addWidget(title_label)

            # 渲染每个历史对局
            for idx in top_indices:
                self.add_history_match(idx, sims[idx], left_monsters, right_monsters)

        except Exception as e:
            logger.error(f"渲染历史对局失败: {str(e)}")

    def add_history_match(self, idx, similarity, left_monsters, right_monsters):
        """添加单个历史对局到面板"""
        # 获取历史数据
        left = self.history_match.past_left[idx]
        right = self.history_match.past_right[idx]
        result = self.history_match.labels[idx]

        # 获取当前对局的左右单位
        cur_left = np.zeros(MONSTER_COUNT, dtype=float)
        cur_right = np.zeros(MONSTER_COUNT, dtype=float)
        for name, entry in left_monsters.items():
            v = entry.text()
            if v.isdigit():
                cur_left[int(name) - 1] = float(v)
        for name, entry in right_monsters.items():
            v = entry.text()
            if v.isdigit():
                cur_right[int(name) - 1] = float(v)

        # 计算当前对局和历史对局的相似度(不镜像和镜像两种情况)
        setL_cur = set(np.where(cur_left > 0)[0])
        setR_cur = set(np.where(cur_right > 0)[0])
        setL_past = set(np.where(left > 0)[0])
        setR_past = set(np.where(right > 0)[0])

        # 判断是否需要镜像历史对局
        should_swap = (len(setL_cur ^ setR_past) + len(setR_cur ^ setL_past)) < (
            len(setL_cur ^ setL_past) + len(setR_cur ^ setR_past)
        )

        # 获取地形名称
        terrain_name = self.history_match.get_terrain_names(idx, should_swap)

        # 创建对局容器
        match_widget = QWidget()
        match_widget.setStyleSheet(
            """
            QWidget {
                background-color: rgba(50, 50, 50, 150);
                border-radius: 10px;
                padding: 0px;
                margin: 5px;
            }
            """
        )
        match_widget.setFixedSize(500, 170)  # 增加高度以容纳地形信息
        match_layout = QVBoxLayout(match_widget)

        # 添加左右阵容
        teams_widget = QWidget()
        teams_layout = QHBoxLayout(teams_widget)

        # 根据是否需要镜像决定显示方向
        if should_swap:
            left_team = self.create_team_widget("右方", right, result == "R")
            right_team = self.create_team_widget("左方", left, result == "L")
        else:
            left_team = self.create_team_widget("左方", left, result == "L")
            right_team = self.create_team_widget("右方", right, result == "R")

        teams_layout.addWidget(left_team)
        teams_layout.addWidget(right_team)
        match_layout.addWidget(teams_widget)

        # 添加地形信息显示
        terrain_label = QLabel(f"地形: {terrain_name}")
        terrain_label.setStyleSheet(
            """
            QLabel {
                color: #CCCCCC;
                font: 10px Microsoft YaHei;
                padding: 2px 5px;
                background-color: rgba(0, 0, 0, 50);
                border-radius: 3px;
                margin: 2px;
            }
            """
        )
        terrain_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        match_layout.addWidget(terrain_label)

        self.history_layout.addWidget(match_widget)

    def create_team_widget(self, side, counts, is_winner):
        """创建单个队伍显示部件"""
        team_widget = QWidget()
        team_widget.setStyleSheet(
            f"""
            QWidget {{
                background-color: {'rgba(250, 250, 50, 150)' if is_winner else 'rgba(50, 50, 50, 100)'};
                border-radius: 8px;
                padding: 0px;
                margin: 0px;
            }}
            """
        )

        layout = QVBoxLayout(team_widget)

        # 显示区域
        ops_widget = QWidget()
        shadow01 = QGraphicsDropShadowEffect()
        shadow01.setBlurRadius(5)  # 模糊半径（控制发光范围）
        shadow01.setColor(QColor(0, 0, 0, 120))  # 发光颜色
        shadow01.setOffset(3)  # 偏移量（0表示均匀四周发光）
        ops_widget.setGraphicsEffect(shadow01)

        ops_widget.setStyleSheet(
            """
                QWidget {
                    background-color: rgba(0, 0, 0, 0);
                    border-radius: 0px;
                    padding: 0px;
                    margin: 0px;
                }
            """
        )
        ops_layout = QHBoxLayout(ops_widget)
        ops_layout.setSpacing(5)
        ops_layout.setContentsMargins(0, 0, 0, 0)

        for i, count in enumerate(counts):
            if count > 0:
                # 创建干员显示
                op_widget = QWidget()
                op_widget.setStyleSheet("background-color: rgba(0, 0, 0, 0); padding: 0px 0;margin: 0px;")
                op_layout = QVBoxLayout(op_widget)
                op_layout.setContentsMargins(0, 0, 0, 0)
                op_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

                # 干员图片
                img_label = QLabel()
                img_label.setFixedSize(60, 60)
                img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                try:
                    pixmap = QPixmap(f"images/{MONSTER_DATA['原始名称'][i+1]}.png")
                    if not pixmap.isNull():
                        pixmap = pixmap.scaled(
                            60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                        )
                        img_label.setPixmap(pixmap)
                except:
                    pass

                # 数量标签
                count_label = QLabel(str(int(count)))
                count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                count_label.setStyleSheet(
                    """
                            color: #EDEDED;
                            font: bold 20px SimHei;
                            min-width: 20px;
                        """
                )

                op_layout.addWidget(img_label, stretch=3)
                op_layout.addWidget(count_label, stretch=1)
                ops_layout.addWidget(op_widget)

        layout.addWidget(ops_widget)
        return team_widget
