from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter, QPixmap


def get_prediction_result_text(left_win_prob):
    winner = "左方" if left_win_prob > 0.5 else "右方"
    if 0.6 > left_win_prob > 0.4:
        winner = "难说"

    if winner == "左方":
        return "预测胜方: 左方", "#d84a3a"
    if winner == "右方":
        return "预测胜方: 右方", "#2389c9"
    return "这一把难说", "#242424"


def create_prediction_share_pixmap(left_monsters, right_monsters, prediction, model_name):
    right_win_prob = prediction
    left_win_prob = 1 - right_win_prob
    winner_text, winner_color = get_prediction_result_text(left_win_prob)

    tile_w, tile_h = 84, 80
    columns_per_side = 3
    panel_w = tile_w * columns_per_side + 24
    gap = 36
    result_h = 116
    footer_h = 24
    margin_x = 42
    margin_y = 22
    left_rows = max(1, (len(left_monsters) + columns_per_side - 1) // columns_per_side)
    right_rows = max(1, (len(right_monsters) + columns_per_side - 1) // columns_per_side)
    rows = max(left_rows, right_rows)
    width = panel_w * 2 + gap + margin_x * 2
    panel_h = 44 + rows * tile_h + 16
    height = margin_y * 2 + panel_h + result_h + footer_h

    pixmap = QPixmap(width, height)
    pixmap.fill(QColor("#f4f1e8"))

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.fillRect(QRectF(0, 0, width, height), QColor("#f4f1e8"))

    subtitle_font = QFont("Microsoft YaHei", 13, QFont.Weight.Bold)
    normal_font = QFont("Microsoft YaHei", 12, QFont.Weight.Bold)
    count_font = QFont("Microsoft YaHei", 15, QFont.Weight.Bold)
    winner_font = QFont("Microsoft YaHei", 24, QFont.Weight.Bold)
    rate_font = QFont("Microsoft YaHei", 18, QFont.Weight.Bold)
    model_font = QFont("Microsoft YaHei", 9)

    left_rect = QRectF(margin_x, margin_y, panel_w, panel_h)
    right_rect = QRectF(margin_x + panel_w + gap, margin_y, panel_w, panel_h)
    draw_prediction_panel(
        painter, left_rect, "左方", left_monsters, "#d84a3a", columns_per_side, tile_w, tile_h,
        subtitle_font, normal_font, count_font
    )
    draw_prediction_panel(
        painter, right_rect, "右方", right_monsters, "#2389c9", columns_per_side, tile_w, tile_h,
        subtitle_font, normal_font, count_font
    )

    result_top = margin_y + panel_h + 36
    painter.setPen(QColor(winner_color))
    painter.setFont(winner_font)
    painter.drawText(
        QRectF(margin_x, result_top, width - margin_x * 2, 42),
        Qt.AlignmentFlag.AlignCenter,
        winner_text,
    )
    painter.setFont(rate_font)
    painter.drawText(
        QRectF(margin_x, result_top + 48, width - margin_x * 2, 34),
        Qt.AlignmentFlag.AlignCenter,
        f"左 {left_win_prob:.2%} | 右 {right_win_prob:.2%}",
    )

    painter.setPen(QColor("#9a9a9a"))
    painter.setFont(model_font)
    painter.drawText(
        QRectF(margin_x, height - margin_y - 2, width - margin_x * 2, 18),
        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        f"model: {model_name}",
    )
    painter.end()
    return pixmap


def draw_prediction_panel(
    painter, rect, side_name, monsters, accent_color, columns, tile_w, tile_h,
    subtitle_font, normal_font, count_font
):
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#ffffff"))
    painter.drawRoundedRect(rect, 14, 14)

    painter.setBrush(QColor(accent_color))
    header_height = 38
    painter.drawRoundedRect(QRectF(rect.x(), rect.y(), rect.width(), header_height), 14, 14)
    painter.drawRect(QRectF(rect.x(), rect.y() + 19, rect.width(), 19))

    painter.setPen(QColor("#ffffff"))
    painter.setFont(subtitle_font)
    painter.drawText(
        QRectF(rect.x() + 18, rect.y() + 5, rect.width() - 36, 30),
        Qt.AlignmentFlag.AlignCenter,
        side_name,
    )

    if not monsters:
        painter.setPen(QColor("#777777"))
        painter.setFont(normal_font)
        painter.drawText(
            QRectF(rect.x(), rect.y() + 66, rect.width(), 40),
            Qt.AlignmentFlag.AlignCenter,
            "无怪物",
        )
        return

    start_x = rect.x() + 12
    start_y = rect.y() + 52
    for idx, (_, monster_name, count) in enumerate(monsters):
        row = idx // columns
        col = idx % columns
        tile_rect = QRectF(start_x + col * tile_w, start_y + row * tile_h, tile_w - 10, tile_h - 4)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ededed"))
        painter.drawRoundedRect(tile_rect, 10, 10)

        monster_pixmap = QPixmap(f"images/{monster_name}.png")
        if monster_pixmap.isNull():
            monster_pixmap = QPixmap("images/empty.png")
        if not monster_pixmap.isNull():
            icon_rect = QRectF(tile_rect.x() + 16, tile_rect.y() + 6, 44, 42)
            scaled = monster_pixmap.scaled(
                int(icon_rect.width()), int(icon_rect.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            draw_x = icon_rect.x() + (icon_rect.width() - scaled.width()) / 2
            draw_y = icon_rect.y() + (icon_rect.height() - scaled.height()) / 2
            painter.drawPixmap(int(draw_x), int(draw_y), scaled)

        painter.setFont(count_font)
        painter.setPen(QColor("#222222"))
        painter.drawText(
            QRectF(tile_rect.x(), tile_rect.y() + 52, tile_rect.width(), 22),
            Qt.AlignmentFlag.AlignCenter,
            f"x{count}",
        )
