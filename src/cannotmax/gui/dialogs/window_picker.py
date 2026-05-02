"""Window picker dialog for selecting capture target."""

import ctypes
from ctypes import wintypes
from typing import Optional

from PyQt6.QtWidgets import QDialog, QLineEdit, QListWidget, QVBoxLayout


def list_visible_window_titles(filter_hwnds: Optional[list[int]] = None) -> list[str]:
    """列出所有**可见**窗口的标题（去重并按字典序排序）。

    参数
    ----
    filter_hwnds: Optional[list[int]]
        若提供，仅枚举这些 hwnd 对应的窗口；否则枚举所有窗口
    """
    EnumWindows = ctypes.windll.user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    IsWindowVisible = ctypes.windll.user32.IsWindowVisible
    GetWindowTextW = ctypes.windll.user32.GetWindowTextW
    GetWindowTextLengthW = ctypes.windll.user32.GetWindowTextLengthW

    titles: list[str] = []

    def foreach(hwnd, lParam):
        # 过滤：如果提供了 filter_hwnds，只处理这些 hwnd
        if filter_hwnds is not None and hwnd not in filter_hwnds:
            return True

        if IsWindowVisible(hwnd):
            length = GetWindowTextLengthW(hwnd)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                GetWindowTextW(hwnd, buff, length + 1)
                t = buff.value.strip()
                if t:
                    titles.append(t)
        return True

    EnumWindows(EnumWindowsProc(foreach), 0)
    titles = sorted(set(titles))
    logger.info(f"窗口列表：{titles}")
    return titles


# ------------------------- 截屏源选择对话框 -------------------------
class WindowPickerDialog(QDialog):
    """列出可见窗口标题，并内置"整屏 (1/2/3)"选项。双击确定。"""

    def __init__(self, parent=None, filter_hwnds: Optional[list[int]] = None):
        """
        参数
        ----
        parent: Parent widget
        filter_hwnds: 若提供，仅枚举这些 hwnd 对应的窗口 (PC 多窗口模式)
                     若为 None，枚举所有窗口 (WIN 模式)
        """
        super().__init__(parent)
        self.setWindowTitle("选择截屏窗口")
        self.resize(520, 480)
        self.selected_title = None
        self._filter_hwnds = filter_hwnds
        self._hwnd_to_title: dict[int, str] = {}

        self.search = QLineEdit(self)
        self.search.setPlaceholderText("输入关键字过滤（支持大小写不敏感）")

        self.listw = QListWidget(self)
        # 预置"整屏（主屏 0/副屏 1/2）"选项在最上面
        self.listw.addItem("【整屏】主屏 (1)")
        self.listw.addItem("【整屏】副屏 (2)")
        self.listw.addItem("【整屏】副屏 (3)")
        self.listw.addItem("—————— 窗口列表 ——————")

        self._all_titles = list_visible_window_titles(filter_hwnds=filter_hwnds)
        if filter_hwnds:
            # Build mapping when filter_hwnds provided
            EnumWindows = ctypes.windll.user32.EnumWindows
            EnumWindowsProc = ctypes.WINFUNCTYPE(
                wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
            )
            IsWindowVisible = ctypes.windll.user32.IsWindowVisible
            GetWindowTextW = ctypes.windll.user32.GetWindowTextW
            GetWindowTextLengthW = ctypes.windll.user32.GetWindowTextLengthW

            def build_map(hwnd, _):
                if filter_hwnds and hwnd not in filter_hwnds:
                    return True
                if IsWindowVisible(hwnd):
                    length = GetWindowTextLengthW(hwnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        GetWindowTextW(hwnd, buff, length + 1)
                        t = buff.value.strip()
                        if t:
                            self._hwnd_to_title[hwnd] = t
                return True

            EnumWindows(EnumWindowsProc(build_map), 0)

        for t in self._all_titles:
            self.listw.addItem(t)

        self.search.textChanged.connect(self._filter)
        self.listw.itemDoubleClicked.connect(self._accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self.search)
        layout.addWidget(self.listw)

    def _filter(self, text: str):
        text_low = (text or "").lower()
        self.listw.clear()
        self.listw.addItem("【整屏】主屏(1)")
        self.listw.addItem("【整屏】副屏(2)")
        self.listw.addItem("【整屏】副屏(3)")
        self.listw.addItem("—————— 窗口列表 ——————")
        for t in self._all_titles:
            if text_low in t.lower():
                self.listw.addItem(t)

    def _accept(self):
        self.accept()

    def get_selection(self):
        item = self.listw.currentItem()
        if not item:
            return None
        text = item.text()
        if text.startswith("【整屏】"):
            # 返回 monitor_index
            if "(1)" in text:
                return {"monitor_index": 1}
            if "(2)" in text:
                return {"monitor_index": 2}
            if "(3)" in text:
                return {"monitor_index": 3}
        elif text.startswith("——————"):
            return None
        else:
            # 返回 window_name 和 hwnd (如果有映射)
            result = {"window_name": text}
            # 查找对应的 hwnd
            for hwnd, title in self._hwnd_to_title.items():
                if title == text:
                    result["hwnd"] = hwnd
                    break
            return result
