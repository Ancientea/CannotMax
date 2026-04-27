class DarkModeStyleFix:
    DARK_TEXT_COLOR = "#313131"
    COMBO_POPUP_BACKGROUND = "#FFFFFF"
    COMBO_POPUP_BORDER = "#CCCCCC"
    COMBO_SELECTION_BACKGROUND = "#F5EA2D"
    COMBO_SELECTION_COLOR = "#313131"
    PLACEHOLDER_COLOR = "#888888"

    @staticmethod
    def get_global_qss() -> str:
        return f"""
            QLabel {{
                color: {DarkModeStyleFix.DARK_TEXT_COLOR};
            }}
            QGroupBox {{
                color: {DarkModeStyleFix.DARK_TEXT_COLOR};
            }}
            QGroupBox::title {{
                color: {DarkModeStyleFix.DARK_TEXT_COLOR};
            }}
            QCheckBox {{
                color: {DarkModeStyleFix.DARK_TEXT_COLOR};
            }}
            QComboBox {{
                color: {DarkModeStyleFix.DARK_TEXT_COLOR};
            }}
            QComboBox QAbstractItemView {{
                color: {DarkModeStyleFix.DARK_TEXT_COLOR};
                background-color: {DarkModeStyleFix.COMBO_POPUP_BACKGROUND};
                selection-background-color: {DarkModeStyleFix.COMBO_SELECTION_BACKGROUND};
                selection-color: {DarkModeStyleFix.COMBO_SELECTION_COLOR};
                border: 1px solid {DarkModeStyleFix.COMBO_POPUP_BORDER};
                outline: none;
            }}
            QComboBox QLineEdit {{
                color: {DarkModeStyleFix.DARK_TEXT_COLOR};
            }}
            QLineEdit {{
                color: {DarkModeStyleFix.DARK_TEXT_COLOR};
            }}
            QPushButton {{
                color: {DarkModeStyleFix.DARK_TEXT_COLOR};
                border: 1px solid #999999;
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QPushButton:hover {{
                border: 1px solid #666666;
            }}
            QPushButton:pressed {{
                border: 1px solid #333333;
            }}
            QScrollArea {{
                color: {DarkModeStyleFix.DARK_TEXT_COLOR};
            }}
        """

    @staticmethod
    def get_combo_box_qss() -> str:
        return f"""
            QComboBox {{
                color: {DarkModeStyleFix.DARK_TEXT_COLOR};
            }}
            QComboBox QAbstractItemView {{
                color: {DarkModeStyleFix.DARK_TEXT_COLOR};
                background-color: {DarkModeStyleFix.COMBO_POPUP_BACKGROUND};
                selection-background-color: {DarkModeStyleFix.COMBO_SELECTION_BACKGROUND};
                selection-color: {DarkModeStyleFix.COMBO_SELECTION_COLOR};
                border: 1px solid {DarkModeStyleFix.COMBO_POPUP_BORDER};
                outline: none;
            }}
            QComboBox QLineEdit {{
                color: {DarkModeStyleFix.DARK_TEXT_COLOR};
            }}
        """

    @staticmethod
    def get_line_edit_qss() -> str:
        return f"""
            QLineEdit {{
                color: {DarkModeStyleFix.DARK_TEXT_COLOR};
            }}
        """

    @staticmethod
    def get_group_box_title_qss() -> str:
        return f"""
            QGroupBox {{
                color: {DarkModeStyleFix.DARK_TEXT_COLOR};
            }}
            QGroupBox::title {{
                color: {DarkModeStyleFix.DARK_TEXT_COLOR};
            }}
        """

    @staticmethod
    def apply(app) -> None:
        if app is None:
            raise ValueError("QApplication instance cannot be None")
        global_qss = DarkModeStyleFix.get_global_qss()
        app.setStyleSheet(global_qss)
