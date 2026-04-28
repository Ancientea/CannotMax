from __future__ import annotations

from datetime import datetime

import pytest

from gradio_ml.infrastructure.models import LogEntry, LogFilter, LogLevel
from gradio_ml.service.log_service import LogService


class TestLogServiceSanitize:
    def test_sanitize_file_path(self):
        svc = LogService(sanitize_patterns={"file_path": r'[A-Z]:\\[^\s]+'})
        result = svc.sanitize("模型路径 C:\\Users\\test\\model.pth 已加载")
        assert "C:\\Users" not in result
        assert "***" in result

    def test_sanitize_api_key(self):
        svc = LogService(sanitize_patterns={"api_key": r'[Aa][Pp][Ii][_]?[Kk][Ee][Yy][:=]\s*\S+'})
        result = svc.sanitize("配置 api_key=sk-12345 完成")
        assert "sk-12345" not in result
        assert "***" in result

    def test_sanitize_token(self):
        svc = LogService(sanitize_patterns={"token": r'[Tt][Oo][Kk][Ee][Nn][:=]\s*\S+'})
        result = svc.sanitize("使用 token=abc123 进行认证")
        assert "abc123" not in result
        assert "***" in result

    def test_no_sanitization_needed(self):
        svc = LogService(sanitize_patterns={"file_path": r'[A-Z]:\\[^\s]+'})
        result = svc.sanitize("训练正常进行")
        assert result == "训练正常进行"


class TestLogServiceGetLogs:
    def test_get_logs_basic(self):
        svc = LogService()
        svc.append(LogEntry(LogLevel.Info, "训练开始", datetime.now()))
        svc.append(LogEntry(LogLevel.Warning, "内存不足", datetime.now()))
        text = svc.get_logs()
        assert "训练开始" in text
        assert "内存不足" in text

    def test_get_logs_with_level_filter(self):
        svc = LogService()
        svc.append(LogEntry(LogLevel.Debug, "debug msg", datetime.now()))
        svc.append(LogEntry(LogLevel.Info, "info msg", datetime.now()))
        svc.append(LogEntry(LogLevel.Error, "error msg", datetime.now()))
        text = svc.get_logs(LogFilter(min_level=LogLevel.Error))
        assert "error msg" in text
        assert "info msg" not in text


class TestLogServiceSearch:
    def test_search_keyword(self):
        svc = LogService()
        svc.append(LogEntry(LogLevel.Info, "epoch 1 done", datetime.now()))
        svc.append(LogEntry(LogLevel.Info, "epoch 2 done", datetime.now()))
        svc.append(LogEntry(LogLevel.Info, "validation", datetime.now()))
        result = svc.search("epoch")
        assert len(result) == 2
