from __future__ import annotations

from datetime import datetime

import pytest

from gradio_ml.infrastructure.models import LogEntry, LogFilter, LogLevel
from gradio_ml.infrastructure.log_buffer import LogBuffer


def _make_entry(level: LogLevel = LogLevel.Info, message: str = "test") -> LogEntry:
    return LogEntry(level=level, message=message, timestamp=datetime.now())


class TestLogBufferAppend:
    def test_append_and_count(self):
        buf = LogBuffer(capacity=5)
        for i in range(3):
            buf.append(_make_entry(message=f"msg{i}"))
        assert buf.count == 3

    def test_ring_buffer_overwrite(self):
        buf = LogBuffer(capacity=3)
        for i in range(5):
            buf.append(_make_entry(message=f"msg{i}"))
        assert buf.count == 3
        entries = buf.get_entries()
        assert entries[0].message == "msg2"
        assert entries[2].message == "msg4"


class TestLogBufferLevelFilter:
    def test_filter_info_and_above(self):
        buf = LogBuffer(capacity=100, rate_threshold=1000)
        buf.append(_make_entry(LogLevel.Debug, "debug"))
        buf.append(_make_entry(LogLevel.Info, "info"))
        buf.append(_make_entry(LogLevel.Warning, "warning"))
        buf.append(_make_entry(LogLevel.Error, "error"))
        result = buf.get_entries(LogFilter(min_level=LogLevel.Warning))
        assert len(result) == 2
        assert all(e.level in (LogLevel.Warning, LogLevel.Error) for e in result)

    def test_filter_error_only(self):
        buf = LogBuffer(capacity=100, rate_threshold=1000)
        buf.append(_make_entry(LogLevel.Info, "info"))
        buf.append(_make_entry(LogLevel.Error, "error"))
        result = buf.get_entries(LogFilter(min_level=LogLevel.Error))
        assert len(result) == 1


class TestLogBufferSearch:
    def test_search_keyword(self):
        buf = LogBuffer(capacity=100, rate_threshold=1000)
        buf.append(_make_entry(LogLevel.Info, "training started"))
        buf.append(_make_entry(LogLevel.Info, "epoch 1 complete"))
        buf.append(_make_entry(LogLevel.Info, "validation done"))
        buf.append(_make_entry(LogLevel.Info, "epoch 2 complete"))
        result = buf.search("epoch")
        assert result == [1, 3]

    def test_search_no_match(self):
        buf = LogBuffer(capacity=100, rate_threshold=1000)
        buf.append(_make_entry(LogLevel.Info, "hello"))
        assert buf.search("xyz") == []

    def test_search_empty_keyword(self):
        buf = LogBuffer(capacity=100, rate_threshold=1000)
        buf.append(_make_entry(LogLevel.Info, "hello"))
        assert buf.search("") == []


class TestLogBufferRateSampling:
    def test_sampling_preserves_error(self):
        buf = LogBuffer(capacity=100, rate_threshold=1, sampling_ratio=2)
        buf._sampling_active = True
        buf.append(_make_entry(LogLevel.Info, "info1"))
        buf.append(_make_entry(LogLevel.Error, "error1"))
        buf.append(_make_entry(LogLevel.Info, "info2"))
        entries = buf.get_entries()
        messages = [e.message for e in entries]
        assert "error1" in messages
