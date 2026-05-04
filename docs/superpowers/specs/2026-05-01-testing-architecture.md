# 测试架构设计

## 三层结构

```
tests/
├── unit/              # 纯逻辑测试，无外部依赖，<1s each
├── integration/       # 需真实 ADB 模拟器，~5s each
└── e2e/               # 完整 Qt GUI + ADB，~30s each
```

### Unit Layer

| 文件 | 覆盖 | 状态 |
|------|------|------|
| `test_connector_factory.py` | 状态机、配置检测、池管理 | 已有，移入 |
| `test_config.py` | 配置加载、默认值回退 | 已有 |
| `test_recognize.py` | crop_ratio、process_regions | 已有 |
| `test_find_monster_zone.py` | Hough 检测、WIN 识别 | 已有 |
| `test_predict.py` | 模型加载、推理准确率 | **新增** |
| `test_auto_fetch_state.py` | 状态机逻辑（mock connector）| **新增** |
| `test_input_panel.py` | 输入面板 set/get monster counts | **新增** |

### Integration Layer

| 文件 | 覆盖 | 状态 |
|------|------|------|
| `test_adb_recognition.py` | ADB 截图→识别→验证结果 | **新增** |
| `test_pc_recognition.py` | PC 截图→识别→验证结果 | **新增** |
| `test_auto_fetch_flow.py` | 状态机单轮循环 | **新增** |

### E2E Layer

| 文件 | 覆盖 | 状态 |
|------|------|------|
| `test_gui_mode_switch.py` | 模式切换 UI 状态、控件启用/禁用 | **新增** |
| `test_gui_recognize.py` | 识别按钮→输入面板填充 | **新增** |
| `test_gui_auto_fetch.py` | 自动获取按钮弹窗（PC/WIN） | **新增** |

## CI 配置

```yaml
# .github/workflows/test.yml
jobs:
  unit:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run pytest tests/unit/ -v

  integration:
    runs-on: [self-hosted, adb]  # 需要模拟器
    steps:
      - run: uv run pytest tests/integration/ -v

  e2e:
    runs-on: [self-hosted, adb]
    steps:
      - run: uv run pytest tests/e2e/ -v
```

## 测试支撑

| 文件 | 用途 |
|------|------|
| `tests/conftest.py` | `qapp` fixture、`adb_connector` fixture |
| `tests/mock_connector.py` | Mock 连接器（返回本地截图） |
| `pytest-xvfb` | CI 无头 GUI |

## 实施顺序

1. 创建 `conftest.py` + `mock_connector.py`
2. 迁移现有测试到 `tests/unit/`
3. `test_predict.py` — 模型推理
4. `test_input_panel.py` — 输入面板
5. `test_auto_fetch_state.py` — 状态机（mock）
6. `test_gui_mode_switch.py` — GUI 模式切换
7. `test_gui_recognize.py` — GUI 识别端到端
8. `test_gui_auto_fetch.py` — GUI 弹窗
9. CI 配置文件
