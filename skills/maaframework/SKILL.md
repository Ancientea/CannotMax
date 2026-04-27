---
name: maaframework
description: Building automation pipelines with MaaFramework — JSON-based task pipelines with image/OCR recognition, multi-platform input control, and custom logic via Python/NodeJS agents
metadata:
  author: kutius
  version: "2026.4.16"
  source: Generated from https://github.com/MaaXYZ/MaaFramework
---

> The skill is based on MaaFramework v5.x, generated at 2026-04-16.

MaaFramework (MaaFW) is a framework for automating tasks on mobile and desktop applications. It combines JSON-defined task pipelines with image recognition, OCR, neural network detection, and platform-specific input simulation. Supports Android (ADB), Windows (Win32), macOS, Linux (Wayland), gamepads, and more.

## Core References

| Topic | Description | Reference |
|-------|-------------|-----------|
| Architecture | Core concepts, terms, resource structure, language bindings | [core-architecture](references/core-architecture.md) |
| Pipeline Protocol | Node lifecycle, execution flow, v1/v2 formats, default properties | [core-pipeline](references/core-pipeline.md) |
| Recognition Algorithms | TemplateMatch, OCR, FeatureMatch, ColorMatch, NeuralNetwork, And/Or | [core-recognition](references/core-recognition.md) |
| Action Types | Click, Swipe, Scroll, key input, Shell, Command, app control | [core-actions](references/core-actions.md) |

## Integration

| Topic | Description | Reference |
|-------|-------------|-----------|
| Setup & Binding | Creating Tasker/Resource/Controller, Python/NodeJS examples, pipeline override | [core-integration](references/core-integration.md) |
| Controller Methods | ADB/Win32/MacOS screencap & input methods, platform-specific config | [core-controllers](references/core-controllers.md) |
| Callback Protocol | Event message types, JSON structures, handling patterns | [core-callbacks](references/core-callbacks.md) |
| Project Interface | `interface.json` — tasks, controllers, resources, options, presets, i18n | [core-project-interface](references/core-project-interface.md) |

## Advanced

| Topic | Description | Reference |
|-------|-------------|-----------|
| Custom Logic | Custom recognition/action via Agent process, Python/NodeJS implementation | [advanced-custom-logic](references/advanced-custom-logic.md) |
| Troubleshooting | Debug options, log files, debug images, common issues | [advanced-troubleshooting](references/advanced-troubleshooting.md) |
