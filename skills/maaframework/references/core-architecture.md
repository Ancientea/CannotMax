---
name: core-architecture
description: MaaFramework architecture, core concepts, and key terminology for building automation workflows
---

# MaaFramework Core Architecture

MaaFramework (MaaFW) is a framework for building automated task pipelines for mobile and desktop applications. It provides image recognition, text recognition, and input simulation across multiple platforms.

## Key Concepts

- **Node**: A single unit in a pipeline — a JSON object conforming to the Pipeline protocol. Contains recognition algorithm, action, and next-node references.
- **Task**: A logical sequence of Nodes connected in order, representing a complete workflow from entry to completion.
- **Entry**: The first Node in a Task.
- **Pipeline**: All Nodes defined in a `pipeline/` folder.
- **Bundle**: A resource folder containing `pipeline/`, `model/`, `image/` subfolders.
- **Resource**: A collection of Bundles loaded in order. Later bundles override earlier ones with the same node names.
- **PI (ProjectInterface)**: A standardized `interface.json` that declares project structure for general UI tools.
- **Agent**: A separate process for CustomRecognition/CustomAction, enabling cross-language logic (e.g., C# GUI + Python custom logic).

## Resource Structure

```
my_resource/
├── image/                  # Template images (scaled to 720p, lossless source)
├── model/
│   └── ocr/               # OCR models (det.onnx, rec.onnx, keys.txt)
│   └── classify/          # Classification models
│   └── detect/            # Detection models (YOLOv8/v11 ONNX)
└── pipeline/              # Task pipeline JSON files
    └── *.json
```

## General Terms

- **Binding**: Glue code converting MaaFW's C API to other languages.
- **ROI**: Region of Interest — image recognition boundary.
- **OCR**: Optical Character Recognition for text detection in screenshots.

## Multi-Language Support

| Language | Status | Package |
|----------|--------|---------|
| C++ | Native | source headers |
| Python | Official, latest | `MaaFw` on PyPI |
| NodeJS | Official, latest | `@maaxyz/maa-node` on npm |
| C# | Official, latest | `Maa.Framework` on NuGet |
| Go | Official, latest | `maa-framework-go` |
| Rust | Official, latest | `maa-framework` on crates.io |

<!--
Source references:
- https://github.com/MaaXYZ/MaaFramework
-->
