---
name: core-recognition
description: Recognition algorithms — TemplateMatch, OCR, FeatureMatch, ColorMatch, NeuralNetwork, And/Or composites
---

# Recognition Algorithms

Each node specifies a `recognition` type. Default is `DirectHit` (no recognition, action executes immediately).

## `DirectHit`

No recognition — action always executes. Use `roi` to set action region.

## `TemplateMatch`

Find a template image in the screenshot.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `template` | string/list | required | Path relative to `image/` folder. Supports directories (recursive). Images must be 720p lossless. |
| `threshold` | double/list | 0.7 | Match threshold |
| `roi` | [x,y,w,h] / string | [0,0,0,0] | Search region. String = node name or `[Anchor]Name` for dynamic ROI |
| `roi_offset` | [x,y,w,h] | [0,0,0,0] | Offset added to roi |
| `order_by` | string | `Horizontal` | Sort: `Horizontal` / `Vertical` / `Score` / `Random` |
| `index` | int | 0 | Which result to use (supports negative indexing) |
| `method` | int | 5 | OpenCV match method. 5=TCCOEFF_NORMED (recommended), 3=TCORR_NORMED, 10001=inverted TSQDIFF_NORMED |
| `green_mask` | bool | false | Mask green (0,255,0) areas from template |

```jsonc
{
    "recognition": "TemplateMatch",
    "template": "confirm_btn.png",
    "threshold": 0.8,
    "roi": [100, 200, 300, 100]
}
```

## `OCR`

Text recognition using PaddleOCR models.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `expected` | string/list | all | Expected text, supports regex |
| `threshold` | double | 0.3 | Confidence threshold |
| `replace` | [from,to]/list | — | Text post-processing replacements |
| `roi` | [x,y,w,h]/string | [0,0,0,0] | Search region |
| `only_rec` | bool | false | Recognition only (no detection, needs precise ROI) |
| `model` | string | — | Model folder path relative to `model/ocr/` |
| `color_filter` | string | — | ColorMatch node name for pre-processing binarization |
| `order_by` | string | `Horizontal` | Sort: adds `Length` and `Expected` options |

```jsonc
{
    "recognition": "OCR",
    "expected": ["Start", "Begin"],
    "threshold": 0.5,
    "roi": [0, 500, 720, 200]
}
```

## `FeatureMatch`

Feature-based image matching — robust to perspective and scale changes.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `template` | string/list | required | Template image path (min 64x64px, needs texture detail) |
| `count` | uint | 4 | Min matching feature points |
| `detector` | string | `SIFT` | `SIFT` / `KAZE` / `AKAZE` / `BRISK` / `ORB` |
| `ratio` | double | 0.6 | KNN matching distance ratio [0-1] |
| `order_by` | string | `Horizontal` | Sort: adds `Area` option |

## `ColorMatch`

Match colors in a region.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | int | 4 | Color space: 4=RGB, 40=HSV, 6=GRAY |
| `lower` | list | required | Lower bound per channel |
| `upper` | list | required | Upper bound per channel |
| `count` | uint | 1 | Min matching pixels |
| `connected` | bool | false | Only count largest connected block |

```jsonc
{
    "recognition": "ColorMatch",
    "method": 40,
    "lower": [100, 50, 50],
    "upper": [130, 255, 255],
    "count": 100
}
```

## `NeuralNetworkClassify`

Classify image content at a fixed position (ONNX models).

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model path relative to `model/classify/` |
| `labels` | list | Category names (for debug only) |
| `expected` | int/list | Expected category indices |

```jsonc
{
    "recognition": "NeuralNetworkClassify",
    "model": "my_classifier.onnx",
    "labels": ["Cat", "Dog", "Mouse"],
    "expected": [0, 2]
}
```

## `NeuralNetworkDetect`

Object detection — find objects at arbitrary positions (YOLOv8/v11 ONNX).

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model path relative to `model/detect/` |
| `labels` | list | Auto-read from model metadata if not set |
| `expected` | int/list | Expected category indices |
| `threshold` | double/list | Confidence threshold, default 0.3 |

## Composite: `And` / `Or`

Combine multiple recognitions.

**`And`** — all sub-recognitions must succeed:
```jsonc
{
    "recognition": "And",
    "all_of": [
        { "sub_name": "icon", "recognition": "TemplateMatch", "template": "icon.png" },
        { "recognition": "OCR", "roi": "icon", "roi_offset": [0,0,100,100], "expected": "Storage" }
    ],
    "box_index": 0
}
```

**`Or`** — first matching sub-recognition wins:
```jsonc
{
    "recognition": "Or",
    "any_of": [
        { "recognition": "TemplateMatch", "template": "confirm.png" },
        { "recognition": "OCR", "expected": ["OK", "Confirm"] }
    ]
}
```

Both support node name references (string) to reuse other nodes' recognition configs.

## `Custom`

Execute a registered custom recognizer:
```jsonc
{
    "recognition": "Custom",
    "custom_recognition": "MyReco",
    "custom_recognition_param": { "key": "value" }
}
```

## Result Sorting

`order_by` controls multi-result ordering:
- `Horizontal`: left→right, top→bottom
- `Vertical`: top→bottom, left→right
- `Score`: highest match score first
- `Area`: largest bounding box first
- `Length`: longest text first (OCR only)
- `Expected`: match `expected` field order
- `Random`: random shuffle

Use `index` to select a specific result after sorting.

<!--
Source references:
- docs/en_us/3.1-PipelineProtocol.md (Algorithm Types section)
-->
