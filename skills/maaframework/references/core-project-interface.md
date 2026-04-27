---
name: core-project-interface
description: ProjectInterface (interface.json) — declaring project structure, tasks, controllers, options, and presets
---

# Project Interface (interface.json)

`interface.json` is the standardized project declaration that general UI tools use to load and run MaaFW projects.

## Structure

```jsonc
{
    "interface_version": 2,
    "name": "MyProject",
    "version": "1.0.0",
    "controller": [ ... ],
    "resource": [ ... ],
    "task": [ ... ],
    "option": { ... },
    "agent": { ... },
    "preset": [ ... ],
    "import": [ ... ]
}
```

## Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `interface_version` | number | Always `2` (required) |
| `name` | string | Unique project ID |
| `label` | string | Display name (supports `$` i18n) |
| `version` | string | Project version |
| `icon` | string | App icon path |
| `github` | string | GitHub repo URL (used for update checking) |
| `controller` | object[] | Controller presets |
| `resource` | object[] | Resource packages |
| `task` | object[] | Executable tasks |
| `option` | object | Configuration options |
| `agent` | object/object[] | Agent subprocess config |
| `preset` | object[] | Preset configurations |
| `import` | string[] | Paths to other PI files to merge |
| `languages` | object | i18n translation file paths |
| `group` | object[] | Task group declarations |
| `global_option` | string[] | Global option keys |

## Controller Config

```jsonc
{
    "name": "Android",
    "label": "$AndroidDevice",
    "type": "Adb",               // Adb | Win32 | MacOS | PlayCover | Gamepad | WlRoots
    "display_short_side": 720,    // Screenshot scaling
    "permission_required": false
}
```

Win32-specific: `class_regex`, `window_regex`, `mouse`, `keyboard`, `screencap`.
ADB: input/screencap auto-detected by MaaFramework.

## Resource Config

```jsonc
{
    "name": "Official",
    "label": "$OfficialServer",
    "path": ["./resource"],
    "controller": ["Android"]    // Optional: restrict to specific controllers
}
```

Multiple paths load sequentially; later resources override earlier ones with the same node names.

## Task Config

```jsonc
{
    "name": "DailyFarm",
    "label": "$DailyFarm",
    "entry": "FarmEntry",        // Pipeline entry node name
    "default_check": false,
    "resource": ["Official"],    // Optional: restrict to specific resources
    "controller": ["Android"],   // Optional: restrict to specific controllers
    "group": ["daily"],          // Optional: task groups
    "pipeline_override": {       // Optional: runtime overrides
        "SomeNode": { "enabled": true }
    },
    "option": ["BattleStage"]    // Optional: option keys
}
```

## Options System

Options let users configure tasks via UI. Types: `select`, `checkbox`, `input`, `switch`.

```jsonc
"option": {
    "BattleStage": {
        "type": "select",
        "label": "$SelectStage",
        "default_case": "3-9",
        "cases": [
            {
                "name": "3-9",
                "pipeline_override": {
                    "EnterStage": { "next": "MainChapter_3" }
                }
            }
        ]
    }
}
```

### Option Merge Priority

`task.option` > `controller.option` > `resource.option` > `global_option`

Options can be scoped to specific controllers/resources via `controller`/`resource` arrays.

## Agent Config

```jsonc
"agent": {
    "child_exec": "python",
    "child_args": ["./agent/main.py"]
}
```

Multiple agents supported via array. The Client injects `PI_*` environment variables (v2.5.0+).

## Presets

Snapshots of predefined task states and option values:

```jsonc
"preset": [{
    "name": "DailyRoutine",
    "label": "$DailyRoutine",
    "task": [
        { "name": "CollectReward", "enabled": true },
        {
            "name": "Battle",
            "enabled": true,
            "option": { "BattleStage": "3-9" }
        }
    ]
}]
```

## Internationalization

Strings starting with `$` are i18n keys. The Client resolves them from translation files specified in `languages`.

## Resource Override Behavior

When later-loaded resources define a node with the same name, top-level keys replace (not merge) the old definition. Arrays are replaced entirely.

<!--
Source references:
- docs/en_us/3.3-ProjectInterfaceV2.md
-->
