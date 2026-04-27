---
name: core-pipeline
description: Task pipeline protocol — node lifecycle, execution flow, and pipeline v1/v2 formats
---

# Pipeline Protocol

The task pipeline is a JSON structure of interconnected Nodes. Each node defines what to recognize, what action to take, and what to do next.

## Basic Format (Pipeline v1)

```jsonc
{
    "NodeA": {
        "recognition": "OCR",       // How to recognize
        "expected": "Start",        // Recognition parameters
        "action": "Click",          // What to do
        "next": ["NodeB", "NodeC"]  // What to check next
    }
}
```

## Pipeline v2 Format

Recognition and action fields are nested under `type`/`param`:

```jsonc
{
    "NodeA": {
        "recognition": {
            "type": "TemplateMatch",
            "param": {
                "template": "A.png",
                "roi": [100, 100, 10, 10]
            }
        },
        "action": {
            "type": "Click",
            "param": {
                "target": true
            }
        },
        "next": ["NodeB"]
    }
}
```

## Execution Flow

1. **Entry**: Start from the entry node via `tasker.post_task(entry_name)`
2. **Sequential Detection**: Try each node in `next` list in order
3. **On Match**: Execute the matched node's action, then enter its `next` list
4. **On Miss**: If no node matches in a round, retry until timeout
5. **On Error**: If action fails or timeout, enter `on_error` list

```
enter node → pre_wait_freezes → pre_delay → action
  → (repeat if repeat > 1: repeat_wait_freezes → repeat_delay → action)
  → post_wait_freezes → post_delay → screenshot → recognize next
  → hit? → enter new node
  → miss? → timeout? → on_error
  → miss? → not timeout → wait rate_limit → screenshot again
```

## Node Lifecycle Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `recognition` | string | `DirectHit` | Recognition algorithm type |
| `action` | string | `DoNothing` | Action to execute |
| `next` | list | `[]` | Next nodes to try (supports `[JumpBack]`, `[Anchor]` prefixes) |
| `on_error` | list | `[]` | Nodes to execute on failure |
| `enabled` | bool | `true` | Whether this node is active |
| `max_hit` | uint | UINT_MAX | Max times this node can be matched |
| `rate_limit` | uint | 1000 | Min ms per recognition round |
| `timeout` | int | 20000 | Ms to wait for `next` list match (-1 = infinite) |
| `pre_delay` | uint | 200 | Ms delay before action |
| `post_delay` | uint | 200 | Ms delay after action |
| `pre_wait_freezes` | uint/obj | 0 | Ms to wait for screen stability before action |
| `post_wait_freezes` | uint/obj | 0 | Ms to wait for screen stability after action |
| `repeat` | uint | 1 | Number of times to repeat the action |
| `repeat_delay` | uint | 0 | Ms delay between repeated actions |
| `inverse` | bool | `false` | Invert recognition result |
| `anchor` | string/list/obj | — | Set anchor name(s) for dynamic reference |
| `focus` | object | null | Node notification configuration |
| `attach` | object | `{}` | Additional custom data (merged with defaults) |

## Node Attributes in `next`/`on_error`

Use prefix syntax or object form:

```jsonc
"next": [
    "SimpleNode",
    "[JumpBack]ExceptionHandler",   // After completion, return to parent
    "[Anchor]DynamicTarget"         // Resolve to last node that set this anchor
]
```

**`[JumpBack]`**: After the node chain completes, return to parent and re-recognize its `next` list. Skipped in error paths.

**`[Anchor]`**: Resolves at runtime to the last node that set the named anchor. Skipped if anchor is unset/cleared.

## Default Properties

Place `default_pipeline.json` in the bundle root (sibling to `pipeline/`):

```jsonc
{
    "Default": {
        "rate_limit": 2000,
        "timeout": 30000
    },
    "TemplateMatch": {
        "recognition": "TemplateMatch",
        "threshold": 0.7
    },
    "Click": {
        "action": "Click",
        "target": true
    }
}
```

Priority: node params > algorithm/action defaults > `Default` object > built-in defaults.

<!--
Source references:
- docs/en_us/3.1-PipelineProtocol.md
-->
