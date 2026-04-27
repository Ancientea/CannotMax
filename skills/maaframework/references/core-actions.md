---
name: core-actions
description: Action types — Click, Swipe, Scroll, key input, app control, Shell, Command, and more
---

# Action Types

Each node specifies an `action` type. Default is `DoNothing`.

## Target Positioning

Most actions accept a `target` field defining where to act:

| Value | Meaning |
|-------|---------|
| `true` | Use this node's recognition result (default) |
| `"NodeName"` | Use another node's recognition result |
| `"[Anchor]Name"` | Use the node corresponding to this anchor |
| `[x, y]` | Fixed coordinate point |
| `[x, y, w, h]` | Fixed area (random point, higher probability near center) |

`target_offset` adds `[x, y, w, h]` to the target position.

## `Click`

Click at the target position.

```jsonc
{
    "action": "Click",
    "target": true,
    "target_offset": [0, 0, 0, 0],
    "contact": 0,      // finger id (Adb) or mouse button (Win32/MacOS)
    "pressure": 1
}
```

## `LongPress`

Long press at target. Additional field: `duration` (ms, default 1000).

## `Swipe`

Linear swipe between two points.

```jsonc
{
    "action": "Swipe",
    "begin": [100, 500],
    "end": [600, 500],
    "duration": 200,
    "end_hold": 0       // ms to wait at end before lifting
}
```

- `end` supports a list for polyline waypoints (single swipe, no lift between points)
- `begin_offset` / `end_offset`: additional offset per coordinate
- `only_hover`: move cursor only, no press/release

## `MultiSwipe`

Multi-finger swipe. Each swipe in `swipes` array has:
- `starting`: ms offset from action start to begin this swipe
- Same `begin`/`end`/`duration` fields as Swipe
- `contact` defaults to array index

```jsonc
{
    "action": "MultiSwipe",
    "swipes": [
        { "begin": [100, 300], "end": [600, 300] },
        { "starting": 500, "begin": [100, 500], "end": [600, 500] }
    ]
}
```

## `Scroll`

Mouse wheel scroll. Move to `target` first, then scroll `dx`/`dy`.

```jsonc
{
    "action": "Scroll",
    "target": [360, 360, 100, 100],
    "dy": 120   // Win32: 120 = one wheel notch
}
```

Supported by Win32, macOS, and custom controllers implementing `scroll`.

## `ClickKey` / `LongPressKey`

Click or long-press a virtual key. `key` accepts int or list of ints.

```jsonc
{ "action": "ClickKey", "key": 13 }  // Enter key (Win32 VK)
```

## `KeyDown` / `KeyUp`

Press/release a key without auto-release. Use together for custom sequences.

## `InputText`

Type text. Some controllers only support ASCII.

```jsonc
{ "action": "InputText", "input_text": "hello" }
```

## `TouchDown` / `TouchMove` / `TouchUp`

Manual touch contact control. Use `contact` to manage multi-touch.

## `StartApp` / `StopApp`

```jsonc
{ "action": "StartApp", "package": "com.example.app" }
{ "action": "StopApp", "package": "com.example.app" }
```

## `StopTask`

Stops the current task chain immediately.

## `Command`

Execute a local system command.

```jsonc
{
    "action": "Command",
    "exec": "python",
    "args": ["{RESOURCE_DIR}/scripts/parse.py", "{IMAGE}", "{BOX}"],
    "detach": false
}
```

Template variables: `{ENTRY}`, `{NODE}`, `{IMAGE}`, `{BOX}`, `{RESOURCE_DIR}`, `{LIBRARY_DIR}`

## `Shell`

Execute a shell command on ADB device (ADB controllers only).

```jsonc
{
    "action": "Shell",
    "cmd": "getprop ro.build.version.sdk",
    "shell_timeout": 20000
}
```

## `Screencap`

Save current screenshot to file.

```jsonc
{
    "action": "Screencap",
    "filename": "my_capture",
    "format": "png"   // png / jpg / jpeg
}
```

Saved to `log_dir/screencap/`. File path available via action detail.

## `Custom`

Execute a registered custom action.

```jsonc
{
    "action": "Custom",
    "custom_action": "MyAct",
    "custom_action_param": { "key": "value" },
    "target": true
}
```

<!--
Source references:
- docs/en_us/3.1-PipelineProtocol.md (Action Types section)
-->
