---
name: core-integration
description: Integration guide — creating Tasker, Resource, Controller and binding them together
---

# Integration Guide

MaaFW requires three core objects: **Controller** (device connection), **Resource** (pipeline/model/image loading), and **Tasker** (execution engine).

## Setup Flow

```
1. Find devices
2. Create Controller → connect
3. Create Resource → load bundles
4. Create Tasker → bind controller + resource
5. Post tasks
```

## Python Example

```python
from maa.resource import Resource
from maa.controller import AdbController
from maa.tasker import Tasker
from maa.toolkit import AdbDevice

# Find devices
devices = AdbDevice.find()
ctrl = AdbController(devices[0].adb_path, devices[0].address)
ctrl.post_connection().wait()

res = Resource()
res.post_bundle("./resource").wait()

tasker = Tasker()
tasker.controller = ctrl
tasker.resource = res

# Check initialization
assert tasker.inited

# Run a task
job = tasker.post_task("MyEntry")
result = job.wait()
print(f"completed: {result.completed}")
```

## NodeJS Example

```typescript
import * as maa from '@maaxyz/maa-node'

const devices = await maa.AdbController.find()
const [name, adb_path, address, screencap_methods, input_methods, config] = devices[0]
const ctrl = new maa.AdbController(adb_path, address, screencap_methods, input_methods, config)
await ctrl.post_connection().wait()

const res = new maa.Resource()
await res.post_bundle('./resource').wait()

const tskr = new maa.Tasker()
tskr.controller = ctrl
tskr.resource = res

if (await tskr.post_task('Task1').wait().success) {
    console.log('success!')
}
```

## Controller Types

| Type | Platform | Create Function |
|------|----------|----------------|
| Adb | Android | `AdbController(adb_path, address, screencap, input, config)` |
| Win32 | Windows | `Win32Controller(hwnd, screencap, mouse, keyboard)` |
| MacOS | macOS 14+ | `MacOSController(...)` |
| PlayCover | macOS | `PlayCoverController(address, uuid)` |
| Gamepad | Windows | `GamepadController(hwnd, type, screencap)` — needs ViGEm driver |
| WlRoots | Linux | `WlRootsController(socket_path)` |
| Custom | Any | `CustomController(callbacks)` |
| Debug | Any | `DbgController(image_path)` — for testing |

## Pipeline Override at Runtime

Override pipeline nodes dynamically when posting tasks:

```python
# Python
tasker.post_task("MyTask", {
    "SomeNode": {
        "threshold": 0.9,
        "enabled": True
    }
})
```

```typescript
// NodeJS
await tskr.post_task('MyTask', {
    SomeNode: {
        threshold: 0.9,
        enabled: true
    }
}).wait()
```

## Async Operations

Most operations (`post_connection`, `post_bundle`, `post_task`, `post_click`, etc.) are async and return operation IDs. Use `status()` to poll or `wait()` to block until complete.

## Global Options

Set before creating objects:

- `LogDir`: Log output path
- `SaveDraw`: Save recognition visualization images
- `DebugMode`: Enable debug details in callbacks
- `SaveOnError`: Save screenshot on task failure
- `DrawQuality`: JPEG quality for debug images (0-100, default 85)

<!--
Source references:
- docs/en_us/1.1-QuickStarted.md
- docs/en_us/2.1-Integration.md
- docs/en_us/2.2-IntegratedInterfaceOverview.md
-->
