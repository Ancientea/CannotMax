---
name: core-callbacks
description: Callback protocol — event message types, JSON structures, and handling patterns
---

# Callback Protocol

MaaFW sends event notifications via `MaaEventCallback`. All callbacks follow the pattern: `message` (type string) + `details_json` (JSON data).

## Callback Signature

```c
void callback(void* handle, const char* message, const char* details_json, void* trans_arg);
```

## Message Categories

### Resource Events

| Message | When |
|---------|------|
| `Resource.Loading.Starting` | Resource load begins |
| `Resource.Loading.Succeeded` | Resource load succeeds |
| `Resource.Loading.Failed` | Resource load fails |

Details: `{ res_id, path, type, hash }` where type is `Bundle`/`OcrModel`/`Pipeline`/`Image`.

### Controller Events

| Message | When |
|---------|------|
| `Controller.Action.Starting` | Controller action begins |
| `Controller.Action.Succeeded` | Controller action succeeds |
| `Controller.Action.Failed` | Controller action fails |

Details: `{ ctrl_id, uuid, action, param, info }`.

### Task Events

| Message | When |
|---------|------|
| `Tasker.Task.Starting` | Task execution begins |
| `Tasker.Task.Succeeded` | Task execution succeeds |
| `Tasker.Task.Failed` | Task execution fails |

Details: `{ task_id, entry, uuid, hash }`.

### Node Events

| Message | When |
|---------|------|
| `Node.Recognition.Starting/Succeeded/Failed` | Recognition phase |
| `Node.Action.Starting/Succeeded/Failed` | Action phase |
| `Node.PipelineNode.Starting/Succeeded/Failed` | Full pipeline node |
| `Node.NextList.Starting/Succeeded/Failed` | Next list recognition |
| `Node.WaitFreezes.Starting/Succeeded/Failed` | Screen freeze wait |
| `Node.RecognitionNode.Starting/Succeeded/Failed` | Recognition-only task |
| `Node.ActionNode.Starting/Succeeded/Failed` | Action-only task |

Node event details include `task_id`, `node_id`, `name`, and `focus` data.

## Handling Pattern

```python
def my_callback(handle, message, details_json, trans_arg):
    details = json.loads(details_json)

    if message == "Tasker.Task.Starting":
        print(f"Task {details['entry']} starting")
    elif message == "Node.Recognition.Succeeded":
        print(f"Node {details['name']} recognized")
    elif message == "Node.Action.Failed":
        print(f"Node {details['name']} action failed")
```

## Important Notes

- Callbacks may be called from different threads — ensure thread safety
- Callbacks should return quickly to avoid blocking the framework
- Add exception handling to prevent callback errors from affecting execution

<!--
Source references:
- docs/en_us/2.3-CallbackProtocol.md
-->
