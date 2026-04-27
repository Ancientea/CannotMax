---
name: advanced-custom-logic
description: Custom recognition and action — extending pipelines with Python/NodeJS code via Agent process
---

# Custom Recognition & Action

MaaFW allows extending pipelines with custom code through the Agent system, which runs custom logic in a separate process.

## Pipeline Configuration

```jsonc
{
    "MyNode": {
        "recognition": "Custom",
        "custom_recognition": "MyReco",
        "custom_recognition_param": { "threshold": 0.8 },
        "action": "Custom",
        "custom_action": "MyAct",
        "custom_action_param": { "mode": "fast" }
    }
}
```

## Python Implementation

```python
from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.custom_action import CustomAction

@AgentServer.custom_recognition("MyReco")
class MyReco(CustomRecognition):
    def analyze(self, context, argv, reco_result):
        # argv.image: current screenshot
        # argv.recognition_param: custom_recognition_param from pipeline
        # Return (hit: bool, box: Rect, detail: str)
        return True, (10, 10, 100, 100), "detected"

@AgentServer.custom_action("MyAct")
class MyAct(CustomAction):
    def run(self, context, argv, box, reco_detail):
        # argv.action_param: custom_action_param from pipeline
        # box: recognition result rect
        # Perform custom logic
        context.controller.post_click(100, 10).wait()
        context.override_next(["TaskA", "TaskB"])  # Dynamically adjust flow
        return True

# Start the Agent server
AgentServer.start_up(sock_id)
```

## NodeJS Implementation

```typescript
import { AgentServer, CustomRecognizer, CustomAction } from '@maaxyz/maa-node'

// Custom Recognition
AgentServer.custom_recognition('MyReco', function (self) {
    // self.image: current screenshot
    // self.param: custom_recognition_param
    // Return [box, detail]
    return [{ x: 0, y: 0, width: 100, height: 100 }, 'result_text']
})

// Custom Action
AgentServer.custom_action('MyAct', function (self) {
    // self.box: recognition result
    // self.param: custom_action_param
    self.context.controller.post_click(100, 10)
    self.context.override_next(['TaskA', 'TaskB'])
    return true
})

AgentServer.start_up(sock_id)
```

## Context API

Inside custom recognition/action, the `context` object provides:

- `context.controller` — access to device control (click, swipe, screenshot, etc.)
- `context.run_recognition(entry, image)` — run another pipeline node's recognition
- `context.run_action(entry, box, detail)` — run another pipeline node's action
- `context.override_next(next_list)` — dynamically change the next node list
- `context.override_pipeline(pipeline_override)` — override pipeline config
- `context.wait_freezes(time)` — wait for screen to stabilize

## Agent Configuration (interface.json)

```jsonc
{
    "agent": {
        "child_exec": "python",
        "child_args": ["./agent/main.py"]
    }
}
```

The Client auto-connects to the Agent process and routes custom recognition/action calls.

## When to Use Custom Logic

- Complex decision-making that can't be expressed in pipeline JSON
- Image pre-processing or post-processing
- Calling external APIs or services
- Dynamic pipeline modification based on runtime state
- Logic requiring loops or conditional branching beyond pipeline capabilities

<!--
Source references:
- docs/en_us/1.1-QuickStarted.md (Approach 2)
- docs/en_us/NodeJS/J1.2-Custom Recognition and Action.md
-->
