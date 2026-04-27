---
name: advanced-troubleshooting
description: Debugging, logging, and troubleshooting MaaFramework issues
---

# Troubleshooting

## Log Files

Default location: `<working_directory>/debug/maafw.log`

If logs seem truncated, also check `debug/maafw.bak.log`.

## Debug Options

Configure in `<working_directory>/config/maa_option.json`:

```json
{
    "logging": true,
    "save_draw": true,
    "stdout_level": 2,
    "save_on_error": true,
    "draw_quality": 85
}
```

- `save_draw`: Save recognition visualization images to `debug/vision/`. Shows ROI regions, hit locations, match scores.
- `stdout_level`: 0=silent, 2=error only, 7=all logs
- `save_on_error`: Auto-screenshot on task failure

Set programmatically via `MaaGlobalSetOption` / `Toolkit.init_option`.

## Debug Images

When `save_draw` is enabled, images are saved to `debug/vision/` with format: `{node_name}_{reco_id}_{timestamp}.jpg`.

These annotate screenshots with ROI regions, hit boxes, and match details.

## Common Issues

### Recognition Not Matching
1. Enable `save_draw` to see what the algorithm "sees"
2. Check template images are 720p lossless
3. Verify ROI coordinates are correct
4. Adjust threshold values
5. Check `order_by` and `index` settings

### Task Timeout
1. Check if `timeout` is sufficient
2. Verify `rate_limit` isn't too high
3. Ensure `next` list nodes have correct recognition configs
4. Check if `enabled: false` is set on needed nodes

### Controller Connection Issues
- **ADB**: Verify `adb devices` shows the device
- **Win32**: Check window handle, class name regex, admin privileges
- **MacOS**: Verify Screen Recording + Accessibility permissions

### Crash Dumps
- **Windows**: `C:\Users\<Username>\AppData\Local\CrashDumps\`
- **Linux/macOS**: Check `dmesg` or system logs

## Reporting Issues

Provide:
- Complete `maafw.log` file
- Pipeline JSON for problematic nodes
- Template images (for TemplateMatch/FeatureMatch)
- Debug images from `debug/vision/`
- MaaFramework version
- Minimal reproduction

<!--
Source references:
- docs/en_us/5.1-Troubleshooting.md
-->
