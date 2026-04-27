---
name: core-controllers
description: Controller types, screencap/input methods, and platform-specific configuration
---

# Controller Methods

## ADB Controller

### Input Methods

Combine via bitwise OR. MaaFW auto-selects the fastest available.

| Name | Speed | Compatibility | Notes |
|------|-------|---------------|-------|
| AdbShell | Slow | High | |
| MinitouchAndAdbKey | Fast | Medium | Key input still uses AdbShell |
| Maatouch | Fast | Medium | |
| EmulatorExtras | Fast | Low | MuMu 12 only |

Priority: EmulatorExtras > Maatouch > MinitouchAndAdbKey > AdbShell

### Screencap Methods

Combine via bitwise OR. MaaFW auto-selects the fastest.

| Name | Speed | Compatibility | Encoding |
|------|-------|---------------|----------|
| EncodeToFileAndPull | Slow | High | Lossless |
| Encode | Slow | High | Lossless |
| RawWithGzip | Medium | High | Lossless |
| RawByNetcat | Fast | Low | Lossless |
| EmulatorExtras | Very Fast | Low | Lossless |

Avoid `MinicapDirect`/`MinicapStream` — lossy JPEG reduces template matching accuracy.

## Win32 Controller

### Screencap Methods

| Name | Speed | Background | Notes |
|------|-------|------------|-------|
| GDI | Fast | No | |
| FramePool | Very Fast | Yes | Win10 1903+, pseudo-minimize support |
| DXGI_DesktopDup | Very Fast | No | Full desktop copy |
| DXGI_DesktopDup_Window | Very Fast | No | Window-cropped desktop copy |
| PrintWindow | Medium | Yes | |
| ScreenDC | Fast | No | |

Predefined combos:
- `All`: all methods
- `Foreground`: `DXGI_DesktopDup_Window | ScreenDC`
- `Background`: `FramePool | PrintWindow`

### Input Methods

| Name | Admin | Background | Notes |
|------|-------|------------|-------|
| Seize | No | No | Direct input |
| SendMessage | Maybe | Yes | |
| PostMessage | Maybe | Yes | |
| SendMessageWithCursorPos | Maybe | Yes | Brief cursor move |
| PostMessageWithCursorPos | Maybe | Yes | Brief cursor move |
| SendMessageWithWindowPos | Maybe | Yes | Brief window move |
| PostMessageWithWindowPos | Maybe | Yes | Brief window move |

**Mouse Lock Follow** mode: for TPS/FPS games. Enable via controller option. Use `post_relative_move()` for camera rotation while active.

## MacOS Controller

Requires macOS 14+, Screen Recording + Accessibility permissions.

| Screencap | Input |
|-----------|-------|
| ScreenCaptureKit (fast, background) | GlobalEvent (high compat, foreground) |
| | PostToPid (medium compat, background) |

## Platform-Specific Notes

- **PlayCover**: Controls iOS apps on macOS via fork PlayCover. No key/text input support.
- **Gamepad**: Xbox360/DualShock4 emulation on Windows. Needs ViGEm Bus Driver.
- **WlRoots**: Linux Wayland compositors. Needs `virtual-keyboard-unstable-v1`, `wlr-screencopy-unstable-v1`, `wlr-virtual-pointer-unstable-v1`.

<!--
Source references:
- docs/en_us/2.4-ControlMethods.md
-->
