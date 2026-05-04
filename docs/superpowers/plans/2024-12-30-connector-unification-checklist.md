# Connector Unification Testing Checklist

## Manual Testing

### 1. ADB Mode
- [ ] App starts, ADB connects automatically (if device available)
- [ ] MAA status shows correct state
- [ ] 识别 button works, shows results

### 2. PC Mode (single window)
- [ ] Switch to PC mode
- [ ] Auto-connects to single 明日方舟 window
- [ ] MAA status updates
- [ ] 识别 works

### 3. PC Mode (multi-window)
- [ ] Open 2 Arknights instances (if possible)
- [ ] Switch to PC mode
- [ ] Dialog shows 2 options
- [ ] Select one, connects to it
- [ ] 识别 works

### 4. WIN Mode
- [ ] Switch to WIN mode
- [ ] Click 选择窗口
- [ ] Select a window
- [ ] 识别 works

### 5. Mode Switching
- [ ] ADB→PC→ADB should reuse ADB connection
- [ ] No crashes during rapid switching

## Code Verification
- [ ] All py_compile checks pass
- [ ] No references to old connector pattern
- [ ] Import chain works: `from src.cannotmax.core.connector import ConnectorFactory`
