import json
with open(r"G:\314\CannotMax-main\simulator\monsters.json", encoding='utf-8') as f:
    data = json.load(f)

# 修复复仇者攻击范围：0.8 → 11.0（绿藤城远程单位）
fixed = 0
for m in data["monsters"]:
    if "复仇者" in m["名字"]:
        old = m.get("攻击范围", {}).get("数值")
        m["攻击范围"]["数值"] = 11.0
        print(f"  修复: {m['名字']} RANGE {old} → 11.0")
        fixed += 1

with open(r"G:\314\CannotMax-main\simulator\monsters.json", 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"共修复 {fixed} 个复仇者")
