import json
with open(r"G:\314\CannotMax-main\simulator\monsters.json", encoding='utf-8') as f:
    data = json.load(f)

fixed = 0
for m in data["monsters"]:
    if "复仇者" in m["名字"] and m.get("攻击范围",{}).get("数值") == 11.0:
        m["攻击范围"]["数值"] = 0.8
        print(f"  回退: {m['名字']} RANGE 11.0 → 0.8")
        fixed += 1

with open(r"G:\314\CannotMax-main\simulator\monsters.json", 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"回退 {fixed} 个")
