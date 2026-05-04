import json
with open(r"G:\314\CannotMax-main\simulator\monsters.json", encoding='utf-8') as f:
    data = json.load(f)
for m in data["monsters"]:
    if "复仇者" in m["名字"]:
        rng = m.get("攻击范围",{})
        atk = m.get("攻击力",{})
        hp = m.get("生命值",{})
        df = m.get("物理防御",{})
        print(f"名字={m['名字']} ATK={atk.get('数值')} HP={hp.get('数值')} DEF={df.get('数值')} RANGE={rng.get('数值')}")
