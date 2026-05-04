import json
with open(r"G:\314\CannotMax-main\simulator\monsters.json", encoding='utf-8') as f:
    data = json.load(f)
for m in data["monsters"]:
    name = m["名字"]
    if any(k in name for k in ["门","高级武装","炮击组长","妒","沙滩车","复仇者","合声","沉沙","沸血","矿脉"]):
        atk = m.get("攻击力",{})
        hp = m.get("生命值",{})
        df = m.get("物理防御",{})
        res = m.get("法术抗性",{})
        spd = m.get("攻击速度",{})
        rng = m.get("攻击范围",{})
        typ = m.get("攻击类型",{})
        print(f"{name}: ATK={atk.get('数值')} HP={hp.get('数值')} DEF={df.get('数值')} RES={res.get('数值') if res else 'N'} SPD={spd.get('数值') if spd else 'N'} RNG={rng.get('数值')} TYPE={typ.get('数值') if typ else 'N'}")
