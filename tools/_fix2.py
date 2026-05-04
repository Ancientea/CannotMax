import json
with open('simulator/monsters.json','r',encoding='utf-8') as f:
    d = json.load(f)

# MONSTER_MAPPING 里可能不在 monsters.json 的额外名字
missing_monsters = {
    '"钳钳生风"': {"攻击力":{"数值":1950},"类型":"物理","生命值":{"数值":8500},"物理防御":{"数值":2500},"法抗":{"数值":90},"攻击范围":{"数值":0.8},"攻击间隔":{"数值":2.0},"移速":{"数值":0.6},"特性":"在家里大家都很尊重它"},
}

existing = {m["名字"] for m in d["monsters"]}
added = 0
for name, data in missing_monsters.items():
    if name not in existing:
        data["名字"] = name
        d["monsters"].append(data)
        added += 1
        print(f"+ {name}")

with open('simulator/monsters.json','w',encoding='utf-8') as f:
    json.dump(d,f,ensure_ascii=False,indent=2)
print(f"total {len(d['monsters'])} monsters")
