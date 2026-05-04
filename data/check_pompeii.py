import json
with open(r"G:\314\CannotMax-main\simulator\monsters.json", encoding='utf-8') as f:
    data = json.load(f)
for m in data["monsters"]:
    if "庞贝" in m["名字"]:
        print(f"名称: {m['名字']}")
        print(f"  ATK={m.get('攻击力')} HP={m.get('生命值')} DEF={m.get('物理防御')} RES={m.get('法术抗性')}")
        print(f"  SPEED={m.get('攻击速度')} RANGE={m.get('攻击范围')} TYPE={m.get('攻击类型')}")
        print()
        break

import csv
with open(r"G:\314\CannotMax-main\monster_greenvine.csv", encoding='utf-8') as f:
    for row in csv.reader(f):
        if "庞贝" in str(row):
            print(f"Greenvine CSV: {row}")
