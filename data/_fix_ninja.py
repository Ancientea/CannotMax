import json
with open(r'G:\314\CannotMax-main\simulator\monsters.json', encoding='utf-8') as f:
    data = json.load(f)

for m in data['monsters']:
    if '水遁' in m['名字']:
        print(f"Before: HP={m['生命值']['数值']} ATK={m['攻击力']['数值']} DEF={m['物理防御']['数值']}")
        m['生命值']['数值'] = 5000
        m['攻击力']['数值'] = 900
        m['物理防御']['数值'] = 500
        print(f"After:  HP={m['生命值']['数值']} ATK={m['攻击力']['数值']} DEF={m['物理防御']['数值']}")

with open(r'G:\314\CannotMax-main\simulator\monsters.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print("Done")
