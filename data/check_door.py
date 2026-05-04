import json
with open(r"G:\314\CannotMax-main\simulator\monsters.json", encoding='utf-8') as f:
    data = json.load(f)
for m in data["monsters"]:
    if "门" in m["名字"] and '"' not in m["名字"]:
        print(json.dumps(m, ensure_ascii=False, indent=2))
        break
for m in data["monsters"]:
    if '"门"' in m["名字"] or '门' == m["名字"]:
        print(json.dumps(m, ensure_ascii=False, indent=2))
        break
