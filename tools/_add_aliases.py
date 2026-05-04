import json
with open('simulator/monsters.json','r',encoding='utf-8') as f:
    d = json.load(f)
# 找蟹蟹爷爷的数据
for m in d['monsters']:
    if '蟹' in m['名字']:
        copy = dict(m)
        copy['名字'] = '"钳钳生风"'
        d['monsters'].append(copy)
        print('Added:', copy['名字'])
        break
with open('simulator/monsters.json','w',encoding='utf-8') as f:
    json.dump(d,f,ensure_ascii=False,indent=2)
