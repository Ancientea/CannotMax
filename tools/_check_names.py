import json,sys
sys.stdout = open('tools/_name_check.txt','w',encoding='utf-8')
d = json.load(open('simulator/monsters.json','r',encoding='utf-8'))
for m in d['monsters']:
    if '钳' in m['名字'] or '蟹' in m['名字']:
        print(m['名字'], '|', m.get('行为','无'))
