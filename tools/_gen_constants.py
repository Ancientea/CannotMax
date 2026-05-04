import csv
rows = []
with open('monster_greenvine.csv','r',encoding='utf-8-sig') as f:
    r = csv.DictReader(f)
    for row in r:
        name = row['名称'].strip().strip('\u201c').strip('\u201d').strip('"').strip()
        hp = float(row.get('生命值',0) or 0)
        atk = float(row.get('攻击力',0) or 0)
        df = float(row.get('防御力',0) or 0)
        mr = float(row.get('法抗',0) or 0)
        speed = float(row.get('移速',1) or 1)
        interval = float(row.get('攻击间隔',2) or 2)
        radius = float(row.get('攻击范围',0.8) or 0.8)
        trait = (row.get('技能/特性','') or '').strip()[:30]
        rows.append((name, atk, df, hp, mr, interval, speed, radius, trait))

with open('constants.py','w',encoding='utf-8') as f:
    f.write('UNIT_CONFIG = {\n')
    for i,(name, atk, df, hp, mr, interval, speed, radius, trait) in enumerate(rows,1):
        f.write(f'    {i}: {{\n')
        f.write(f'        "name": "{name}",\n')
        f.write(f'        "attack": {atk},\n')
        f.write(f'        "defense": {df},\n')
        f.write(f'        "health": {hp},\n')
        f.write(f'        "magic_resist": {mr},\n')
        f.write(f'        "attack_interval": {interval},\n')
        f.write(f'        "move_speed": {speed}/2,\n')
        f.write(f'        "attack_radius": {radius},\n')
        f.write(f'        "effect": "{trait}",\n')
        f.write(f'        "icon": "images/{i}.png"\n')
        f.write(f'    }},\n')
    f.write('}\n')
print(f'{len(rows)} entries written')
