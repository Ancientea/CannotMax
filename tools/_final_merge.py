"""最终合并：CSV数值 + Wiki攻击细节 + 智能推断 → monsters.json"""
import csv, json

# 加载 Wiki 提取数据
with open('tools/_wiki_ranges.json', 'r', encoding='utf-8') as f:
    WIKI = json.load(f)

# 魔法关键词（Wiki没覆盖时的备用推断）
MAGIC_KW = ["术师","法师","萨卡兹","巫","术","凋零","灼","燃","星术","烈酒","乐","音","合唱"]

monsters = []
hit_wiki = 0
hit_fallback = 0

with open('monster_greenvine.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row.get('名称','').strip().strip('"').strip("'").strip()
        orig = row.get('原始名称','').strip().strip('"').strip("'").strip()
        if not name:
            continue

        hp = float(row.get('生命值', 1000) or 1000)
        atk = float(row.get('攻击力', 100) or 100)
        defense = float(row.get('防御力', 0) or 0)
        mres = float(row.get('法抗', 0) or 0)
        speed = float(row.get('移速', 1.0) or 1.0)
        trait = (row.get('技能/特性','') or '').strip().strip('"').strip("'").strip()

        # 查找 Wiki 数据
        wiki = WIKI.get(name) or WIKI.get(orig)
        if wiki:
            atk_range = wiki.get('攻击范围', 0.8)
            atk_interval = wiki.get('攻击间隔', 2.0)
            atk_type = wiki.get('类型', '物理')
            wiki_trait = wiki.get('特性', '')
            if wiki_trait and not trait:
                trait = wiki_trait
            hit_wiki += 1
        else:
            # 智能推断
            atk_range = 0.8
            atk_interval = 2.0
            atk_type = '物理'

            # 关键词推断攻击类型
            for kw in MAGIC_KW:
                if kw in name or kw in orig or kw in trait:
                    atk_type = '法术'
                    break

            # 关键词推断远程
            if any(kw in name for kw in ['炮','射','术','投','喷','弓','弩','法师','合唱','狙击']):
                atk_range = 3.2
                atk_interval = 3.5
            elif trait and '远程' in trait:
                atk_range = 2.5
            hit_fallback += 1

        if not atk_type:
            atk_type = '物理'

        monsters.append({
            '名字': name,
            '攻击力': {'数值': atk},
            '类型': atk_type,
            '生命值': {'数值': hp},
            '物理防御': {'数值': defense},
            '法抗': {'数值': mres},
            '攻击范围': {'数值': atk_range},
            '攻击间隔': {'数值': atk_interval},
            '移速': {'数值': speed},
            '特性': trait or ''
        })

with open('simulator/monsters.json', 'w', encoding='utf-8') as f:
    json.dump({'monsters': monsters}, f, ensure_ascii=False, indent=2)

# 统计
ranged = sum(1 for m in monsters if m['攻击范围']['数值'] > 0.8)
magic = sum(1 for m in monsters if m['类型'] == '法术')
print(f'总数: {len(monsters)} | Wiki: {hit_wiki} | 推断: {hit_fallback} '
      f'| 远程: {ranged} | 法术: {magic}')

# 列出推断（非Wiki）的怪物
print('\n非Wiki覆盖的怪物:')
for m in monsters:
    name = m['名字']
    if name not in WIKI and (m['原始名称'] if '原始名称' in m else '') not in {k for k in WIKI}:
        pass

# 列出所有怪物摘要
with open('tools/_monster_summary.txt', 'w', encoding='utf-8') as out:
    out.write(f'{"名字":<20} {"HP":>8} {"ATK":>6} {"DEF":>6} {"MR":>4} '
              f'{"范围":>5} {"间隔":>5} {"类型":<4} {"来源"}\n')
    out.write('-'*80 + '\n')
    for m in monsters:
        src = 'Wiki' if m['名字'] in WIKI else '推断'
        out.write(f'{m["名字"]:<20} {m["生命值"]["数值"]:>8.0f} '
                  f'{m["攻击力"]["数值"]:>6.0f} {m["物理防御"]["数值"]:>6.0f} '
                  f'{m["法抗"]["数值"]:>4.0f} {m["攻击范围"]["数值"]:>5.1f} '
                  f'{m["攻击间隔"]["数值"]:>5.1f} {m["类型"]:<4} {src}\n')
print('详细列表已保存 tools/_monster_summary.txt')
