"""一站式生成 monsters.json — 包含协同、行为"""
import csv, json

# 协同配置 (原始名称)
SYNERGY = {
    "玉双剑": ["炭长矛", "枣大刀"],
    "腐败骑士": ["凋零骑士"],
    "凋零骑士": ["腐败骑士"],
    "凯尔希": ["Mon3tr"],
}

# 行为配置 (原始名称)
from collections import OrderedDict as od
BEHAVIORS = od()
BS = []

def bh(name, *behaviors): BS.append((name, list(behaviors)))

bh("高能源石虫", {"类型":"死亡自爆","radius":1.25,"damage_ratio":4.0,"dmg_type":"物理"})
bh("冰爆源石虫", {"类型":"死亡自爆","radius":1.65,"damage_ratio":2.0,"dmg_type":"物理","extra_effect":"CHILL","extra_duration":10})
bh("枯朽之种", {"类型":"攻击自杀"})
bh("变异巨岩蛛", {"类型":"定期召唤","monster_name":"畸变赘生物","interval":5.0,"count":1,"max_total":30}, {"类型":"死亡召唤","monster_name":"畸变赘生物","count":4})
bh("水手重艇", {"类型":"死亡召唤","monster_name":"过气水手","count":1})
bh("瘴", {"类型":"重生","hp_ratio":1.0,"count":1})
bh("码头水手", {"类型":"攻击晕眩","duration":7.0,"every_n_hits":4})
bh("狙击步兵", {"类型":"攻击晕眩","duration":5.0,"every_n_hits":3})
bh("荒原劫掠者", {"类型":"攻击回血","hp_ratio":0.05,"every_n_hits":2})
bh("弧光镜卫", {"类型":"受击降防","def_per_hit":50,"mr_per_hit":2,"max_stacks":80})
bh("诡核集养者", {"类型":"半血分裂"})
bh("岁相", {"类型":"多目标","max_targets":2})
bh("R-11突击动力装甲", {"类型":"多目标","max_targets":4})
bh("凋零骑士", {"类型":"多目标","max_targets":2})
bh("\"复仇者\"", {"类型":"半血狂暴","atk_mult":3.8,"dmg_reduction":0.6,"threshold":0.5},{"类型":"重生","hp_ratio":0.5,"count":1})
bh("\"自在\"", {"类型":"重生","hp_ratio":1.0,"count":1},{"类型":"多目标","max_targets":2})
bh("扎罗，\"狼之主\"", {"类型":"重生","hp_ratio":1.0,"count":1})
bh("杰斯顿·威廉姆斯", {"类型":"重生","hp_ratio":1.0,"count":1})
bh("朗姆酒推荐者", {"类型":"死亡自爆","radius":2.0,"damage_ratio":2.8,"dmg_type":"物理"})
bh("宿主流浪者", {"类型":"持续回血","hp_per_second":250})
bh("狂暴宿主组长", {"类型":"持续掉血","hp_ratio_per_second":0.02})
bh("复仇者", {"类型":"半血狂暴","atk_mult":3.8,"dmg_reduction":0.6,"threshold":0.5})
bh("矿脉守卫", {"类型":"反伤","reflect_ratio":1.0,"dmg_type":"法术"})
bh("田鼷", {"类型":"受击加速","speed_mult":4.0,"duration":5.0,"cooldown":10.0})
bh("提亚卡乌好战者", {"类型":"攻击降防","def_reduce":10,"max_stacks":99})
bh("冰原术师", {"类型":"攻击寒冷","duration":10.0})
bh("萨卡兹枯朽吞噬者", {"类型":"范围内秒杀","radius":1.0,"max_kills":1,"cooldown":20.0})
bh("暴走食人花", {"类型":"范围内秒杀","radius":1.0,"max_kills":1,"cooldown":20.0})
bh("水遁忍者", {"类型":"旋转AOE","radius":1.0,"damage_ratio":1.2,"duration":60.0,"cooldown":60.0,"dmg_type":"物理"})
bh("沉沙", {"类型":"刀机制","sword_count":4,"atk_per_sword":0.7})
bh("\"门\"", {"类型":"死亡秒杀"})
bh("残党萨克斯手", {"类型":"多目标","max_targets":4})
bh("\"投石机\"", {"类型":"多目标","max_targets":2})
bh("\"盛怒\"", {"类型":"多目标","max_targets":2})
bh("萨卡兹枯朽宿卫", {"类型":"多目标","max_targets":2})  # 高仇恨→模拟多目标
bh("腐败骑士", {"类型":"半血狂暴","atk_mult":1.8,"dmg_reduction":0.0,"threshold":0.01})

bh_map = {n: b for n, b in BS}

# 生成怪物列表
monsters = []
with open('monster_greenvine.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row['原始名称'].strip().strip('"').strip("'").strip()
        if not name: continue
        
        hp = max(float(row.get('生命值', 0) or 0), 100)
        atk = max(float(row.get('攻击力', 0) or 0), 10)
        defense = float(row.get('防御力', 0) or 0)
        mr = float(row.get('法抗', 0) or 0)
        speed = float(row.get('移速', 1.0) or 1.0)
        interval = float(row.get('攻击间隔', 2.0) or 2.0)
        radius = float(row.get('攻击范围', 0.8) or 0.8)
        atk_type = (row.get('攻击类型', '') or '').strip() or '物理'
        trait = (row.get('技能/特性', '') or '').strip().strip('"').strip("'").strip()
        
        m = {
            '名字': name,
            '攻击力': {'数值': atk},
            '类型': atk_type,
            '生命值': {'数值': hp},
            '物理防御': {'数值': defense},
            '法抗': {'数值': mr},
            '攻击范围': {'数值': radius},
            '攻击间隔': {'数值': interval},
            '移速': {'数值': speed},
            '特性': trait or ''
        }
        if name in SYNERGY:
            m['协同'] = SYNERGY[name]
        if name in bh_map:
            m['行为'] = bh_map[name]
        monsters.append(m)

# 额外怪物（不在CSV中的衍生物等）
def add_extra(name, atk, atk_type, hp, defense, mr, radius, interval, speed, trait, synergy=None, behaviors=None):
    if any(m['名字'] == name for m in monsters): return
    m = {'名字':name,'攻击力':{'数值':atk},'类型':atk_type,'生命值':{'数值':hp},'物理防御':{'数值':defense},'法抗':{'数值':mr},'攻击范围':{'数值':radius},'攻击间隔':{'数值':interval},'移速':{'数值':speed},'特性':trait}
    if synergy: m['协同'] = synergy
    if behaviors: m['行为'] = behaviors
    monsters.append(m)

add_extra("畸变赘生物", 200, '物理', 800, 0, 0, 0.8, 2.0, 0.8, '大喷蛛衍生物')
add_extra("大君之赐", 300, '法术', 2000, 100, 30, 3.0, 3.0, 0.5, '重生术师衍生物')
add_extra("枣大刀", 1800, '物理', 60000, 600, 0, 0.8, 2.0, 1.0, '侠客三人行·协同')
add_extra("炭长矛", 1000, '物理', 120000, 300, 0, 0.8, 2.0, 0.9, '侠客三人行·协同')
add_extra("玉双剑", 700, '法术', 38000, 300, 35, 0.8, 2.0, 0.5, '侠客三人行·协同', synergy=["炭长矛","枣大刀"])

save = {'monsters': monsters}
with open('simulator/monsters.json', 'w', encoding='utf-8') as f:
    json.dump(save, f, ensure_ascii=False, indent=2)

bh_count = sum(1 for m in monsters if m.get('行为'))
syn_count = sum(1 for m in monsters if m.get('协同'))
print(f'{len(monsters)} monsters, {bh_count} w/ behaviors, {syn_count} w/ synergy')
