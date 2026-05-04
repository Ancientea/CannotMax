"""从Wiki选手信息页面提取的完整数值"""
# 格式: 显示名 -> {hp, atk, def, mr, range, interval, type, trait}
WIKI_FULL = {}

def add_full(name, hp=None, atk=None, defense=None, mr=None, 
             atk_range=None, atk_interval=None, atk_type=None, trait=None):
    d = {}
    if hp is not None: d['生命值'] = hp
    if atk is not None: d['攻击力'] = atk
    if defense is not None: d['防御力'] = defense
    if mr is not None: d['法抗'] = mr
    if atk_range is not None: d['攻击范围'] = atk_range
    if atk_interval is not None: d['攻击间隔'] = atk_interval
    if atk_type: d['类型'] = atk_type
    if trait: d['特性'] = trait
    WIKI_FULL[name] = d

# ===== 绿藤城 选手信息（有完整数值的）=====
add_full("群集之瘴", 10000, 750, 200, atk_range=0.9)
add_full('"食器时代"', 3000, 800, 3000, atk_range=0.8, trait="嘲讽+1; 优先高仇恨")
add_full("极饿先锋", 25000, 1050, defense=0, mr=20, atk_range=0.8, trait="吞噬:强制击杀1名高仇恨敌人,+700防")
add_full("龙卷风忍者", 12000, 750, 350, atk_range=0.8, atk_type="物理", trait="漩涡:持续旋转,AOE 120%物伤")
add_full('"睡眠不足"', 8000, 500, 100, mr=60, atk_range=0.8)
add_full('"情绪失控"', 16000, 1100, 600, mr=25, atk_range=1.9, atk_type="法术")
add_full("匪帮欢乐船", 18000, 1300, 700, atk_range=0.8, trait="嘲讽+1; 接触伤害100%物伤; 死亡召唤过气水手")
add_full("易爆种子", hp=1200, atk=1750, atk_range=0.8, trait="攻击后自爆")
add_full("止戈者", 10000, 1500, 500, mr=30, atk_range=0.8, atk_type="物理", trait="出场4断刃,每刀+70%攻击,耗尽多砍一刀")
add_full("速胜卫士", atk=1000, defense=2500, atk_range=0.8, trait="受击-50防/-2法抗,最多80层; 召唤交通亭")
add_full('"染血之盾"', 5000, 1000, 5000, mr=90, atk_range=0.8)
add_full('"交通亭"量产型', defense=2500, mr=70, atk_range=0.8, trait="受击-50防/-1法抗,最多110层; 死亡AOE+弹出速胜卫士")
add_full("神拳机甲", atk=1600, defense=1000, mr=20, atk_range=0.8, atk_type="物理", trait="超负载神拳:锁定目标10秒连击+晕眩")
add_full("神射机甲", atk=1200, defense=1000, mr=20, atk_range=0.8, atk_type="物理", trait="打4目标,50%攻击力")
add_full("度假区越野车", 15000, 750, 500, mr=10, atk_range=0.8, trait="载客3:搭载非领袖地面友军")
add_full("橡胶弹狙击手", 3500, 300, defense=100, mr=10, atk_range=3.8, atk_type="物理", trait="嘲讽-1; 每3击晕眩5s")
add_full("迷茫的合声", 60000, 300, atk_range=8.0, trait="超大范围")

# ===== 领袖特辑 =====
add_full("岁相", 130000, 1200, defense=550, mr=30, atk_range=3.5, atk_interval=3.267, atk_type="物理", trait="巨型;双目标;远山惊雷AOE;天坠;十方吐纳龙息")
add_full('"萨米的意志"', 100000, 800, 800, mr=40, atk_range=5.0, atk_type="物理", trait="巨型;整列冰凌;半血减伤60%;自然涌动晕眩")
add_full("枣大刀", 60000, 1800, 600, atk_range=0.8, atk_type="物理", trait="单刀赴会:AOE 80%物伤")
add_full("炭长矛", 120000, atk=1000, defense=300, atk_range=0.8)
add_full("玉双剑", 38000, 700, 300, mr=35, atk_range=0.8, trait="全队减伤至60%;治疗最低血量队友")
add_full("腐败骑士", 70000, 1500, 1000, atk_range=0.8, atk_type="物理", trait="横扫5列; 蓄力锤300%; 队友死+80%攻击")
add_full("凋零骑士", 70000, 700, 500, atk_range=3.5, atk_type="法术", trait="双目标; 爆炸箭160%法术AOE; 队友死+80%攻击")
add_full("奎隆，摩诃萨埵权化", 300000, 3500, 1200, mr=70, atk_range=0.8, atk_interval=6.0, atk_type="物理", trait="惩五戒:5连击90%物伤")
add_full('"自在"', 40000, 600, 800, mr=55, atk_range=3.5, atk_type="法术", trait="二阶段重生; 纬地经天; 破桎而出盾+爆")
add_full('依然"狼之主"', hp=50000, atk=1500, atk_range=0.8, trait="二阶段重生; 溶血骇惧3目标流血; 二阶段二连击")
add_full('"火与钢"', 57500, 800, 650, mr=60, atk_range=0.8, atk_interval=1.0, atk_type="物理", trait="复仇者:半血+280%攻-60%伤;二阶段;冲锋")

# ===== 青草城/蜜果城 =====
add_full("喷射背包客", 3500, 380, 200, mr=20, atk_range=0.8)
add_full("黑土游击弩手", 5000, 430, atk_range=3.4, atk_type="物理")
add_full("赛场无赖射手", hp=4500, atk_range=3.8, atk_type="物理", trait="每3击AOE周围4格")
add_full("过激的竞猜者", hp=8000, atk_range=0.8, trait="每2击治疗5%最大生命")
add_full("爆裂冰法师", hp=5000, atk=300, atk_range=0.8, atk_type="法术")
add_full('"配重投石机"', 75000, 1500, defense=600, atk_range=1.0, atk_type="物理", trait="超晕眩击:25s晕眩")
add_full("臭嗓门战士", hp=15000, atk_range=0.8, trait="嘲讽+1; 沉默抗性")
add_full("最普通的鼷", 2000, 340, atk_range=0.8)
add_full("变异食肉植物", 13000, atk=0, defense=250, mr=10, atk_range=1.0, trait="吞噬:秒杀1.0范围内敌人")
add_full("高敏感积藏者", 8000, 190, 150, mr=35, atk_range=2.8, trait="半血分裂")
add_full('"铜舌"', 15000, 1200, 700, mr=20, atk_range=1.0)
add_full('"巢穴"', hp=12000, atk_range=0.8, trait="30秒内+400%HP+80%攻击(受伤/攻击后停)")
add_full("狂躁觅食兽", 25000, 600, 20, atk_range=0.8, atk_type="物理", trait="加速:每0.33s+33.3%移速;撞击伤害;冲锋秒杀")
add_full("勇敢的壳", hp=8000, atk_range=0.8)
add_full("荒原刺背兽", hp=9000, atk_range=1.0, trait="背面减伤75%; 失血12%倍数→扇形AOE240%")
add_full("石头脑袋", hp=10000, atk_range=1.0)
add_full("勤奋的钻头", hp=12000, atk_range=0.8)
add_full("改装排污车", hp=10000, atk_range=0.8, trait="每3击扩大范围4.0+污染区域100真伤/秒")
add_full("覆面大锤客", hp=12000, atk_range=0.8)
add_full("街头乐队鼓手", hp=6000, atk_range=1.0, trait="每3击AOE半径2.0")
add_full("街头乐队贝斯手", hp=8000, atk=500, atk_range=2.6, atk_type="法术", trait="Free Bird:持续施法15s,150法伤/秒+80灼燃")
add_full("街头乐队吉他手", hp=10000, atk=700, atk_range=2.6, atk_type="法术", trait="Free Bird+:持续施法15s,200法伤/秒+80灼燃")

# ===== 其他有特殊机制的 =====
add_full("大喷蛛", 15000, 600, 300, atk_range=3.0, atk_type="物理", trait="每5秒召唤畸变赘生物;死亡召唤4只")
add_full("假酒海盗", 10500, 1000, 500, mr=20, atk_range=0.8, trait="酒桶:280%物伤+酒区域(友方+100攻速+80%闪避)")
add_full("无德决斗家", 17000, 600, 500, mr=60, atk_range=1.0, atk_type="法术", trait="SP:150%法伤+缴械15s")
add_full("凋零萨卡兹", hp=7500, atk=450, atk_range=3.5, atk_type="法术", trait="普攻附加25%凋亡损伤; 凋零技能220%凋亡")
add_full("星星魔法师", 7000, 380, 250, mr=50, atk_range=4.5, atk_interval=3.0, atk_type="法术")
add_full("点灯骑士", 7500, 350, 300, mr=60, atk_range=3.2, atk_interval=5.0, atk_type="法术", trait="微光之触:蓄力7s→250%法术AOE")
add_full("易爆源石虫", 2460, 260, atk_range=0.8, atk_type="物理", trait="死亡自爆:400%物伤半径1.25")
add_full("自爆冰虫", 3250, 300, atk_range=0.8, atk_type="物理", trait="死亡冰爆:200%物伤+10s寒冷半径1.65")
add_full("失业萨克斯手", 17500, 800, 650, mr=20, atk_range=0.8, trait="宣泄怨气:四方150%物伤")
add_full("过气水手", 10000, 800, 800, mr=20, atk_range=0.8, trait="晕眩击:每4击晕眩7s")
add_full("巨大雪球投手", 10000, 800, 700, mr=20, atk_range=1.0)
add_full("雪境精锐", hp=6000, atk_range=0.8)
add_full("醒的墨魉", hp=3000, atk=290, defense=120, mr=10, atk_range=0.8)
add_full("高普尼克", 30000, 2000, 200, mr=50, atk_range=0.8, trait="晕眩冻结浮空抗性")
add_full("食腐野兽", hp=5000, atk_range=0.8)
add_full("安逸的驮兽", 65000, 1800, 50, mr=50, atk_range=0.8)
add_full("榴弹佣兵", 3200, 500, 50, mr=40, atk_range=1.0, atk_type="物理")
add_full("重生术师", hp=4000, atk_range=3.2, atk_type="法术", trait="召唤大君之赐")
add_full("炮击组长", 5000, 550, 150, atk_range=7.0, atk_interval=4.5, atk_type="物理")
# ===== 补齐：CSV有但Wiki遗漏或名字不匹配的 =====
add_full('"情绪失控"', 16000, 1100, 600, mr=25, atk_range=1.9, atk_type="法术")
add_full('"自在"', 40000, 600, 800, mr=55, atk_range=3.5, atk_type="法术", trait="二阶段重生; 纬地经天; 破桎而出")
add_full('"萨米的意志"', 100000, 800, 800, mr=40, atk_range=5.0, atk_type="物理", trait="巨型;整列冰凌;半血减伤60%;自然涌动")
add_full('依然"狼之主"', 50000, 1500, atk_range=0.8, trait="二阶段重生; 溶血骇惧; 二连击")
add_full('"食器时代"', 3000, 800, 3000, atk_range=0.8, trait="嘲讽+1; 优先高仇恨")
add_full('"交通亭"量产型', defense=2500, mr=70, atk_range=0.8, trait="受击-50防/-1法抗; 死亡AOE+弹出速胜卫士")
add_full('圣徒卡门', hp=30000, atk=1000, atk_range=0.8, trait="领袖")
add_full('调停的意志', hp=30000, atk=800, atk_range=0.8)
add_full('奔跑吧！躯壳！', hp=8000, atk=400, atk_range=0.8)
add_full('衣架射手', hp=3500, atk=300, atk_range=3.2, atk_type="物理")
add_full('劈柴骑士', 6500, 975, 800, atk_range=0.8, atk_type="物理", trait="友方死亡+10%攻+5攻速,上限10层")
add_full('锁链拳手', 4000, 450, 300, atk_range=0.8, atk_type="物理", trait="双形态:禁锢/解放")
add_full('狗Pro', 3000, 370, defense=0, mr=20, atk_range=0.8, atk_type="物理")
add_full('"庞贝"', 40000, 230, 220, mr=70, atk_range=6.0, atk_type="法术", trait="大型源石虫,全屏范围")
add_full('杰斯顿·威廉姆斯', 25000, 700, 400, mr=20, atk_range=0.8, atk_type="物理", trait="二阶段:露出真面目")
add_full('硕鼷', 2200, 340, atk_range=0.8)
add_full('超级洗地机', 25000, 800, 1200, mr=20, atk_range=0.8, atk_type="物理")
add_full('标枪恐鱼', 4000, 400, atk_range=1.0, atk_type="物理")
add_full('"火与钢"', 57500, 800, 650, mr=60, atk_range=0.8, atk_interval=1.0, atk_type="物理", trait="半血+280%攻-60%伤")
add_full('"自在"', 40000, 600, 800, mr=55, atk_range=3.5, atk_type="法术", trait="二阶段;纬地经天;破桎而出")
add_full('"铜舌"', 15000, 1200, 700, mr=20, atk_range=1.0)
add_full('醒的墨魉', 3000, 290, 120, mr=10, atk_range=0.8)
add_full('"染血之盾"', 5000, 1000, 5000, mr=90, atk_range=0.8)
add_full('保鲜膜骑士', 2800, 300, 100, atk_range=0.8)
add_full('拳击宗师', 4000, 320, 250, mr=50, atk_range=0.8, atk_type="物理")
add_full("高能源石虫", 1230, 390, atk_range=0.8, atk_interval=1.7, atk_type="物理", trait="死亡自爆1560物伤")
add_full("灼热源石虫", 1200, 50, atk_range=2.8, atk_interval=2.0, atk_type="法术", trait="灼燃")
add_full("冰爆源石虫", 3250, 300, atk_range=0.8, atk_interval=1.7, atk_type="物理", trait="死亡冰爆+10s寒冷")
add_full("萨卡兹大剑手", 3750, 900, 230, mr=50, atk_range=0.8, atk_interval=2.0, atk_type="物理")
add_full("重装防御者", hp=3000, atk=900, defense=800, atk_range=0.8, atk_interval=2.6, atk_type="物理")
add_full("复仇者", 9000, 720, 230, mr=50, atk_range=0.8, atk_interval=2.3, atk_type="物理")
add_full("狂暴宿主组长", 15000, atk=2625, defense=230, mr=30, atk_range=0.8, atk_interval=1.3, atk_type="物理")
add_full("泥岩巨像", 50000, 4500, 700, mr=30, atk_range=0.8, atk_interval=7.0, atk_type="物理")
add_full("高塔术师", 7000, 600, 160, mr=65, atk_range=3.2, atk_interval=7.0, atk_type="法术")
add_full("冰原术师", 5000, 250, 200, mr=50, atk_range=3.2, atk_interval=4.5, atk_type="法术")
add_full("提亚卡乌好战者", 2750, 480, 380, mr=10, atk_range=0.8, atk_interval=1.0, atk_type="物理", trait="破甲10")
add_full("矿脉守卫", 25000, 400, 300, mr=10, atk_range=0.8, atk_interval=4.0, atk_type="法术", trait="反伤")
add_full("反装甲步兵", 3200, 500, 50, mr=40, atk_range=1.0, atk_interval=2.0, atk_type="物理")
add_full("码头水手", 10000, 800, 800, mr=20, atk_range=0.8)
add_full("萨卡兹枯朽宿卫", 5000, 1000, 5000, mr=90, atk_range=0.8)

print(f"WIKI_FULL: {len(WIKI_FULL)} 条目")
import json
with open('tools/_wiki_full.json', 'w', encoding='utf-8') as f:
    json.dump(WIKI_FULL, f, ensure_ascii=False, indent=2)
print("已保存 tools/_wiki_full.json")
