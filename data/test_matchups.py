import sys,os,json,time,io
sys.path.insert(0,r'G:\314\CannotMax-main')
os.chdir(r'G:\314\CannotMax-main')
from simulator.battle_field import Battlefield
from simulator.utils import Faction, REVERSE_MONSTER_MAPPING
from simulator.monsters import MonsterFactory
import traceback

with open("simulator/monsters.json",encoding='utf-8') as f:
    mdata = json.load(f)["monsters"]

def name_lookup(partial):
    for m in mdata:
        if partial in m["名字"]:
            return m["名字"]
    return None

def names(parts):
    return {name_lookup(p):c for p,c in parts}

def test(n, label, left_parts, right_parts, expected, rounds=50):
    l = names(left_parts)
    r = names(right_parts)
    wl=wr=0
    t0=time.time()
    null=io.StringIO();old=sys.stdout;sys.stdout=null
    errors=0
    for _ in range(rounds):
        try:
            bf=Battlefield(mdata)
            bf.setup_battle(l,r,mdata)
            while True:
                res=bf.run_one_frame()
                if res is not None:
                    if res==Faction.LEFT:wl+=1
                    else:wr+=1
                    break
        except Exception as e:
            errors+=1
            if errors<=3:
                traceback.print_exc()
    sys.stdout=old
    left_rate=100*wl/rounds
    match="✅" if (expected=="L" and left_rate>50) or (expected=="R" and left_rate<50) else "❌"
    print(f"{n:>3} {match} {label:<50} 预期{expected}胜 左胜率{left_rate:.0f}% L{wl}/R{wr} [{time.time()-t0:.1f}s]" + (f" 崩{errors}次" if errors else ""))

print("="*90)
print(f"{'编号':>3} {'状态':<3} {'对局':<50} 结果")

# 之前的问题对局
test(12, "散华×4 vs 炮击×3", [("散华",4)],[("炮击",3)], "L", 50)
test(93, "R31×2 vs 交通亭×2", [("R-31",2)],[("交通亭",2)], "L", 50)
test(103,"妒×4+水遁×3 vs 庞贝×2", [("妒",4),("水遁",3)],[("庞贝",2)], "R", 30)
test(201,"门×3+妒×4 vs 高级武装×3+炮击×4", [("门",3),("妒",4)],[("高级武装",3),("炮击",4)], "L", 30)
test(55, "矿脉×5+炮击×4 vs 沉沙×3+沸血×3", [("矿脉",5),("炮击",4)],[("沉沙",3),("沸血",3)], "L", 30)
test(86, "沙滩车×4+复仇者×6 vs 高级武装×2+合声×5", [("沙滩车",4),("复仇者",6)],[("高级武装",2),("合声",5)], "R", 30)
test(72, "暴走食人花×3 vs 合声×6", [("暴走",3)],[("合声",6)], "R", 30)
test(111,"高塔术师×3 vs 暴走×2", [("高塔",3)],[("暴走",2)], "R", 30)
