import sys,os,json,time,io,traceback
sys.path.insert(0,r'G:\314\CannotMax-main')
os.chdir(r'G:\314\CannotMax-main')
from simulator.battle_field import Battlefield
from simulator.utils import Faction

with open("simulator/monsters.json",encoding='utf-8') as f:
    mdata = json.load(f)["monsters"]

def name_lookup(partial):
    for m in mdata:
        if partial in m["名字"]:
            return m["名字"]
    return None

def test(label, left_parts, right_parts, expected, rounds=30):
    l={name_lookup(p):c for p,c in left_parts}
    r={name_lookup(p):c for p,c in right_parts}
    wl=wr=0;t0=time.time()
    null=io.StringIO();old=sys.stdout;sys.stdout=null
    errors=0
    for _ in range(rounds):
        try:
            bf=Battlefield(mdata);bf.setup_battle(l,r,mdata)
            while True:
                res=bf.run_one_frame()
                if res is not None:
                    if res==Faction.LEFT:wl+=1
                    else:wr+=1
                    break
        except Exception as e:
            errors+=1
            if errors<=1:traceback.print_exc()
    sys.stdout=old
    rate=100*wl/rounds
    match="OK" if (expected=="L" and rate>50) or (expected=="R" and rate<50) else "MIS"
    with open(r'G:\314\CannotMax-main\data\test_result2.txt','a',encoding='utf-8') as f:
        f.write(f"{match} {label:<55} {expected} 左{rate:.0f}% L{wl}/R{wr} [{time.time()-t0:.1f}s]" + (f" 崩{errors}" if errors else "") + "\n")

with open(r'G:\314\CannotMax-main\data\test_result2.txt','w',encoding='utf-8') as f:
    f.write("Quick test after fix\n")

test("#55 矿脉×5+炮击×4 vs 沉沙×3+沸血×3", [("矿脉",5),("炮击",4)],[("沉沙",3),("沸血",3)], "L", 15)
test("#103 妒×4+水遁×3 vs 庞贝×2", [("妒",4),("水遁",3)],[("庞贝",2)], "R", 15)
