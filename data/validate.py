import sys,os,json,time,io
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
os.chdir(os.path.join(os.path.dirname(__file__),'..'))
from simulator.battle_field import Battlefield
from simulator.utils import Faction
with open("simulator/monsters.json",encoding='utf-8') as f:mdata=json.load(f)["monsters"]
def f(p):
    for m in mdata:
        if p in m["名字"]:return m["名字"]
    return None
def test(label,left_parts,right_parts,n=10):
    l={};r={}
    for p,c in left_parts:l[f(p)]=c
    for p,c in right_parts:r[f(p)]=c
    wl=wr=0;t0=time.time()
    null=io.StringIO();old=sys.stdout;sys.stdout=null
    for _ in range(n):
        bf=Battlefield(mdata);bf.setup_battle(l,r,mdata)
        while True:
            res=bf.run_one_frame()
            if res is not None:
                if res==Faction.LEFT:wl+=1
                else:wr+=1
                break
    sys.stdout=old
    print(f"  {label}: R{100*wr/n:.0f}% ({wr}W/{wl}L) [{time.time()-t0:.1f}s]")

# 检查庞贝类的实际属性
bf=Battlefield(mdata)
bf.setup_battle({f("庞贝"):1},{f("水遁"):1},mdata)
null=io.StringIO();old=sys.stdout;sys.stdout=null
for _ in range(50):bf.run_one_frame()
sys.stdout=old
for m in bf.alive_monsters:
    if m.is_alive:
        t=type(m).__name__
        print(f"\n{m.name} type={t}")
        print(f"  atk_pow={m.attack_power} get_atk={m.get_attack_power()} mult={m.attack_multiplier}")
        print(f"  atk_type={m.attack_type} range={m.attack_range} speed={m.attack_speed}")
        if hasattr(m,'_self_destruct_timer'):
            print(f"  self_destruct_timer={m._self_destruct_timer:.1f}")
        if hasattr(m,'_speed_boosted'):
            print(f"  speed_boosted={m._speed_boosted}")
        for b in m.behaviors:
            print(f"  behavior: {type(b).__name__}")

print("\n=== 1v3 小规模测试 ===")
test("庞贝1 vs 妒2+水遁1",[("妒",2),("水遁",1)],[("庞贝",1)],10)
test("庞贝2 vs 妒3+水遁2",[("妒",3),("水遁",2)],[("庞贝",2)],10)
