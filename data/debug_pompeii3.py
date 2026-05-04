import sys,os,json,time,io
sys.path.insert(0,r'G:\314\CannotMax-main')
os.chdir(r'G:\314\CannotMax-main')
from simulator.battle_field import Battlefield
from simulator.utils import Faction

with open("simulator/monsters.json",encoding='utf-8') as f:
    mdata = json.load(f)["monsters"]

def n(part):
    for m in mdata:
        if part in m["名字"]:
            return m["名字"]
    return None

l={n("妒"):4, n("水遁"):3}
r={n("庞贝"):2}

out=[]
bf=Battlefield(mdata)
bf.setup_battle(l,r,mdata)
frame=0
while True:
    res=bf.run_one_frame()
    frame+=1
    if res is not None:
        out.append(f"Winner={'Left' if res==Faction.LEFT else 'Right'} frames={frame}")
        for m in sorted(bf.monsters,key=lambda x:(x.faction.value,x.name)):
            hp=f"HP={m.health:.0f}/{m.max_health:.0f}" if m.is_alive else f"HP=0/{m.max_health:.0f}"
            dmg=m.damage_dealt if hasattr(m,'damage_dealt') else 0
            taken=m.damage_taken if hasattr(m,'damage_taken') else 0
            kills=m.kill_count if hasattr(m,'kill_count') else 0
            out.append(f"  {m.faction.name} {m.name}{m.id} {hp} DMG={dmg:.0f} TAKEN={taken:.0f} KILL={kills}")
        break

with open(r'G:\314\CannotMax-main\data\debug_out.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(out))
print("Done")
