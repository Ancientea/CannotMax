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
print(f"Left: {l}")
print(f"Right: {r}")

bf=Battlefield(mdata)
bf.setup_battle(l,r,mdata)

frame=0
last_log=0
while True:
    res=bf.run_one_frame()
    frame+=1
    if frame%50==0:
        alive_l=sum(1 for m in bf.alive_monsters if m.faction==Faction.LEFT and m.is_alive)
        alive_r=sum(1 for m in bf.alive_monsters if m.faction==Faction.RIGHT and m.is_alive)
        # Show HP of庞贝
        pompeii=[m for m in bf.alive_monsters if m.faction==Faction.RIGHT and m.is_alive]
        hp_str=" ".join([f"庞贝{p.id}:HP={p.health:.0f}/{p.max_health:.0f}" for p in pompeii])
        print(f"  frame {frame}: L_alive={alive_l} R_alive={alive_r} {hp_str}")
    if res is not None:
        print(f"\nWinner: {'Left' if res==Faction.LEFT else 'Right'} in {frame} frames")
        # Show damages
        for m in bf.monsters:
            dead="" if m.is_alive else "[DEAD]"
            print(f"  {m.name}{m.id} {dead}: DMG_DEALT={m.damage_dealt:.0f} DMG_TAKEN={m.damage_taken:.0f} KILLS={m.kill_count}")
        break
