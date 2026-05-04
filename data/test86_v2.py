import sys,os,json,io
sys.path.insert(0,r'G:\314\CannotMax-main')
os.chdir(r'G:\314\CannotMax-main')
from simulator.battle_field import Battlefield
from simulator.utils import Faction
with open('simulator/monsters.json',encoding='utf-8') as f:mdata=json.load(f)['monsters']
def n(p):
    for m in mdata:
        if p in m['名字']:return m['名字']
    return None
l={n('沙滩车'):4,n('复仇者'):6}
r={n('高级武装'):2,n('合声'):5}
wl=wr=0
for i in range(10):
    bf=Battlefield(mdata);bf.setup_battle(l,r,mdata)
    null=io.StringIO();old=sys.stdout;sys.stdout=null
    while True:
        res=bf.run_one_frame()
        if res is not None:
            if res==Faction.LEFT:wl+=1
            else:wr+=1
            break
    sys.stdout=old
with open(r'G:\314\CannotMax-main\data\test86_v2.txt','w') as f:
    f.write(f'#86 fix: LEFT {wl}/10 RIGHT {wr}/10')
print('done')
