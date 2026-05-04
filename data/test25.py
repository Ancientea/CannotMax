import sys,os,json,io,time
sys.path.insert(0,r'G:\314\CannotMax-main')
os.chdir(r'G:\314\CannotMax-main')
from simulator.battle_field import Battlefield
from simulator.utils import Faction
with open('simulator/monsters.json',encoding='utf-8') as f:mdata=json.load(f)['monsters']
def n(p):
    for m in mdata:
        if p in m['名字']:return m['名字']
    return None

cases=[
 ('#86',{n('沙滩车'):4,n('复仇者'):6},{n('高级武装'):2,n('合声'):5},'R',5),
 ('#55',{n('矿脉'):5,n('炮击'):4},{n('沉沙'):3,n('沸血'):3},'L',5),
 ('#201',{n('门'):3,n('妒'):4},{n('高级武装'):3,n('炮击'):4},'L',5),
 ('#103',{n('妒'):4,n('水遁'):3},{n('庞贝'):2},'R',5),
 ('#12',{n('散华'):4},{n('炮击'):3},'L',5),
]
t0=time.time()
out=[]
for tag,l,r,exp,n in cases:
    wl=wr=0;null=io.StringIO();old=sys.stdout;sys.stdout=null
    for _ in range(n):
        bf=Battlefield(mdata);bf.setup_battle(l,r,mdata)
        while True:
            res=bf.run_one_frame()
            if res is not None:
                if res==Faction.LEFT:wl+=1
                else:wr+=1
                break
    sys.stdout=old
    rate=100*wl/n
    ok='OK' if (exp=='L' and rate>=50) or (exp=='R' and rate<=50) else 'MIS'
    out.append(f'{tag} {ok} 预期{exp}胜 左{rate:.0f}% L{wl}/R{wr}')
out.append(f'total {time.time()-t0:.0f}s')
with open(r'G:\314\CannotMax-main\data\test25.txt','w',encoding='utf-8') as f:
    f.write('\n'.join(out))
print('done')
