import sys,io
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
with open(r'G:\314\CannotMax-main\main.py',encoding='utf-8') as f:
    for i,line in enumerate(f,1):
        if 'def _stop_batch_sim' in line:
            for j in range(i,min(i+12,9999)):
                print(f"{j}: {open(r'G:\314\CannotMax-main\main.py',encoding='utf-8').readlines()[j-1].rstrip()}")
            break
