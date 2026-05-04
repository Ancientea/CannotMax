import csv,os
os.chdir(r'G:\314\CannotMax-main')
for path in ['arknights.csv', 'arknights_filtered.csv', 'monster_greenvine.csv','数据_train/arknights备用.csv']:
    try:
        with open(path, encoding='utf-8-sig') as f:
            rows = list(csv.reader(f))
            ncols = max(len(r) for r in rows) if rows else 0
            print(f'{path}: {len(rows)} rows, max {ncols} cols')
            if len(rows) > 11:
                r = rows[11]
                nz = [(i,v) for i,v in enumerate(r) if v and v not in ('0','0.0')]
                print(f'  Row11 nz: {nz[:8]}')
    except Exception as e:
        print(f'{path}: ERROR {e}')
