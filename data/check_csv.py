import csv
with open(r'G:\314\CannotMax-main\simulator\arknights.csv', encoding='utf-8') as f:
    rows = list(csv.reader(f))
    row0 = rows[0]
    print(f'Total cols: {len(row0)}')
    
    # Match #12 should be row 11 (0-indexed) if each row is a separate match
    # Or row 22-23 if alternating left/right
    for match_num in [12, 55, 86, 103, 201]:
        r = rows[match_num - 1]  # 1-indexed to 0-indexed
        nz = [(i,float(v)) for i,v in enumerate(r[:69]) if float(v)>0]
        print(f'Match {match_num} (row {match_num-1}): nz={nz} winner={r[-1]}')

open(r'G:\314\CannotMax-main\data\csv_check.txt','w').write('done')
