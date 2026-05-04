import csv
with open(r"G:\314\CannotMax-main\monster_greenvine.csv", encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    for row in reader:
        name = row[1] if len(row)>1 else ''
        if '复仇' in name or '沙滩' in name or '合声' in name or '全封闭' in name:
            print(f"name={name}")
            for i,v in enumerate(row):
                if v:
                    print(f"  col[{i}]={v}")
            print()
