import csv
with open(r"G:\314\CannotMax-main\monster_greenvine.csv", encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    # Find header
    for row in reader:
        if '庞贝' in str(row):
            print("庞贝 Greenvine 原始行:")
            for i,v in enumerate(row):
                print(f"  col[{i}]={v}")
            break
