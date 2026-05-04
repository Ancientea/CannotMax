import csv
with open(r"G:\314\CannotMax-main\monster_greenvine.csv", encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    rows = list(reader)
    
# Print header row (should be first row)
print("=== HEADER ===")
for i,v in enumerate(rows[0]):
    print(f"  col[{i}]={v}")

print("\n=== 复仇者(63号) ===")
for row in rows:
    if row[0]=='63':
        for i,v in enumerate(row):
            if v:
                print(f"  col[{i}]={v}")
        break

print("\n=== 复仇者(17号) ===")
for row in rows:
    if row[0]=='17':
        for i,v in enumerate(row):
            if v:
                print(f"  col[{i}]={v}")
        break
