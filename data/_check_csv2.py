with open(r'G:\314\CannotMax-main\monster_greenvine.csv',encoding='utf-8-sig') as f:
    for line in f:
        if '水遁' in line or '妒' in line or '睡眠不足' in line:
            print(line.rstrip())
