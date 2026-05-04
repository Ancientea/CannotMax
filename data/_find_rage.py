with open(r'G:\314\CannotMax-main\simulator\monsters.py', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if 'rage_mode' in line:
            print(f"{i}: {line.rstrip()[:100]}")
