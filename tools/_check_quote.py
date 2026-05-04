import csv,sys
sys.stdout = open('tools/_quote_out.txt','w',encoding='utf-8')
with open('monster_greenvine.csv','r',encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        if '钳' in row['名称']:
            print(repr(row['名称']))
