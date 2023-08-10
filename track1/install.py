import os

re = './requirements.txt'
with open(re,'r') as f:
    for i,l in enumerate(f.readlines()):
        l = l.strip()
        if not i:
            last = l
            continue
        cmd ='pip3 install '+l+' '+last
        print(cmd)
        os.system(cmd)
        