import sys
import numpy as np

l = int(sys.argv[1])
P_list=np.random.dirichlet(np.ones(l-1),size=1)[0]
print(P_list)

with open("Relation-M.csv","w+") as f:
    for i in range(l):
        li=list(range(l))
        li.remove(i)
        size=np.random.randint(1,l)
        choice=np.random.choice(li,size, replace=False,p=P_list)
        for j in choice:
            f.write(f"{i},{j}\n")
