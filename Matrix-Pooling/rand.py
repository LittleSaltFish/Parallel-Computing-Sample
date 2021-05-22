import random
import sys
r=int(sys.argv[1])
c=int(sys.argv[2])
with open("./MatrixRaw.csv","w") as f:
    for i in range(r):
        for j in range(c):
            f.write(str(random.randint(0,100)))
            if j !=c-1:
                f.write(",")
        f.write("\n")
