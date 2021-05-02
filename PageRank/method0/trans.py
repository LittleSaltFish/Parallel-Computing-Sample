import sys
import numpy as np

#read from command line
node_n = int(sys.argv[1])    #size of nodes
out_p = float(sys.argv[2])   #mod_value to avoid spider trap

M=np.zeros([node_n,node_n], dtype="float64")

M_raw=np.loadtxt("./Relation-M.csv",delimiter=",",dtype="int")
print(f"read:\n{M_raw}")

for i in M_raw:
    M[i[1]][i[0]]=1.0

print(f"transfer to Adjacency-Matrix:\n{M}")
np.savetxt('Adjacency-M.csv',M,fmt="%d",delimiter=",")

sumM=np.sum(M,axis=0)
print(sumM)

for i in range(node_n):
    for j in range(node_n):
        M[i][j]=1/sumM[j] if M[i][j]==1 else 0

print(f"transfer to Value-Matrix:\n{M}")
np.savetxt('Value-M.csv',M,fmt="%f",delimiter=",")

for i in range(node_n):
    for j in range(node_n):
        M[i][j]=(1-out_p)*M[i][j]+out_p*(1/node_n)

print(f"transfer to Power-iterations-Matrix:\n{M}")
np.savetxt('Power-iterations-M.csv',M,fmt="%f",delimiter=",")

