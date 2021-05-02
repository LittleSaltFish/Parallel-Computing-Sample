import matplotlib.pyplot as plt
import sys
from mpi4py import MPI

sys.path.append("/home/mpi/code/mpi4py/trapParallel")

from trapParallel_2_noD_f import PnoD
from trapParallel_2_mD_f import PmD

res_noD=[]
time_noD=[]
comm = MPI.COMM_WORLD


for i in range(20,10000,20):
    ret=PnoD(0.0,1.0,i)
    res_noD.append(ret[0])
    time_noD.append(ret[1])
print(res_noD)
print(time_noD)

plt.figure(1)
x1=range(20,10000,20)
plt.plot(x1,res_noD,label='result')
plt.xlabel('result')
plt.ylabel('value')
plt.title('result-value')
plt.legend()
# plt.show()

if comm.rank == 0:
    plt.savefig('plot-value-noD.png', bbox_inches='tight')

plt.figure(2)
x2=range(20,10000,20)
plt.plot(x2,time_noD,label='time_noD')
plt.xlabel('time_noD')
plt.ylabel('value')
plt.title('result-time')
plt.legend()
# plt.show()

if comm.rank == 0:
    plt.savefig('plot-time-noD.png', bbox_inches='tight')


res_mD=[]
time_mD=[]
for i in range(20,10000,20):
    print(i)
    ret=PmD(0.0,1.0,i)
    res_mD.append(ret[0])
    time_mD.append(ret[1])
print(res_mD)
print(time_mD)

plt.figure(3)
x1=range(20,10000,20)
plt.plot(x1,res_mD,label='result')
plt.xlabel('result')
plt.ylabel('value')
plt.title('result-value')
plt.legend()
# plt.show()
if comm.rank == 0:
    plt.savefig('plot-value-mD.png', bbox_inches='tight')

plt.figure(4)
x2=range(20,10000,20)
plt.plot(x2,time_mD,label='time_mD')
plt.xlabel('time_mD')
plt.ylabel('value')
plt.title('result-time')
plt.legend()
# plt.show()
if comm.rank == 0:
    plt.savefig('plot-time-mD.png', bbox_inches='tight')
