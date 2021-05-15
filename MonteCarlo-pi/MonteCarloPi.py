import numpy as np
import sys
from mpi4py import MPI
import random


accuracy = float(sys.argv[1])
# 精度，即测试次数

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

count = np.zeros(1)
ans = np.zeros(size)

for i in range(int(accuracy / size)):
    x = random.uniform(0, 2)
    y = random.uniform(0, 2)
    if (x - 1) ** 2 + (y - 1) ** 2 <= 1:
        count[0] += 1

comm.Gather(count, ans, root=0)

if rank == 0:
    print(ans)
    print(4*sum(ans)/accuracy)
