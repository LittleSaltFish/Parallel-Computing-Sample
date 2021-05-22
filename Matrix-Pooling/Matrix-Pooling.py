import numpy as np
from mpi4py import MPI
import sys
import math

from numpy.lib.type_check import common_type


def readMR():
    """MatrixRaw"""
    MR = np.loadtxt("./MatrixRaw.csv", delimiter=",")
    print("read MatrixRaw\n", MR)
    return MR


def readMK():
    """convolutional kernel"""
    MK = np.loadtxt("./ConvolutionalKernel.csv", delimiter=",")
    print("read ConvolutionalKernel\n", MK)
    return MK


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()


# initialize as np arrays
MR_r = np.array([0])
MR_c = np.array([0])
MK_r = np.array([0])
MK_c = np.array([0])
FinalAns_r = np.array([0])
FinalAns_c = np.array([0])

Flag = int(sys.argv[1])
MK_r[0] = int(sys.argv[2])
MK_c[0] = int(sys.argv[3])

MR = readMR() if rank == 0 else None

MR_r[0] = int(np.size(MR, 0)) if rank == 0 else 0
MR_c[0] = int(np.size(MR, 1)) if rank == 0 else 0
FinalAns_r[0] = int(MR_r[0] / MK_r[0])
FinalAns_c[0] = int(MR_c[0] / MK_c[0])


# communicate local array size to all processes
comm.Bcast(MR_r, root=0)
comm.Bcast(MR_c, root=0)
comm.Bcast(MK_r, root=0)
comm.Bcast(MK_c, root=0)


if rank != 0:
    MR = np.zeros(MR_r[0] * MR_c[0]).reshape(MR_r[0], MR_c[0])

ans = np.zeros(FinalAns_r[0] * FinalAns_c[0]).reshape(FinalAns_r[0], FinalAns_c[0])

comm.Bcast(MR, root=0)

if FinalAns_r[0] * FinalAns_c[0] % size != 0:
    print("NOT DIVIDE")
    comm.Abort()

local_count = int(FinalAns_r[0] * FinalAns_c[0] / size)
local_ans = []
for i in range(local_count):
    start_r = int((i + local_count * rank) / FinalAns_c) * MK_r[0]
    start_c = int((i + local_count * rank) % FinalAns_c) * MK_c[0]
    tmp = MR[start_r : start_r + MK_r[0], start_c : start_c + MK_c[0]]
    # print(f"{i}\n",tmp)
    if Flag == 0:
        local_ans.append(np.max(tmp))
    else:
        local_ans.append(np.mean(tmp))
local_ans = np.array(local_ans)

comm.Gather(local_ans, ans, root=0)

if rank == 0:
    ans = ans.reshape(FinalAns_r[0], FinalAns_c[0])
    ans = ans.astype(int)
    print(ans)
    np.savetxt("ans.csv", ans, delimiter=",", fmt="%.2f")
