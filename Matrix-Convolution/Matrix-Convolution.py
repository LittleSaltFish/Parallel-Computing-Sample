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


def PaddingMR(Stride, Dilated, MR_r, MR_c, MK_r, MK_c):
    MNew_r = Stride * (MR_r - 1) + MK_r
    MNew_c = Stride * (MR_c - 1) + MK_c
    MNew_r += Dilated
    MNew_c += Dilated
    MRNew = np.zeros(MNew_r * MNew_c).reshape(MNew_r, MNew_c)

    start_r = int((MNew_r - MR_r) / 2)
    end_r = start_r + MR_r
    start_c = int((MNew_c - MR_c) / 2)
    end_c = start_c + MR_c

    MRNew[start_r:end_r, start_c:end_c] = MR
    print(MRNew)
    return MRNew


# MR=readMR()
# MK=readMK()
# size_MK=np.size(MK,0)
# PaddingMR(2,MR,size_MK)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

Stride = int(sys.argv[1])
Padding = int(sys.argv[2])
Dilated = int(sys.argv[3])

# initialize as np arrays
MR_r_old = np.array([0])
MR_c_old = np.array([0])
MR_r = np.array([0])
MR_c = np.array([0])
MK_r = np.array([0])
MK_c = np.array([0])
FinalAns_r = np.array([0])
FinalAns_c = np.array([0])

MR = readMR() if rank == 0 else None
MK = readMK() if rank == 0 else None

MR_r_old[0] = np.size(MR, 0) if rank == 0 else 0
MR_c_old[0] = np.size(MR, 1) if rank == 0 else 0
MK_r[0] = np.size(MK, 0) if rank == 0 else 0
MK_c[0] = np.size(MK, 1) if rank == 0 else 0
if Padding == 1 and rank == 0:
    MR = PaddingMR(Stride, Dilated, MR_r_old[0], MR_c_old[0], MK_r[0], MK_c[0])
    FinalAns_r[0] = MR_r_old[0]
    FinalAns_c[0] = MR_c_old[0]
elif Padding == 0 and rank == 0:
    FinalAns_r[0] = (MR_r_old[0] - MK_r[0] + 1 - (MK_r - 1) * (Dilated - 1)) / Stride
    FinalAns_c[0] = (MR_c_old[0] - MK_c[0] + 1 - (MK_c - 1) * (Dilated - 1)) / Stride

MR_r[0] = np.size(MR, 0) if rank == 0 else 0
MR_c[0] = np.size(MR, 1) if rank == 0 else 0

# communicate local array size to all processes
comm.Bcast(MR_r, root=0)
comm.Bcast(MR_c, root=0)
comm.Bcast(MK_r, root=0)
comm.Bcast(MK_c, root=0)
comm.Bcast(FinalAns_r, root=0)
comm.Bcast(FinalAns_c, root=0)


if rank != 0:
    MR = np.zeros(MR_r[0] * MR_c[0]).reshape(MR_r[0], MR_c[0])
    MK = np.zeros(MK_r[0] * MK_c[0]).reshape(MK_r[0], MK_c[0])

ans = np.zeros(FinalAns_r[0] * FinalAns_c[0]).reshape(FinalAns_r[0], FinalAns_c[0])

comm.Bcast(MR, root=0)
comm.Bcast(MK, root=0)

if FinalAns_r[0] * FinalAns_c[0] % size != 0:
    print("NOT DIVIDE")
    comm.Abort()

local_count = int(FinalAns_r[0] * FinalAns_c[0] / size)
local_ans = []
for i in range(local_count):
    start_r = int((i + local_count * rank) / FinalAns_c) * Stride
    start_c = int((i + local_count * rank) % FinalAns_c) * Stride
    tmp = []
    for j in range(MK_r[0]):
        for k in range(MK_c[0]):
            tmp.append(MR[start_r + j * Dilated, start_c + k * Dilated])
    tmp = np.array(tmp)
    tmp = tmp.reshape(MK_r[0], MK_c[0])
    # print(f"{rank}-tmp:\n",tmp)
    local_ans.append((tmp * MK).sum())
    # print((tmp * MK).sum())
local_ans = np.array(local_ans)

comm.Gather(local_ans, ans, root=0)

if rank == 0:
    ans = ans.reshape(FinalAns_r[0], FinalAns_c[0])
    ans = ans.astype(int)
    print(ans)
    np.savetxt("ans.csv", ans, delimiter=",", fmt="%.2f")
