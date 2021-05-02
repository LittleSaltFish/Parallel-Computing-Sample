import numpy as np
from mpi4py import MPI
import sys


def readM():
    M = np.loadtxt("./M.csv", delimiter=",")
    print("read", M)
    return M


def readV():
    V = np.loadtxt("./V.csv", delimiter=",")
    print("read", V)
    return V


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

# arbitrary example vectors, generated to be evenly divided by the number of
# processes for convenience

M = readM() if comm.rank == 0 else None
V = readV() if comm.rank == 0 else None

# initialize as np arrays
local_n = np.array([0])
M_r = np.array([0])
M_c = np.array([0])
V_r = np.array([0])

if rank == 0:
    local_V=V
    M_r[0], M_c[0] = M.shape
    V_r[0] = V.shape[0]

    if V_r[0] != M_c[0]:
        print("vector length mismatch")
        comm.Abort()

    # test for conformability
    # currently, our program cannot handle sizes that are not evenly divided by
    # the number of processors
    if M_r[0] % size != 0:
        print("the number of processors must evenly divide n.")
        comm.Abort()

    # length of each process's portion of the original vector
    local_n = np.array([int(M_r[0] / size)])

dot = np.zeros([size, local_n[0]], dtype="float64") if rank == 0 else None

# communicate local array size to all processes
comm.Bcast(local_n, root=0)
comm.Bcast(M_c, root=0)
comm.Bcast(V_r, root=0)

# initialize as np arrays
local_M = np.zeros([local_n[0], M_c[0]], dtype="float64")
local_V = np.zeros(V_r, dtype="float64") if rank!=0 else local_V

# divide up vectors
comm.Scatter(M, local_M, root=0)
comm.Bcast(local_V, root=0)

print(name, rank, "get", local_M)
print(name, rank, "get", local_V)

# local computation of dot product
local_dot = np.array(local_M.dot(local_V))

comm.Gather(local_dot, dot, root=0)

if rank == 0:
    print(np.dot(M, V), "computed serially")
    print(dot.reshape(1, -1), "computed serially")
