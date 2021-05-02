import numpy as np
from mpi4py import MPI

def readM1():
    M1 = np.loadtxt("./M1.csv", delimiter=",")
    print("read M1", M1)
    return M1


def readM2():
    M2 = np.loadtxt("./M2.csv", delimiter=",")
    print("read M2", M2)
    return M2


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

M1 = readM1() if comm.rank == 0 else None
M2 = readM2() if comm.rank == 0 else None

# initialize as np arrays
M1_r = np.array([0])
M1_c = np.array([0])
M2_r = np.array([0])
M2_c = np.array([0])
local_n = np.array([0])
dot = None

if rank == 0:

    # test for conformability
    M1_r[0], M1_c[0] = M1.shape
    M2_r[0], M2_c[0] = M2.shape
    if M1_c[0] != M2_r[0]:
        print("vector length mismatch")
        comm.Abort()

    # currently, our program cannot handle sizes that are not evenly divided by
    # the number of processors
    if M1_r[0] % size != 0:
        print("the number of processors must evenly divide n.")
        comm.Abort()

    # length of each process's portion of the original matrix
    local_n = np.array([int(M1_r[0] / size)])

    dot = np.zeros([M1_r[0], M2_c[0]], dtype="float64").reshape(
        (size, local_n[0], M2_c[0])
    )

    local_M2 = M2

# communicate local array size to all processes
comm.Bcast(local_n, root=0)
comm.Bcast(M1_c, root=0)
comm.Bcast(M2_c, root=0)
comm.Bcast(M2_r, root=0)


# initialize as np arrays
local_M1 = np.zeros([local_n[0], M1_c[0]], dtype="float64")
local_M2 = np.zeros([M2_r[0], M2_c[0]], dtype="float64") if rank != 0 else local_M2

# divide up matrix
comm.Scatter(M1, local_M1, root=0)
comm.Bcast(local_M2, root=0)

print(f"{name} {rank} get M1: {local_M1}")
print(f"{name} {rank} get M2: {local_M2}")

# local computation of dot product
local_dot = np.array(local_M1.dot(local_M2))

comm.Gather(local_dot, dot, root=0)

if rank == 0:
    print(np.dot(M1, M2), "computed serially")
    print(dot, "computed in parallel")
