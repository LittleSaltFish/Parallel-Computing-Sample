import numpy as np
from mpi4py import MPI
from numpy.lib.utils import source
from mpi4py.MPI import ANY_SOURCE


def readM1():
    M1 = np.loadtxt("./M1.csv", delimiter=",")
    # print("read M1:\n", M1)
    return M1


def readM2():
    M2 = np.loadtxt("./M2.csv", delimiter=",")
    # print("read M2:\n", M2)
    return M2


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

# # MPI.BSEND_OVERHEAD gives the extra overhead in buffered mode
# BUFSISE = 2000 + MPI.BSEND_OVERHEAD
# buf = bytearray(BUFSISE)

# # Attach a user-provided buffer for sending in buffered mode
# MPI.Attach_buffer(buf)

M1 = readM1() if comm.rank == 0 else None
M2 = readM2() if comm.rank == 0 else None

# initialize as np arrays
# use numpy array to increase the effectiveness of Bcast/Send
M1_r = np.array([0])
M1_c = np.array([0])
M2_r = np.array([0])
M2_c = np.array([0])
local_n1 = np.array([0])
local_n2 = np.array([0])
dot = None

if rank == 0:

    # test for conformability
    M1_r[0], M1_c[0] = M1.shape
    M2_r[0], M2_c[0] = M2.shape
    if M1_c[0] != M2_r[0]:
        print("vector length mismatch")
        comm.Abort()

    size_q = int((size - 1) ** 0.5)
    # currently, our program cannot handle sizes that are not evenly divided by
    # the square number of processors
    if M1_r[0] % size_q != 0 or M2_c[0] % size_q != 0:
        print("the number of processors must evenly divide n.")
        comm.Abort()
    if size_q ** 2 != size - 1:
        print("the number of processors must be quarted.")
        comm.Abort()

    # length of each process's portion of the original vector
    local_n1 = np.array([int(M1_r[0] / size_q)])
    local_n2 = np.array([int(M2_c[0] / size_q)])

    # final answer
    dot = np.zeros([M1_r[0], M2_c[0]], dtype="float64").reshape(
        size_q, size_q, local_n1[0], local_n2[0]
    )


# communicate local array size to all processes
comm.Bcast(local_n1, root=0)
comm.Bcast(local_n2, root=0)
comm.Bcast(M1_c, root=0)
comm.Bcast(M2_r, root=0)


# initialize as np arrays
local_M1 = np.zeros([local_n1[0], M1_c[0]], dtype="float64")
local_M2 = np.zeros([M2_r[0], local_n2[0]], dtype="float64")

# divide up matrix
if rank == 0:
    dcount=1
    for i in range(size_q):
        for j in range(size_q):
            st_1 = int(i * local_n1[0])
            ed_1 = int((i + 1) * local_n1[0])
            st_2 = int(j * local_n2[0])
            ed_2 = int((j + 1) * local_n2[0])
            comm.Send(np.ascontiguousarray(M1[st_1:ed_1, :]), dest=dcount, tag=1)
            comm.Send(np.ascontiguousarray(M2[:, st_2:ed_2]), dest=dcount, tag=2)
            # data stored discontinuity (discontinual row/col of matrix) cannot be Bcast or Send
            # use np.ascontiguousarray() to transfer it into continuity stored data
            dcount+=1
    rcount=0
    for i in range(size_q):
        for j in range(size_q):
            rcount+=1
            comm.Recv(dot[i][j], source=rcount)
else:
    comm.Recv(local_M1, source=0, tag=1)
    comm.Recv(local_M2, source=0, tag=2)
    print(f"{name} {rank} get M1: \n{local_M1}")
    print(f"{name} {rank} get M2: \n{local_M2}")

# local computation of dot product
# to test the parallel answer
if rank != 0:
    local_dot = np.array(local_M1.dot(local_M2))
    comm.Send(local_dot, dest=0,tag=rank)

if rank == 0:
    print(np.dot(M1, M2), "computed serially")
    print(dot, "computed in parallel")
