import numpy as np
from mpi4py import MPI
import random


def read():
    li = np.loadtxt("./input.txt", encoding="utf-8", delimiter=",")
    print("read\n", li)
    return np.ndarray.tolist(li)


def FS(li):
    if li!=[]:
        x = random.choice(li)
        l1 = []
        l2 = []
        for i in li:
            if i >= x:
                l2.append(i)
            else:
                l1.append(i)
        return np.array(l1), np.array(l2)
    else:
        return np.array([]), np.array([])


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

size_li = np.array([0])
local_li = None
flag = np.array([True])

# 读文件
if rank == 0:
    li = read()
    size_li[0] = len(li)
    li = [li]


# comm.Barrier()
# start = MPI.Wtime()

# 每个进程创建本地空间 
comm.Bcast(size_li, root=0)
local_li = np.zeros(size_li[0])
local_0 = np.zeros(1)
local_1 = np.zeros(size_li[0])
local_2 = np.zeros(size_li[0])

if rank == 0:
    while flag[0]:
        size_1 = np.array([0])
        size_2 = np.array([0])
        tmp = []
        n_rank = 1
        if len(li)%(size-1)!=0:
            li.extend([[] for _ in range((size-1)-len(li)%(size-1))])
        for i in li:
            local_0[0] = len(i)
            comm.Send(local_0, dest=n_rank, tag=0)
            comm.Send(np.array(i), dest=n_rank, tag=1)
            print(f"{name} {rank} Send to {n_rank}: \n{i}")
            n_rank += 1
            if n_rank == size:
                for j in range(1, size):
                    comm.Recv(local_1, source=j, tag=1)
                    comm.Recv(local_2, source=j, tag=2)
                    comm.Recv(size_1, source=j, tag=3)
                    comm.Recv(size_2, source=j, tag=4)
                    # print(f"{name} {rank} Recv: \n{local_1}")
                    # print(f"{name} {rank} Recv: \n{local_2}")
                    if size_1[0] != 0:
                        tmp.append(np.ndarray.tolist(local_1[: size_1[0]]))
                    if size_2[0] != 0:
                        tmp.append(np.ndarray.tolist(local_2[: size_2[0]]))
                    print(f"tmp from {j}:{tmp[-2:]}")
                print(f"tmp:{tmp}")
                n_rank = 1
        li = tmp

        flag[0] = False
        print("li:", li)
        for i in li:
            if len(i) > 1:
                flag[0] = True
        comm.Bcast(flag, root=0)
        print(flag[0])
        print("-" * 100)

else:
    while flag[0]:
        comm.Recv(local_0, source=0, tag=0)
        # print("get len:", local_0[0])

        comm.Recv(local_li, source=0, tag=1)
        local_tmp = np.ndarray.tolist(local_li)[: int(local_0[0])]
        # print(f"{name} {rank} Recv: \n{local_tmp}")

        local_1, local_2 = FS(local_tmp)
        # print("local_1", local_1)
        # print("local_2", local_2)

        comm.Send(np.array([len(local_1)]), dest=0, tag=3)
        comm.Send(np.array([len(local_2)]), dest=0, tag=4)

        comm.Send(local_1, dest=0, tag=1)
        comm.Send(local_2, dest=0, tag=2)
        # print(f"{name} {rank} Send: \n{local_1}")
        # print(f"{name} {rank} Send: \n{local_2}")


if rank == 0:
    # end = MPI.Wtime()
    # print(sorted(read()))
    print(li)
    # print(end-start)
