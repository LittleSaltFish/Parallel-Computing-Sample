# example to run: mpiexec -n 4 python3 ./trapParallel_2_noD.py 0.0 1.0 9

import numpy
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#takes in command-line arguments [a,b,n]
a = float(sys.argv[1])
b = float(sys.argv[2])
n = int(sys.argv[3])

#we arbitrarily define a function to integrate
def f(x):
    return x*x

#this is the serial version of the trapezoidal rule
#parallelization occurs by dividing the range among processes
def integrateRange(a, b, n):
    integral = -(f(a) + f(b))/2.0
    # n+1 endpoints, but n trapazoids
    for x in numpy.linspace(a,b,n+1):
            integral = integral + f(x)
    integral = integral* (b-a)/n
    return integral


comm.Barrier()
start = MPI.Wtime()

#h is the step size. n is the total number of trapezoids
h = (b-a)/n
#local_n is the number of trapezoids each process will calculate
#note that size must divide n

local_n = int(n/size)
left_n = n%size

if rank in range(left_n):
    local_a = a + rank*(local_n+1)*h
    local_b = local_a + (local_n+1)*h

else:
#we calculate the interval that each process handles
#local_a is the starting point and local_b is the endpoint
    local_a = a + (rank-left_n)*local_n*h+left_n*(local_n+1)*h
    local_b = local_a + local_n*h

#initializing variables. mpi4py requires that we pass numpy objects.
integral = numpy.zeros(1,dtype='float64')
total = numpy.zeros(1,dtype='float64')

# perform local computation. Each process integrates its own interval
integral[0] = integrateRange(local_a, local_b, local_n)

# communication
# root node receives results with a collective "reduce"
comm.Reduce(integral, total, op=MPI.SUM, root=0)


comm.Barrier()
end = MPI.Wtime()
time=end-start

# root process print(s results)
if comm.rank == 0:
    print( "With n =", n, "trapezoids, our estimate of the integral from", a, "to", b, "is", total,"use",time)

