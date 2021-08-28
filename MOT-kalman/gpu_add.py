import numpy as np
import math
import pycuda.driver as cuda
from numpy import linalg as la
from pycuda import driver, compiler, gpuarray, tools
from pycuda.compiler import SourceModule
from pycuda.autoinit import context
# -- initialize the device
import pycuda.autoinit



# define the (square) matrix sizes
BLK_SIZE = 32
M = 5
N = 32


def matrixAdd(a, b, M, N):
    #C++ double datatype
    a_cpu = a.astype(np.float64)
    b_cpu = b.astype(np.float64)
    
    #to Device
    a_gpu = gpuarray.to_gpu(a_cpu)
    b_gpu = gpuarray.to_gpu(b_cpu)
    
    mod = SourceModule("""
    #define BLOCK_DIM 32

    __global__ void matAdd(double *a, double *b, double *c, int m, int n){

            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int idy = blockIdx.y * blockDim.y + threadIdx.y;

        if (idx < n && idy <m)
                c[idy * n + idx] =  a[idy * n + idx] +  b[idy * n + idx];


    }
    """)

    # define size of blocks and tiles sub-matrix
    # (we assume that the block size is same as tile size)
    TILE_SIZE = 32
    #BLK_SIZE = TILE_SIZE

    # get the kernel code from the template
    # by specifying the constants MATRIX_SIZE and BLOCK_SIZE

    # get the kernel code from the template
    # by specifying the constants MATRIX_SIZE and BLOCK_SIZE

    # compile the kernel code

    #mod = compiler.SourceModule(kernel_code_template)
    # create empty gpu array for the result (C = A * B)

    c_gpu = gpuarray.empty((M, N), np.float64)

    # get the kernel function from the compiled module
    matrixadd = mod.get_function("matAdd")
    #u_M = np.uint32(M)
    #u_N = np.uint32(N)
    #u_K = np.uint32(K)
    # call the kernel on the card


    matrixadd(
        # inputs
        a_gpu, b_gpu,
        # output
        c_gpu,
        # sizes
        np.int32(M), np.int32(N),
        # grid of multiple blocks
        grid=(int(math.ceil(float(M / BLK_SIZE))), int(math.ceil(float(N / BLK_SIZE)))),
        # block of multiple threads
        block=(32, 32, 1),
    )

    context.synchronize()
    return c_gpu.get()

'''
# create two random square matrices
a_cpu = np.random.randn(M, N).astype(np.float64)
b_cpu = np.random.randn(M, N).astype(np.float64)

# compute reference on the CPU to verify GPU computation
c_cpu = np.add(a_cpu, b_cpu)

# transfer host (CPU) memory to device (GPU) memory
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)
c_gpu = matrixAdd(a_gpu, b_gpu, M, N)


# print the results
print("-" * 80)
print("Matrix A (GPU):")
print(a_gpu.get())

print("-" * 80)
print("Matrix B (GPU):")
print(b_gpu.get())

print("-" * 80)
print("Matrix C (GPU):")
print(c_gpu.get())

print("-" * 80)
print("CPU-GPU difference:")
print(c_cpu - c_gpu.get())
'''
