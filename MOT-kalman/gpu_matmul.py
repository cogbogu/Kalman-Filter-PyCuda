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
K = 33


def matmul(a, b, M, N, K):
    a_cpu = a.astype(np.float64)
    b_cpu = b.astype(np.float64)
    a_gpu = gpuarray.to_gpu(a_cpu)
    b_gpu = gpuarray.to_gpu(b_cpu)
    
    mod = SourceModule("""
    #define BLK_SIZE 32
    __global__ void MatrixMulKernel(double *A, double *B, double *C, int M, int N, int K)
    {
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int row = by * BLK_SIZE + ty;
        int col = bx * BLK_SIZE + tx;

        __shared__ double sa[BLK_SIZE][BLK_SIZE+1];
        __shared__ double sb[BLK_SIZE][BLK_SIZE+1];

        double sum = 0;


        for(int tile = 0; tile < (BLK_SIZE + K - 1)/BLK_SIZE; tile+=1){
                if(tile * BLK_SIZE + tx < K && row < M)
                        sa[ty][tx] = A[row * K + tx + tile*BLK_SIZE];
                else
                        sa[ty][tx] = 0.0;

                if(tile * BLK_SIZE + ty < K && col < N)
                        sb[ty][tx] = B[(tile*BLK_SIZE + ty) * N + col];
                else
                        sb[ty][tx] = 0.0;

                __syncthreads();

                for(int i = 0; i < BLK_SIZE; ++i){
                        sum += sa[ty][i] * sb[i][tx];
                }

                __syncthreads();

        }

        if(row < M && col < N)
                C[row*N +col] = sum;

    }
    """)

    # define size of blocks and tiles sub-matrix
    # (we assume that the block size is same as tile size)
    TILE_SIZE = 32
    #BLK_SIZE = TILE_SIZE
    
    # get the kernel code from the template
    # by specifying the constants MATRIX_SIZE and BLOCK_SIZE

    # compile the kernel code
    
    #mod = compiler.SourceModule(kernel_code_template)
    # create empty gpu array for the result (C = A * B)
    c_gpu = gpuarray.empty((M, N), np.float64)

    # get the kernel function from the compiled module
    matrixmul = mod.get_function("MatrixMulKernel")
    #u_M = np.uint32(M)
    #u_N = np.uint32(N)
    #u_K = np.uint32(K)
    # call the kernel on the card
    

    matrixmul(
        # inputs
        a_gpu, b_gpu,
        # output
        c_gpu,
        # sizes
        np.int32(M), np.int32(N), np.int32(K),
        # grid of multiple blocks
        grid=(int(math.ceil(float(M / BLK_SIZE))), int(math.ceil(float(K / BLK_SIZE)))),
        # block of multiple threads
        block=(32, 32, 1),
    )
    context.synchronize()
    
    return c_gpu.get()

'''
# create two random square matrices
a_cpu = np.random.randn(M, K).astype(np.float64)
b_cpu = np.random.randn(K, N).astype(np.float64)

# compute reference on the CPU to verify GPU computation

c_cpu = np.dot(a_cpu, b_cpu)

# transfer host (CPU) memory to device (GPU) memory
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)

c_gpu = matmul(toDevice(a_cpu), toDevice(b_cpu), M, N, K)


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
