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


def transpose_kernel(inpt, M, N):
    input_cpu = inpt.astype(np.float64)
    input_gpu = gpuarray.to_gpu(input_cpu)
    mod = SourceModule("""
    #define BLOCK_DIM 32
    __global__ void transpose(double *odata, double *idata, int height, int width)
    {
	__shared__ double block[BLOCK_DIM][BLOCK_DIM+1];
	
	// read the matrix tile into shared memory
        // load one element per thread from device memory (idata) and store it
        // in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

        // synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}
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
    output_gpu = gpuarray.empty((N, M), np.float64)

    # get the kernel function from the compiled module
    trans = mod.get_function("transpose")
    
    # call the kernel on the card


    trans(
        # output
        output_gpu,
        # input
        input_gpu,
        # width & height
        np.int32(M), np.int32(N),
        # grid of multiple blocks
        grid=(int(math.ceil(float(N / BLK_SIZE))), int(math.ceil(float(M / BLK_SIZE)))),
        # block of multiple threads
        block=(32, 32, 1),
    )
    context.synchronize()
    return output_gpu.get()

'''
# create two random square matrices
input_cpu = np.random.randn(M, N).astype(np.float64)

# compute reference on the CPU to verify GPU computation
output_cpu = np.transpose(input_cpu)

# transfer host (CPU) memory to device (GPU) memory

input_gpu = gpuarray.to_gpu(input_cpu)

output_gpu = transpose_kernel(input_gpu, M, N)


# print the results
print("-" * 80)
print("input Matrix (GPU):")
print(input_gpu.get())


print("-" * 80)
print("Output Matrix  (GPU):")
print(output_gpu.get())

print("-" * 80)
print("CPU-GPU difference:")
print(output_cpu - output_gpu.get())
'''

