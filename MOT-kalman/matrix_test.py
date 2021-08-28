
import numpy as np
from gpu_FMT import *
import numpy as np
import math
import pycuda.driver as cuda
from numpy import linalg as la
from pycuda import driver, compiler, gpuarray, tools
from pycuda.compiler import SourceModule
from pycuda.autoinit import context
# -- initialize the device
import pycuda.autoinit


A = np.matrix([[1, 3, 0, 0], [0, 1, 0, 0],
                            [0, 0, 1, 3],  [0, 0, 0, 1]])
print(A*A.T)
res = FMT(A, A, A.shape[0], A.shape[1], A.shape[1])

print(res)
# Number of rows and columns of the said matrix

print(A.shape)
print(A.shape[0])

