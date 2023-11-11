import numba
import numpy as np
from numba import cuda, float32


@cuda.jit
def compute_hist(src, hist):
    local_hist = cuda.shared.array(256, dtype=np.uint32)

    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    block_x = bx * bw + tx
    block_y = by * bh + ty

    if tx < 256:
        local_hist[tx] = 0
    cuda.syncthreads()

    if block_x < src.shape[0] and block_y < src.shape[1]:
        value = src[block_x, block_y, 2]
        cuda.atomic.add(local_hist, value, 1)
    cuda.syncthreads()

    if tx < 256:
        cuda.atomic.add(hist, tx, local_hist[tx])


@cuda.jit
def compute_cdf(hist, cdf, hist_sum):
    shist = cuda.shared.array(256, dtype=np.uint32)
    scdf = cuda.shared.array(256, dtype=np.float32)

    tx = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + tx

    if idx < hist.shape[0]:
        shist[tx] = hist[idx]
    else:
        shist[tx] = 0
    cuda.syncthreads()

    if tx == 0:
        scdf[tx] = shist[tx]
    for i in range(1, 256):
        cuda.syncthreads()
        if tx >= i:
            scdf[tx] = scdf[tx] + shist[tx - i]
    cuda.syncthreads()

    if idx < hist.shape[0]:
        cdf[idx] = min(255, (round((scdf[tx] / hist_sum) * 255)))


@cuda.jit
def equalize_hist(src, cdf, dst):
    x, y = cuda.grid(2)
    if x < src.shape[0] and y < src.shape[1]:
        value = src[x, y, 2]
        new_value = cdf[value]
        dst[x, y, 0] = src[x, y, 0]
        dst[x, y, 1] = src[x, y, 1]
        dst[x, y, 2] = min(255, max(0, new_value))
