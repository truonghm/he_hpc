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

    tx = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + tx

    if idx < hist.shape[0]:
        shist[tx] = hist[idx]
    else:
        shist[tx] = 0
    cuda.syncthreads()

    stride = 1
    while stride < 256:
        index = (tx + 1) * stride * 2 - 1
        if index < 256:
            shist[index] += shist[index - stride]
        stride *= 2
        cuda.syncthreads()

    stride = 128
    while stride > 0:
        index = (tx + 1) * stride * 2 - 1
        if index + stride < 256:
            shist[index + stride] += shist[index]
        stride //= 2
        cuda.syncthreads()

    if idx < hist.shape[0]:
        cdf[idx] = min(255, (round((shist[tx] / hist_sum) * 255)))



@cuda.jit
def equalize_hist(src, cdf, dst):
    x, y = cuda.grid(2)
    if x < src.shape[0] and y < src.shape[1]:
        value = src[x, y, 2]
        new_value = cdf[value]
        dst[x, y, 0] = src[x, y, 0]
        dst[x, y, 1] = src[x, y, 1]
        dst[x, y, 2] = min(255, max(0, new_value))

@cuda.jit
def compute_adjusted_hist(src, hist, lambda_val):
    local_hist = cuda.shared.array(256, dtype=np.uint32)

    x, y = cuda.grid(2)
    tx = cuda.threadIdx.x

    if tx < 256:
        local_hist[tx] = 0
    cuda.syncthreads()

    if x < src.shape[0] and y < src.shape[1]:
        value = src[x, y, 2] 
        cuda.atomic.add(local_hist, value, 1)
    cuda.syncthreads()

    if tx < 256:
        original_hist_val = local_hist[tx] / float(src.shape[0] * src.shape[1])
        uniform_hist_val = 1.0 / 256.0
        weight_original = 1.0 / (1 + lambda_val)
        weight_uniform = lambda_val / (1 + lambda_val)
        adjusted_hist_val = weight_original * original_hist_val + weight_uniform * uniform_hist_val
        cuda.atomic.add(hist, tx, int(adjusted_hist_val * (src.shape[0] * src.shape[1])))

@cuda.jit
def compute_weighted_hist(src, hist, W, lambda_, u):
    local_hist = cuda.shared.array(256, dtype=np.float32)

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

    if x < src.shape[0] and y < src.shape[1]:
        pixel_value = src[x, y, 2]
        weight = W[block_x * block_y]
        cuda.atomic.add(local_hist, pixel_value, weight)
    cuda.syncthreads()

    if tx < 256:
        weighted_hist_value = (1 - lambda_) * local_hist[tx] + lambda_ * u[tx]
        cuda.atomic.add(hist, tx, weighted_hist_value)