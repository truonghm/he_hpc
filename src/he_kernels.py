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

    cuda.syncthreads()

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
def compute_adjusted_hist(src, hist, lambda_):
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
        weight_original = 1.0 / (1 + lambda_)
        weight_uniform = lambda_ / (1 + lambda_)
        adjusted_hist_val = (
            weight_original * original_hist_val + weight_uniform * uniform_hist_val
        )
        cuda.atomic.add(
            hist, tx, int(adjusted_hist_val * (src.shape[0] * src.shape[1]))
        )
    
    cuda.syncthreads()


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

    cuda.syncthreads()


@cuda.jit
def compute_esi_hist(src, hist_low, hist_high, Xa, Tc):
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
        clipped_value = min(local_hist[tx], Tc)
        if tx < Xa:
            cuda.atomic.add(hist_low, tx, clipped_value)
        else:
            cuda.atomic.add(hist_high, tx - Xa, clipped_value)

    cuda.syncthreads()



@cuda.jit
def compute_esi_cdf(
    hist_low, hist_high, cdf_low, cdf_high, hist_sum_low, hist_sum_high
):
    shist_low = cuda.shared.array(256, dtype=np.float32)
    shist_high = cuda.shared.array(256, dtype=np.uint32)

    tx = cuda.threadIdx.x
    idx = cuda.blockIdx.x * cuda.blockDim.x + tx

    if idx < hist_low.shape[0]:
        shist_low[tx] = hist_low[idx]
    else:
        shist_low[tx] = 0

    if idx < hist_high.shape[0]:
        shist_high[tx] = hist_high[idx]
    else:
        shist_high[tx] = 0

    cuda.syncthreads()

    stride = 1
    while stride < 256:
        index = (tx + 1) * stride * 2 - 1
        if index < 256:
            shist_low[index] += shist_low[index - stride]
        stride *= 2
        cuda.syncthreads()

    stride = 1
    while stride < 256:
        index = (tx + 1) * stride * 2 - 1
        if index < 256:
            shist_high[index] += shist_high[index - stride]
        stride *= 2
        cuda.syncthreads()

    if idx < hist_low.shape[0]:
        cdf_low[idx] = min(255, (int((shist_low[tx] / hist_sum_low) * 255)))
    if idx < hist_high.shape[0]:
        cdf_high[idx] = min(255, (int((shist_high[tx] / hist_sum_high) * 255)))


@cuda.jit
def equalize_esi_hist(src, cdf_low, cdf_high, Xa, dst):
    x, y = cuda.grid(2)
    if x < src.shape[0] and y < src.shape[1]:
        value = src[x, y, 2]
        if value < Xa:
            new_value = cdf_low[value]
        else:
            new_value = cdf_high[value - Xa]
        dst[x, y, 0] = src[x, y, 0]
        dst[x, y, 1] = src[x, y, 1]
        dst[x, y, 2] = min(255, max(0, new_value))
