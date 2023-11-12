# import os
# import cv2
from numba import cuda
import numpy as np
import time
import math
from matplotlib.image import imsave, imread
from src.metrics import ame, entropy
from src.utils import read_img, OutputPath, Algorithm
from src.rgb_hsv import rgb_to_hsv, hsv_to_rgb
from src import he_kernels as kernels
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="images/input1.png")
    args = parser.parse_args()
    
    
    input_path = args.input

    image = read_img(input_path)
    print("Image dtype:", image.dtype)
    print("Min value:", image.min())
    print("Max value:", image.max())

    h, w, c = image.shape

    # W = np.random.rand(256) 
    W = np.ones(256, dtype=np.float32)

    for i in range(128):
        W[i] = 2
    for i in range(128, 256):
        W[i] = 0.5

    u = np.full(256, 1.0 / 256) 
    lambda_ = 0.8


    image_gpu = cuda.to_device(image)
    hsv_gpu = cuda.device_array_like(image_gpu)
    he_gpu = cuda.device_array_like(image_gpu)
    final_output_gpu = cuda.device_array_like(image_gpu)
    block_size = (16, 16)
    grid_size_x = math.ceil(h / block_size[0])
    grid_size_y = math.ceil(w / block_size[1])
    grid_size = (grid_size_x, grid_size_y)

    rgb_to_hsv[grid_size, block_size](image_gpu, hsv_gpu)


    # hist = np.zeros((256,), dtype=np.uint32)
    # hist_gpu = cuda.to_device(hist)
    # kernels.compute_weighted_hist[grid_size, block_size](hsv_gpu, hist_gpu, weights_gpu)

    hist_gpu = cuda.device_array(256, dtype=np.float32)
    W_gpu = cuda.to_device(W)
    u_gpu = cuda.to_device(u)
    kernels.compute_weighted_hist[grid_size, block_size](hsv_gpu, hist_gpu, W_gpu, lambda_, u_gpu)


    hist = hist_gpu.copy_to_host()
    hist_sum = hist.sum()
    cdf_gpu = cuda.device_array_like(hist_gpu)

    cdf_block_size = 256
    cdf_grid_size = 1
    kernels.compute_cdf[cdf_grid_size, cdf_block_size](hist_gpu, cdf_gpu, hist_sum)
    kernels.equalize_hist[grid_size, block_size](hsv_gpu, cdf_gpu, he_gpu)
    
    hsv_to_rgb[grid_size, block_size](he_gpu, final_output_gpu)
    final_output = final_output_gpu.copy_to_host()

    imsave(OutputPath.WHE, final_output, )

    print("AME: ", ame(image, final_output))
    print("Entropy: ", entropy(final_output))
