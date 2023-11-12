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


def compute_thresholds(src):
    L = 256
    h = np.histogram(src, bins=256)[0]
    exposure = np.sum(h * np.arange(256)) / np.sum(h) / L
    print(exposure)
    Xa = int(L * (1 - exposure))
    Tc = int(np.sum(h) / L)
    print(Xa, Tc)
    print(np.mean(h))
    print(np.max(h))
    return Xa, Tc

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

    image_gpu = cuda.to_device(image)
    hsv_gpu = cuda.device_array_like(image_gpu)
    he_gpu = cuda.device_array_like(image_gpu)
    final_output_gpu = cuda.device_array_like(image_gpu)
    block_size = (16, 16)
    grid_size_x = math.ceil(h / block_size[0])
    grid_size_y = math.ceil(w / block_size[1])
    grid_size = (grid_size_x, grid_size_y)

    rgb_to_hsv[grid_size, block_size](image_gpu, hsv_gpu)
    hist_low = np.zeros((256,), dtype=np.uint32)
    hist_high = np.zeros((256,), dtype=np.uint32)
    hist_low_gpu = cuda.to_device(hist_low)
    hist_high_gpu = cuda.to_device(hist_high)

    hsv_output = hsv_gpu.copy_to_host()
    Xa, Tc = compute_thresholds(hsv_output)

    kernels.compute_esi_hist[grid_size, block_size](hsv_gpu, hist_low_gpu, hist_high_gpu, Xa, Tc)


    hist_low = hist_low_gpu.copy_to_host()
    hist_high = hist_high_gpu.copy_to_host()
    hist_sum_low = hist_low.sum()
    hist_sum_high = hist_high.sum()
    cdf_low_gpu = cuda.device_array_like(hist_low_gpu)
    cdf_high_gpu = cuda.device_array_like(hist_high_gpu)

    cdf_block_size = 256
    cdf_grid_size = 1
    kernels.compute_esi_cdf[cdf_grid_size, cdf_block_size](hist_low_gpu, hist_high_gpu, cdf_low_gpu, cdf_high_gpu, hist_sum_low, hist_sum_high)

    kernels.equalize_esi_hist[grid_size, block_size](hsv_gpu, cdf_low_gpu, cdf_high_gpu, Xa, he_gpu)

    
    hsv_to_rgb[grid_size, block_size](he_gpu, final_output_gpu)
    final_output = final_output_gpu.copy_to_host()

    imsave(OutputPath.EIHE, final_output, )

    print("AME: ", ame(image, final_output))
    print("Entropy: ", entropy(final_output))
