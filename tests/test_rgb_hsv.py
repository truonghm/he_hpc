import argparse
import math
import os
import time

import cv2
import numpy as np
from matplotlib.image import imread, imsave
from numba import cuda

from src.he_kernels import compute_cdf, compute_hist, equalize_hist
from src.metrics import ame, entropy
from src.rgb_hsv import hsv_to_rgb, rgb_to_hsv
from src.utils import Algorithm, OutputPath, read_img

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
    hsv_output = hsv_gpu.copy_to_host()
    hsv_output_cv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    imsave("images/test_hsv_cv2.png", hsv_output_cv)
    imsave("images/test_hsv.png", hsv_output)

    hsv_to_rgb[grid_size, block_size](hsv_gpu, final_output_gpu)
    final_output = final_output_gpu.copy_to_host()

    final_output_cv = cv2.cvtColor(hsv_output_cv, cv2.COLOR_HSV2RGB)


    imsave("images/test_rgb_hsv_cv2.png", final_output_cv)
    imsave("images/test_rgb_hsv.png", final_output, )

    print("AME: ", ame(image, final_output))
    print("Entropy: ", entropy(final_output))

    print("AME between cv2 and cuda: ", ame(final_output_cv, final_output))
    print("AME between cv2 and cuda for hsv: ", ame(hsv_output_cv, hsv_output))
