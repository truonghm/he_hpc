import os
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
    parser.add_argument("-i", "--input", type=str, default="images/input2.png")
    parser.add_argument("-l", "--lambda-val", type=float, default=0.5)
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
    
    hist = np.zeros((256,), dtype=np.float32)
    hist_gpu = cuda.to_device(hist)
    kernels.compute_adjusted_hist[grid_size, block_size](hsv_gpu, hist_gpu, args.lambda_val)
    hist = hist_gpu.copy_to_host()
    hist_sum = hist.sum()
    cdf_gpu = cuda.device_array_like(hist_gpu)

    cdf_block_size = 256
    # cdf_grid_size = (hist.size + cdf_block_size - 1) // cdf_block_size
    cdf_grid_size = 1
    kernels.compute_cdf[cdf_grid_size, cdf_block_size](hist_gpu, cdf_gpu, hist_sum)
    # cdf = (np.cumsum(hist).astype('float32') / hist.sum()) * 255
    # cdf = np.round(cdf).astype('uint8')
    # cdf_gpu = cuda.to_device(cdf)
    kernels.equalize_hist[grid_size, block_size](hsv_gpu, cdf_gpu, he_gpu)
    
    hsv_to_rgb[grid_size, block_size](he_gpu, final_output_gpu)
    final_output = final_output_gpu.copy_to_host()

    output_path = OutputPath.AHE.replace(".png", f"_lambda_{args.lambda_val}.png")
    imsave(output_path, final_output, )

    print("AME: ", ame(image, final_output))
    print("Entropy: ", entropy(final_output))
