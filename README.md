# Histogram Equalization using CUDA

## Main objective

Histogram Equalization techniques:

1. HE : classical Histogram Equalization (see google or old labwork 4).
1. AHE : Adjusted Histogram Equalization (see [1]).
1. WHE : Weighted Histogram Equalization (see [1]).
1. EIHE : (see [2]).
1. MMSICHE : (see [3]).
1. CLAHE : Contrast Local Adaptive Histogram Equalization (see [4] or google).
1. ACLAHE : Automatic Contrast Local Adaptive Histogram Equalization (see [4]).

### Metrics

- AME which is the sum of the absolute value of the difference per pixel. This is a reduction made from the transform of two images. Easy to do onto Cuda.
- Entropy (see [1]) : Relies on the global histogram of the image.
