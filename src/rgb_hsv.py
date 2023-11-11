import numba
from numba import cuda, float32


@cuda.jit
def rgb_to_hsv(src, dst):
    x, y = cuda.grid(2)
    if x < src.shape[0] and y < src.shape[1]:
        r, g, b = src[x, y, 0], src[x, y, 1], src[x, y, 2]
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        df = max_val - min_val
        if max_val == min_val:
            h = 0
        elif max_val == r:
            h = int(((60 * ((g - b) / df) + 360) % 360) / 2)
        elif max_val == g:
            h = int(((60 * ((b - r) / df) + 120) % 360) / 2)
        elif max_val == b:
            h = int(((60 * ((r - g) / df) + 240) % 360) / 2)
        if max_val == 0:
            s = 0
        else:
            s = df / max_val
        v = max_val

        s = int(s * 255.0)
        v = int(v * 255.0)

        dst[x, y, 0] = h
        dst[x, y, 1] = s
        dst[x, y, 2] = v


@cuda.jit
def hsv_to_rgb(src, dst):
    x, y = cuda.grid(2)
    if x < src.shape[0] and y < src.shape[1]:
        h, s, v = src[x, y, 0], src[x, y, 1], src[x, y, 2]

        h = float(h) * 2
        s = float(s) / 255.0
        v = float(v) / 255.0

        c = v * s
        x_ = c * (1 - abs(((h / 60) % 2) - 1))
        m = v - c

        if 0 <= h < 60:
            r_, g_, b_ = c, x_, 0
        elif 60 <= h < 120:
            r_, g_, b_ = x_, c, 0
        elif 120 <= h < 180:
            r_, g_, b_ = 0, c, x_
        elif 180 <= h < 240:
            r_, g_, b_ = 0, x_, c
        elif 240 <= h < 300:
            r_, g_, b_ = x_, 0, c
        elif 300 <= h < 360:
            r_, g_, b_ = c, 0, x_
        else:
            r_, g_, b_ = 0, 0, 0

        r, g, b = (r_ + m) * 255.0, (g_ + m) * 255.0, (b_ + m) * 255.0

        r, g, b = min(255, max(0, r)), min(255, max(0, g)), min(255, max(0, b))

        dst[x, y, 0] = int(r)
        dst[x, y, 1] = int(g)
        dst[x, y, 2] = int(b)
