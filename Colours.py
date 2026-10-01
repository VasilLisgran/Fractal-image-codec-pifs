import numpy as np
from PIL import Image

def rgb_to_ycbcr(img_rgb):
    matrix = np.array([
        [ 0.29900,  0.58700,  0.11400],
        [-0.16874, -0.33126,  0.50000],
        [ 0.50000, -0.41869, -0.08131]
    ])
    ycbcr = np.dot(img_rgb.astype(np.float64), matrix.T)
    ycbcr[:, :, 1] += 128
    ycbcr[:, :, 2] += 128
    return np.clip(ycbcr, 0, 255)

def ycbcr_to_rgb(img_ycbcr):
    matrix = np.array([
        [1.0,  0.00000,  1.40200],
        [1.0, -0.34414, -0.71414],
        [1.0,  1.77200,  0.00000]
    ])
    ycbcr = img_ycbcr.astype(np.float64)
    ycbcr[:, :, 1] -= 128
    ycbcr[:, :, 2] -= 128
    rgb = np.dot(ycbcr, matrix.T)
    return np.clip(rgb, 0, 255).astype(np.uint8)

def subsample_2x(channel):
    H, W = channel.shape
    c = channel[:H // 2 * 2, :W // 2 * 2]
    return c.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))

def upsample_2x(channel, target_shape):
    H, W = target_shape
    upsampled = np.repeat(np.repeat(channel, 2, axis=0), 2, axis=1)
    return upsampled[:H, :W]
