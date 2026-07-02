"""Week 2 examples: convolution, blur kernels, and boundary effects."""

import numpy as np
from scipy import ndimage, signal
from skimage import data


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def motion_kernel(size: int) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=float)
    kernel[size // 2, :] = 1.0
    return kernel / kernel.sum()


def main() -> None:
    x = np.array([0, 0, 1, 3, 2, 0, 0], dtype=float)
    h = np.array([0.25, 0.5, 0.25])
    y = np.convolve(x, h, mode="same")

    print("1D signal x:")
    print(x)
    print("kernel h:")
    print(h)
    print("same-size convolution y:")
    print(y)
    print()

    image = data.camera().astype(float) / 255.0
    g = gaussian_kernel(size=15, sigma=2.4)
    blurred = signal.convolve2d(image, g, mode="same", boundary="symm")

    print("image shape:", image.shape)
    print("gaussian kernel sum:", g.sum())
    print("blurred image range:", blurred.min(), blurred.max())
    print()

    small = image[145:153, 150:158]
    k = np.ones((3, 3)) / 9
    constant = ndimage.convolve(small, k, mode="constant", cval=0.0)
    reflect = ndimage.convolve(small, k, mode="reflect")

    print("boundary demo crop shape:", small.shape)
    print("constant boundary first row:")
    print(np.round(constant[0], 3))
    print("reflect boundary first row:")
    print(np.round(reflect[0], 3))


if __name__ == "__main__":
    main()

