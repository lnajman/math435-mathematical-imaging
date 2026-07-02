"""Week 3 examples: Fourier transform, spectra, filtering, and deblurring."""

import numpy as np
from scipy import ndimage
from skimage import data


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    axis = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def fft_log_magnitude(image: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    return np.log1p(np.abs(spectrum))


def high_frequency_energy(image: np.ndarray, radius_fraction: float = 0.18) -> float:
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    power = np.abs(spectrum) ** 2
    rows, cols = image.shape
    rr, cc = np.ogrid[:rows, :cols]
    center_r, center_c = rows // 2, cols // 2
    radius = radius_fraction * min(rows, cols)
    low_mask = (rr - center_r) ** 2 + (cc - center_c) ** 2 <= radius**2
    high_power = power[~low_mask].sum()
    total_power = power.sum()
    return float(high_power / total_power)


def main() -> None:
    n = 128
    t = np.arange(n) / n
    signal = np.sin(2 * np.pi * 5 * t) + 0.45 * np.sin(2 * np.pi * 23 * t)
    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(n, d=1 / n)
    strongest = np.argsort(spectrum)[-4:][::-1]

    print("Strongest 1D frequencies:")
    for index in strongest:
        print(f"frequency={frequencies[index]:.0f}, magnitude={spectrum[index]:.3f}")
    print()

    image = data.camera().astype(float) / 255.0
    image = image[80:336, 90:346]
    blurred = ndimage.convolve(image, gaussian_kernel(21, 3.0), mode="reflect")

    print("image shape:", image.shape)
    print("spectrum shape:", fft_log_magnitude(image).shape)
    print("original high-frequency energy:", round(high_frequency_energy(image), 4))
    print("blurred high-frequency energy:", round(high_frequency_energy(blurred), 4))


if __name__ == "__main__":
    main()
