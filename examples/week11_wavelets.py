"""Week 11 examples: wavelets and multiscale image representation."""

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter
from skimage import data


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def wavelet_compress(image: np.ndarray, fraction: float, wavelet: str = "db2", level: int = 3) -> tuple[np.ndarray, int]:
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, mode="periodization")
    array, slices = pywt.coeffs_to_array(coeffs)
    keep = max(1, int(fraction * array.size))
    threshold = np.partition(np.abs(array).ravel(), -keep)[-keep]
    compressed_array = np.where(np.abs(array) >= threshold, array, 0.0)
    compressed_coeffs = pywt.array_to_coeffs(compressed_array, slices, output_format="wavedec2")
    reconstruction = pywt.waverec2(compressed_coeffs, wavelet=wavelet, mode="periodization")
    return np.clip(reconstruction[: image.shape[0], : image.shape[1]], 0.0, 1.0), keep


def estimate_noise_sigma(coeffs: list) -> float:
    finest_diagonal = coeffs[-1][2]
    return float(np.median(np.abs(finest_diagonal)) / 0.6745)


def wavelet_denoise(
    noisy: np.ndarray,
    wavelet: str = "db2",
    level: int = 3,
    threshold_scale: float = 1.0,
) -> tuple[np.ndarray, float]:
    coeffs = pywt.wavedec2(noisy, wavelet=wavelet, level=level, mode="periodization")
    sigma = estimate_noise_sigma(coeffs)
    threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(noisy.size))
    denoised_coeffs = [coeffs[0]]
    for horizontal, vertical, diagonal in coeffs[1:]:
        denoised_coeffs.append(
            tuple(pywt.threshold(detail, threshold, mode="soft") for detail in (horizontal, vertical, diagonal))
        )
    reconstruction = pywt.waverec2(denoised_coeffs, wavelet=wavelet, mode="periodization")
    return np.clip(reconstruction[: noisy.shape[0], : noisy.shape[1]], 0.0, 1.0), threshold


def main() -> None:
    image = data.camera().astype(float)[80:208, 128:256] / 255.0
    coeffs = pywt.wavedec2(image, wavelet="db2", level=3, mode="periodization")
    array, _ = pywt.coeffs_to_array(coeffs)

    print("Wavelet coefficient summary")
    print(f"image shape: {image.shape}")
    print(f"coefficient array shape: {array.shape}")
    print(f"non-negligible coefficients above 0.01: {np.count_nonzero(np.abs(array) > 0.01)} / {array.size}")

    print()
    print("Wavelet compression")
    for fraction in [0.10, 0.03, 0.01]:
        reconstruction, keep = wavelet_compress(image, fraction=fraction)
        print(f"  keep={100*fraction:4.1f}% ({keep:4d} coefficients), RMSE={rmse(reconstruction, image):.4f}")

    rng = np.random.default_rng(43511)
    noisy = np.clip(image + 0.08 * rng.standard_normal(image.shape), 0.0, 1.0)
    gaussian = np.clip(gaussian_filter(noisy, sigma=1.0, mode="reflect"), 0.0, 1.0)
    denoised, threshold = wavelet_denoise(noisy, threshold_scale=0.30)

    print()
    print("Denoising comparison")
    print(f"estimated wavelet threshold: {threshold:.4f}")
    print(f"  noisy RMSE:    {rmse(noisy, image):.4f}")
    print(f"  Gaussian RMSE: {rmse(gaussian, image):.4f}")
    print(f"  wavelet RMSE:  {rmse(denoised, image):.4f}")


if __name__ == "__main__":
    main()
