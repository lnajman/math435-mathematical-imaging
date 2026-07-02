"""Week 12 examples: model-based versus data-driven imaging."""

import numpy as np
import pywt
from scipy.ndimage import gaussian_filter
from skimage import data


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def extract_random_patches(image: np.ndarray, patch_size: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows, cols = image.shape
    patches = np.empty((count, patch_size * patch_size))
    for index in range(count):
        row = rng.integers(0, rows - patch_size + 1)
        col = rng.integers(0, cols - patch_size + 1)
        patches[index] = image[row : row + patch_size, col : col + patch_size].reshape(-1)
    return patches


def fit_patch_pca(patches: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = patches.mean(axis=0)
    _, singular_values, components = np.linalg.svd(patches - mean, full_matrices=False)
    return mean, components, singular_values


def pca_patch_denoise(
    noisy: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    patch_size: int,
    n_components: int,
) -> np.ndarray:
    rows, cols = noisy.shape
    estimate = np.zeros_like(noisy)
    weights = np.zeros_like(noisy)
    basis = components[:n_components]
    for row in range(rows - patch_size + 1):
        for col in range(cols - patch_size + 1):
            patch = noisy[row : row + patch_size, col : col + patch_size].reshape(-1)
            scores = (patch - mean) @ basis.T
            reconstructed = mean + scores @ basis
            estimate[row : row + patch_size, col : col + patch_size] += reconstructed.reshape(patch_size, patch_size)
            weights[row : row + patch_size, col : col + patch_size] += 1.0
    return np.clip(estimate / np.maximum(weights, 1.0), 0.0, 1.0)


def wavelet_denoise(noisy: np.ndarray, threshold_scale: float = 0.30) -> np.ndarray:
    coeffs = pywt.wavedec2(noisy, wavelet="db2", level=3, mode="periodization")
    sigma = float(np.median(np.abs(coeffs[-1][2])) / 0.6745)
    threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(noisy.size))
    denoised_coeffs = [coeffs[0]]
    for horizontal, vertical, diagonal in coeffs[1:]:
        denoised_coeffs.append(
            tuple(pywt.threshold(detail, threshold, mode="soft") for detail in (horizontal, vertical, diagonal))
        )
    reconstruction = pywt.waverec2(denoised_coeffs, wavelet="db2", mode="periodization")
    return np.clip(reconstruction[: noisy.shape[0], : noisy.shape[1]], 0.0, 1.0)


def main() -> None:
    rng = np.random.default_rng(43512)
    image = data.camera().astype(float) / 255.0
    train_image = image[:320, :320]
    test_image = image[352:448, 256:352]
    patch_size = 7

    patches = extract_random_patches(train_image, patch_size=patch_size, count=6000, seed=43512)
    mean, components, singular_values = fit_patch_pca(patches)

    noisy = np.clip(test_image + 0.08 * rng.standard_normal(test_image.shape), 0.0, 1.0)
    gaussian = np.clip(gaussian_filter(noisy, sigma=1.0, mode="reflect"), 0.0, 1.0)
    wavelet = wavelet_denoise(noisy)
    learned = pca_patch_denoise(noisy, mean, components, patch_size=patch_size, n_components=16)

    print("Denoising comparison on a held-out crop")
    print("method        RMSE")
    for name, estimate in [
        ("noisy", noisy),
        ("Gaussian", gaussian),
        ("wavelet", wavelet),
        ("learned PCA", learned),
    ]:
        print(f"{name:10s}  {rmse(estimate, test_image):.4f}")

    print()
    print("Learned prior capacity: number of PCA components")
    for n_components in [4, 8, 12, 16, 24, 32]:
        estimate = pca_patch_denoise(noisy, mean, components, patch_size=patch_size, n_components=n_components)
        print(f"  components={n_components:2d}, RMSE={rmse(estimate, test_image):.4f}")

    coins = data.coins().astype(float) / 255.0
    shifted_test = coins[40:136, 40:136]
    shifted_noisy = np.clip(shifted_test + 0.08 * rng.standard_normal(shifted_test.shape), 0.0, 1.0)
    shifted_learned = pca_patch_denoise(shifted_noisy, mean, components, patch_size=patch_size, n_components=16)

    print()
    print("Same learned prior on a different image family")
    print(f"  noisy RMSE:      {rmse(shifted_noisy, shifted_test):.4f}")
    print(f"  Gaussian RMSE:   {rmse(gaussian_filter(shifted_noisy, sigma=1.0, mode='reflect'), shifted_test):.4f}")
    print(f"  learned PCA RMSE:{rmse(shifted_learned, shifted_test):.4f}")


if __name__ == "__main__":
    main()
