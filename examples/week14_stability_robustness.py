"""Week 14 examples: stability, robustness, and limitations."""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import data, draw


Array = np.ndarray


def rmse(estimate: Array, reference: Array) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def relative_norm(values: Array, reference: Array) -> float:
    return float(np.linalg.norm(values) / max(np.linalg.norm(reference), 1e-12))


def gaussian_kernel2d(size: int, sigma: float) -> Array:
    axis = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def centered_kernel_fft(kernel: Array, shape: tuple[int, int]) -> Array:
    padded = np.zeros(shape)
    kh, kw = kernel.shape
    padded[:kh, :kw] = kernel
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return np.fft.fft2(padded)


def blur_periodic(image: Array, kernel_fft: Array) -> Array:
    return np.real(np.fft.ifft2(np.fft.fft2(image) * kernel_fft))


def data_residual(estimate: Array, observation: Array, kernel_fft: Array) -> float:
    residual = blur_periodic(estimate, kernel_fft) - observation
    return float(np.linalg.norm(residual) / np.sqrt(residual.size))


def tikhonov_deblur_raw(observation: Array, kernel_fft: Array, lam: float) -> Array:
    numerator = np.conj(kernel_fft) * np.fft.fft2(observation)
    denominator = np.abs(kernel_fft) ** 2 + lam
    return np.real(np.fft.ifft2(numerator / denominator))


def tikhonov_deblur(observation: Array, kernel_fft: Array, lam: float) -> Array:
    return np.clip(tikhonov_deblur_raw(observation, kernel_fft, lam), 0.0, 1.0)


def extract_random_patches(image: Array, patch_size: int, count: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    rows, cols = image.shape
    patches = np.empty((count, patch_size * patch_size))
    for index in range(count):
        row = rng.integers(0, rows - patch_size + 1)
        col = rng.integers(0, cols - patch_size + 1)
        patches[index] = image[row : row + patch_size, col : col + patch_size].reshape(-1)
    return patches


def fit_patch_pca(patches: Array) -> tuple[Array, Array, Array]:
    mean = patches.mean(axis=0)
    _, singular_values, components = np.linalg.svd(patches - mean, full_matrices=False)
    return mean, components, singular_values


def pca_patch_denoise(
    noisy: Array,
    mean: Array,
    components: Array,
    patch_size: int,
    n_components: int,
) -> Array:
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


def make_deblurring_problem(seed: int = 43514) -> tuple[Array, Array, Array, Array]:
    rng = np.random.default_rng(seed)
    image = data.camera().astype(float) / 255.0
    clean = image[132:228, 178:274]
    kernel = gaussian_kernel2d(size=21, sigma=2.4)
    kernel_fft = centered_kernel_fft(kernel, clean.shape)
    blurred = blur_periodic(clean, kernel_fft)
    observation = np.clip(blurred + 0.015 * rng.standard_normal(clean.shape), 0.0, 1.0)
    return clean, observation, kernel, kernel_fft


def lambda_sensitivity() -> list[dict[str, float]]:
    clean, observation, _, kernel_fft = make_deblurring_problem()
    rows = []
    for lam in np.logspace(-5, -0.4, 18):
        estimate = tikhonov_deblur(observation, kernel_fft, lam)
        rows.append(
            {
                "lambda": float(lam),
                "rmse": rmse(estimate, clean),
                "residual": data_residual(estimate, observation, kernel_fft),
                "contrast": local_contrast(estimate),
            }
        )
    return rows


def perturbation_amplification() -> list[dict[str, float]]:
    clean, observation, _, kernel_fft = make_deblurring_problem()
    rows, cols = clean.shape
    rr, cc = np.indices(clean.shape)
    target = np.sqrt(1e-5)
    magnitudes = np.abs(kernel_fft).copy()
    magnitudes[0, 0] = np.inf
    freq_row, freq_col = np.unravel_index(int(np.argmin(np.abs(magnitudes - target))), clean.shape)
    sinusoid = np.cos(2.0 * np.pi * (freq_row * rr / rows + freq_col * cc / cols))
    perturbation = 0.002 * sinusoid / np.sqrt(np.mean(sinusoid**2))
    perturbed_observation = observation + perturbation

    rows_out = []
    for lam in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        base = tikhonov_deblur_raw(observation, kernel_fft, lam)
        perturbed = tikhonov_deblur_raw(perturbed_observation, kernel_fft, lam)
        rows_out.append(
            {
                "lambda": lam,
                "input_change": relative_norm(perturbed_observation - observation, observation),
                "output_change": relative_norm(perturbed - base, base),
                "rmse": rmse(perturbed, clean),
                "frequency_row": float(freq_row),
                "frequency_col": float(freq_col),
                "blur_magnitude": float(abs(kernel_fft[freq_row, freq_col])),
            }
        )
    return rows_out


def local_contrast(image: Array, center: tuple[int, int] = (48, 48), radius: int = 4) -> float:
    rr, cc = np.indices(image.shape)
    distance = np.sqrt((rr - center[0]) ** 2 + (cc - center[1]) ** 2)
    inner = image[distance <= radius]
    ring = image[(distance > radius + 3) & (distance <= radius + 10)]
    return float(inner.mean() - ring.mean())


def small_feature_case(seed: int = 43514) -> dict[str, Array | float]:
    rng = np.random.default_rng(seed)
    base = gaussian_filter(rng.random((96, 96)), sigma=7.0)
    base = (base - base.min()) / (base.max() - base.min())
    clean = 0.30 + 0.35 * base
    rr, cc = draw.disk((48, 48), radius=4, shape=clean.shape)
    clean[rr, cc] = np.clip(clean[rr, cc] + 0.32, 0.0, 1.0)

    kernel = gaussian_kernel2d(size=19, sigma=2.1)
    kernel_fft = centered_kernel_fft(kernel, clean.shape)
    observation = np.clip(blur_periodic(clean, kernel_fft) + 0.012 * rng.standard_normal(clean.shape), 0.0, 1.0)
    weak = tikhonov_deblur(observation, kernel_fft, lam=5e-4)
    strong = tikhonov_deblur(observation, kernel_fft, lam=5e-2)
    smoothed = gaussian_filter(strong, sigma=1.2, mode="wrap")

    return {
        "clean": clean,
        "observation": observation,
        "weak": weak,
        "strong": strong,
        "smoothed": smoothed,
        "kernel_fft": kernel_fft,
        "clean_contrast": local_contrast(clean),
        "weak_contrast": local_contrast(weak),
        "strong_contrast": local_contrast(strong),
        "smoothed_contrast": local_contrast(smoothed),
    }


def learned_prior_domain_test(seed: int = 43514) -> list[dict[str, float | str]]:
    rng = np.random.default_rng(seed)
    camera = data.camera().astype(float) / 255.0
    training = camera[:320, :320]
    mean, components, _ = fit_patch_pca(extract_random_patches(training, patch_size=7, count=4500, seed=seed))

    cases = [
        ("camera", camera[352:448, 256:352]),
        ("moon", data.moon().astype(float)[300:396, 100:196] / 255.0),
        ("coins", data.coins().astype(float)[40:136, 40:136] / 255.0),
    ]

    rows = []
    for name, clean in cases:
        noisy = np.clip(clean + 0.07 * rng.standard_normal(clean.shape), 0.0, 1.0)
        gaussian = np.clip(gaussian_filter(noisy, sigma=1.0, mode="reflect"), 0.0, 1.0)
        learned = pca_patch_denoise(noisy, mean, components, patch_size=7, n_components=16)
        rows.append(
            {
                "domain": name,
                "noisy_rmse": rmse(noisy, clean),
                "gaussian_rmse": rmse(gaussian, clean),
                "learned_rmse": rmse(learned, clean),
            }
        )
    return rows


def main() -> None:
    print("Tikhonov parameter sensitivity")
    rows = lambda_sensitivity()
    best = min(rows, key=lambda row: row["rmse"])
    print(f"  best lambda by RMSE: {best['lambda']:.3e}, RMSE={best['rmse']:.4f}")
    for row in rows[::4]:
        print(f"  lambda={row['lambda']:.3e}, RMSE={row['rmse']:.4f}, residual={row['residual']:.4f}")

    print()
    print("Small perturbation amplification")
    for row in perturbation_amplification():
        gain = row["output_change"] / max(row["input_change"], 1e-12)
        print(
            f"  lambda={row['lambda']:.0e}, input change={row['input_change']:.5f}, "
            f"output change={row['output_change']:.5f}, gain={gain:.1f}"
        )

    print()
    print("Small feature visibility")
    feature = small_feature_case()
    for key in ["clean_contrast", "weak_contrast", "strong_contrast", "smoothed_contrast"]:
        print(f"  {key}: {feature[key]:.4f}")

    print()
    print("Learned prior across image families")
    for row in learned_prior_domain_test():
        print(
            f"  {row['domain']:6s} noisy={row['noisy_rmse']:.4f}, "
            f"Gaussian={row['gaussian_rmse']:.4f}, learned={row['learned_rmse']:.4f}"
        )


if __name__ == "__main__":
    main()
