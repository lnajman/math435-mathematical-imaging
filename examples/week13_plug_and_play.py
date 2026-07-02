"""Week 13 examples: plug-and-play and learned regularization."""

from collections.abc import Callable

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import data


Array = np.ndarray


def rmse(estimate: Array, reference: Array) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


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


def adjoint_blur_periodic(image: Array, kernel_fft: Array) -> Array:
    return np.real(np.fft.ifft2(np.fft.fft2(image) * np.conj(kernel_fft)))


def data_gradient(estimate: Array, observation: Array, kernel_fft: Array) -> Array:
    residual = blur_periodic(estimate, kernel_fft) - observation
    return adjoint_blur_periodic(residual, kernel_fft)


def data_residual(estimate: Array, observation: Array, kernel_fft: Array) -> float:
    residual = blur_periodic(estimate, kernel_fft) - observation
    return float(np.linalg.norm(residual) / np.sqrt(residual.size))


def tikhonov_deblur(observation: Array, kernel_fft: Array, lam: float) -> Array:
    numerator = np.conj(kernel_fft) * np.fft.fft2(observation)
    denominator = np.abs(kernel_fft) ** 2 + lam
    return np.clip(np.real(np.fft.ifft2(numerator / denominator)), 0.0, 1.0)


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


def pnp_gradient_descent(
    observation: Array,
    kernel_fft: Array,
    denoiser: Callable[[Array], Array],
    iterations: int,
    step_size: float = 1.0,
    clean: Array | None = None,
) -> tuple[Array, dict[str, list[float]]]:
    estimate = observation.copy()
    history: dict[str, list[float]] = {"rmse": [], "residual": [], "change": []}
    previous = estimate.copy()

    for _ in range(iterations + 1):
        if clean is not None:
            history["rmse"].append(rmse(estimate, clean))
        history["residual"].append(data_residual(estimate, observation, kernel_fft))
        history["change"].append(float(np.linalg.norm(estimate - previous) / np.sqrt(estimate.size)))

        previous = estimate.copy()
        descent = estimate - step_size * data_gradient(estimate, observation, kernel_fft)
        estimate = np.clip(denoiser(descent), 0.0, 1.0)

    return estimate, history


def make_camera_problem(seed: int = 43513) -> tuple[Array, Array, Array, Array]:
    rng = np.random.default_rng(seed)
    image = data.camera().astype(float) / 255.0
    clean = image[132:228, 178:274]
    kernel = gaussian_kernel2d(size=21, sigma=2.4)
    kernel_fft = centered_kernel_fft(kernel, clean.shape)
    blurred = blur_periodic(clean, kernel_fft)
    observation = np.clip(blurred + 0.015 * rng.standard_normal(clean.shape), 0.0, 1.0)
    return clean, observation, kernel, kernel_fft


def make_moon_problem(seed: int = 43513) -> tuple[Array, Array, Array, Array]:
    rng = np.random.default_rng(seed)
    image = data.moon().astype(float) / 255.0
    clean = image[300:396, 100:196]
    kernel = gaussian_kernel2d(size=21, sigma=2.4)
    kernel_fft = centered_kernel_fft(kernel, clean.shape)
    blurred = blur_periodic(clean, kernel_fft)
    observation = np.clip(blurred + 0.015 * rng.standard_normal(clean.shape), 0.0, 1.0)
    return clean, observation, kernel, kernel_fft


def fit_camera_patch_prior() -> tuple[Array, Array, Array]:
    image = data.camera().astype(float) / 255.0
    train_image = image[:320, :320]
    patches = extract_random_patches(train_image, patch_size=7, count=5000, seed=43513)
    return fit_patch_pca(patches)


def run_demo(iterations: int = 24) -> dict[str, object]:
    clean, observation, kernel, kernel_fft = make_camera_problem()
    mean, components, singular_values = fit_camera_patch_prior()

    tikhonov = tikhonov_deblur(observation, kernel_fft, lam=0.002)
    gaussian_denoiser = lambda values: gaussian_filter(values, sigma=0.65, mode="wrap")
    learned_denoiser = lambda values: pca_patch_denoise(values, mean, components, patch_size=7, n_components=18)

    pnp_gaussian, gaussian_history = pnp_gradient_descent(
        observation,
        kernel_fft,
        gaussian_denoiser,
        iterations=iterations,
        step_size=1.0,
        clean=clean,
    )
    pnp_learned, learned_history = pnp_gradient_descent(
        observation,
        kernel_fft,
        learned_denoiser,
        iterations=iterations,
        step_size=1.0,
        clean=clean,
    )

    return {
        "clean": clean,
        "observation": observation,
        "kernel": kernel,
        "kernel_fft": kernel_fft,
        "singular_values": singular_values,
        "tikhonov": tikhonov,
        "pnp_gaussian": pnp_gaussian,
        "pnp_learned": pnp_learned,
        "gaussian_history": gaussian_history,
        "learned_history": learned_history,
        "patch_mean": mean,
        "patch_components": components,
    }


def denoiser_strength_sweep() -> list[dict[str, float]]:
    clean, observation, _, kernel_fft = make_camera_problem()
    rows = []
    for sigma in [0.0, 0.35, 0.65, 1.0, 1.45]:
        denoiser = (lambda values, s=sigma: values) if sigma == 0.0 else (lambda values, s=sigma: gaussian_filter(values, sigma=s, mode="wrap"))
        estimate, history = pnp_gradient_descent(
            observation,
            kernel_fft,
            denoiser,
            iterations=24,
            step_size=1.0,
            clean=clean,
        )
        rows.append(
            {
                "sigma": sigma,
                "rmse": rmse(estimate, clean),
                "residual": history["residual"][-1],
                "change": history["change"][-1],
            }
        )
    return rows


def distribution_shift_demo() -> dict[str, Array]:
    mean, components, _ = fit_camera_patch_prior()
    clean, observation, _, kernel_fft = make_moon_problem()
    gaussian_denoiser = lambda values: gaussian_filter(values, sigma=0.65, mode="wrap")
    learned_denoiser = lambda values: pca_patch_denoise(values, mean, components, patch_size=7, n_components=18)
    pnp_gaussian, _ = pnp_gradient_descent(observation, kernel_fft, gaussian_denoiser, iterations=24, clean=clean)
    pnp_learned, _ = pnp_gradient_descent(observation, kernel_fft, learned_denoiser, iterations=24, clean=clean)
    return {
        "clean": clean,
        "observation": observation,
        "pnp_gaussian": pnp_gaussian,
        "pnp_learned": pnp_learned,
    }


def main() -> None:
    demo = run_demo()
    clean = demo["clean"]
    observation = demo["observation"]
    kernel_fft = demo["kernel_fft"]

    print("Plug-and-play deblurring on a camera crop")
    print("method          RMSE    data residual")
    for name in ["observation", "tikhonov", "pnp_gaussian", "pnp_learned"]:
        estimate = demo[name]
        print(f"{name:13s}  {rmse(estimate, clean):.4f}  {data_residual(estimate, observation, kernel_fft):.4f}")

    print()
    print("Gaussian denoiser strength sweep")
    for row in denoiser_strength_sweep():
        print(
            f"  sigma={row['sigma']:.2f}, RMSE={row['rmse']:.4f}, "
            f"data residual={row['residual']:.4f}, fixed-point change={row['change']:.5f}"
        )

    shifted = distribution_shift_demo()
    print()
    print("Camera-trained learned denoiser on a different image family")
    for name in ["observation", "pnp_gaussian", "pnp_learned"]:
        print(f"{name:13s}  RMSE={rmse(shifted[name], shifted['clean']):.4f}")


if __name__ == "__main__":
    main()
