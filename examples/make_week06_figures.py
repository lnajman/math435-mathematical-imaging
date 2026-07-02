"""Generate static figures for Week 6 slides."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import data


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = values.min()
    vmax = values.max()
    if vmax == vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def gaussian_kernel1d(radius: int, sigma: float) -> np.ndarray:
    axis = np.arange(-radius, radius + 1)
    kernel = np.exp(-(axis**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def gaussian_kernel2d(size: int, sigma: float) -> np.ndarray:
    axis = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def convolution_matrix(n: int, sigma: float) -> np.ndarray:
    radius = max(3, int(np.ceil(4 * sigma)))
    kernel = gaussian_kernel1d(radius, sigma)
    matrix = np.zeros((n, n))
    for col in range(n):
        impulse = np.zeros(n)
        impulse[col] = 1.0
        matrix[:, col] = np.convolve(impulse, kernel, mode="same")
    return matrix


def test_signal(n: int) -> np.ndarray:
    t = np.linspace(0, 1, n, endpoint=False)
    signal = (
        0.58 * np.sin(2 * np.pi * 3 * t)
        + 0.24 * np.sin(2 * np.pi * 14 * t)
        + 0.20 * (t > 0.43)
        - 0.18 * (t > 0.72)
    )
    return signal / np.max(np.abs(signal))


def tikhonov_solution(matrix: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n = matrix.shape[1]
    lhs = matrix.T @ matrix + lam * np.eye(n)
    rhs = matrix.T @ y
    return np.linalg.solve(lhs, rhs)


def centered_kernel_fft(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    padded = np.zeros(shape)
    kh, kw = kernel.shape
    padded[:kh, :kw] = kernel
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return np.fft.fft2(padded)


def frequency_tikhonov(noisy: np.ndarray, kernel: np.ndarray, lam: float) -> np.ndarray:
    h_fft = centered_kernel_fft(kernel, noisy.shape)
    y_fft = np.fft.fft2(noisy)
    estimate = np.real(np.fft.ifft2(np.conj(h_fft) * y_fft / (np.abs(h_fft) ** 2 + lam)))
    return np.clip(estimate, 0.0, 1.0)


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_tikhonov_path(path: Path) -> None:
    rng = np.random.default_rng(43506)
    n = 128
    matrix = convolution_matrix(n, sigma=2.8)
    clean = test_signal(n)
    blurred = matrix @ clean
    noisy = blurred + 0.018 * rng.standard_normal(n)
    lambdas = [1e-6, 1e-3, 1e-1]
    estimates = [tikhonov_solution(matrix, noisy, lam) for lam in lambdas]

    fig, axes = plt.subplots(1, 5, figsize=(13.4, 3.3), dpi=150)
    panels = [
        ("true signal", clean, "#24536b", (-1.35, 1.35)),
        ("blurred + noise", noisy, "#9a5b2f", (-1.35, 1.35)),
    ]
    panels.extend(
        [
            (f"lambda={lam:g}", estimate, "#2f7a54", (-1.35, 1.35))
            for lam, estimate in zip(lambdas, estimates)
        ]
    )
    for ax, (title, values, color, ylim) in zip(axes, panels):
        ax.plot(values, color=color, linewidth=1.8)
        ax.set_title(title, fontsize=13, color="#1f4f63", pad=9)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.grid(True, alpha=0.2)
    fig.text(
        0.5,
        0.02,
        "lambda controls how strongly the inverse is damped",
        ha="center",
        fontsize=14,
        color="#202428",
    )
    save(fig, path)


def make_filter_factors(path: Path) -> None:
    n = 128
    matrix = convolution_matrix(n, sigma=2.8)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    lambdas = [1e-6, 1e-4, 1e-2, 1e-1]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=150)
    axes[0].semilogy(singular_values, color="#24536b", linewidth=2.5)
    axes[0].set_title("singular values", fontsize=15, color="#1f4f63", pad=9)
    axes[0].set_xlabel("index")
    axes[0].set_ylabel("sigma_i")
    axes[0].grid(True, which="both", alpha=0.22)

    for lam in lambdas:
        factors = singular_values / (singular_values**2 + lam)
        axes[1].semilogy(factors, linewidth=2.2, label=f"lambda={lam:g}")
    axes[1].set_title("Tikhonov inverse factors", fontsize=15, color="#1f4f63", pad=9)
    axes[1].set_xlabel("index")
    axes[1].set_ylabel("sigma_i / (sigma_i^2 + lambda)")
    axes[1].grid(True, which="both", alpha=0.22)
    axes[1].legend(frameon=False)
    fig.text(
        0.5,
        0.015,
        "Tikhonov replaces explosive division by a damped inverse",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_parameter_tradeoff(path: Path) -> None:
    rng = np.random.default_rng(43506)
    n = 128
    matrix = convolution_matrix(n, sigma=2.8)
    clean = test_signal(n)
    blurred = matrix @ clean
    noisy = blurred + 0.018 * rng.standard_normal(n)
    lambdas = np.logspace(-8, 0, 70)
    residuals = []
    solution_norms = []
    errors = []
    for lam in lambdas:
        estimate = tikhonov_solution(matrix, noisy, lam)
        residuals.append(np.linalg.norm(matrix @ estimate - noisy))
        solution_norms.append(np.linalg.norm(estimate))
        errors.append(rmse(estimate, clean))

    best_index = int(np.argmin(errors))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=150)
    axes[0].loglog(lambdas, errors, color="#24536b", linewidth=2.5)
    axes[0].scatter([lambdas[best_index]], [errors[best_index]], color="#9a5b2f", s=55, zorder=3)
    axes[0].set_title("reconstruction error", fontsize=15, color="#1f4f63", pad=9)
    axes[0].set_xlabel("lambda")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(True, which="both", alpha=0.22)

    axes[1].loglog(residuals, solution_norms, color="#2f7a54", linewidth=2.5)
    axes[1].scatter([residuals[best_index]], [solution_norms[best_index]], color="#9a5b2f", s=55, zorder=3)
    axes[1].set_title("L-curve view", fontsize=15, color="#1f4f63", pad=9)
    axes[1].set_xlabel("||Ax_lambda - y||")
    axes[1].set_ylabel("||x_lambda||")
    axes[1].grid(True, which="both", alpha=0.22)
    fig.text(
        0.5,
        0.015,
        f"best shown lambda is about {lambdas[best_index]:.1e}; in practice the true image is unknown",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_bias_variance(path: Path) -> None:
    rng = np.random.default_rng(43506)
    n = 90
    matrix = convolution_matrix(n, sigma=2.6)
    clean = test_signal(n)
    blurred = matrix @ clean
    lambdas = np.logspace(-7, -0.2, 46)
    noise_level = 0.02
    trials = 180

    means = []
    variances = []
    mse = []
    for lam in lambdas:
        estimates = []
        for _ in range(trials):
            noisy = blurred + noise_level * rng.standard_normal(n)
            estimates.append(tikhonov_solution(matrix, noisy, lam))
        stack = np.asarray(estimates)
        mean_estimate = stack.mean(axis=0)
        bias2 = np.mean((mean_estimate - clean) ** 2)
        variance = np.mean(np.var(stack, axis=0))
        means.append(bias2)
        variances.append(variance)
        mse.append(bias2 + variance)

    fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=150)
    ax.loglog(lambdas, means, label="bias^2", color="#9a5b2f", linewidth=2.5)
    ax.loglog(lambdas, variances, label="variance", color="#24536b", linewidth=2.5)
    ax.loglog(lambdas, mse, label="bias^2 + variance", color="#2f7a54", linewidth=2.8)
    best = int(np.argmin(mse))
    ax.scatter([lambdas[best]], [mse[best]], color="#8a2942", s=60, zorder=3)
    ax.set_title("Bias-variance tradeoff", fontsize=16, color="#1f4f63", pad=10)
    ax.set_xlabel("lambda")
    ax.set_ylabel("mean squared contribution")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)
    fig.text(
        0.5,
        0.02,
        "larger lambda lowers variance but increases bias",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_image_tikhonov(path: Path) -> None:
    rng = np.random.default_rng(43506)
    image = data.camera().astype(float) / 255.0
    image = image[120:376, 120:376]
    kernel = gaussian_kernel2d(25, 3.5)
    blurred = ndimage.convolve(image, kernel, mode="reflect")
    noisy = np.clip(blurred + 0.018 * rng.standard_normal(image.shape), 0, 1)
    lambdas = [1e-5, 5e-2, 1.0]
    estimates = [frequency_tikhonov(noisy, kernel, lam) for lam in lambdas]

    fig, axes = plt.subplots(1, 5, figsize=(13.4, 3.5), dpi=150)
    panels = [("original", image), ("blurred + noise", noisy)]
    panels.extend([(f"lambda={lam:g}", estimate) for lam, estimate in zip(lambdas, estimates)])
    for ax, (title, values) in zip(axes, panels):
        ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=13, color="#1f4f63", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    errors = [rmse(estimate, image) for estimate in estimates]
    fig.text(
        0.5,
        0.02,
        "too little regularization leaves noise; too much leaves blur",
        ha="center",
        fontsize=14,
        color="#202428",
    )
    for ax, error in zip(axes[2:], errors):
        ax.set_xlabel(f"RMSE={error:.3f}", fontsize=11)
    save(fig, path)


if __name__ == "__main__":
    base = Path("assets/figures")
    make_tikhonov_path(base / "week06-tikhonov-path.png")
    make_filter_factors(base / "week06-filter-factors.png")
    make_parameter_tradeoff(base / "week06-parameter-tradeoff.png")
    make_bias_variance(base / "week06-bias-variance.png")
    make_image_tikhonov(base / "week06-image-tikhonov.png")
