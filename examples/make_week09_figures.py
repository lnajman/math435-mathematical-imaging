"""Generate static figures for Week 9 slides."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter
from skimage import data


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Soft-thresholding proximal operator for threshold * ||.||_1."""
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def quadratic_energy(point: np.ndarray, matrix: np.ndarray, center: np.ndarray) -> float:
    diff = point - center
    return float(0.5 * diff @ matrix @ diff)


def gradient_descent_path(
    start: np.ndarray,
    matrix: np.ndarray,
    center: np.ndarray,
    step_size: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    point = start.astype(float).copy()
    points = [point.copy()]
    energies = [quadratic_energy(point, matrix, center)]
    for _ in range(iterations):
        gradient = matrix @ (point - center)
        point = point - step_size * gradient
        points.append(point.copy())
        energies.append(quadratic_energy(point, matrix, center))
    return np.array(points), np.array(energies)


def make_convexity(path: Path) -> None:
    x = np.linspace(-3.0, 3.0, 600)
    convex = 0.5 * (x - 0.35) ** 2 + 0.7
    nonsmooth = 0.45 * (x - 0.3) ** 2 + 1.3 * np.abs(x + 0.2)
    nonconvex = 0.18 * x**2 + 0.55 * np.sin(2.7 * x) + 1.9

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), dpi=150)
    panels = [
        ("smooth convex", convex, "#24536b"),
        ("nonsmooth convex", nonsmooth, "#9a5b2f"),
        ("nonconvex", nonconvex, "#6f5a9b"),
    ]
    for ax, (title, values, color) in zip(axes, panels):
        ax.plot(x, values, color=color, linewidth=2.5)
        ax.set_title(title, fontsize=14, color="#1f4f63", pad=9)
        ax.set_xlabel("unknown value")
        ax.set_ylabel("energy")
        ax.grid(True, alpha=0.22)
        ax.set_yticks([])
    fig.text(
        0.5,
        0.015,
        "convexity makes local descent trustworthy; nonsmoothness changes the algorithm",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_gradient_descent_path(path: Path) -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 10.0]])
    center = np.array([0.9, -0.45])
    start = np.array([-1.7, 1.15])
    path_slow, _ = gradient_descent_path(start, matrix, center, 0.055, 28)
    path_fast, _ = gradient_descent_path(start, matrix, center, 0.175, 28)

    x = np.linspace(-2.2, 1.5, 240)
    y = np.linspace(-1.2, 1.35, 240)
    xx, yy = np.meshgrid(x, y)
    energy = 0.5 * ((xx - center[0]) ** 2 + 10.0 * (yy - center[1]) ** 2)

    fig, ax = plt.subplots(figsize=(7.2, 5.5), dpi=150)
    ax.contour(xx, yy, energy, levels=18, colors="#d6dde1", linewidths=1)
    ax.plot(path_slow[:, 0], path_slow[:, 1], "-o", markersize=3.2, color="#24536b", label="safe small steps")
    ax.plot(path_fast[:, 0], path_fast[:, 1], "-o", markersize=3.2, color="#9a5b2f", label="larger steps")
    ax.scatter([center[0]], [center[1]], s=80, color="#202428", marker="*", label="minimizer")
    ax.set_title("Gradient descent follows the negative gradient", fontsize=16, color="#1f4f63", pad=10)
    ax.set_xlabel("coordinate 1")
    ax.set_ylabel("coordinate 2")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.18)
    save(fig, path)


def make_step_sizes(path: Path) -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 10.0]])
    center = np.array([0.9, -0.45])
    start = np.array([-1.7, 1.15])
    lipschitz = float(np.linalg.eigvalsh(matrix).max())
    settings = [
        (0.4 / lipschitz, "small", "#24536b"),
        (1.0 / lipschitz, "near 1/L", "#2f7a54"),
        (1.8 / lipschitz, "aggressive", "#9a5b2f"),
        (2.1 / lipschitz, "too large", "#7f2c2c"),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150)
    for step_size, label, color in settings:
        _, energies = gradient_descent_path(start, matrix, center, step_size, 35)
        ax.plot(energies, label=f"{label}: tau={step_size:.3f}", color=color, linewidth=2.3)
    ax.set_yscale("log")
    ax.set_title("Step size controls convergence", fontsize=16, color="#1f4f63", pad=10)
    ax.set_xlabel("iteration")
    ax.set_ylabel("energy")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    save(fig, path)


def make_soft_threshold(path: Path) -> None:
    x = np.linspace(-3.0, 3.0, 700)
    threshold = 0.9
    prox = soft_threshold(x, threshold)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=150)
    axes[0].plot(x, np.abs(x), color="#9a5b2f", linewidth=2.5)
    axes[0].plot([0, 0], [0, 1.0], color="#202428", linewidth=3.5)
    axes[0].fill_betweenx([0, 1.0], -0.14, 0.14, color="#202428", alpha=0.12)
    axes[0].set_title("Absolute value is nonsmooth at 0", fontsize=14, color="#1f4f63", pad=9)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("|x|")
    axes[0].grid(True, alpha=0.22)

    axes[1].plot(x, x, color="#b9c3c8", linewidth=1.8, linestyle="--", label="identity")
    axes[1].plot(x, prox, color="#24536b", linewidth=2.7, label="soft threshold")
    axes[1].axvline(-threshold, color="#9a5b2f", linewidth=1.5, linestyle=":")
    axes[1].axvline(threshold, color="#9a5b2f", linewidth=1.5, linestyle=":")
    axes[1].set_title("Proximal map for l1", fontsize=14, color="#1f4f63", pad=9)
    axes[1].set_xlabel("input")
    axes[1].set_ylabel("output")
    axes[1].grid(True, alpha=0.22)
    axes[1].legend(frameon=False)
    fig.text(
        0.5,
        0.015,
        "the proximal operator shrinks small coefficients to zero and reduces large ones",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def blur1d(values: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter(values, sigma=sigma, mode="wrap")


def ista_sparse_signal(
    observed: np.ndarray,
    lam: float,
    step_size: float,
    iterations: int,
    sigma: float,
) -> tuple[np.ndarray, list[float], list[int]]:
    estimate = np.zeros_like(observed)
    energies: list[float] = []
    active_counts: list[int] = []
    for _ in range(iterations):
        residual = blur1d(estimate, sigma) - observed
        gradient = blur1d(residual, sigma)
        estimate = soft_threshold(estimate - step_size * gradient, step_size * lam)
        energies.append(0.5 * float(np.sum((blur1d(estimate, sigma) - observed) ** 2)) + lam * float(np.sum(np.abs(estimate))))
        active_counts.append(int(np.count_nonzero(np.abs(estimate) > 1e-3)))
    return estimate, energies, active_counts


def make_ista_sparse_signal(path: Path) -> None:
    rng = np.random.default_rng(43509)
    n = 180
    x = np.linspace(0, 1, n)
    clean = np.zeros(n)
    locations = np.array([18, 42, 75, 109, 143, 162])
    amplitudes = np.array([1.0, -0.75, 0.55, 0.9, -0.65, 0.5])
    clean[locations] = amplitudes
    observed = blur1d(clean, sigma=3.0) + 0.035 * rng.standard_normal(n)
    estimate, energies, active_counts = ista_sparse_signal(observed, lam=0.035, step_size=0.95, iterations=80, sigma=3.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), dpi=150)
    axes[0].plot(x, observed, color="#b8bfc4", linewidth=1.8, label="blurred noisy observation")
    axes[0].stem(x, clean, linefmt="#202428", markerfmt="ko", basefmt=" ", label="true sparse signal")
    axes[0].stem(x, estimate, linefmt="#9a5b2f", markerfmt="o", basefmt=" ", label="ISTA estimate")
    axes[0].set_title("ISTA for a sparse inverse problem", fontsize=14, color="#1f4f63", pad=9)
    axes[0].set_xlabel("position")
    axes[0].set_ylabel("amplitude")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(True, alpha=0.22)

    axes[1].plot(energies, color="#24536b", linewidth=2.4, label="energy")
    axes[1].set_title("Energy decreases through iterations", fontsize=14, color="#1f4f63", pad=9)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("energy")
    axes[1].grid(True, alpha=0.22)
    twin = axes[1].twinx()
    twin.plot(active_counts, color="#9a5b2f", linewidth=1.9, linestyle="--", label="active coefficients")
    twin.set_ylabel("active coefficients")
    save(fig, path)


def ista_dct_deblur(
    observed: np.ndarray,
    lam: float,
    step_size: float,
    iterations: int,
    blur_sigma: float,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    coefficients = dctn(observed, norm="ortho")
    energies: list[float] = []
    for _ in range(iterations):
        image = idctn(coefficients, norm="ortho")
        blurred = gaussian_filter(image, sigma=blur_sigma, mode="reflect")
        residual = blurred - observed
        gradient_image = gaussian_filter(residual, sigma=blur_sigma, mode="reflect")
        gradient_coefficients = dctn(gradient_image, norm="ortho")
        coefficients = soft_threshold(coefficients - step_size * gradient_coefficients, step_size * lam)
        image = idctn(coefficients, norm="ortho")
        energies.append(
            0.5 * float(np.sum((gaussian_filter(image, sigma=blur_sigma, mode="reflect") - observed) ** 2))
            + lam * float(np.sum(np.abs(coefficients)))
        )
    return np.clip(idctn(coefficients, norm="ortho"), 0.0, 1.0), coefficients, energies


def make_image_ista(path: Path) -> None:
    rng = np.random.default_rng(43509)
    image = data.camera().astype(float)[96:224, 128:256] / 255.0
    observed = np.clip(gaussian_filter(image, sigma=1.1, mode="reflect") + 0.035 * rng.standard_normal(image.shape), 0, 1)
    estimate, coefficients, energies = ista_dct_deblur(observed, lam=0.0045, step_size=0.95, iterations=65, blur_sigma=1.1)
    coefficient_magnitude = np.log1p(np.abs(np.fft.fftshift(coefficients)))

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.6), dpi=150)
    panels = [
        ("original", image, "gray"),
        ("blurred + noisy", observed, "gray"),
        ("ISTA estimate", estimate, "gray"),
        ("log DCT coefficients", coefficient_magnitude, "magma"),
    ]
    for ax, (title, values, cmap) in zip(axes, panels):
        ax.imshow(values, cmap=cmap, vmin=0 if cmap == "gray" else None, vmax=1 if cmap == "gray" else None)
        ax.set_title(title, fontsize=12.5, color="#1f4f63", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(
        0.5,
        0.02,
        f"proximal gradient handles data fidelity and sparse regularization separately; final energy={energies[-1]:.2f}",
        ha="center",
        fontsize=12.5,
        color="#202428",
    )
    save(fig, path)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    base = Path("assets/figures")
    make_convexity(base / "week09-convexity.png")
    make_gradient_descent_path(base / "week09-gradient-descent-path.png")
    make_step_sizes(base / "week09-step-sizes.png")
    make_soft_threshold(base / "week09-soft-threshold.png")
    make_ista_sparse_signal(base / "week09-ista-sparse-signal.png")
    make_image_ista(base / "week09-ista-image.png")
