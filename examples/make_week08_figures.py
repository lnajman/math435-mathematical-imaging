"""Generate static figures for Week 8 slides."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage.restoration import denoise_tv_chambolle


def periodic_laplacian_eigenvalues(shape: tuple[int, ...]) -> np.ndarray:
    grids = np.meshgrid(
        *[2 * np.pi * np.fft.fftfreq(size) for size in shape],
        indexing="ij",
    )
    values = np.zeros(shape)
    for grid in grids:
        values += 2 - 2 * np.cos(grid)
    return values


def quadratic_smooth(noisy: np.ndarray, lam: float) -> np.ndarray:
    eig = periodic_laplacian_eigenvalues(noisy.shape)
    estimate = np.real(np.fft.ifftn(np.fft.fftn(noisy) / (1 + lam * eig)))
    return np.clip(estimate, 0.0, 1.0)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    if image.ndim == 1:
        return np.abs(np.diff(image, append=image[-1]))
    vertical = np.diff(image, axis=0, append=image[-1:, :])
    horizontal = np.diff(image, axis=1, append=image[:, -1:])
    return np.sqrt(vertical**2 + horizontal**2)


def total_variation(image: np.ndarray) -> float:
    return float(np.sum(gradient_magnitude(image)))


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_penalties(path: Path) -> None:
    g = np.linspace(-3.0, 3.0, 600)
    l2 = 0.5 * g**2
    tv = np.abs(g)
    eps_tv = np.sqrt(g**2 + 0.12**2)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), dpi=150)
    axes[0].plot(g, l2, label=r"$\frac{1}{2} g^2$", color="#24536b", linewidth=2.5)
    axes[0].plot(g, tv, label=r"$|g|$", color="#9a5b2f", linewidth=2.5)
    axes[0].plot(g, eps_tv, label=r"$\sqrt{g^2+\epsilon^2}$", color="#2f7a54", linewidth=2.0, linestyle="--")
    axes[0].set_title("Gradient penalties", fontsize=15, color="#1f4f63", pad=9)
    axes[0].set_xlabel("gradient g")
    axes[0].set_ylabel("penalty")
    axes[0].grid(True, alpha=0.22)
    axes[0].legend(frameon=False)

    axes[1].plot(g, g, label=r"$g$", color="#24536b", linewidth=2.5)
    axes[1].plot(g, np.sign(g), label=r"$\mathrm{sign}(g)$", color="#9a5b2f", linewidth=2.5)
    axes[1].plot(g, g / np.sqrt(g**2 + 0.12**2), label="smoothed TV derivative", color="#2f7a54", linewidth=2.0, linestyle="--")
    axes[1].set_title("Derivative behavior", fontsize=15, color="#1f4f63", pad=9)
    axes[1].set_xlabel("gradient g")
    axes[1].set_ylabel("derivative")
    axes[1].grid(True, alpha=0.22)
    axes[1].legend(frameon=False)
    fig.text(
        0.5,
        0.015,
        "TV grows linearly with gradient size; quadratic smoothing grows quadratically",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_step_signal(path: Path) -> None:
    rng = np.random.default_rng(43508)
    n = 220
    x = np.linspace(0, 1, n)
    clean = np.zeros(n)
    clean[x > 0.18] = 0.35
    clean[x > 0.45] = 0.9
    clean[x > 0.72] = 0.2
    noisy = np.clip(clean + 0.13 * rng.standard_normal(n), 0, 1)
    l2 = quadratic_smooth(noisy, 18.0)
    tv = denoise_tv_chambolle(noisy, weight=0.18)

    fig, ax = plt.subplots(figsize=(10.8, 4.4), dpi=150)
    ax.plot(x, clean, label="clean", color="#202428", linewidth=2.5)
    ax.plot(x, noisy, label="noisy", color="#b8bfc4", linewidth=1.2)
    ax.plot(x, l2, label="quadratic smoothing", color="#24536b", linewidth=2.2)
    ax.plot(x, tv, label="TV denoising", color="#9a5b2f", linewidth=2.2)
    ax.set_title("Step signal: l2 smoothing versus TV", fontsize=16, color="#1f4f63", pad=10)
    ax.set_xlabel("position")
    ax.set_ylabel("intensity")
    ax.set_ylim(-0.08, 1.08)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.text(
        0.5,
        0.015,
        "TV keeps sharp jumps better; quadratic smoothing rounds them",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_image_comparison(path: Path) -> None:
    rng = np.random.default_rng(43508)
    image = data.camera().astype(float) / 255.0
    image = image[80:336, 90:346]
    noisy = np.clip(image + 0.10 * rng.standard_normal(image.shape), 0, 1)
    l2 = quadratic_smooth(noisy, 1.2)
    tv = denoise_tv_chambolle(noisy, weight=0.11)

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.6), dpi=150)
    panels = [
        ("original", image),
        ("noisy", noisy),
        (f"l2 smooth\nRMSE={rmse(l2, image):.3f}", l2),
        (f"TV\nRMSE={rmse(tv, image):.3f}", tv),
    ]
    for ax, (title, values) in zip(axes, panels):
        ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=13, color="#1f4f63", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(
        0.5,
        0.02,
        "TV reduces noise while preserving stronger edges better than quadratic smoothing",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_gradient_maps(path: Path) -> None:
    rng = np.random.default_rng(43508)
    image = data.camera().astype(float) / 255.0
    image = image[80:336, 90:346]
    noisy = np.clip(image + 0.10 * rng.standard_normal(image.shape), 0, 1)
    l2 = quadratic_smooth(noisy, 1.2)
    tv = denoise_tv_chambolle(noisy, weight=0.11)
    maps = [gradient_magnitude(values) for values in [image, noisy, l2, tv]]
    vmax = np.percentile(maps[1], 99)

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.6), dpi=150)
    titles = ["original gradients", "noisy gradients", "l2 gradients", "TV gradients"]
    for ax, title, values in zip(axes, titles, maps):
        ax.imshow(values, cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=13, color="#1f4f63", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(
        0.5,
        0.02,
        "edge preservation can be read through gradient magnitude maps",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_parameter_effects(path: Path) -> None:
    rng = np.random.default_rng(43508)
    image = data.camera().astype(float) / 255.0
    image = image[120:376, 120:376]
    noisy = np.clip(image + 0.10 * rng.standard_normal(image.shape), 0, 1)
    weights = [0.03, 0.10, 0.28]
    estimates = [denoise_tv_chambolle(noisy, weight=weight) for weight in weights]

    fig, axes = plt.subplots(1, 5, figsize=(13.4, 3.5), dpi=150)
    panels = [("original", image), ("noisy", noisy)]
    panels.extend([(f"weight={weight:g}\nRMSE={rmse(estimate, image):.3f}", estimate) for weight, estimate in zip(weights, estimates)])
    for ax, (title, values) in zip(axes, panels):
        ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12.5, color="#1f4f63", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(
        0.5,
        0.02,
        "the TV weight controls the noise-edge tradeoff",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_staircasing(path: Path) -> None:
    rng = np.random.default_rng(43508)
    n = 260
    x = np.linspace(0, 1, n)
    clean = 0.15 + 0.65 * x + 0.08 * np.sin(2 * np.pi * 3 * x)
    noisy = np.clip(clean + 0.045 * rng.standard_normal(n), 0, 1)
    l2 = quadratic_smooth(noisy, 22.0)
    tv = denoise_tv_chambolle(noisy, weight=0.12)

    fig, ax = plt.subplots(figsize=(10.8, 4.4), dpi=150)
    ax.plot(x, clean, label="clean smooth signal", color="#202428", linewidth=2.5)
    ax.plot(x, noisy, label="noisy", color="#c0c5c8", linewidth=1.2)
    ax.plot(x, l2, label="quadratic smoothing", color="#24536b", linewidth=2.2)
    ax.plot(x, tv, label="TV", color="#9a5b2f", linewidth=2.2)
    ax.set_title("Staircasing on a smooth signal", fontsize=16, color="#1f4f63", pad=10)
    ax.set_xlabel("position")
    ax.set_ylabel("intensity")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.text(
        0.5,
        0.015,
        "TV may approximate smooth ramps by piecewise-flat plateaus",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


if __name__ == "__main__":
    base = Path("assets/figures")
    make_penalties(base / "week08-penalties.png")
    make_step_signal(base / "week08-step-l2-tv.png")
    make_image_comparison(base / "week08-image-l2-tv.png")
    make_gradient_maps(base / "week08-gradient-maps.png")
    make_parameter_effects(base / "week08-parameter-effects.png")
    make_staircasing(base / "week08-staircasing.png")
