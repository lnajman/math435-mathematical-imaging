"""Generate static figures for Week 4 slides."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
from skimage import data


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = values.min()
    vmax = values.max()
    if vmax == vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def camera_crop() -> np.ndarray:
    image = data.camera().astype(float) / 255.0
    return image[80:336, 90:346]


def add_gaussian_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(image + sigma * rng.standard_normal(image.shape), 0.0, 1.0)


def add_poisson_noise(image: np.ndarray, peak_photons: float, rng: np.random.Generator) -> np.ndarray:
    counts = rng.poisson(peak_photons * image)
    return np.clip(counts / peak_photons, 0.0, 1.0)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_noise_models(path: Path) -> None:
    rng = np.random.default_rng(43504)
    image = camera_crop()
    gaussian = add_gaussian_noise(image, 0.08, rng)
    poisson_low = add_poisson_noise(image, 18, rng)
    poisson_high = add_poisson_noise(image, 120, rng)

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.5), dpi=150)
    panels = [
        ("clean image", image),
        ("Gaussian noise", gaussian),
        ("Poisson: low light", poisson_low),
        ("Poisson: more photons", poisson_high),
    ]
    for ax, (title, values) in zip(axes, panels):
        ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=14, color="#1f4f63", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(
        0.5,
        0.025,
        "Gaussian noise has roughly constant variance; Poisson noise depends on intensity",
        ha="center",
        fontsize=14,
        color="#202428",
    )
    save(fig, path)


def make_noise_histograms(path: Path) -> None:
    rng = np.random.default_rng(43504)
    image = camera_crop()
    gaussian = add_gaussian_noise(image, 0.08, rng)
    gaussian_residual = (gaussian - image).ravel()

    true_intensities = np.array([0.15, 0.45, 0.8])
    peak_photons = 50
    samples = rng.poisson(peak_photons * true_intensities[:, None], size=(3, 7000))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    axes[0].hist(gaussian_residual, bins=70, density=True, color="#1f4f63", alpha=0.85)
    axes[0].axvline(0, color="#9a5b2f", linewidth=2)
    axes[0].set_title("Gaussian residuals", fontsize=15, color="#1f4f63", pad=9)
    axes[0].set_xlabel("observed - true intensity")
    axes[0].set_ylabel("density")
    axes[0].grid(True, alpha=0.2)

    colors = ["#24536b", "#9a5b2f", "#2f7a54"]
    for intensity, counts, color in zip(true_intensities, samples, colors):
        axes[1].hist(
            counts,
            bins=np.arange(counts.min(), counts.max() + 2) - 0.5,
            density=True,
            alpha=0.48,
            color=color,
            label=f"lambda={peak_photons * intensity:.1f}",
        )
    axes[1].set_title("Poisson photon counts", fontsize=15, color="#1f4f63", pad=9)
    axes[1].set_xlabel("count")
    axes[1].set_ylabel("probability")
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.2)
    fig.text(
        0.5,
        0.015,
        "Poisson variance equals its mean, so brighter pixels fluctuate more in counts",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_snr_curves(path: Path) -> None:
    intensities = np.linspace(0.02, 1.0, 200)
    gaussian_sigma = 0.08
    photon_levels = [12, 40, 140]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    gaussian_snr = intensities / gaussian_sigma
    axes[0].plot(intensities, gaussian_snr, color="#1f4f63", linewidth=2.5)
    axes[0].set_title("Gaussian: SNR grows linearly", fontsize=15, color="#1f4f63", pad=9)
    axes[0].set_xlabel("true intensity")
    axes[0].set_ylabel("SNR = signal / sigma")
    axes[0].grid(True, alpha=0.2)

    for photons in photon_levels:
        poisson_snr = np.sqrt(photons * intensities)
        axes[1].plot(intensities, poisson_snr, linewidth=2.5, label=f"peak={photons}")
    axes[1].set_title("Poisson: SNR grows with sqrt(count)", fontsize=15, color="#1f4f63", pad=9)
    axes[1].set_xlabel("true intensity")
    axes[1].set_ylabel("SNR")
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.2)
    fig.text(
        0.5,
        0.015,
        "Low light is statistically hard because the signal arrives as a small number of random counts",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_likelihood_curves(path: Path) -> None:
    x = np.linspace(0.01, 1.0, 400)
    y_gaussian = 0.55
    sigma = 0.09
    gaussian_nll = 0.5 * ((y_gaussian - x) / sigma) ** 2
    gaussian_nll -= gaussian_nll.min()

    observed_count = 18
    peak_photons = 36
    poisson_mean = peak_photons * x
    poisson_nll = poisson_mean - observed_count * np.log(poisson_mean)
    poisson_nll -= poisson_nll.min()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    axes[0].plot(x, gaussian_nll, color="#1f4f63", linewidth=2.5)
    axes[0].axvline(y_gaussian, color="#9a5b2f", linestyle="--", linewidth=2)
    axes[0].set_title("Gaussian negative log-likelihood", fontsize=15, color="#1f4f63", pad=9)
    axes[0].set_xlabel("candidate true intensity x")
    axes[0].set_ylabel("relative cost")
    axes[0].set_ylim(0, 35)
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(x, poisson_nll, color="#2f7a54", linewidth=2.5)
    axes[1].axvline(observed_count / peak_photons, color="#9a5b2f", linestyle="--", linewidth=2)
    axes[1].set_title("Poisson negative log-likelihood", fontsize=15, color="#1f4f63", pad=9)
    axes[1].set_xlabel("candidate true intensity x")
    axes[1].set_ylabel("relative cost")
    axes[1].set_ylim(0, 35)
    axes[1].grid(True, alpha=0.2)
    fig.text(
        0.5,
        0.015,
        "A likelihood turns a noise model into a data-fidelity term for reconstruction",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_averaging(path: Path) -> None:
    rng = np.random.default_rng(43504)
    image = camera_crop()
    sigma = 0.12
    noisy_stack = [
        add_gaussian_noise(image, sigma, rng)
        for _ in range(16)
    ]
    averages = [
        noisy_stack[0],
        np.mean(noisy_stack[:2], axis=0),
        np.mean(noisy_stack[:4], axis=0),
        np.mean(noisy_stack[:16], axis=0),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.5), dpi=150)
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("clean", fontsize=14, color="#1f4f63", pad=9)
    for ax, values, count in zip(axes[1:], averages, [1, 2, 4, 16]):
        ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        error = np.sqrt(np.mean((values - image) ** 2))
        ax.set_title(f"average {count}\nRMSE={error:.3f}", fontsize=13, color="#1f4f63", pad=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(
        0.5,
        0.015,
        "Independent Gaussian noise averages down like 1/sqrt(number of images)",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


if __name__ == "__main__":
    base = Path("assets/figures")
    make_noise_models(base / "week04-noise-models.png")
    make_noise_histograms(base / "week04-noise-histograms.png")
    make_snr_curves(base / "week04-snr-curves.png")
    make_likelihood_curves(base / "week04-likelihood-curves.png")
    make_averaging(base / "week04-averaging.png")
