"""Generate static figures for Week 11 slides."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter
from skimage import data


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_haar_atoms(path: Path) -> None:
    x = np.linspace(0, 1, 800, endpoint=False)
    scaling = np.ones_like(x)
    wavelet = np.where(x < 0.5, 1.0, -1.0)
    localized = np.zeros_like(x)
    mask = (x >= 0.25) & (x < 0.75)
    localized[mask] = np.where(x[mask] < 0.5, 1.0, -1.0)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), dpi=150, sharey=True)
    panels = [
        ("scaling function", scaling, "#24536b"),
        ("Haar wavelet", wavelet, "#9a5b2f"),
        ("shifted and scaled wavelet", localized, "#2f7a54"),
    ]
    for ax, (title, values, color) in zip(axes, panels):
        ax.plot(x, values, color=color, linewidth=2.5)
        ax.fill_between(x, 0, values, color=color, alpha=0.16)
        ax.axhline(0, color="#202428", linewidth=1)
        ax.set_title(title, fontsize=14, color="#1f4f63", pad=9)
        ax.set_xlabel("position")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("amplitude")
    fig.text(
        0.5,
        0.015,
        "wavelets are localized patterns that can be shifted and scaled",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def reconstruct_1d_component(coeffs: list[np.ndarray], keep_index: int, wavelet: str, length: int) -> np.ndarray:
    selected = []
    for index, coeff in enumerate(coeffs):
        selected.append(coeff if index == keep_index else np.zeros_like(coeff))
    return pywt.waverec(selected, wavelet=wavelet, mode="periodization")[:length]


def make_1d_multiscale(path: Path) -> None:
    n = 256
    x = np.linspace(0, 1, n, endpoint=False)
    signal = 0.35 + 0.45 * (x > 0.34) - 0.25 * (x > 0.68)
    signal += 0.15 * np.sin(2 * np.pi * 6 * x) * ((x > 0.48) & (x < 0.9))
    signal += 0.12 * np.exp(-((x - 0.16) / 0.035) ** 2)
    coeffs = pywt.wavedec(signal, wavelet="haar", level=3, mode="periodization")
    components = [
        ("original signal", signal, "#202428"),
        ("coarse approximation A3", reconstruct_1d_component(coeffs, 0, "haar", n), "#24536b"),
        ("detail D3", reconstruct_1d_component(coeffs, 1, "haar", n), "#9a5b2f"),
        ("detail D2", reconstruct_1d_component(coeffs, 2, "haar", n), "#2f7a54"),
        ("detail D1", reconstruct_1d_component(coeffs, 3, "haar", n), "#6f5a9b"),
    ]

    fig, axes = plt.subplots(len(components), 1, figsize=(11.0, 8.2), dpi=150, sharex=True)
    for ax, (title, values, color) in zip(axes, components):
        ax.plot(x, values, color=color, linewidth=2.0)
        ax.set_title(title, fontsize=12.5, color="#1f4f63", pad=6)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("position")
    fig.text(
        0.5,
        0.012,
        "coarse scales capture broad structure; fine details capture localized changes",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_image_subbands(path: Path) -> None:
    image = data.camera().astype(float)[80:208, 128:256] / 255.0
    coeffs = pywt.wavedec2(image, wavelet="haar", level=2, mode="periodization")
    array, _ = pywt.coeffs_to_array(coeffs)
    display = np.sign(array) * np.log1p(np.abs(array))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), dpi=150)
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("image patch", fontsize=15, color="#1f4f63", pad=9)
    axes[1].imshow(display, cmap="RdBu_r")
    axes[1].set_title("2-level wavelet coefficient layout", fontsize=15, color="#1f4f63", pad=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1].text(4, 12, "LL", color="white", fontsize=12, weight="bold")
    axes[1].text(42, 12, "LH/HL/HH", color="white", fontsize=10)
    axes[1].text(72, 30, "finer details", color="white", fontsize=10)
    fig.text(
        0.5,
        0.02,
        "2D wavelets split an image into coarse approximation and directional details",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_coefficient_decay(path: Path) -> None:
    image = data.camera().astype(float)[80:208, 128:256] / 255.0
    pixels = np.abs(image.ravel())
    dct_values = np.abs(dctn(image, norm="ortho").ravel())
    wavelet_coeffs, _ = pywt.coeffs_to_array(pywt.wavedec2(image, wavelet="db2", level=3, mode="periodization"))
    wavelet_values = np.abs(wavelet_coeffs.ravel())

    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=150)
    for label, values, color in [
        ("pixels", pixels, "#6f5a9b"),
        ("DCT", dct_values, "#24536b"),
        ("wavelet", wavelet_values, "#9a5b2f"),
    ]:
        sorted_values = np.sort(values)[::-1]
        ax.loglog(sorted_values + 1e-8, label=label, color=color, linewidth=2.3)
    ax.set_title("Coefficient decay in different representations", fontsize=16, color="#1f4f63", pad=10)
    ax.set_xlabel("sorted coefficient index")
    ax.set_ylabel("magnitude")
    ax.grid(True, alpha=0.22, which="both")
    ax.legend(frameon=False)
    save(fig, path)


def wavelet_compress(image: np.ndarray, fraction: float, wavelet: str = "db2", level: int = 3) -> tuple[np.ndarray, int]:
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, mode="periodization")
    array, slices = pywt.coeffs_to_array(coeffs)
    keep = max(1, int(fraction * array.size))
    threshold = np.partition(np.abs(array).ravel(), -keep)[-keep]
    compressed_array = np.where(np.abs(array) >= threshold, array, 0.0)
    compressed_coeffs = pywt.array_to_coeffs(compressed_array, slices, output_format="wavedec2")
    reconstruction = pywt.waverec2(compressed_coeffs, wavelet=wavelet, mode="periodization")
    return np.clip(reconstruction[: image.shape[0], : image.shape[1]], 0.0, 1.0), keep


def make_wavelet_compression(path: Path) -> None:
    image = data.camera().astype(float)[80:208, 128:256] / 255.0
    panels = [("original", image)]
    for fraction in [0.10, 0.03, 0.01]:
        reconstruction, keep = wavelet_compress(image, fraction)
        panels.append((f"{100*fraction:.0f}% kept\nRMSE={rmse(reconstruction, image):.3f}", reconstruction))

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.6), dpi=150)
    for ax, (title, values) in zip(axes, panels):
        ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12.5, color="#1f4f63", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(
        0.5,
        0.02,
        "keeping only the largest wavelet coefficients gives a multiscale approximation",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


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


def make_wavelet_denoising(path: Path) -> None:
    rng = np.random.default_rng(43511)
    image = data.camera().astype(float)[80:208, 128:256] / 255.0
    noisy = np.clip(image + 0.08 * rng.standard_normal(image.shape), 0.0, 1.0)
    gaussian = np.clip(gaussian_filter(noisy, sigma=1.0, mode="reflect"), 0.0, 1.0)
    wavelet, _ = wavelet_denoise(noisy, threshold_scale=0.30)

    panels = [
        ("original", image),
        (f"noisy\nRMSE={rmse(noisy, image):.3f}", noisy),
        (f"Gaussian\nRMSE={rmse(gaussian, image):.3f}", gaussian),
        (f"wavelet threshold\nRMSE={rmse(wavelet, image):.3f}", wavelet),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.6), dpi=150)
    for ax, (title, values) in zip(axes, panels):
        ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12.5, color="#1f4f63", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(
        0.5,
        0.02,
        "wavelet thresholding removes small detail coefficients while keeping larger structures",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_threshold_sweep(path: Path) -> None:
    rng = np.random.default_rng(43511)
    image = data.camera().astype(float)[80:208, 128:256] / 255.0
    noisy = np.clip(image + 0.08 * rng.standard_normal(image.shape), 0.0, 1.0)
    scales = [0.20, 0.30, 0.60]
    panels = [("noisy", noisy)]
    for scale in scales:
        reconstruction, threshold = wavelet_denoise(noisy, threshold_scale=scale)
        panels.append((f"scale={scale:g}\nRMSE={rmse(reconstruction, image):.3f}", reconstruction))

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.6), dpi=150)
    for ax, (title, values) in zip(axes, panels):
        ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12.5, color="#1f4f63", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(
        0.5,
        0.02,
        "larger thresholds remove more noise but also remove more fine detail",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


if __name__ == "__main__":
    base = Path("assets/figures")
    make_haar_atoms(base / "week11-haar-atoms.png")
    make_1d_multiscale(base / "week11-1d-multiscale.png")
    make_image_subbands(base / "week11-image-subbands.png")
    make_coefficient_decay(base / "week11-coefficient-decay.png")
    make_wavelet_compression(base / "week11-wavelet-compression.png")
    make_wavelet_denoising(base / "week11-wavelet-denoising.png")
    make_threshold_sweep(base / "week11-threshold-sweep.png")
