"""Generate static figures for Week 3 slides."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import data


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    axis = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = values.min()
    vmax = values.max()
    if vmax == vmin:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def fft_log_magnitude(image: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    return np.log1p(np.abs(spectrum))


def centered_kernel_fft(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    padded = np.zeros(shape)
    kh, kw = kernel.shape
    padded[:kh, :kw] = kernel
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return np.fft.fft2(padded)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_1d_frequencies(path: Path) -> None:
    n = 256
    t = np.arange(n) / n
    low = np.sin(2 * np.pi * 5 * t)
    high = 0.45 * np.sin(2 * np.pi * 23 * t)
    signal = low + high
    frequencies = np.fft.rfftfreq(n, d=1 / n)
    magnitude = np.abs(np.fft.rfft(signal))

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.2), dpi=150)
    panels = [
        ("low frequency", low),
        ("high frequency", high),
        ("sum", signal),
    ]
    for ax, (title, values) in zip(axes[:3], panels):
        ax.plot(t, values, color="#1f4f63", linewidth=1.8)
        ax.set_title(title, fontsize=14, color="#1f4f63", pad=9)
        ax.set_ylim(-1.6, 1.6)
        ax.set_xticks([])
        ax.grid(True, alpha=0.2)
    axes[3].stem(frequencies, magnitude, basefmt=" ")
    axes[3].set_xlim(0, 32)
    axes[3].set_title("Fourier magnitude", fontsize=14, color="#9a5b2f", pad=9)
    axes[3].set_xlabel("frequency")
    axes[3].grid(True, alpha=0.2)
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.035, "the Fourier transform measures which oscillations are present",
             ha="center", fontsize=14, color="#202428")
    save(fig, path)


def make_image_spectrum(path: Path) -> None:
    image = data.camera().astype(float) / 255.0
    image = image[80:336, 90:346]
    spectrum = fft_log_magnitude(image)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=150)
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("image", fontsize=15, color="#1f4f63", pad=10)
    axes[1].imshow(spectrum, cmap="magma")
    axes[1].set_title("log Fourier magnitude", fontsize=15, color="#1f4f63", pad=10)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, 0.02, "the center contains low frequencies; far from center are fine oscillations",
             ha="center", fontsize=14, color="#202428")
    save(fig, path)


def make_magnitude_phase(path: Path) -> None:
    image = data.camera().astype(float) / 255.0
    image = image[80:336, 90:346]
    fft_image = np.fft.fft2(image)
    magnitude = np.abs(fft_image)
    phase = np.angle(fft_image)
    magnitude_only = normalize01(np.real(np.fft.ifft2(magnitude)))
    phase_only = normalize01(np.real(np.fft.ifft2(np.exp(1j * phase))))

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.5), dpi=150)
    panels = [
        ("original", image, "gray"),
        ("log magnitude", np.log1p(np.fft.fftshift(magnitude)), "magma"),
        ("magnitude only", magnitude_only, "gray"),
        ("phase only", phase_only, "gray"),
    ]
    for ax, (title, values, cmap) in zip(axes, panels):
        ax.imshow(values, cmap=cmap, vmin=0 if cmap == "gray" else None, vmax=1 if cmap == "gray" else None)
        ax.set_title(title, fontsize=14, color="#1f4f63", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, 0.025, "phase carries much of the geometry; magnitude describes frequency strength",
             ha="center", fontsize=14, color="#202428")
    save(fig, path)


def make_convolution_theorem(path: Path) -> None:
    image = data.camera().astype(float) / 255.0
    image = image[80:336, 90:346]
    kernel = gaussian_kernel(25, 4.0)
    blurred = ndimage.convolve(image, kernel, mode="reflect")
    kernel_response = np.fft.fftshift(np.abs(centered_kernel_fft(kernel, image.shape)))

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.5), dpi=150)
    panels = [
        ("image", image, "gray"),
        ("kernel", kernel, "viridis"),
        ("blurred image", blurred, "gray"),
        ("|FFT(kernel)|", kernel_response, "magma"),
    ]
    for ax, (title, values, cmap) in zip(axes, panels):
        ax.imshow(values, cmap=cmap, vmin=0 if cmap == "gray" else None, vmax=1 if cmap == "gray" else None)
        ax.set_title(title, fontsize=14, color="#1f4f63", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, 0.025, "convolution in space corresponds to multiplication in frequency",
             ha="center", fontsize=14, color="#202428")
    save(fig, path)


def make_low_high_pass(path: Path) -> None:
    image = data.camera().astype(float) / 255.0
    image = image[80:336, 90:346]
    rows, cols = image.shape
    rr, cc = np.ogrid[:rows, :cols]
    center_r, center_c = rows // 2, cols // 2
    radius = 28
    low_mask = (rr - center_r) ** 2 + (cc - center_c) ** 2 <= radius**2
    high_mask = ~low_mask

    shifted = np.fft.fftshift(np.fft.fft2(image))
    low_pass = normalize01(np.real(np.fft.ifft2(np.fft.ifftshift(shifted * low_mask))))
    high_pass = normalize01(np.real(np.fft.ifft2(np.fft.ifftshift(shifted * high_mask))))

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.5), dpi=150)
    panels = [
        ("original", image, "gray"),
        ("low-pass mask", low_mask.astype(float), "magma"),
        ("low-pass image", low_pass, "gray"),
        ("high-pass image", high_pass, "gray"),
    ]
    for ax, (title, values, cmap) in zip(axes, panels):
        ax.imshow(values, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=14, color="#1f4f63", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, 0.025, "low frequencies carry smooth structure; high frequencies carry rapid changes",
             ha="center", fontsize=14, color="#202428")
    save(fig, path)


def make_inverse_filtering(path: Path) -> None:
    image = data.camera().astype(float) / 255.0
    image = image[120:376, 120:376]
    kernel = gaussian_kernel(21, 3.0)
    blurred = ndimage.convolve(image, kernel, mode="reflect")
    rng = np.random.default_rng(12)
    noisy = np.clip(blurred + 0.015 * rng.standard_normal(image.shape), 0, 1)

    H = centered_kernel_fft(kernel, image.shape)
    Y = np.fft.fft2(noisy)
    naive = normalize01(np.real(np.fft.ifft2(Y / (H + 1e-4))))
    regularized = normalize01(np.real(np.fft.ifft2(Y * np.conj(H) / (np.abs(H) ** 2 + 5e-3))))

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.5), dpi=150)
    panels = [
        ("original", image),
        ("blurred + noise", noisy),
        ("naive inverse", naive),
        ("regularized inverse", regularized),
    ]
    for ax, (title, values) in zip(axes, panels):
        ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=14, color="#1f4f63", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, 0.025, "division by small frequency responses amplifies noise",
             ha="center", fontsize=14, color="#202428")
    save(fig, path)


if __name__ == "__main__":
    base = Path("assets/figures")
    make_1d_frequencies(base / "week03-1d-frequencies.png")
    make_image_spectrum(base / "week03-image-spectrum.png")
    make_magnitude_phase(base / "week03-magnitude-phase.png")
    make_convolution_theorem(base / "week03-convolution-theorem.png")
    make_low_high_pass(base / "week03-low-high-pass.png")
    make_inverse_filtering(base / "week03-inverse-filtering.png")
