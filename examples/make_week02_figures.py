"""Generate static figures for Week 2 slides."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal
from skimage import data


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def motion_kernel(size: int) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=float)
    kernel[size // 2, :] = 1.0
    return kernel / kernel.sum()


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_1d_convolution(path: Path) -> None:
    x = np.array([0, 0, 1, 3, 2, 0, 0], dtype=float)
    h = np.array([0.25, 0.5, 0.25])
    y = np.convolve(x, h, mode="same")

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.4), dpi=150)
    for ax, values, title, color in [
        (axes[0], x, "signal x", "#1f4f63"),
        (axes[1], h, "kernel h", "#9a5b2f"),
        (axes[2], y, "blurred y = h * x", "#126c83"),
    ]:
        ax.stem(np.arange(len(values)), values, basefmt=" ")
        ax.set_title(title, fontsize=15, color=color, pad=10)
        ax.set_ylim(-0.1, max(3.3, values.max() + 0.4))
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("index")
    axes[1].set_xlim(-0.5, 2.5)
    fig.subplots_adjust(bottom=0.24)
    fig.text(0.5, 0.04, "convolution replaces each value by a weighted neighborhood average",
             ha="center", fontsize=14, color="#202428")
    save(fig, path)


def make_kernels(path: Path) -> None:
    kernels = [
        ("identity", np.pad([[1.0]], ((3, 3), (3, 3)))),
        ("box average", np.ones((7, 7)) / 49),
        ("gaussian", gaussian_kernel(7, 1.2)),
        ("motion", motion_kernel(7)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.2), dpi=150)
    for ax, (title, kernel) in zip(axes, kernels):
        im = ax.imshow(kernel, cmap="viridis")
        ax.set_title(title, fontsize=14, color="#1f4f63", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for (i, j), value in np.ndenumerate(kernel):
            if value > 0:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7, color="white")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.text(0.5, 0.02, "a point spread function describes how one bright point is spread",
             ha="center", fontsize=14, color="#202428")
    save(fig, path)


def make_real_blur(path: Path) -> None:
    image = data.camera().astype(float) / 255.0
    image = image[80:400, 90:410]
    box = np.ones((13, 13)) / (13 * 13)
    gaussian = gaussian_kernel(21, 3.0)
    motion = motion_kernel(25)

    images = [
        ("original", image),
        ("box blur", signal.convolve2d(image, box, mode="same", boundary="symm")),
        ("gaussian blur", signal.convolve2d(image, gaussian, mode="same", boundary="symm")),
        ("motion blur", signal.convolve2d(image, motion, mode="same", boundary="symm")),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.7), dpi=150)
    for ax, (title, values) in zip(axes, images):
        ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=14, color="#1f4f63", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, 0.025, "different kernels produce different physical blur models",
             ha="center", fontsize=14, color="#202428")
    save(fig, path)


def make_boundary_effects(path: Path) -> None:
    image = data.camera().astype(float) / 255.0
    image = image[90:218, 315:443]
    kernel = np.ones((17, 17)) / (17 * 17)
    modes = [
        ("original crop", image),
        ("zero padding", ndimage.convolve(image, kernel, mode="constant", cval=0.0)),
        ("reflect", ndimage.convolve(image, kernel, mode="reflect")),
        ("wrap", ndimage.convolve(image, kernel, mode="wrap")),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.7), dpi=150)
    for ax, (title, values) in zip(axes, modes):
        ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=14, color="#1f4f63", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, 0.025, "boundary conditions decide what the kernel sees outside the image",
             ha="center", fontsize=14, color="#202428")
    save(fig, path)


def make_deblur_instability(path: Path) -> None:
    n = 256
    t = np.linspace(0, 1, n, endpoint=False)
    clean = np.sin(2 * np.pi * 6 * t) + 0.45 * np.sin(2 * np.pi * 34 * t)
    clean = clean / np.max(np.abs(clean))
    sigma = 3.0
    ax = np.arange(-18, 19)
    h = np.exp(-(ax**2) / (2 * sigma**2))
    h = h / h.sum()
    blurred = np.convolve(clean, h, mode="same")
    rng = np.random.default_rng(7)
    noisy = blurred + 0.035 * rng.standard_normal(n)

    # Simple frequency-domain inverse with a tiny stabilizer to avoid division by zero.
    pad_h = np.zeros(n)
    pad_h[: len(h)] = h
    pad_h = np.roll(pad_h, -len(h) // 2)
    H = np.fft.fft(pad_h)
    naive = np.real(np.fft.ifft(np.fft.fft(noisy) / (H + 1e-3)))
    naive = naive / max(1.0, np.max(np.abs(naive)))

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.3), dpi=150)
    panels = [
        ("clean signal", clean),
        ("blurred", blurred),
        ("blurred + noise", noisy),
        ("naive inverse", naive),
    ]
    for axp, (title, values) in zip(axes, panels):
        axp.plot(t, values, color="#1f4f63", linewidth=1.5)
        axp.set_title(title, fontsize=14, color="#1f4f63", pad=9)
        axp.set_ylim(-1.25, 1.25)
        axp.set_xticks([])
        axp.set_yticks([])
        axp.grid(True, alpha=0.2)
    fig.text(0.5, 0.025, "deblurring tries to undo attenuation, so noise can be amplified",
             ha="center", fontsize=14, color="#202428")
    save(fig, path)


if __name__ == "__main__":
    base = Path("assets/figures")
    make_1d_convolution(base / "week02-1d-convolution.png")
    make_kernels(base / "week02-kernels.png")
    make_real_blur(base / "week02-real-blur.png")
    make_boundary_effects(base / "week02-boundary-effects.png")
    make_deblur_instability(base / "week02-deblur-instability.png")
