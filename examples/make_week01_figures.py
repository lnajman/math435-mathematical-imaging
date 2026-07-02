"""Generate static figures for Week 1 slides."""

from pathlib import Path

import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
from skimage import data


def make_sampling_vectorization(path: Path) -> None:
    n = 18
    grid = np.linspace(-2.5, 2.5, n)
    x, y = np.meshgrid(grid, grid)
    image = np.exp(-0.55 * (x**2 + y**2)) + 0.45 * np.exp(
        -1.8 * ((x - 1.05) ** 2 + (y + 0.75) ** 2)
    )
    image += 0.18 * np.sin(2.5 * x) * np.cos(1.8 * y)
    image = (image - image.min()) / (image.max() - image.min())

    vector = image.reshape(-1)

    fig = plt.figure(figsize=(12.8, 5.0), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.78], wspace=0.34)

    ax0 = fig.add_subplot(gs[0, 0])
    fine = 220
    fine_grid = np.linspace(-2.5, 2.5, fine)
    xf, yf = np.meshgrid(fine_grid, fine_grid)
    smooth = np.exp(-0.55 * (xf**2 + yf**2)) + 0.45 * np.exp(
        -1.8 * ((xf - 1.05) ** 2 + (yf + 0.75) ** 2)
    )
    smooth += 0.18 * np.sin(2.5 * xf) * np.cos(1.8 * yf)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    ax0.imshow(smooth, cmap="viridis", origin="lower")
    ax0.set_title("continuous model", fontsize=15, color="#1f4f63", pad=10)
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(image, cmap="viridis", origin="lower", interpolation="nearest")
    ax1.set_title("sampled image", fontsize=15, color="#1f4f63", pad=10)
    ax1.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax1.grid(which="minor", color="white", linewidth=0.55, alpha=0.7)
    ax1.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(vector[:, None], cmap="viridis", aspect="auto", origin="lower")
    ax2.set_title("vector x", fontsize=15, color="#1f4f63", pad=10)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_ylabel("pixel index", fontsize=11, color="#65717a")

    fig.text(0.5, 0.03, "sampling and vectorization:  u(x,y)  ->  U[i,j]  ->  x = vec(U)",
             ha="center", va="center", fontsize=14, color="#202428")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_real_image_vectorization(path: Path) -> None:
    image = data.camera().astype(float) / 255.0
    patch = image[170:202, 240:272]
    vector = patch.reshape(-1)

    fig = plt.figure(figsize=(12.8, 5.25), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.95, 0.82], wspace=0.34)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax0.set_title("real image", fontsize=15, color="#1f4f63", pad=10)
    ax0.add_patch(
        plt.Rectangle((240, 170), 32, 32, fill=False, edgecolor="#d4a24c", linewidth=2.5)
    )
    ax0.set_xticks([])
    ax0.set_yticks([])

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(patch, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax1.set_title("pixel patch", fontsize=15, color="#1f4f63", pad=10)
    ax1.set_xticks(np.arange(-0.5, patch.shape[1], 1), minor=True)
    ax1.set_yticks(np.arange(-0.5, patch.shape[0], 1), minor=True)
    ax1.grid(which="minor", color="#d4a24c", linewidth=0.4, alpha=0.65)
    ax1.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(vector[:, None], cmap="gray", aspect="auto", vmin=0, vmax=1, origin="lower")
    ax2.set_title("vectorized patch", fontsize=15, color="#1f4f63", pad=10)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_ylabel("pixel index", fontsize=11, color="#65717a")

    fig.text(
        0.5,
        0.035,
        "a real image is numerical data:  photograph  ->  pixel array  ->  vector",
        ha="center",
        va="center",
        fontsize=14,
        color="#202428",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_real_image_masking(path: Path) -> None:
    image = data.camera().astype(float) / 255.0
    image = image[145:273, 150:278]
    mask = np.zeros_like(image, dtype=bool)
    mask[::2, ::2] = True
    observed = np.where(mask, image, 1.0)
    zero_filled = np.where(mask, image, 0.0)

    fig = plt.figure(figsize=(12.8, 4.6), dpi=150)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.18)

    panels = [
        ("unknown x", image, "gray"),
        ("sampling mask", mask.astype(float), "gray"),
        ("observed y", observed, "gray"),
        ("zero-filled view", zero_filled, "gray"),
    ]

    for idx, (title, values, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(values, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=14, color="#1f4f63", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.text(
        0.5,
        0.035,
        "masking model:  y = Mx  keeps only selected pixels",
        ha="center",
        va="center",
        fontsize=14,
        color="#202428",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_sampling_vectorization(Path("assets/figures/week01-sampling-vectorization.png"))
    make_real_image_vectorization(Path("assets/figures/week01-real-image-vectorization.png"))
    make_real_image_masking(Path("assets/figures/week01-real-image-masking.png"))
