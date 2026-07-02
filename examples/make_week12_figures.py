"""Generate static figures for Week 12 slides."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.ndimage import gaussian_filter
from skimage import data


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def extract_random_patches(image: np.ndarray, patch_size: int, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows, cols = image.shape
    patches = np.empty((count, patch_size * patch_size))
    for index in range(count):
        row = rng.integers(0, rows - patch_size + 1)
        col = rng.integers(0, cols - patch_size + 1)
        patches[index] = image[row : row + patch_size, col : col + patch_size].reshape(-1)
    return patches


def fit_patch_pca(patches: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = patches.mean(axis=0)
    _, singular_values, components = np.linalg.svd(patches - mean, full_matrices=False)
    return mean, components, singular_values


def pca_patch_denoise(
    noisy: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
    patch_size: int,
    n_components: int,
) -> np.ndarray:
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


def wavelet_denoise(noisy: np.ndarray, threshold_scale: float = 0.30) -> np.ndarray:
    coeffs = pywt.wavedec2(noisy, wavelet="db2", level=3, mode="periodization")
    sigma = float(np.median(np.abs(coeffs[-1][2])) / 0.6745)
    threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(noisy.size))
    denoised_coeffs = [coeffs[0]]
    for horizontal, vertical, diagonal in coeffs[1:]:
        denoised_coeffs.append(
            tuple(pywt.threshold(detail, threshold, mode="soft") for detail in (horizontal, vertical, diagonal))
        )
    reconstruction = pywt.waverec2(denoised_coeffs, wavelet="db2", mode="periodization")
    return np.clip(reconstruction[: noisy.shape[0], : noisy.shape[1]], 0.0, 1.0)


def prepare_patch_prior() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = data.camera().astype(float) / 255.0
    train_image = image[:320, :320]
    test_image = image[352:448, 256:352]
    patches = extract_random_patches(train_image, patch_size=7, count=6000, seed=43512)
    mean, components, singular_values = fit_patch_pca(patches)
    return train_image, test_image, mean, components, singular_values


def make_two_philosophies(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.2), dpi=150)
    ax.axis("off")

    def box(x, y, text, color):
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=color, edgecolor="#1f4f63", linewidth=1.5),
        )

    ax.text(0.25, 0.93, "Model-based imaging", ha="center", fontsize=17, color="#1f4f63", weight="bold")
    ax.text(0.75, 0.93, "Data-driven imaging", ha="center", fontsize=17, color="#1f4f63", weight="bold")

    for x, labels, color in [
        (0.25, ["forward model A", "data y", "handcrafted prior R", "optimization algorithm", "reconstruction"], "#eef5f7"),
        (0.75, ["training examples", "loss function", "learned parameters theta", "learned operator f_theta", "reconstruction"], "#f7f1ea"),
    ]:
        ys = [0.78, 0.62, 0.46, 0.30, 0.14]
        for y, label in zip(ys, labels):
            box(x, y, label, color)
        for y0, y1 in zip(ys[:-1], ys[1:]):
            ax.annotate("", xy=(x, y1 + 0.055), xytext=(x, y0 - 0.055), arrowprops=dict(arrowstyle="->", color="#202428", linewidth=1.6))

    ax.text(0.5, 0.02, "Both use assumptions; they differ in where those assumptions come from", ha="center", fontsize=13, color="#202428")
    save(fig, path)


def make_neural_operator(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 4.8), dpi=150)
    ax.axis("off")
    labels = ["input image y", "linear filters", "nonlinearity", "linear filters", "nonlinearity", "output image"]
    xs = np.linspace(0.08, 0.92, len(labels))
    colors = ["#eef5f7", "#f7f1ea", "#f5f7ee", "#f7f1ea", "#f5f7ee", "#eef5f7"]
    for x, label, color in zip(xs, labels, colors):
        ax.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=11.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor="#1f4f63", linewidth=1.4),
        )
    for x0, x1 in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(x1 - 0.06, 0.55), xytext=(x0 + 0.06, 0.55), arrowprops=dict(arrowstyle="->", color="#202428", linewidth=1.7))
    ax.text(0.5, 0.84, r"A neural network is a parameterized operator $f_\theta$", ha="center", fontsize=17, color="#1f4f63", weight="bold")
    ax.text(0.5, 0.22, r"Training chooses $\theta$ so that $f_\theta(y)$ is close to desired outputs on examples", ha="center", fontsize=13, color="#202428")
    save(fig, path)


def make_pca_atoms(path: Path) -> None:
    _, _, mean, components, _ = prepare_patch_prior()
    patch_size = 7
    fig, axes = plt.subplots(3, 5, figsize=(9.4, 6.0), dpi=150)
    panels = [mean.reshape(patch_size, patch_size)] + [components[index].reshape(patch_size, patch_size) for index in range(14)]
    titles = ["mean patch"] + [f"atom {index + 1}" for index in range(14)]
    for ax, values, title in zip(axes.ravel(), panels, titles):
        ax.imshow(values, cmap="gray")
        ax.set_title(title, fontsize=10.5, color="#1f4f63", pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, 0.02, "a simple learned prior: principal patch directions learned from training patches", ha="center", fontsize=13, color="#202428")
    save(fig, path)


def make_capacity_curve(path: Path) -> None:
    rng = np.random.default_rng(43512)
    _, test_image, mean, components, singular_values = prepare_patch_prior()
    noisy = np.clip(test_image + 0.08 * rng.standard_normal(test_image.shape), 0.0, 1.0)
    counts = [2, 4, 8, 12, 16, 24, 32, 40]
    errors = [
        rmse(pca_patch_denoise(noisy, mean, components, patch_size=7, n_components=count), test_image)
        for count in counts
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), dpi=150)
    axes[0].plot(singular_values[:30], "-o", color="#24536b", linewidth=2.2, markersize=4)
    axes[0].set_title("learned patch spectrum", fontsize=15, color="#1f4f63", pad=9)
    axes[0].set_xlabel("component index")
    axes[0].set_ylabel("singular value")
    axes[0].grid(True, alpha=0.22)

    axes[1].plot(counts, errors, "-o", color="#9a5b2f", linewidth=2.2, markersize=5)
    axes[1].set_title("capacity tradeoff", fontsize=15, color="#1f4f63", pad=9)
    axes[1].set_xlabel("PCA components used")
    axes[1].set_ylabel("RMSE on noisy denoising task")
    axes[1].grid(True, alpha=0.22)
    fig.text(0.5, 0.02, "too few components oversmooth; too many components start preserving noise", ha="center", fontsize=13, color="#202428")
    save(fig, path)


def make_denoising_comparison(path: Path) -> None:
    rng = np.random.default_rng(43512)
    _, test_image, mean, components, _ = prepare_patch_prior()
    noisy = np.clip(test_image + 0.08 * rng.standard_normal(test_image.shape), 0.0, 1.0)
    gaussian = np.clip(gaussian_filter(noisy, sigma=1.0, mode="reflect"), 0.0, 1.0)
    wavelet = wavelet_denoise(noisy)
    learned = pca_patch_denoise(noisy, mean, components, patch_size=7, n_components=16)

    panels = [
        ("clean", test_image),
        (f"noisy\nRMSE={rmse(noisy, test_image):.3f}", noisy),
        (f"Gaussian\nRMSE={rmse(gaussian, test_image):.3f}", gaussian),
        (f"wavelet\nRMSE={rmse(wavelet, test_image):.3f}", wavelet),
        (f"learned PCA\nRMSE={rmse(learned, test_image):.3f}", learned),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(13.4, 3.4), dpi=150)
    for ax, (title, values) in zip(axes, panels):
        ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12.2, color="#1f4f63", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, 0.02, "handcrafted and learned priors make different assumptions about the same noisy input", ha="center", fontsize=12.5, color="#202428")
    save(fig, path)


def make_distribution_shift(path: Path) -> None:
    rng = np.random.default_rng(43512)
    _, camera_test, mean, components, _ = prepare_patch_prior()
    coins = data.coins().astype(float) / 255.0
    coins_test = coins[40:136, 40:136]
    cases = [("camera-like crop", camera_test), ("coins crop", coins_test)]
    names = ["noisy", "Gaussian", "learned PCA"]

    fig, axes = plt.subplots(2, 4, figsize=(11.8, 6.0), dpi=150)
    for row, (case_name, clean) in enumerate(cases):
        noisy = np.clip(clean + 0.08 * rng.standard_normal(clean.shape), 0.0, 1.0)
        gaussian = np.clip(gaussian_filter(noisy, sigma=1.0, mode="reflect"), 0.0, 1.0)
        learned = pca_patch_denoise(noisy, mean, components, patch_size=7, n_components=16)
        images = [clean, noisy, gaussian, learned]
        titles = [case_name, f"noisy\n{rmse(noisy, clean):.3f}", f"Gaussian\n{rmse(gaussian, clean):.3f}", f"learned PCA\n{rmse(learned, clean):.3f}"]
        for ax, image, title in zip(axes[row], images, titles):
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title, fontsize=11.5, color="#1f4f63", pad=7)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.text(0.5, 0.02, "a learned prior can depend strongly on whether test data resembles training data", ha="center", fontsize=12.5, color="#202428")
    save(fig, path)


if __name__ == "__main__":
    base = Path("assets/figures")
    make_two_philosophies(base / "week12-two-philosophies.png")
    make_neural_operator(base / "week12-neural-operator.png")
    make_pca_atoms(base / "week12-pca-atoms.png")
    make_capacity_curve(base / "week12-capacity-curve.png")
    make_denoising_comparison(base / "week12-denoising-comparison.png")
    make_distribution_shift(base / "week12-distribution-shift.png")
