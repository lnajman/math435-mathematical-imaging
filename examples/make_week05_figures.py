"""Generate static figures for Week 5 slides."""

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


def gaussian_kernel1d(radius: int, sigma: float) -> np.ndarray:
    axis = np.arange(-radius, radius + 1)
    kernel = np.exp(-(axis**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def convolution_matrix(n: int, sigma: float, radius: int | None = None) -> np.ndarray:
    if radius is None:
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
        0.55 * np.sin(2 * np.pi * 3 * t)
        + 0.28 * np.sin(2 * np.pi * 13 * t)
        + 0.22 * (t > 0.45)
        - 0.18 * (t > 0.72)
    )
    return signal / np.max(np.abs(signal))


def truncated_svd_solution(matrix: np.ndarray, y: np.ndarray, keep: int) -> np.ndarray:
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    coefficients = u.T @ y
    filtered = np.zeros_like(coefficients)
    filtered[:keep] = coefficients[:keep] / singular_values[:keep]
    return vh.T @ filtered


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_hadamard_checks(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=150)
    ax.axis("off")
    checks = [
        ("Existence", "at least one solution\nmatches the data"),
        ("Uniqueness", "only one solution\nmatches the data"),
        ("Stability", "small data errors cause\nsmall solution errors"),
    ]
    colors = ["#24536b", "#9a5b2f", "#2f7a54"]
    for index, ((title, body), color) in enumerate(zip(checks, colors)):
        x0 = 0.05 + index * 0.315
        box = plt.Rectangle((x0, 0.25), 0.26, 0.5, facecolor="#f7f8f8", edgecolor=color, linewidth=3)
        ax.add_patch(box)
        ax.text(x0 + 0.13, 0.60, title, ha="center", va="center", fontsize=18, color=color, weight="bold")
        ax.text(x0 + 0.13, 0.42, body, ha="center", va="center", fontsize=14, color="#202428")
    ax.text(
        0.5,
        0.08,
        "Hadamard well-posedness requires all three; inverse imaging often fails stability or uniqueness",
        ha="center",
        fontsize=14,
        color="#202428",
    )
    save(fig, path)


def make_nonuniqueness(path: Path) -> None:
    image = data.camera().astype(float) / 255.0
    image = image[70:326, 100:356]
    texture_source = data.coins().astype(float) / 255.0
    texture_source = texture_source[30:286, 30:286]
    texture_source = normalize01(texture_source)

    mask = np.ones_like(image, dtype=bool)
    mask[80:176, 92:180] = False

    candidate_a = image.copy()
    candidate_b = image.copy()
    candidate_b[~mask] = texture_source[~mask]

    observed = image.copy()
    observed[~mask] = 1.0
    mask_image = mask.astype(float)

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.5), dpi=150)
    panels = [
        ("candidate 1", candidate_a, "gray"),
        ("candidate 2", candidate_b, "gray"),
        ("observed pixels", observed, "gray"),
        ("sampling mask", mask_image, "gray"),
    ]
    for ax, (title, values, cmap) in zip(axes, panels):
        ax.imshow(values, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=14, color="#1f4f63", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(
        0.5,
        0.025,
        "Different hidden regions can produce exactly the same recorded pixels",
        ha="center",
        fontsize=14,
        color="#202428",
    )
    save(fig, path)


def make_singular_values(path: Path) -> None:
    n = 90
    sigmas = [1.2, 2.4, 4.8]
    fig, ax = plt.subplots(figsize=(8.8, 5.1), dpi=150)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.23)
    for sigma in sigmas:
        matrix = convolution_matrix(n, sigma)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        ax.semilogy(singular_values, linewidth=2.4, label=f"blur sigma={sigma}")
    ax.set_title("Singular values of 1D blur operators", fontsize=16, color="#1f4f63", pad=10)
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(frameon=False)
    fig.text(
        0.5,
        0.055,
        "stronger blur creates smaller singular values, making inversion more sensitive",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_noise_amplification(path: Path) -> None:
    rng = np.random.default_rng(43505)
    n = 128
    matrix = convolution_matrix(n, sigma=2.4)
    clean = test_signal(n)
    blurred = matrix @ clean
    noisy = blurred + 1e-5 * rng.standard_normal(n)
    naive = np.linalg.pinv(matrix, rcond=1e-8) @ noisy
    truncated = truncated_svd_solution(matrix, noisy, keep=32)

    x_axis = np.arange(n)
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.3), dpi=150)
    panels = [
        ("true signal", clean, "#24536b"),
        ("blurred + noise", noisy, "#9a5b2f"),
        ("unstable inverse", naive, "#8a2942"),
        ("truncated SVD", truncated, "#2f7a54"),
    ]
    for index, (ax, (title, values, color)) in enumerate(zip(axes, panels)):
        ax.plot(x_axis, values, color=color, linewidth=1.8)
        ax.set_title(title, fontsize=14, color="#1f4f63", pad=9)
        ax.set_ylim(-8, 8) if index == 2 else ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([])
        ax.grid(True, alpha=0.2)
    fig.text(
        0.5,
        0.025,
        "a mathematically valid inverse can be useless when noise is amplified",
        ha="center",
        fontsize=14,
        color="#202428",
    )
    save(fig, path)


def make_svd_coefficients(path: Path) -> None:
    rng = np.random.default_rng(43505)
    n = 128
    matrix = convolution_matrix(n, sigma=3.2)
    clean = test_signal(n)
    blurred = matrix @ clean
    noise = 0.012 * rng.standard_normal(n)
    noisy = blurred + noise

    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    clean_coefficients = np.abs(u.T @ blurred)
    noisy_coefficients = np.abs(u.T @ noisy)
    inverted_noise = np.abs(u.T @ noise) / singular_values

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.25, wspace=0.22)
    axes[0].semilogy(singular_values, color="#24536b", linewidth=2.4)
    axes[0].set_title("singular values", fontsize=15, color="#1f4f63", pad=9)
    axes[0].set_xlabel("index")
    axes[0].set_ylabel("value")
    axes[0].grid(True, which="both", alpha=0.2)

    axes[1].semilogy(clean_coefficients + 1e-14, label="clean data coeff.", color="#2f7a54", linewidth=2.0)
    axes[1].semilogy(noisy_coefficients + 1e-14, label="noisy data coeff.", color="#9a5b2f", linewidth=2.0)
    axes[1].semilogy(inverted_noise + 1e-14, label="noise divided by s_i", color="#8a2942", linewidth=2.0)
    axes[1].set_title("SVD coefficient view", fontsize=15, color="#1f4f63", pad=9)
    axes[1].set_xlabel("index")
    axes[1].set_ylabel("magnitude")
    axes[1].grid(True, which="both", alpha=0.2)
    axes[1].legend(frameon=False)
    fig.text(
        0.5,
        0.055,
        "small singular values expose directions where noise dominates the reconstruction",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


if __name__ == "__main__":
    base = Path("assets/figures")
    make_hadamard_checks(base / "week05-hadamard-checks.png")
    make_nonuniqueness(base / "week05-nonuniqueness.png")
    make_singular_values(base / "week05-singular-values.png")
    make_noise_amplification(base / "week05-noise-amplification.png")
    make_svd_coefficients(base / "week05-svd-coefficients.png")
