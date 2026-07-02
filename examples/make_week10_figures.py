"""Generate static figures for Week 10 slides."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dctn, idctn
from skimage import data


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_sparse_problem(measurements: int = 32, unknowns: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(43510)
    matrix = rng.standard_normal((64, unknowns))
    matrix = matrix[:measurements] / np.sqrt(measurements)
    matrix = matrix / np.linalg.norm(matrix, axis=0, keepdims=True)

    true_signal = np.zeros(unknowns)
    support = np.array([7, 19, 35, 52, 70, 88])
    true_signal[support] = np.array([1.2, -0.9, 0.75, 1.0, -0.65, 0.85])
    measurements_vector = matrix @ true_signal
    return matrix, true_signal, measurements_vector


def minimum_l2_solution(matrix: np.ndarray, measurements_vector: np.ndarray) -> np.ndarray:
    gram = matrix @ matrix.T
    return matrix.T @ np.linalg.solve(gram + 1e-10 * np.eye(gram.shape[0]), measurements_vector)


def ista_l1(
    matrix: np.ndarray,
    measurements_vector: np.ndarray,
    lam: float,
    iterations: int = 1200,
) -> tuple[np.ndarray, list[float]]:
    lipschitz = float(np.linalg.norm(matrix, 2) ** 2)
    step_size = 0.95 / lipschitz
    estimate = np.zeros(matrix.shape[1])
    energies: list[float] = []
    for _ in range(iterations):
        residual = matrix @ estimate - measurements_vector
        gradient = matrix.T @ residual
        estimate = soft_threshold(estimate - step_size * gradient, step_size * lam)
        energies.append(0.5 * float(np.sum((matrix @ estimate - measurements_vector) ** 2)) + lam * float(np.sum(np.abs(estimate))))
    return estimate, energies


def make_sparse_vs_dense(path: Path) -> None:
    rng = np.random.default_rng(43510)
    n = 72
    sparse = np.zeros(n)
    sparse[[6, 18, 31, 44, 57, 66]] = [1.0, -0.8, 0.7, 1.15, -0.65, 0.9]
    dense = 0.18 * rng.standard_normal(n)
    dense += 0.25 * np.sin(np.linspace(0, 5 * np.pi, n))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.0), dpi=150)
    axes[0].stem(np.arange(n), sparse, linefmt="#24536b", markerfmt="o", basefmt=" ")
    axes[0].set_title("sparse signal", fontsize=15, color="#1f4f63", pad=9)
    axes[0].set_xlabel("coefficient index")
    axes[0].set_ylabel("value")
    axes[0].grid(True, alpha=0.22)
    axes[0].text(0.02, 0.92, "6 nonzero entries", transform=axes[0].transAxes, fontsize=12)

    axes[1].stem(np.arange(n), dense, linefmt="#9a5b2f", markerfmt="o", basefmt=" ")
    axes[1].set_title("dense signal", fontsize=15, color="#1f4f63", pad=9)
    axes[1].set_xlabel("coefficient index")
    axes[1].grid(True, alpha=0.22)
    axes[1].text(0.02, 0.92, "many small entries", transform=axes[1].transAxes, fontsize=12)
    fig.text(
        0.5,
        0.015,
        "sparsity means most coefficients are exactly zero or negligible in a chosen representation",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_l1_l2_geometry(path: Path) -> None:
    x = np.linspace(-1.0, 1.35, 400)
    y = (1.0 - x) / 2.0
    l2_solution = np.array([0.2, 0.4])
    l1_solution = np.array([0.0, 0.5])

    theta = np.linspace(0, 2 * np.pi, 500)
    l2_radius = np.linalg.norm(l2_solution)
    circle = np.c_[l2_radius * np.cos(theta), l2_radius * np.sin(theta)]
    diamond_radius = np.sum(np.abs(l1_solution))
    diamond = np.array(
        [
            [diamond_radius, 0],
            [0, diamond_radius],
            [-diamond_radius, 0],
            [0, -diamond_radius],
            [diamond_radius, 0],
        ]
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=150)
    ax.plot(x, y, color="#202428", linewidth=2.2, label=r"data constraint $x_1+2x_2=1$")
    ax.plot(circle[:, 0], circle[:, 1], color="#24536b", linewidth=2.4, label="l2 ball at first contact")
    ax.plot(diamond[:, 0], diamond[:, 1], color="#9a5b2f", linewidth=2.4, label="l1 ball at first contact")
    ax.scatter(*l2_solution, s=70, color="#24536b", zorder=5)
    ax.scatter(*l1_solution, s=70, color="#9a5b2f", zorder=5)
    ax.annotate("l2 solution\nspread over both coordinates", l2_solution + np.array([0.08, -0.12]), fontsize=11, color="#24536b")
    ax.annotate("l1 solution\non an axis", l1_solution + np.array([0.08, 0.08]), fontsize=11, color="#9a5b2f")
    ax.axhline(0, color="#d7dee2", linewidth=1)
    ax.axvline(0, color="#d7dee2", linewidth=1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.75, 1.15)
    ax.set_ylim(-0.65, 0.95)
    ax.set_title("Geometry: l1 favors coordinate axes", fontsize=16, color="#1f4f63", pad=10)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, alpha=0.18)
    save(fig, path)


def make_l1_l2_recovery(path: Path) -> None:
    matrix, true_signal, measurements_vector = make_sparse_problem()
    l2_solution = minimum_l2_solution(matrix, measurements_vector)
    l1_solution, _ = ista_l1(matrix, measurements_vector, lam=0.015)

    fig, axes = plt.subplots(3, 1, figsize=(11.4, 7.6), dpi=150, sharex=True)
    panels = [
        ("true sparse signal", true_signal, "#202428"),
        ("minimum l2-norm solution", l2_solution, "#24536b"),
        ("l1-regularized solution", l1_solution, "#9a5b2f"),
    ]
    for ax, (title, values, color) in zip(axes, panels):
        ax.stem(np.arange(len(values)), values, linefmt=color, markerfmt="o", basefmt=" ")
        ax.set_title(title, fontsize=13.5, color="#1f4f63", pad=8)
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.2)
        ax.text(
            0.985,
            0.82,
            f"active={np.count_nonzero(np.abs(values) > 1e-2)}",
            ha="right",
            transform=ax.transAxes,
            fontsize=11,
        )
    axes[-1].set_xlabel("coefficient index")
    fig.text(
        0.5,
        0.012,
        "both reconstructions use the same measurements, but l1 better matches the sparse prior",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_measurement_sweep(path: Path) -> None:
    unknowns = 96
    counts = [12, 16, 20, 24, 28, 32, 40, 56, 64]
    l2_errors = []
    l1_errors = []
    l1_active = []
    for measurements in counts:
        matrix, true_signal, measurements_vector = make_sparse_problem(measurements=measurements, unknowns=unknowns)
        l2_solution = minimum_l2_solution(matrix, measurements_vector)
        l1_solution, _ = ista_l1(matrix, measurements_vector, lam=0.015, iterations=1500)
        l2_errors.append(np.linalg.norm(l2_solution - true_signal) / np.linalg.norm(true_signal))
        l1_errors.append(np.linalg.norm(l1_solution - true_signal) / np.linalg.norm(true_signal))
        l1_active.append(np.count_nonzero(np.abs(l1_solution) > 1e-2))

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=150)
    ax.plot(counts, l2_errors, "-o", color="#24536b", linewidth=2.4, label="minimum l2-norm")
    ax.plot(counts, l1_errors, "-o", color="#9a5b2f", linewidth=2.4, label="l1 regularized")
    ax.axhline(0.05, color="#d0d8dc", linewidth=1.4, linestyle="--")
    ax.set_title("More measurements make sparse recovery easier", fontsize=16, color="#1f4f63", pad=10)
    ax.set_xlabel("number of measurements")
    ax.set_ylabel("relative reconstruction error")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    twin = ax.twinx()
    twin.plot(counts, l1_active, color="#2f7a54", linewidth=1.9, linestyle=":", label="active l1 entries")
    twin.set_ylabel("active l1 entries")
    save(fig, path)


def make_dct_compression(path: Path) -> None:
    image = data.camera().astype(float)[80:208, 128:256] / 255.0
    coefficients = dctn(image, norm="ortho")
    magnitudes = np.abs(coefficients).ravel()
    sorted_magnitudes = np.sort(magnitudes)[::-1]

    reconstructions = []
    labels = []
    for fraction in [0.10, 0.03, 0.01]:
        count = max(1, int(fraction * coefficients.size))
        threshold = sorted_magnitudes[count - 1]
        kept = np.where(np.abs(coefficients) >= threshold, coefficients, 0.0)
        reconstruction = np.clip(idctn(kept, norm="ortho"), 0.0, 1.0)
        reconstructions.append(reconstruction)
        labels.append(f"{100*fraction:.0f}% kept\nRMSE={rmse(reconstruction, image):.3f}")

    fig, axes = plt.subplots(1, 5, figsize=(13.8, 3.6), dpi=150)
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("original", fontsize=12.5, color="#1f4f63", pad=8)
    axes[1].imshow(np.log1p(np.abs(np.fft.fftshift(coefficients))), cmap="magma")
    axes[1].set_title("log DCT\nmagnitude", fontsize=12.5, color="#1f4f63", pad=8)
    for ax, reconstruction, label in zip(axes[2:], reconstructions, labels):
        ax.imshow(reconstruction, cmap="gray", vmin=0, vmax=1)
        ax.set_title(label, fontsize=12.5, color="#1f4f63", pad=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(
        0.5,
        0.02,
        "many natural-image patches are approximately sparse in transform domains",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def dct_sparse_inpaint(
    observed: np.ndarray,
    mask: np.ndarray,
    lam: float,
    iterations: int = 180,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    coefficients = np.zeros_like(observed)
    energies: list[float] = []
    for _ in range(iterations):
        image = idctn(coefficients, norm="ortho")
        residual = (image - observed) * mask
        gradient = dctn(residual, norm="ortho")
        coefficients = soft_threshold(coefficients - 0.95 * gradient, 0.95 * lam)
        image = idctn(coefficients, norm="ortho")
        energies.append(0.5 * float(np.sum(((image - observed) * mask) ** 2)) + lam * float(np.sum(np.abs(coefficients))))
    return np.clip(idctn(coefficients, norm="ortho"), 0.0, 1.0), coefficients, energies


def make_sparse_image_reconstruction(path: Path) -> None:
    rng = np.random.default_rng(43510)
    image = data.camera().astype(float)[96:160, 128:192] / 255.0
    mask = rng.random(image.shape) < 0.40
    observed = image * mask
    reconstruction, coefficients, energies = dct_sparse_inpaint(observed, mask, lam=0.008)

    fig, axes = plt.subplots(1, 5, figsize=(13.6, 3.4), dpi=150)
    panels = [
        ("original", image, "gray"),
        (f"observed\n{mask.mean():.0%} pixels", observed, "gray"),
        (f"sparse DCT\nRMSE={rmse(reconstruction, image):.3f}", reconstruction, "gray"),
        ("sample mask", mask.astype(float), "gray"),
        ("log DCT\ncoefficients", np.log1p(np.abs(np.fft.fftshift(coefficients))), "magma"),
    ]
    for ax, (title, values, cmap) in zip(axes, panels):
        ax.imshow(values, cmap=cmap, vmin=0 if cmap == "gray" else None, vmax=1 if cmap == "gray" else None)
        ax.set_title(title, fontsize=12.3, color="#1f4f63", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(
        0.5,
        0.02,
        f"random samples plus a sparse transform prior reconstruct missing information; final energy={energies[-1]:.2f}",
        ha="center",
        fontsize=12.5,
        color="#202428",
    )
    save(fig, path)


if __name__ == "__main__":
    base = Path("assets/figures")
    make_sparse_vs_dense(base / "week10-sparse-vs-dense.png")
    make_l1_l2_geometry(base / "week10-l1-l2-geometry.png")
    make_l1_l2_recovery(base / "week10-l1-l2-recovery.png")
    make_measurement_sweep(base / "week10-measurement-sweep.png")
    make_dct_compression(base / "week10-dct-compression.png")
    make_sparse_image_reconstruction(base / "week10-sparse-image-reconstruction.png")
