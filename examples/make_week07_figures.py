"""Generate static figures for Week 7 slides."""

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


def periodic_laplacian_matrix_eigenvalues(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    row_freq = 2 * np.pi * np.fft.fftfreq(rows)
    col_freq = 2 * np.pi * np.fft.fftfreq(cols)
    rr, cc = np.meshgrid(row_freq, col_freq, indexing="ij")
    return 4 - 2 * np.cos(rr) - 2 * np.cos(cc)


def quadratic_denoise(noisy: np.ndarray, lam: float) -> np.ndarray:
    eig = periodic_laplacian_matrix_eigenvalues(noisy.shape)
    estimate = np.real(np.fft.ifft2(np.fft.fft2(noisy) / (1 + lam * eig)))
    return np.clip(estimate, 0, 1)


def smoothness_energy(image: np.ndarray) -> float:
    vertical = np.roll(image, -1, axis=0) - image
    horizontal = np.roll(image, -1, axis=1) - image
    return float(np.sum(vertical**2 + horizontal**2))


def quadratic_energy(u: np.ndarray, y: np.ndarray, lam: float) -> float:
    return 0.5 * float(np.sum((u - y) ** 2)) + 0.5 * lam * smoothness_energy(u)


def gradient(u: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    laplacian_operator = (
        4 * u
        - np.roll(u, 1, axis=0)
        - np.roll(u, -1, axis=0)
        - np.roll(u, 1, axis=1)
        - np.roll(u, -1, axis=1)
    )
    return (u - y) + lam * laplacian_operator


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_energy_terms(path: Path) -> None:
    x = np.linspace(-2.5, 2.5, 600)
    y_obs = 0.8
    lam = 0.55
    data_term = 0.5 * (x - y_obs) ** 2
    regularizer = 0.5 * lam * x**2
    total = data_term + regularizer

    fig, ax = plt.subplots(figsize=(9.4, 4.9), dpi=150)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.86, bottom=0.24)
    ax.plot(x, data_term, label="data fit", color="#24536b", linewidth=2.5)
    ax.plot(x, regularizer, label="regularizer", color="#9a5b2f", linewidth=2.5)
    ax.plot(x, total, label="energy", color="#2f7a54", linewidth=3.0)
    ax.axvline(y_obs, color="#24536b", linestyle="--", alpha=0.45)
    minimizer = y_obs / (1 + lam)
    ax.scatter([minimizer], [0.5 * (minimizer - y_obs) ** 2 + 0.5 * lam * minimizer**2],
               color="#8a2942", s=65, zorder=3)
    ax.set_title("Energy = data fit + regularization", fontsize=16, color="#1f4f63", pad=10)
    ax.set_xlabel("candidate x")
    ax.set_ylabel("cost")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False)
    fig.text(
        0.5,
        0.055,
        "a variational model chooses the image that minimizes an energy",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_convexity(path: Path) -> None:
    x = np.linspace(-2.6, 2.6, 220)
    y = np.linspace(-2.2, 2.2, 220)
    xx, yy = np.meshgrid(x, y)
    convex = 0.65 * (xx - 0.3) ** 2 + 1.2 * (yy + 0.2) ** 2 + 0.35 * xx * yy
    nonconvex = 0.15 * (xx**2 + yy**2) + 0.8 * np.sin(2.4 * xx) * np.cos(2.0 * yy)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), dpi=150)
    panels = [
        ("convex energy", convex, "#24536b"),
        ("nonconvex energy", nonconvex, "#9a5b2f"),
    ]
    for ax, (title, values, color) in zip(axes, panels):
        contour = ax.contourf(xx, yy, values, levels=32, cmap="viridis")
        ax.contour(xx, yy, values, levels=12, colors="white", alpha=0.35, linewidths=0.7)
        ax.set_title(title, fontsize=15, color="#1f4f63", pad=9)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")
        fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    fig.text(
        0.5,
        0.015,
        "convex energies have no deceptive local minima; nonconvex energies can have many",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_gradient_descent(path: Path) -> None:
    x = np.linspace(-3.0, 3.0, 240)
    y = np.linspace(-2.6, 2.6, 240)
    xx, yy = np.meshgrid(x, y)
    matrix = np.array([[3.0, 0.9], [0.9, 1.2]])
    center = np.array([0.4, -0.7])
    grid = np.stack([xx - center[0], yy - center[1]], axis=-1)
    values = 0.5 * np.einsum("...i,ij,...j->...", grid, matrix, grid)

    point = np.array([-2.4, 2.1])
    step = 0.18
    path_points = [point.copy()]
    for _ in range(24):
        grad = matrix @ (point - center)
        point = point - step * grad
        path_points.append(point.copy())
    path_points = np.asarray(path_points)

    fig, ax = plt.subplots(figsize=(7.4, 5.6), dpi=150)
    ax.contourf(xx, yy, values, levels=36, cmap="viridis")
    ax.contour(xx, yy, values, levels=14, colors="white", alpha=0.42, linewidths=0.75)
    ax.plot(path_points[:, 0], path_points[:, 1], color="#ffffff", linewidth=2.4)
    ax.scatter(path_points[:, 0], path_points[:, 1], color="#8a2942", s=18, zorder=3)
    ax.scatter([center[0]], [center[1]], color="#f7c948", s=90, edgecolor="#202428", zorder=4)
    ax.set_title("Gradient descent follows downhill directions", fontsize=16, color="#1f4f63", pad=10)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")
    fig.text(
        0.5,
        0.02,
        "each step moves opposite the gradient of the energy",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_euler_lagrange_stencil(path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=150)
    for ax in axes:
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
        ax.set_aspect("equal")
        ax.axis("off")

    ax = axes[0]
    for i in range(5):
        for j in range(5):
            face = "#f7f8f8"
            edge = "#d4d8da"
            if i == 2 and j == 2:
                face, edge = "#24536b", "#24536b"
            elif abs(i - 2) + abs(j - 2) == 1:
                face, edge = "#9a5b2f", "#9a5b2f"
            ax.add_patch(plt.Rectangle((i, j), 1, 1, facecolor=face, edgecolor=edge, linewidth=1.5))
    ax.text(2.5, 2.5, "u", ha="center", va="center", color="white", fontsize=20, weight="bold")
    ax.text(2.5, 3.5, "N", ha="center", va="center", color="white", fontsize=15, weight="bold")
    ax.text(2.5, 1.5, "S", ha="center", va="center", color="white", fontsize=15, weight="bold")
    ax.text(1.5, 2.5, "W", ha="center", va="center", color="white", fontsize=15, weight="bold")
    ax.text(3.5, 2.5, "E", ha="center", va="center", color="white", fontsize=15, weight="bold")
    ax.set_title("finite-difference neighborhood", fontsize=15, color="#1f4f63", pad=9)

    ax = axes[1]
    ax.text(0.05, 3.8, "Energy", fontsize=16, color="#1f4f63", weight="bold")
    ax.text(0.05, 3.25, r"$E(u)=\frac{1}{2}\|u-y\|^2+\frac{\lambda}{2}\|\nabla u\|^2$", fontsize=18)
    ax.text(0.05, 2.35, "Optimality", fontsize=16, color="#1f4f63", weight="bold")
    ax.text(0.05, 1.8, r"$(u-y)-\lambda\Delta u=0$", fontsize=20)
    ax.text(0.05, 0.9, "Discrete version", fontsize=16, color="#1f4f63", weight="bold")
    ax.text(0.05, 0.35, r"$(u-y)+\lambda L u=0$", fontsize=20)
    fig.text(
        0.5,
        0.015,
        "Euler-Lagrange equations are optimality conditions for an energy",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


def make_quadratic_denoising(path: Path) -> None:
    rng = np.random.default_rng(43507)
    image = data.camera().astype(float) / 255.0
    image = image[80:336, 90:346]
    noisy = np.clip(image + 0.10 * rng.standard_normal(image.shape), 0, 1)
    lambdas = [0.15, 1.2, 8.0]
    estimates = [quadratic_denoise(noisy, lam) for lam in lambdas]

    fig, axes = plt.subplots(1, 5, figsize=(13.4, 3.5), dpi=150)
    panels = [("original", image), ("noisy", noisy)]
    panels.extend([(f"lambda={lam:g}", estimate) for lam, estimate in zip(lambdas, estimates)])
    for ax, (title, values) in zip(axes, panels):
        ax.imshow(values, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=13, color="#1f4f63", pad=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax, estimate in zip(axes[2:], estimates):
        ax.set_xlabel(f"RMSE={np.sqrt(np.mean((estimate - image) ** 2)):.3f}", fontsize=11)
    fig.text(
        0.5,
        0.02,
        "quadratic smoothness reduces noise but blurs edges as lambda grows",
        ha="center",
        fontsize=14,
        color="#202428",
    )
    save(fig, path)


def make_gradient_descent_energy(path: Path) -> None:
    rng = np.random.default_rng(43507)
    image = data.camera().astype(float) / 255.0
    image = image[96:224, 120:248]
    noisy = np.clip(image + 0.10 * rng.standard_normal(image.shape), 0, 1)
    lam = 1.5
    step = 0.08
    current = noisy.copy()
    energies = [quadratic_energy(current, noisy, lam)]
    snapshots = [current.copy()]
    for iteration in range(80):
        current = current - step * gradient(current, noisy, lam)
        energies.append(quadratic_energy(current, noisy, lam))
        if iteration in [2, 9, 39, 79]:
            snapshots.append(current.copy())

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), dpi=150)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.27, wspace=0.22)
    axes[0].semilogy(energies, color="#24536b", linewidth=2.5)
    axes[0].set_title("energy decreases during descent", fontsize=15, color="#1f4f63", pad=9)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("energy")
    axes[0].grid(True, which="both", alpha=0.22)

    labels = ["iter 0", "iter 3", "iter 10", "iter 40", "iter 80"]
    strip = np.concatenate(snapshots, axis=1)
    axes[1].imshow(strip, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("iterates", fontsize=15, color="#1f4f63", pad=9)
    axes[1].set_xticks([(i + 0.5) * image.shape[1] for i in range(len(snapshots))], labels, rotation=25)
    axes[1].set_yticks([])
    for i in range(1, len(snapshots)):
        axes[1].axvline(i * image.shape[1] - 0.5, color="white", linewidth=1.2)
    fig.text(
        0.5,
        0.055,
        "gradient descent gives an algorithmic view of variational reconstruction",
        ha="center",
        fontsize=13,
        color="#202428",
    )
    save(fig, path)


if __name__ == "__main__":
    base = Path("assets/figures")
    make_energy_terms(base / "week07-energy-terms.png")
    make_convexity(base / "week07-convexity.png")
    make_gradient_descent(base / "week07-gradient-descent-path.png")
    make_euler_lagrange_stencil(base / "week07-euler-lagrange-stencil.png")
    make_quadratic_denoising(base / "week07-quadratic-denoising.png")
    make_gradient_descent_energy(base / "week07-gradient-descent-energy.png")
