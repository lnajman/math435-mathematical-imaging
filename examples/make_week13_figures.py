"""Generate static figures for Week 13 slides."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np

from week13_plug_and_play import (
    data_residual,
    denoiser_strength_sweep,
    distribution_shift_demo,
    pnp_gradient_descent,
    rmse,
    run_demo,
)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_pnp_loop(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.8), dpi=150)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        (0.16, 0.62, "current image\n$x^k$", "#eef5f7"),
        (0.42, 0.62, "data-consistency\nstep using $A,y$", "#f5f7ee"),
        (0.68, 0.62, "denoiser\n$D_\\sigma$", "#f7f1ea"),
        (0.42, 0.25, "next image\n$x^{k+1}$", "#eef5f7"),
    ]
    for x, y, text, color in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=13,
            bbox=dict(boxstyle="round,pad=0.45", facecolor=color, edgecolor="#1f4f63", linewidth=1.6),
        )

    arrows = [
        ((0.24, 0.62), (0.33, 0.62)),
        ((0.51, 0.62), (0.59, 0.62)),
        ((0.68, 0.53), (0.48, 0.32)),
        ((0.34, 0.25), (0.16, 0.53)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#202428", linewidth=1.8))

    ax.text(0.5, 0.90, "Plug-and-play reconstruction loop", ha="center", fontsize=18, color="#1f4f63", weight="bold")
    ax.text(0.5, 0.08, "The denoiser may be handcrafted or learned; the forward model still enforces measured data.", ha="center", fontsize=12.5, color="#202428")
    save(fig, path)


def make_prox_vs_denoiser(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.0), dpi=150)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.25, 0.90, "Classical proximal step", ha="center", fontsize=17, color="#1f4f63", weight="bold")
    ax.text(0.75, 0.90, "Plug-and-play step", ha="center", fontsize=17, color="#1f4f63", weight="bold")

    left = [
        (0.25, 0.68, "$z$ from data step"),
        (0.25, 0.50, "$\\operatorname{prox}_{\\lambda R}(z)$"),
        (0.25, 0.32, "minimize a known\nregularized objective"),
    ]
    right = [
        (0.75, 0.68, "$z$ from data step"),
        (0.75, 0.50, "$D_\\sigma(z)$"),
        (0.75, 0.32, "call a denoising\noperator instead"),
    ]
    for items, color in [(left, "#eef5f7"), (right, "#f7f1ea")]:
        for x, y, text in items:
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=12.5,
                bbox=dict(boxstyle="round,pad=0.45", facecolor=color, edgecolor="#1f4f63", linewidth=1.4),
            )
        ax.annotate("", xy=(items[1][0], 0.56), xytext=(items[0][0], 0.62), arrowprops=dict(arrowstyle="->", color="#202428", linewidth=1.6))
        ax.annotate("", xy=(items[2][0], 0.38), xytext=(items[1][0], 0.44), arrowprops=dict(arrowstyle="->", color="#202428", linewidth=1.6))

    ax.text(0.5, 0.08, "The algorithmic shape is familiar; the mathematical interpretation changes.", ha="center", fontsize=12.5, color="#202428")
    save(fig, path)


def make_deblurring_comparison(path: Path) -> None:
    demo = run_demo()
    clean = demo["clean"]
    observation = demo["observation"]
    kernel_fft = demo["kernel_fft"]
    panels = [
        ("clean", clean),
        (f"blurred + noise\nRMSE={rmse(observation, clean):.3f}", observation),
        (f"Tikhonov\nRMSE={rmse(demo['tikhonov'], clean):.3f}", demo["tikhonov"]),
        (f"PnP Gaussian\nRMSE={rmse(demo['pnp_gaussian'], clean):.3f}", demo["pnp_gaussian"]),
        (f"PnP learned\nRMSE={rmse(demo['pnp_learned'], clean):.3f}", demo["pnp_learned"]),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(13.4, 3.4), dpi=150)
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12.0, color="#1f4f63", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
    residual_text = (
        "data residuals: "
        f"Tikhonov {data_residual(demo['tikhonov'], observation, kernel_fft):.3f}, "
        f"PnP Gaussian {data_residual(demo['pnp_gaussian'], observation, kernel_fft):.3f}, "
        f"PnP learned {data_residual(demo['pnp_learned'], observation, kernel_fft):.3f}"
    )
    fig.text(0.5, 0.02, residual_text, ha="center", fontsize=11.5, color="#202428")
    save(fig, path)


def make_iteration_curve(path: Path) -> None:
    demo = run_demo()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=150)
    iterations = np.arange(len(demo["gaussian_history"]["rmse"]))
    axes[0].plot(iterations, demo["gaussian_history"]["rmse"], "-o", label="PnP Gaussian", color="#24536b", linewidth=2.0, markersize=4)
    axes[0].plot(iterations, demo["learned_history"]["rmse"], "-o", label="PnP learned", color="#9a5b2f", linewidth=2.0, markersize=4)
    axes[0].set_title("reconstruction error", fontsize=15, color="#1f4f63", pad=9)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(True, alpha=0.22)
    axes[0].legend()

    axes[1].plot(iterations, demo["gaussian_history"]["change"], "-o", label="PnP Gaussian", color="#24536b", linewidth=2.0, markersize=4)
    axes[1].plot(iterations, demo["learned_history"]["change"], "-o", label="PnP learned", color="#9a5b2f", linewidth=2.0, markersize=4)
    axes[1].set_title("fixed-point change", fontsize=15, color="#1f4f63", pad=9)
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel(r"$\|x^{k+1}-x^k\|/\sqrt{n}$")
    axes[1].grid(True, alpha=0.22)
    axes[1].legend()
    fig.text(0.5, 0.02, "PnP algorithms are often monitored as fixed-point iterations, not only objective minimizers.", ha="center", fontsize=12.5, color="#202428")
    save(fig, path)


def make_strength_tradeoff(path: Path) -> None:
    rows = denoiser_strength_sweep()
    sigmas = [row["sigma"] for row in rows]
    errors = [row["rmse"] for row in rows]
    residuals = [row["residual"] for row in rows]

    fig, ax1 = plt.subplots(figsize=(8.8, 5.2), dpi=150)
    fig.subplots_adjust(left=0.10, right=0.90, top=0.84, bottom=0.24)
    ax1.plot(sigmas, errors, "-o", color="#24536b", linewidth=2.2, label="RMSE")
    ax1.set_xlabel("Gaussian denoiser sigma")
    ax1.set_ylabel("RMSE", color="#24536b")
    ax1.tick_params(axis="y", labelcolor="#24536b")
    ax1.grid(True, alpha=0.22)

    ax2 = ax1.twinx()
    ax2.plot(sigmas, residuals, "-s", color="#9a5b2f", linewidth=2.2, label="data residual")
    ax2.set_ylabel("data residual", color="#9a5b2f")
    ax2.tick_params(axis="y", labelcolor="#9a5b2f")

    ax1.set_title("denoising strength tradeoff", fontsize=16, color="#1f4f63", pad=10)
    fig.text(
        0.5,
        0.055,
        "Stronger denoising can improve stability but may move farther from the measured data.",
        ha="center",
        fontsize=12.5,
        color="#202428",
    )
    save(fig, path)


def make_failure_mode(path: Path) -> None:
    shifted = distribution_shift_demo()
    clean = shifted["clean"]
    panels = [
        ("clean moon crop", clean),
        (f"blurred + noise\nRMSE={rmse(shifted['observation'], clean):.3f}", shifted["observation"]),
        (f"PnP Gaussian\nRMSE={rmse(shifted['pnp_gaussian'], clean):.3f}", shifted["pnp_gaussian"]),
        (f"camera-learned PnP\nRMSE={rmse(shifted['pnp_learned'], clean):.3f}", shifted["pnp_learned"]),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.4, 3.5), dpi=150)
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11.5, color="#1f4f63", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, 0.02, "A learned denoiser can be useful, but its training distribution matters.", ha="center", fontsize=12.5, color="#202428")
    save(fig, path)


if __name__ == "__main__":
    base = Path("assets/figures")
    make_pnp_loop(base / "week13-pnp-loop.png")
    make_prox_vs_denoiser(base / "week13-prox-vs-denoiser.png")
    make_deblurring_comparison(base / "week13-pnp-deblurring.png")
    make_iteration_curve(base / "week13-iteration-curve.png")
    make_strength_tradeoff(base / "week13-strength-tradeoff.png")
    make_failure_mode(base / "week13-failure-mode.png")
