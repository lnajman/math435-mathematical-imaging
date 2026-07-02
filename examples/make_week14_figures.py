"""Generate static figures for Week 14 slides."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib").resolve()))

import matplotlib.pyplot as plt
import numpy as np

from week14_stability_robustness import (
    data_residual,
    lambda_sensitivity,
    learned_prior_domain_test,
    make_deblurring_problem,
    perturbation_amplification,
    rmse,
    small_feature_case,
    tikhonov_deblur,
)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_reliability_loop(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.6), dpi=150)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        (0.50, 0.80, "reconstruction\nmethod", "#eef5f7"),
        (0.22, 0.56, "measurement\nperturbations", "#f7f1ea"),
        (0.50, 0.42, "robustness\nchecks", "#f5f7ee"),
        (0.78, 0.56, "domain-shift\ntests", "#f7f1ea"),
        (0.50, 0.16, "decision:\ntrust, revise, or reject", "#eef5f7"),
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

    for start, end in [
        ((0.43, 0.74), (0.28, 0.61)),
        ((0.57, 0.74), (0.72, 0.61)),
        ((0.30, 0.52), (0.42, 0.45)),
        ((0.70, 0.52), (0.58, 0.45)),
        ((0.50, 0.35), (0.50, 0.24)),
    ]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color="#202428", linewidth=1.8))

    ax.text(0.5, 0.94, "Reliability is tested, not assumed", ha="center", fontsize=18, color="#1f4f63", weight="bold")
    ax.text(0.5, 0.03, "A beautiful image is not enough: test stability, robustness, and consequences.", ha="center", fontsize=12.5, color="#202428")
    save(fig, path)


def make_lambda_sensitivity(path: Path) -> None:
    clean, observation, _, kernel_fft = make_deblurring_problem()
    rows = lambda_sensitivity()
    lambdas = np.asarray([row["lambda"] for row in rows])
    errors = np.asarray([row["rmse"] for row in rows])
    residuals = np.asarray([row["residual"] for row in rows])
    best_lambda = float(lambdas[int(np.argmin(errors))])

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=150)
    axes[0].plot(lambdas, errors, "-o", color="#24536b", linewidth=2.1, markersize=4, label="RMSE")
    axes[0].plot(lambdas, residuals, "-o", color="#9a5b2f", linewidth=2.1, markersize=4, label="data residual")
    axes[0].axvline(best_lambda, color="#202428", linestyle="--", linewidth=1.3, label="best RMSE")
    axes[0].set_xscale("log")
    axes[0].set_title("parameter sensitivity", fontsize=15, color="#1f4f63", pad=9)
    axes[0].set_xlabel("Tikhonov lambda")
    axes[0].set_ylabel("score")
    axes[0].grid(True, alpha=0.22)
    axes[0].legend()

    panel_lambdas = [1e-5, best_lambda, 2e-1]
    panel_titles = ["too weak", "near best", "too strong"]
    for index, (lam, title) in enumerate(zip(panel_lambdas, panel_titles)):
        estimate = tikhonov_deblur(observation, kernel_fft, lam)
        inset = axes[1].inset_axes([0.02 + 0.325 * index, 0.15, 0.30, 0.68])
        inset.imshow(estimate, cmap="gray", vmin=0, vmax=1)
        inset.set_title(f"{title}\n$\\lambda$={lam:.1e}", fontsize=10.5, color="#1f4f63")
        inset.set_xticks([])
        inset.set_yticks([])
    axes[1].axis("off")
    axes[1].set_title("visual effect", fontsize=15, color="#1f4f63", pad=9)
    fig.subplots_adjust(bottom=0.16)
    save(fig, path)


def make_perturbation_amplification(path: Path) -> None:
    rows = perturbation_amplification()
    lambdas = np.asarray([row["lambda"] for row in rows])
    gains = np.asarray([row["output_change"] / row["input_change"] for row in rows])

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=150)
    ax.plot(lambdas, gains, "-o", color="#9a5b2f", linewidth=2.2, markersize=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("tiny perturbation, large reconstruction change", fontsize=16, color="#1f4f63", pad=10)
    ax.set_xlabel("Tikhonov lambda")
    ax.set_ylabel("output-change / input-change")
    ax.grid(True, alpha=0.22, which="both")
    ax.text(0.03, 0.08, "weak regularization\ncan amplify invisible changes", transform=ax.transAxes, fontsize=12.5, color="#202428")
    save(fig, path)


def make_small_feature(path: Path) -> None:
    feature = small_feature_case()
    panels = [
        ("clean\ncontrast={:.3f}".format(feature["clean_contrast"]), feature["clean"]),
        ("blurred + noise", feature["observation"]),
        ("weak regularization\ncontrast={:.3f}".format(feature["weak_contrast"]), feature["weak"]),
        ("strong regularization\ncontrast={:.3f}".format(feature["strong_contrast"]), feature["strong"]),
        ("post-smoothed\ncontrast={:.3f}".format(feature["smoothed_contrast"]), feature["smoothed"]),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(13.4, 3.4), dpi=150)
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11.5, color="#1f4f63", pad=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.plot([48], [48], marker="o", color="#c7432b", markersize=4)
    fig.text(0.5, 0.02, "A small but important feature can be weakened by regularization or post-processing.", ha="center", fontsize=12.5, color="#202428")
    save(fig, path)


def make_domain_robustness(path: Path) -> None:
    rows = learned_prior_domain_test()
    domains = [row["domain"] for row in rows]
    noisy = [row["noisy_rmse"] for row in rows]
    gaussian = [row["gaussian_rmse"] for row in rows]
    learned = [row["learned_rmse"] for row in rows]
    x = np.arange(len(domains))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=150)
    ax.bar(x - width, noisy, width, label="noisy", color="#9ba7ad")
    ax.bar(x, gaussian, width, label="Gaussian", color="#24536b")
    ax.bar(x + width, learned, width, label="camera-learned PCA", color="#9a5b2f")
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.set_ylabel("RMSE")
    ax.set_title("robustness across image families", fontsize=16, color="#1f4f63", pad=10)
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend()
    fig.subplots_adjust(bottom=0.16)
    save(fig, path)


def make_ethics_axes(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 5.6), dpi=150)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.92, "Questions before deployment", ha="center", fontsize=18, color="#1f4f63", weight="bold")
    items = [
        (0.22, 0.68, "What data\ntrained it?", "#eef5f7"),
        (0.50, 0.68, "What failures\nwere tested?", "#f7f1ea"),
        (0.78, 0.68, "Who is harmed\nif it is wrong?", "#eef5f7"),
        (0.35, 0.36, "Can uncertainty\nbe communicated?", "#f5f7ee"),
        (0.65, 0.36, "Can the result be\naudited?", "#f5f7ee"),
    ]
    for x, y, text, color in items:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=13,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, edgecolor="#1f4f63", linewidth=1.5),
        )
    ax.text(0.5, 0.10, "Ethics is not separate from mathematics: modeling choices shape evidence.", ha="center", fontsize=13, color="#202428")
    save(fig, path)


if __name__ == "__main__":
    base = Path("assets/figures")
    make_reliability_loop(base / "week14-reliability-loop.png")
    make_lambda_sensitivity(base / "week14-lambda-sensitivity.png")
    make_perturbation_amplification(base / "week14-perturbation-amplification.png")
    make_small_feature(base / "week14-small-feature-risk.png")
    make_domain_robustness(base / "week14-domain-robustness.png")
    make_ethics_axes(base / "week14-ethics-questions.png")
