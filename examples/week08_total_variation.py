"""Week 8 examples: total variation denoising and edge preservation."""

import numpy as np
from skimage import data
from skimage.restoration import denoise_tv_chambolle


def periodic_laplacian_eigenvalues(shape: tuple[int, ...]) -> np.ndarray:
    grids = np.meshgrid(
        *[2 * np.pi * np.fft.fftfreq(size) for size in shape],
        indexing="ij",
    )
    values = np.zeros(shape)
    for grid in grids:
        values += 2 - 2 * np.cos(grid)
    return values


def quadratic_smooth(noisy: np.ndarray, lam: float) -> np.ndarray:
    eig = periodic_laplacian_eigenvalues(noisy.shape)
    estimate = np.real(np.fft.ifftn(np.fft.fftn(noisy) / (1 + lam * eig)))
    return np.clip(estimate, 0.0, 1.0)


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    if image.ndim == 1:
        return np.abs(np.diff(image, append=image[-1]))
    vertical = np.diff(image, axis=0, append=image[-1:, :])
    horizontal = np.diff(image, axis=1, append=image[:, -1:])
    return np.sqrt(vertical**2 + horizontal**2)


def total_variation(image: np.ndarray) -> float:
    return float(np.sum(gradient_magnitude(image)))


def smoothness_energy(image: np.ndarray) -> float:
    gradient = gradient_magnitude(image)
    return float(np.sum(gradient**2))


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def main() -> None:
    rng = np.random.default_rng(43508)
    image = data.camera().astype(float)[80:336, 90:346] / 255.0
    noisy = np.clip(image + 0.10 * rng.standard_normal(image.shape), 0, 1)

    l2 = quadratic_smooth(noisy, lam=1.2)
    tv = denoise_tv_chambolle(noisy, weight=0.11)

    print("method       RMSE      TV norm    l2 gradient energy")
    for name, estimate in [
        ("noisy", noisy),
        ("l2", l2),
        ("TV", tv),
    ]:
        print(
            f"{name:8s}  {rmse(estimate, image):8.4f}"
            f"  {total_variation(estimate):9.1f}"
            f"  {smoothness_energy(estimate):18.1f}"
        )

    n = 220
    x = np.linspace(0, 1, n)
    clean = np.zeros(n)
    clean[x > 0.18] = 0.35
    clean[x > 0.45] = 0.9
    clean[x > 0.72] = 0.2
    noisy_signal = np.clip(clean + 0.13 * rng.standard_normal(n), 0, 1)
    l2_signal = quadratic_smooth(noisy_signal, 18.0)
    tv_signal = denoise_tv_chambolle(noisy_signal, weight=0.18)

    print()
    print("1D step signal RMSE")
    print("  noisy:", round(rmse(noisy_signal, clean), 4))
    print("  l2:", round(rmse(l2_signal, clean), 4))
    print("  TV:", round(rmse(tv_signal, clean), 4))


if __name__ == "__main__":
    main()
