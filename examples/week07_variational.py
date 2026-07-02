"""Week 7 examples: variational energies, optimality, and gradient descent."""

import numpy as np
from skimage import data


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


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def main() -> None:
    rng = np.random.default_rng(43507)
    image = data.camera().astype(float)[80:208, 90:218] / 255.0
    noisy = np.clip(image + 0.10 * rng.standard_normal(image.shape), 0, 1)
    lam = 1.5

    closed_form = quadratic_denoise(noisy, lam)
    residual = np.linalg.norm(gradient(closed_form, noisy, lam))

    print("noisy RMSE:", round(rmse(noisy, image), 4))
    print("closed-form quadratic denoising RMSE:", round(rmse(closed_form, image), 4))
    print("Euler-Lagrange residual norm:", f"{residual:.3e}")
    print()

    current = noisy.copy()
    step = 0.08
    current_iteration = 0
    print("gradient descent energy")
    for iteration in [0, 1, 2, 5, 10, 20, 40, 80]:
        while current_iteration < iteration:
            current = current - step * gradient(current, noisy, lam)
            current_iteration += 1
        energy = quadratic_energy(current, noisy, lam)
        error = rmse(current, image)
        print(f"  iter={iteration:2d}, energy={energy:10.4f}, RMSE={error:.4f}")


if __name__ == "__main__":
    main()
