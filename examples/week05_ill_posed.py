"""Week 5 examples: ill-posed inverse problems and singular values."""

import numpy as np


def gaussian_kernel1d(radius: int, sigma: float) -> np.ndarray:
    axis = np.arange(-radius, radius + 1)
    kernel = np.exp(-(axis**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def convolution_matrix(n: int, sigma: float) -> np.ndarray:
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


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def truncated_svd_solution(matrix: np.ndarray, y: np.ndarray, keep: int) -> np.ndarray:
    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    coefficients = u.T @ y
    filtered = np.zeros_like(coefficients)
    filtered[:keep] = coefficients[:keep] / singular_values[:keep]
    return vh.T @ filtered


def main() -> None:
    rng = np.random.default_rng(43505)
    n = 128
    clean = test_signal(n)

    for sigma in [1.2, 2.4, 3.2]:
        matrix = convolution_matrix(n, sigma=sigma)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        condition_number = singular_values[0] / singular_values[-1]
        print(f"blur sigma={sigma}")
        print("  largest singular value:", round(float(singular_values[0]), 6))
        print("  smallest singular value:", f"{singular_values[-1]:.3e}")
        print("  condition number:", f"{condition_number:.3e}")

    matrix = convolution_matrix(n, sigma=3.2)
    blurred = matrix @ clean
    noisy = blurred + 0.012 * rng.standard_normal(n)

    pseudo_inverse = np.linalg.pinv(matrix, rcond=1e-13) @ noisy
    truncated_20 = truncated_svd_solution(matrix, noisy, keep=20)
    truncated_32 = truncated_svd_solution(matrix, noisy, keep=32)

    print()
    print("reconstruction RMSE")
    print("  blurred data:", round(rmse(blurred, clean), 4))
    print("  pseudo-inverse:", round(rmse(pseudo_inverse, clean), 4))
    print("  truncated SVD keep=20:", round(rmse(truncated_20, clean), 4))
    print("  truncated SVD keep=32:", round(rmse(truncated_32, clean), 4))


if __name__ == "__main__":
    main()
