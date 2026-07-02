"""Week 6 examples: Tikhonov regularization and parameter effects."""

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
        0.58 * np.sin(2 * np.pi * 3 * t)
        + 0.24 * np.sin(2 * np.pi * 14 * t)
        + 0.20 * (t > 0.43)
        - 0.18 * (t > 0.72)
    )
    return signal / np.max(np.abs(signal))


def tikhonov_solution(matrix: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n = matrix.shape[1]
    lhs = matrix.T @ matrix + lam * np.eye(n)
    rhs = matrix.T @ y
    return np.linalg.solve(lhs, rhs)


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def main() -> None:
    rng = np.random.default_rng(43506)
    n = 128
    matrix = convolution_matrix(n, sigma=2.8)
    clean = test_signal(n)
    blurred = matrix @ clean
    noisy = blurred + 0.018 * rng.standard_normal(n)

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    print("blur matrix condition number:", f"{singular_values[0] / singular_values[-1]:.3e}")
    print("normal matrix condition number:", f"{(singular_values[0] ** 2) / (singular_values[-1] ** 2):.3e}")
    print()

    lambdas = np.logspace(-8, 0, 17)
    rows = []
    for lam in lambdas:
        estimate = tikhonov_solution(matrix, noisy, lam)
        lhs_condition = (singular_values[0] ** 2 + lam) / (singular_values[-1] ** 2 + lam)
        rows.append((lam, rmse(estimate, clean), np.linalg.norm(matrix @ estimate - noisy), lhs_condition))

    best = min(rows, key=lambda row: row[1])
    print("lambda        RMSE      residual norm    cond(A^T A + lambda I)")
    for lam, error, residual, condition in rows:
        marker = " <-- best" if lam == best[0] else ""
        print(f"{lam:9.1e}  {error:8.4f}  {residual:13.4f}  {condition:21.3e}{marker}")


if __name__ == "__main__":
    main()
