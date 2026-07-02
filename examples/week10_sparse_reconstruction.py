"""Week 10 examples: sparse reconstruction and l1 regularization."""

import numpy as np
from scipy.fft import dctn, idctn
from skimage import data


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


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


def main() -> None:
    matrix, true_signal, measurements_vector = make_sparse_problem()
    l2_solution = minimum_l2_solution(matrix, measurements_vector)
    l1_solution, energies = ista_l1(matrix, measurements_vector, lam=0.015)

    print("Underdetermined sparse recovery")
    print(f"measurements: {matrix.shape[0]}")
    print(f"unknowns:      {matrix.shape[1]}")
    print(f"true nonzeros: {np.count_nonzero(true_signal)}")
    print()
    print("method    relative error   residual norm   active entries")
    for name, estimate in [
        ("l2", l2_solution),
        ("l1", l1_solution),
    ]:
        relative_error = np.linalg.norm(estimate - true_signal) / np.linalg.norm(true_signal)
        residual = np.linalg.norm(matrix @ estimate - measurements_vector)
        active = np.count_nonzero(np.abs(estimate) > 1e-2)
        print(f"{name:5s}    {relative_error:13.4f}   {residual:13.4e}   {active:14d}")

    print()
    print(f"l1 energy: {energies[0]:.4f} -> {energies[-1]:.4f}")

    rng = np.random.default_rng(43510)
    image = data.camera().astype(float)[96:160, 128:192] / 255.0
    mask = rng.random(image.shape) < 0.40
    observed = image * mask
    reconstruction, coefficients, image_energies = dct_sparse_inpaint(observed, mask, lam=0.008)

    print()
    print("DCT-sparse image reconstruction from random pixels")
    print(f"observed pixel fraction: {mask.mean():.2f}")
    print(f"zero-filled RMSE:        {rmse(observed, image):.4f}")
    print(f"sparse reconstruction RMSE: {rmse(reconstruction, image):.4f}")
    print(f"active DCT coefficients: {np.count_nonzero(np.abs(coefficients) > 1e-3)}")
    print(f"image energy: {image_energies[0]:.2f} -> {image_energies[-1]:.2f}")


if __name__ == "__main__":
    main()
