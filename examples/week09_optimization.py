"""Week 9 examples: optimization methods for variational imaging."""

import numpy as np
from scipy.fft import dctn, idctn
from scipy.ndimage import gaussian_filter
from skimage import data


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def quadratic_energy(point: np.ndarray, matrix: np.ndarray, center: np.ndarray) -> float:
    diff = point - center
    return float(0.5 * diff @ matrix @ diff)


def gradient_descent(
    start: np.ndarray,
    matrix: np.ndarray,
    center: np.ndarray,
    step_size: float,
    iterations: int,
) -> list[float]:
    point = start.astype(float).copy()
    energies = [quadratic_energy(point, matrix, center)]
    for _ in range(iterations):
        gradient = matrix @ (point - center)
        point = point - step_size * gradient
        energies.append(quadratic_energy(point, matrix, center))
    return energies


def ista_dct_deblur(
    observed: np.ndarray,
    lam: float,
    step_size: float,
    iterations: int,
    blur_sigma: float,
) -> tuple[np.ndarray, list[float], list[int]]:
    coefficients = dctn(observed, norm="ortho")
    energies: list[float] = []
    active_counts: list[int] = []
    for _ in range(iterations):
        image = idctn(coefficients, norm="ortho")
        residual = gaussian_filter(image, sigma=blur_sigma, mode="reflect") - observed
        gradient_image = gaussian_filter(residual, sigma=blur_sigma, mode="reflect")
        gradient_coefficients = dctn(gradient_image, norm="ortho")
        coefficients = soft_threshold(coefficients - step_size * gradient_coefficients, step_size * lam)
        image = idctn(coefficients, norm="ortho")
        energies.append(
            0.5 * float(np.sum((gaussian_filter(image, sigma=blur_sigma, mode="reflect") - observed) ** 2))
            + lam * float(np.sum(np.abs(coefficients)))
        )
        active_counts.append(int(np.count_nonzero(np.abs(coefficients) > 1e-3)))
    return np.clip(idctn(coefficients, norm="ortho"), 0.0, 1.0), energies, active_counts


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def main() -> None:
    matrix = np.array([[1.0, 0.0], [0.0, 10.0]])
    center = np.array([0.9, -0.45])
    start = np.array([-1.7, 1.15])
    lipschitz = float(np.linalg.eigvalsh(matrix).max())

    print("Gradient descent on a convex quadratic")
    print(f"Lipschitz constant L = {lipschitz:.1f}; stable steps should be below 2/L = {2/lipschitz:.3f}")
    for label, step_size in [
        ("small", 0.4 / lipschitz),
        ("near 1/L", 1.0 / lipschitz),
        ("aggressive", 1.8 / lipschitz),
        ("too large", 2.1 / lipschitz),
    ]:
        energies = gradient_descent(start, matrix, center, step_size, iterations=25)
        trend = "decreased" if energies[-1] < energies[0] else "increased"
        print(f"  {label:10s} step={step_size:.3f}: E0={energies[0]:.4f}, E25={energies[-1]:.4f} ({trend})")

    print()
    print("Soft thresholding")
    values = np.array([-2.0, -0.7, -0.2, 0.0, 0.3, 0.9, 2.3])
    threshold = 0.6
    print("  input: ", values)
    print("  output:", soft_threshold(values, threshold))

    print()
    print("ISTA with sparse DCT regularization on a small image")
    rng = np.random.default_rng(43509)
    image = data.camera().astype(float)[96:224, 128:256] / 255.0
    observed = np.clip(gaussian_filter(image, sigma=1.1, mode="reflect") + 0.035 * rng.standard_normal(image.shape), 0, 1)
    estimate, energies, active_counts = ista_dct_deblur(observed, lam=0.0045, step_size=0.95, iterations=65, blur_sigma=1.1)
    print(f"  observed RMSE: {rmse(observed, image):.4f}")
    print(f"  ISTA RMSE:     {rmse(estimate, image):.4f}")
    print(f"  initial energy: {energies[0]:.2f}")
    print(f"  final energy:   {energies[-1]:.2f}")
    print(f"  active DCT coefficients: {active_counts[0]} -> {active_counts[-1]}")


if __name__ == "__main__":
    main()
