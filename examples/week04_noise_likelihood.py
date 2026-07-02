"""Week 4 examples: Gaussian noise, Poisson noise, SNR, and likelihoods."""

import numpy as np
from skimage import data


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - reference) ** 2)))


def gaussian_negative_log_likelihood(x: np.ndarray, y: float, sigma: float) -> np.ndarray:
    return 0.5 * ((y - x) / sigma) ** 2


def poisson_negative_log_likelihood(x: np.ndarray, count: int, peak_photons: float) -> np.ndarray:
    mean_count = np.maximum(peak_photons * x, 1e-12)
    return mean_count - count * np.log(mean_count)


def main() -> None:
    rng = np.random.default_rng(43504)
    image = data.camera().astype(float)[80:336, 90:346] / 255.0

    sigma = 0.08
    gaussian_observation = np.clip(image + sigma * rng.standard_normal(image.shape), 0.0, 1.0)

    peak_photons = 35
    poisson_counts = rng.poisson(peak_photons * image)
    poisson_observation = poisson_counts / peak_photons

    print("Image crop shape:", image.shape)
    print("Gaussian sigma:", sigma)
    print("Gaussian observation RMSE:", round(rmse(gaussian_observation, image), 4))
    print("Poisson peak photons:", peak_photons)
    print("Poisson observation RMSE:", round(rmse(poisson_observation, image), 4))
    print()

    bright = image > 0.75
    dark = (image > 0.05) & (image < 0.2)
    print("Gaussian residual std, dark pixels:", round(float((gaussian_observation - image)[dark].std()), 4))
    print("Gaussian residual std, bright pixels:", round(float((gaussian_observation - image)[bright].std()), 4))
    print("Poisson count std, dark pixels:", round(float(poisson_counts[dark].std()), 4))
    print("Poisson count std, bright pixels:", round(float(poisson_counts[bright].std()), 4))
    print()

    candidates = np.linspace(0.01, 1.0, 500)
    y = 0.57
    gaussian_cost = gaussian_negative_log_likelihood(candidates, y, sigma)
    gaussian_minimizer = candidates[np.argmin(gaussian_cost)]

    observed_count = 20
    poisson_cost = poisson_negative_log_likelihood(candidates, observed_count, peak_photons)
    poisson_minimizer = candidates[np.argmin(poisson_cost)]

    print("Gaussian likelihood minimizer:", round(float(gaussian_minimizer), 3))
    print("Gaussian observed value:", y)
    print("Poisson likelihood minimizer:", round(float(poisson_minimizer), 3))
    print("Poisson count / peak photons:", round(observed_count / peak_photons, 3))


if __name__ == "__main__":
    main()
