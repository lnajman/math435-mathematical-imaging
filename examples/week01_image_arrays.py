"""Week 1 examples: arrays, vectorization, and simple observation models."""

import numpy as np


def main() -> None:
    rows, cols = 4, 5
    image = np.arange(rows * cols).reshape(rows, cols)
    vector = image.reshape(-1)

    print("Image as an array:")
    print(image)
    print()

    print("Image as a vector:")
    print(vector)
    print()

    mask = vector % 3 != 0
    observations = vector[mask]
    print("Observed entries under a toy mask:")
    print(observations)


if __name__ == "__main__":
    main()

