#!/usr/bin/env python3
"""Build student notebooks for MATH 435."""

import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


def make_notebook(cells: list[nbf.NotebookNode]) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    return notebook


def cell_key(cell: nbf.NotebookNode) -> tuple[str, str]:
    return (cell.cell_type, cell.source)


def stable_cell_id(name: str, index: int, cell: nbf.NotebookNode) -> str:
    text = f"{name}:{index}:{cell.cell_type}:{cell.source}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def preserve_cell_ids(path: Path, cells: list[nbf.NotebookNode]) -> None:
    old_ids: dict[tuple[str, str], list[str]] = {}
    if path.exists():
        old_notebook = nbf.read(path, as_version=4)
        for old_cell in old_notebook.cells:
            old_ids.setdefault(cell_key(old_cell), []).append(old_cell.get("id", ""))

    used_ids: set[str] = set()
    for index, cell in enumerate(cells):
        previous_ids = old_ids.get(cell_key(cell), [])
        candidate = previous_ids.pop(0) if previous_ids else stable_cell_id(path.name, index, cell)
        if not candidate:
            candidate = stable_cell_id(path.name, index, cell)
        while candidate in used_ids:
            candidate = hashlib.sha1(f"{candidate}:{index}".encode("utf-8")).hexdigest()[:8]
        cell["id"] = candidate
        used_ids.add(candidate)


def write_notebook(name: str, cells: list[nbf.NotebookNode]) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOK_DIR / name
    preserve_cell_ids(path, cells)
    nbf.write(make_notebook(cells), path)
    print(f"Wrote {path.relative_to(ROOT)}")


def week01_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Week 1 - Image Formation as Arrays

            This notebook accompanies the first MATH 435 slide deck.

            ## Goal

            By the end, you should be able to:

            - read an image as a numerical array;
            - inspect shape, intensity range, and individual pixels;
            - flatten an image into a vector and reshape it back;
            - model simple missing-pixel observations with a mask;
            - explain why masking recorded pixels is not the same as undoing scene occlusion.
            """
        ),
        md(
            """
            ## Setup

            The notebook uses NumPy, scikit-image, and Plotly.

            If you run this outside Google Colab and a package is missing, install the course requirements from the repository root:

            ```bash
            python3 -m pip install -r requirements.txt
            ```
            """
        ),
        code(
            """
            import numpy as np
            from skimage import data
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots


            def show_gray(image, title="", zmin=None, zmax=None, height=450, text_auto=False):
                fig = px.imshow(
                    image,
                    color_continuous_scale="gray",
                    origin="upper",
                    aspect="equal",
                    zmin=zmin,
                    zmax=zmax,
                    text_auto=text_auto,
                    binary_string=False,
                )
                fig.update_layout(
                    title=title,
                    height=height,
                    margin=dict(l=20, r=20, t=55, b=20),
                )
                fig.update_coloraxes(colorbar_title="intensity")
                fig.show()


            def show_image_grid(images, titles, zmin=0, zmax=1, height=430):
                fig = make_subplots(rows=1, cols=len(images), subplot_titles=titles)
                for index, image in enumerate(images, start=1):
                    fig.add_trace(
                        go.Heatmap(
                            z=image,
                            colorscale="Gray",
                            zmin=zmin,
                            zmax=zmax,
                            showscale=index == len(images),
                        ),
                        row=1,
                        col=index,
                    )
                    fig.update_xaxes(showticklabels=False, row=1, col=index)
                    fig.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=index)
                fig.update_layout(height=height, margin=dict(l=20, r=20, t=60, b=20))
                fig.show()
            """
        ),
        md(
            """
            ## Step 1 - A Tiny Image Array

            A grayscale image can be represented by a matrix of intensity values.
            """
        ),
        code(
            """
            small_image = np.arange(20).reshape(4, 5)

            print("shape:", small_image.shape)
            print(small_image)

            show_gray(
                small_image,
                title="A 4 by 5 image array",
                zmin=small_image.min(),
                zmax=small_image.max(),
                height=350,
                text_auto=True,
            )
            """
        ),
        md(
            """
            ### Exercise 1

            Change the number of rows and columns below. Before running the cell, predict:

            - the shape of the array;
            - the first row;
            - the last value.
            """
        ),
        code(
            """
            # TODO: change these values.
            rows = 6
            cols = 6

            exercise_image = np.arange(rows * cols).reshape(rows, cols)

            print("shape:", exercise_image.shape)
            print("first row:", exercise_image[0])
            print("last value:", exercise_image[-1, -1])
            show_gray(exercise_image, title="Exercise image", height=380, text_auto=True)
            """
        ),
        md(
            """
            ## Step 2 - Vectorizing an Image

            Many imaging models write the unknown image as a vector `x`.
            Vectorization does not change the information; it changes the shape.
            """
        ),
        code(
            """
            vector = small_image.reshape(-1)
            restored = vector.reshape(small_image.shape)

            print("vector shape:", vector.shape)
            print(vector)
            print("restored equals original:", np.array_equal(restored, small_image))
            """
        ),
        md(
            """
            ### Exercise 2

            For row-major vectorization, the pixel at row `r` and column `c` goes to index

            ```text
            r * number_of_columns + c
            ```

            Check this formula below.
            """
        ),
        code(
            """
            # TODO: change row and col, then compare the matrix value and vector value.
            row = 2
            col = 3

            flat_index = row * small_image.shape[1] + col

            print("matrix value:", small_image[row, col])
            print("flat index:", flat_index)
            print("vector value:", vector[flat_index])
            """
        ),
        md(
            """
            ## Step 3 - A Real Image

            We now load a real grayscale image from `skimage.data`.
            Intensities are scaled to the interval `[0, 1]`.
            """
        ),
        code(
            """
            image = data.camera().astype(float) / 255.0

            print("shape:", image.shape)
            print("dtype:", image.dtype)
            print("min:", image.min())
            print("max:", image.max())

            show_gray(image, title="Real grayscale image as an array", zmin=0, zmax=1, height=520)
            """
        ),
        md(
            """
            ## Step 4 - Looking at Pixels

            A small crop lets us inspect individual intensity values.
            """
        ),
        code(
            """
            # TODO: change r0, c0, and size to inspect another part of the image.
            r0 = 145
            c0 = 150
            size = 10

            crop = image[r0 : r0 + size, c0 : c0 + size]

            print("crop shape:", crop.shape)
            show_gray(
                np.round(crop, 2),
                title="Zoomed crop with intensity values",
                zmin=0,
                zmax=1,
                height=460,
                text_auto=".2f",
            )
            """
        ),
        md(
            """
            ## Step 5 - A Simple Observation Mask

            A mask can model which recorded pixels are available. Here we remove a rectangular region from the recorded image.
            """
        ),
        code(
            """
            mask = np.ones_like(image, dtype=bool)
            mask[120:310, 160:320] = False

            masked_observation = image.copy()
            masked_observation[~mask] = 0.0

            show_image_grid(
                [image, masked_observation],
                ["original image", "masked observation"],
                zmin=0,
                zmax=1,
                height=500,
            )

            flat_image = image.reshape(-1)
            flat_mask = mask.reshape(-1)
            observations = flat_image[flat_mask]

            print("unknown image length:", flat_image.size)
            print("observed pixel count:", observations.size)
            print("observed fraction:", round(float(flat_mask.mean()), 3))
            """
        ),
        md(
            """
            ### Exercise 3

            Change the mask rule below. Try observing every second, fourth, or tenth pixel.

            What changes in the observation vector?
            """
        ),
        code(
            """
            # TODO: change observed_every.
            observed_every = 4

            toy_vector = small_image.reshape(-1)
            toy_mask = np.arange(toy_vector.size) % observed_every != 0
            toy_observations = toy_vector[toy_mask]

            print("toy vector:", toy_vector)
            print("mask:", toy_mask.astype(int))
            print("observations:", toy_observations)
            print("observed fraction:", round(float(toy_mask.mean()), 3))
            """
        ),
        md(
            """
            ## Checkpoint

            Answer in your own words:

            1. What is the difference between an image array and an image vector?
            2. What information does `image.shape` give?
            3. What does a binary mask select?
            4. Why is a missing recorded pixel not the same as seeing behind an occluding object?
            """
        ),
    ]


def week02_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Week 2 - Convolution and Blur

            This notebook accompanies the second MATH 435 slide deck.

            ## Goal

            By the end, you should be able to:

            - compute a simple 1D convolution;
            - interpret a blur kernel as a point spread function;
            - apply 2D convolution to a real image;
            - compare boundary conditions;
            - see why naive deblurring amplifies noise.
            """
        ),
        md(
            """
            ## Setup

            The notebook uses NumPy, SciPy, scikit-image, and Plotly.

            If you run this outside Google Colab and a package is missing, install the course requirements from the repository root:

            ```bash
            python3 -m pip install -r requirements.txt
            ```
            """
        ),
        code(
            """
            import numpy as np
            from scipy import ndimage, signal
            from skimage import data
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots


            def gaussian_kernel(size, sigma):
                axis = np.arange(-(size // 2), size // 2 + 1)
                xx, yy = np.meshgrid(axis, axis)
                kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                return kernel / kernel.sum()


            def motion_kernel(size):
                kernel = np.zeros((size, size), dtype=float)
                kernel[size // 2, :] = 1.0
                return kernel / kernel.sum()


            def show_heatmaps(images, titles, colorscale="Gray", zmin=None, zmax=None, height=430):
                fig = make_subplots(rows=1, cols=len(images), subplot_titles=titles)
                for index, image in enumerate(images, start=1):
                    fig.add_trace(
                        go.Heatmap(
                            z=image,
                            colorscale=colorscale,
                            zmin=zmin,
                            zmax=zmax,
                            showscale=index == len(images),
                        ),
                        row=1,
                        col=index,
                    )
                    fig.update_xaxes(showticklabels=False, row=1, col=index)
                    fig.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=index)
                fig.update_layout(height=height, margin=dict(l=20, r=20, t=60, b=20))
                fig.show()


            def show_signal_panels(series, titles, height=360):
                fig = make_subplots(rows=1, cols=len(series), subplot_titles=titles)
                for index, values in enumerate(series, start=1):
                    x_values = np.arange(len(values))
                    fig.add_trace(go.Bar(x=x_values, y=values), row=1, col=index)
                    fig.update_xaxes(title_text="index", row=1, col=index)
                fig.update_layout(height=height, showlegend=False, margin=dict(l=20, r=20, t=60, b=45))
                fig.show()
            """
        ),
        md(
            """
            ## Step 1 - 1D Convolution by Hand

            Start with a short signal and a three-tap smoothing kernel.
            """
        ),
        code(
            """
            x = np.array([0, 0, 1, 3, 2, 0, 0], dtype=float)
            h = np.array([0.25, 0.5, 0.25])
            y = np.convolve(x, h, mode="same")

            print("x:", x)
            print("h:", h)
            print("y:", y)

            show_signal_panels(
                [x, h, y],
                ["signal x", "kernel h", "same-size convolution y"],
            )
            """
        ),
        md(
            """
            ### Exercise 1

            Compute the central output value by hand, then compare with Python.
            """
        ),
        code(
            """
            central_index = 3
            by_hand = 0.25 * x[central_index - 1] + 0.5 * x[central_index] + 0.25 * x[central_index + 1]

            print("by hand:", by_hand)
            print("from convolution:", y[central_index])
            """
        ),
        md(
            """
            ## Step 2 - Blur Kernels as Point Spread Functions

            A point spread function describes how one ideal bright point is spread by the imaging system.
            """
        ),
        code(
            """
            identity = np.zeros((7, 7))
            identity[3, 3] = 1.0

            box = np.ones((7, 7)) / 49
            gaussian = gaussian_kernel(7, 1.2)
            motion = motion_kernel(7)

            kernels = [identity, box, gaussian, motion]
            titles = ["identity", "box average", "gaussian", "motion"]

            show_heatmaps(kernels, titles, colorscale="Viridis", zmin=0, height=360)

            for title, kernel in zip(titles, kernels):
                print(title, "sum =", round(float(kernel.sum()), 6))
            """
        ),
        md(
            """
            ### Exercise 2

            Build a new kernel. Keep the sum equal to 1 if you want average brightness to be preserved.
            """
        ),
        code(
            """
            # TODO: change these values and rerun.
            kernel_size = 9
            sigma = 2.0

            student_kernel = gaussian_kernel(kernel_size, sigma)

            print("shape:", student_kernel.shape)
            print("sum:", student_kernel.sum())
            show_heatmaps([student_kernel], ["student gaussian kernel"], colorscale="Viridis", zmin=0, height=420)
            """
        ),
        md(
            """
            ## Step 3 - Blur a Real Image

            A fixed blur kernel gives a linear imaging operator.
            """
        ),
        code(
            """
            image = data.camera().astype(float) / 255.0
            image = image[80:400, 90:410]

            box13 = np.ones((13, 13)) / (13 * 13)
            gaussian21 = gaussian_kernel(21, 3.0)
            motion25 = motion_kernel(25)

            box_blur = signal.convolve2d(image, box13, mode="same", boundary="symm")
            gaussian_blur = signal.convolve2d(image, gaussian21, mode="same", boundary="symm")
            motion_blur = signal.convolve2d(image, motion25, mode="same", boundary="symm")

            show_heatmaps(
                [image, box_blur, gaussian_blur, motion_blur],
                ["original", "box blur", "gaussian blur", "motion blur"],
                zmin=0,
                zmax=1,
                height=430,
            )
            """
        ),
        md(
            """
            ### Exercise 3

            Change the blur parameters. Which details disappear first?
            """
        ),
        code(
            """
            # TODO: change kernel_size and sigma.
            kernel_size = 31
            sigma = 5.0

            experimental_kernel = gaussian_kernel(kernel_size, sigma)
            experimental_blur = signal.convolve2d(image, experimental_kernel, mode="same", boundary="symm")

            show_heatmaps(
                [image, experimental_blur],
                ["original", f"gaussian blur: size={kernel_size}, sigma={sigma}"],
                zmin=0,
                zmax=1,
                height=460,
            )
            """
        ),
        md(
            """
            ## Step 4 - Boundary Conditions

            Near the image boundary, the kernel asks for pixels outside the image. Boundary conditions decide what those pixels mean.
            """
        ),
        code(
            """
            boundary_crop = data.camera().astype(float)[90:218, 315:443] / 255.0
            boundary_kernel = np.ones((17, 17)) / (17 * 17)

            zero_padding = ndimage.convolve(boundary_crop, boundary_kernel, mode="constant", cval=0.0)
            reflect = ndimage.convolve(boundary_crop, boundary_kernel, mode="reflect")
            wrap = ndimage.convolve(boundary_crop, boundary_kernel, mode="wrap")

            show_heatmaps(
                [boundary_crop, zero_padding, reflect, wrap],
                ["original crop", "zero padding", "reflect", "wrap"],
                zmin=0,
                zmax=1,
                height=430,
            )
            """
        ),
        md(
            """
            ### Exercise 4

            Change `mode` to `"nearest"` or `"mirror"` in the cell below. Compare it with `"constant"`, `"reflect"`, and `"wrap"`.
            """
        ),
        code(
            """
            # TODO: try "nearest", "mirror", "constant", "reflect", or "wrap".
            boundary_mode = "nearest"

            boundary_result = ndimage.convolve(boundary_crop, boundary_kernel, mode=boundary_mode, cval=0.0)

            show_heatmaps(
                [boundary_crop, boundary_result],
                ["original crop", f"mode={boundary_mode}"],
                zmin=0,
                zmax=1,
                height=460,
            )
            """
        ),
        md(
            """
            ## Step 5 - Naive Deblurring Is Unstable

            A blur removes or attenuates high-frequency information. Trying to invert that attenuation can amplify noise.
            """
        ),
        code(
            """
            n = 256
            t = np.linspace(0, 1, n, endpoint=False)

            clean = np.sin(2 * np.pi * 6 * t) + 0.45 * np.sin(2 * np.pi * 34 * t)
            clean = clean / np.max(np.abs(clean))

            axis = np.arange(-18, 19)
            blur_kernel = np.exp(-(axis**2) / (2 * 3.0**2))
            blur_kernel = blur_kernel / blur_kernel.sum()

            blurred = np.convolve(clean, blur_kernel, mode="same")

            rng = np.random.default_rng(7)
            noise_level = 0.035
            noisy = blurred + noise_level * rng.standard_normal(n)

            padded_kernel = np.zeros(n)
            padded_kernel[: len(blur_kernel)] = blur_kernel
            padded_kernel = np.roll(padded_kernel, -(len(blur_kernel) // 2))

            H = np.fft.fft(padded_kernel)
            stabilizer = 1e-3
            naive_inverse = np.real(np.fft.ifft(np.fft.fft(noisy) / (H + stabilizer)))
            naive_inverse = naive_inverse / max(1.0, np.max(np.abs(naive_inverse)))

            fig = make_subplots(
                rows=1,
                cols=4,
                subplot_titles=["clean", "blurred", "blurred + noise", "naive inverse"],
            )
            for index, values in enumerate([clean, blurred, noisy, naive_inverse], start=1):
                fig.add_trace(go.Scatter(x=t, y=values, mode="lines"), row=1, col=index)
                fig.update_yaxes(range=[-1.25, 1.25], row=1, col=index)
            fig.update_layout(height=360, showlegend=False, margin=dict(l=20, r=20, t=60, b=35))
            fig.show()
            """
        ),
        md(
            """
            ### Exercise 5

            The stabilizer limits noise amplification. Try changing it by powers of ten.
            """
        ),
        code(
            """
            # TODO: add or remove stabilizer values.
            stabilizers = [1e-1, 1e-2, 1e-3]

            for value in stabilizers:
                estimate = np.real(np.fft.ifft(np.fft.fft(noisy) / (H + value)))
                estimate = estimate / max(1.0, np.max(np.abs(estimate)))
                rms_error = np.sqrt(np.mean((estimate - clean) ** 2))
                print(f"stabilizer={value:g}, RMS error={rms_error:.3f}")
            """
        ),
        md(
            """
            ## Checkpoint

            Answer in your own words:

            1. What does a normalized blur kernel preserve?
            2. Why is a point spread function a useful model?
            3. Which boundary mode would you choose for a natural photograph?
            4. Why does naive inverse filtering amplify noise?
            """
        ),
    ]


def week03_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Week 3 - Fourier Transform in Imaging

            This notebook accompanies the third MATH 435 slide deck.

            ## Goal

            By the end, you should be able to:

            - identify simple 1D frequency components;
            - display a 2D Fourier magnitude spectrum;
            - compare magnitude and phase information;
            - use the convolution theorem to explain blur;
            - experiment with low-pass, high-pass, and deblurring filters.
            """
        ),
        md(
            """
            ## Setup

            The notebook uses NumPy, SciPy, scikit-image, and Plotly.

            If you run this outside Google Colab and a package is missing, install the course requirements from the repository root:

            ```bash
            python3 -m pip install -r requirements.txt
            ```
            """
        ),
        code(
            """
            import numpy as np
            from scipy import ndimage
            from skimage import data
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots


            def normalize01(values):
                values = np.asarray(values, dtype=float)
                vmin = values.min()
                vmax = values.max()
                if vmax == vmin:
                    return np.zeros_like(values)
                return (values - vmin) / (vmax - vmin)


            def gaussian_kernel(size, sigma):
                axis = np.arange(-(size // 2), size // 2 + 1)
                xx, yy = np.meshgrid(axis, axis)
                kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                return kernel / kernel.sum()


            def fft_log_magnitude(image):
                spectrum = np.fft.fftshift(np.fft.fft2(image))
                return np.log1p(np.abs(spectrum))


            def centered_kernel_fft(kernel, shape):
                padded = np.zeros(shape)
                kh, kw = kernel.shape
                padded[:kh, :kw] = kernel
                padded = np.roll(padded, -(kh // 2), axis=0)
                padded = np.roll(padded, -(kw // 2), axis=1)
                return np.fft.fft2(padded)


            def show_heatmaps(images, titles, colorscales=None, zmin=None, zmax=None, height=430):
                if colorscales is None:
                    colorscales = ["Gray"] * len(images)
                fig = make_subplots(rows=1, cols=len(images), subplot_titles=titles)
                for index, (image, colorscale) in enumerate(zip(images, colorscales), start=1):
                    fig.add_trace(
                        go.Heatmap(
                            z=image,
                            colorscale=colorscale,
                            zmin=zmin,
                            zmax=zmax,
                            showscale=index == len(images),
                        ),
                        row=1,
                        col=index,
                    )
                    fig.update_xaxes(showticklabels=False, row=1, col=index)
                    fig.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=index)
                fig.update_layout(height=height, margin=dict(l=20, r=20, t=60, b=20))
                fig.show()


            def show_lines(series, titles, x_values=None, height=360):
                fig = make_subplots(rows=1, cols=len(series), subplot_titles=titles)
                for index, values in enumerate(series, start=1):
                    x_axis = np.arange(len(values)) if x_values is None else x_values
                    fig.add_trace(go.Scatter(x=x_axis, y=values, mode="lines"), row=1, col=index)
                    fig.update_xaxes(title_text="index", row=1, col=index)
                fig.update_layout(height=height, showlegend=False, margin=dict(l=20, r=20, t=60, b=45))
                fig.show()
            """
        ),
        md(
            """
            ## Step 1 - 1D Frequencies

            A Fourier transform measures which oscillations are present in a signal.
            """
        ),
        code(
            """
            n = 256
            t = np.arange(n) / n

            low = np.sin(2 * np.pi * 5 * t)
            high = 0.45 * np.sin(2 * np.pi * 23 * t)
            signal = low + high

            frequencies = np.fft.rfftfreq(n, d=1 / n)
            magnitude = np.abs(np.fft.rfft(signal))

            show_lines([low, high, signal], ["low frequency", "high frequency", "sum"], x_values=t)

            fig = go.Figure(go.Bar(x=frequencies, y=magnitude))
            fig.update_layout(
                title="Fourier magnitude",
                xaxis_title="frequency",
                yaxis_title="magnitude",
                xaxis_range=[0, 32],
                height=360,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()
            """
        ),
        md(
            """
            ### Exercise 1

            Change the two frequencies below. Before looking at the spectrum, predict where the largest peaks should be.
            """
        ),
        code(
            """
            # TODO: change these values.
            frequency_1 = 8
            frequency_2 = 31
            amplitude_2 = 0.35

            experiment = (
                np.sin(2 * np.pi * frequency_1 * t)
                + amplitude_2 * np.sin(2 * np.pi * frequency_2 * t)
            )
            experiment_magnitude = np.abs(np.fft.rfft(experiment))

            fig = make_subplots(rows=1, cols=2, subplot_titles=["signal", "Fourier magnitude"])
            fig.add_trace(go.Scatter(x=t, y=experiment, mode="lines"), row=1, col=1)
            fig.add_trace(go.Bar(x=frequencies, y=experiment_magnitude), row=1, col=2)
            fig.update_xaxes(range=[0, 40], row=1, col=2)
            fig.update_layout(height=360, showlegend=False, margin=dict(l=20, r=20, t=60, b=45))
            fig.show()
            """
        ),
        md(
            """
            ## Step 2 - 2D Fourier Transform of an Image

            Images have horizontal and vertical frequencies. We usually display the centered log magnitude.
            """
        ),
        code(
            """
            image = data.camera().astype(float) / 255.0
            image = image[80:336, 90:346]
            spectrum = fft_log_magnitude(image)

            print("image shape:", image.shape)
            print("spectrum shape:", spectrum.shape)

            show_heatmaps(
                [image, spectrum],
                ["image", "centered log Fourier magnitude"],
                colorscales=["Gray", "Magma"],
                height=460,
            )
            """
        ),
        md(
            """
            ### Exercise 2

            Inspect another crop. Do smooth regions and textured regions produce different spectra?
            """
        ),
        code(
            """
            # TODO: change the crop coordinates.
            row_start = 0
            col_start = 0
            crop_size = 192

            full_image = data.camera().astype(float) / 255.0
            crop = full_image[row_start : row_start + crop_size, col_start : col_start + crop_size]
            crop_spectrum = fft_log_magnitude(crop)

            show_heatmaps(
                [crop, crop_spectrum],
                ["selected crop", "crop spectrum"],
                colorscales=["Gray", "Magma"],
                height=460,
            )
            """
        ),
        md(
            """
            ## Step 3 - Magnitude and Phase

            A Fourier coefficient has magnitude and phase. Magnitude measures strength; phase measures alignment.
            """
        ),
        code(
            """
            F = np.fft.fft2(image)
            magnitude_image = np.abs(F)
            phase_image = np.angle(F)

            magnitude_only = normalize01(np.real(np.fft.ifft2(magnitude_image)))
            phase_only = normalize01(np.real(np.fft.ifft2(np.exp(1j * phase_image))))

            show_heatmaps(
                [image, np.log1p(np.fft.fftshift(magnitude_image)), magnitude_only, phase_only],
                ["original", "log magnitude", "magnitude only", "phase only"],
                colorscales=["Gray", "Magma", "Gray", "Gray"],
                height=430,
            )
            """
        ),
        md(
            """
            ### Exercise 3

            In one sentence: which reconstruction is more recognizable, magnitude-only or phase-only?
            """
        ),
        code(
            """
            answer = "TODO: write your answer here."
            print(answer)
            """
        ),
        md(
            """
            ## Step 4 - Convolution Theorem

            The convolution theorem says that convolution in image space becomes multiplication in Fourier space.
            """
        ),
        code(
            """
            blur_kernel = gaussian_kernel(25, 4.0)
            blurred = ndimage.convolve(image, blur_kernel, mode="reflect")
            kernel_response = np.fft.fftshift(np.abs(centered_kernel_fft(blur_kernel, image.shape)))

            show_heatmaps(
                [image, blur_kernel, blurred, kernel_response],
                ["image", "blur kernel", "blurred image", "|FFT(kernel)|"],
                colorscales=["Gray", "Viridis", "Gray", "Magma"],
                height=430,
            )
            """
        ),
        md(
            """
            ### Exercise 4

            Increase `sigma`. What happens to the blurred image and to the kernel frequency response?
            """
        ),
        code(
            """
            # TODO: change sigma.
            sigma = 6.0
            kernel_size = 31

            experiment_kernel = gaussian_kernel(kernel_size, sigma)
            experiment_blur = ndimage.convolve(image, experiment_kernel, mode="reflect")
            experiment_response = np.fft.fftshift(np.abs(centered_kernel_fft(experiment_kernel, image.shape)))

            show_heatmaps(
                [experiment_blur, experiment_response],
                [f"blurred: sigma={sigma}", "|FFT(kernel)|"],
                colorscales=["Gray", "Magma"],
                height=430,
            )
            """
        ),
        md(
            """
            ## Step 5 - Low-Pass and High-Pass Filtering

            A frequency mask can keep low frequencies or high frequencies.
            """
        ),
        code(
            """
            rows, cols = image.shape
            rr, cc = np.ogrid[:rows, :cols]
            center_r, center_c = rows // 2, cols // 2
            cutoff_radius = 28

            low_mask = (rr - center_r) ** 2 + (cc - center_c) ** 2 <= cutoff_radius**2
            high_mask = ~low_mask

            shifted = np.fft.fftshift(np.fft.fft2(image))
            low_pass = normalize01(np.real(np.fft.ifft2(np.fft.ifftshift(shifted * low_mask))))
            high_pass = normalize01(np.real(np.fft.ifft2(np.fft.ifftshift(shifted * high_mask))))

            show_heatmaps(
                [image, low_mask.astype(float), low_pass, high_pass],
                ["original", "low-pass mask", "low-pass image", "high-pass image"],
                colorscales=["Gray", "Magma", "Gray", "Gray"],
                height=430,
            )
            """
        ),
        md(
            """
            ### Exercise 5

            Change the cutoff radius. How does it affect smooth structures and edges?
            """
        ),
        code(
            """
            # TODO: change cutoff_radius.
            cutoff_radius = 12

            low_mask = (rr - center_r) ** 2 + (cc - center_c) ** 2 <= cutoff_radius**2
            low_pass = normalize01(np.real(np.fft.ifft2(np.fft.ifftshift(shifted * low_mask))))

            show_heatmaps(
                [low_mask.astype(float), low_pass],
                [f"low-pass mask: radius={cutoff_radius}", "filtered image"],
                colorscales=["Magma", "Gray"],
                height=430,
            )
            """
        ),
        md(
            """
            ## Step 6 - Deblurring and Noise Amplification

            In frequency space, naive deblurring divides by the blur response. If the response is small, noise can explode.
            """
        ),
        code(
            """
            deblur_image = data.camera().astype(float)[120:376, 120:376] / 255.0
            deblur_kernel = gaussian_kernel(21, 3.0)
            deblur_blurred = ndimage.convolve(deblur_image, deblur_kernel, mode="reflect")

            rng = np.random.default_rng(12)
            deblur_noisy = np.clip(deblur_blurred + 0.015 * rng.standard_normal(deblur_image.shape), 0, 1)

            H = centered_kernel_fft(deblur_kernel, deblur_image.shape)
            Y = np.fft.fft2(deblur_noisy)

            naive = normalize01(np.real(np.fft.ifft2(Y / (H + 1e-4))))
            regularization = 5e-3
            regularized = normalize01(np.real(np.fft.ifft2(Y * np.conj(H) / (np.abs(H) ** 2 + regularization))))

            show_heatmaps(
                [deblur_image, deblur_noisy, naive, regularized],
                ["original", "blurred + noise", "naive inverse", "regularized inverse"],
                colorscales=["Gray", "Gray", "Gray", "Gray"],
                height=430,
            )

            print("naive RMS error:", np.sqrt(np.mean((naive - deblur_image) ** 2)))
            print("regularized RMS error:", np.sqrt(np.mean((regularized - deblur_image) ** 2)))
            """
        ),
        md(
            """
            ### Exercise 6

            Change the regularization parameter. What is the tradeoff between sharpness and noise?
            """
        ),
        code(
            """
            # TODO: add or remove values.
            regularization_values = [1e-2, 5e-3, 1e-3]

            for value in regularization_values:
                estimate = normalize01(np.real(np.fft.ifft2(Y * np.conj(H) / (np.abs(H) ** 2 + value))))
                rms_error = np.sqrt(np.mean((estimate - deblur_image) ** 2))
                print(f"regularization={value:g}, RMS error={rms_error:.3f}")
            """
        ),
        md(
            """
            ## Checkpoint

            Answer in your own words:

            1. What does the Fourier magnitude measure?
            2. Where are low frequencies after `fftshift`?
            3. What does the convolution theorem say?
            4. Why is deblurring unstable when the blur response is small?
            """
        ),
    ]


def main() -> None:
    write_notebook("week01_image_formation.ipynb", week01_cells())
    write_notebook("week02_convolution_blur.ipynb", week02_cells())
    write_notebook("week03_fourier_imaging.ipynb", week03_cells())


if __name__ == "__main__":
    main()
