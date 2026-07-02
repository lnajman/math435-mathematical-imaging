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


def week04_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Week 4 - Noise Models and Likelihoods

            This notebook accompanies the fourth MATH 435 slide deck.

            ## Goal

            By the end, you should be able to:

            - simulate Gaussian and Poisson noise on an image;
            - compare constant-variance and signal-dependent noise;
            - compute simple SNR quantities;
            - plot Gaussian and Poisson negative log-likelihoods;
            - explain why a likelihood becomes a data-fidelity term.
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
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots


            def normalize01(values):
                values = np.asarray(values, dtype=float)
                vmin = values.min()
                vmax = values.max()
                if vmax == vmin:
                    return np.zeros_like(values)
                return (values - vmin) / (vmax - vmin)


            def rmse(estimate, reference):
                return float(np.sqrt(np.mean((estimate - reference) ** 2)))


            def add_gaussian_noise(image, sigma, rng):
                return np.clip(image + sigma * rng.standard_normal(image.shape), 0.0, 1.0)


            def add_poisson_noise(image, peak_photons, rng):
                counts = rng.poisson(peak_photons * image)
                return np.clip(counts / peak_photons, 0.0, 1.0), counts


            def show_heatmaps(images, titles, colorscales=None, zmin=0, zmax=1, height=430):
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


            def gaussian_nll(candidate_x, observed_y, sigma):
                return 0.5 * ((observed_y - candidate_x) / sigma) ** 2


            def poisson_nll(candidate_x, observed_count, peak_photons):
                mean_count = np.maximum(peak_photons * candidate_x, 1e-12)
                return mean_count - observed_count * np.log(mean_count)
            """
        ),
        md(
            """
            ## Steps

            ### 1. Load an Image and Add Noise

            We start with a clean grayscale image. Then we create one Gaussian observation and two Poisson observations.
            """
        ),
        code(
            """
            rng = np.random.default_rng(43504)

            image = data.camera().astype(float) / 255.0
            image = image[80:336, 90:346]

            gaussian_sigma = 0.08
            poisson_low_photons = 18
            poisson_high_photons = 120

            gaussian_noisy = add_gaussian_noise(image, gaussian_sigma, rng)
            poisson_low, poisson_low_counts = add_poisson_noise(image, poisson_low_photons, rng)
            poisson_high, poisson_high_counts = add_poisson_noise(image, poisson_high_photons, rng)

            show_heatmaps(
                [image, gaussian_noisy, poisson_low, poisson_high],
                [
                    "clean image",
                    f"Gaussian sigma={gaussian_sigma}",
                    f"Poisson peak={poisson_low_photons}",
                    f"Poisson peak={poisson_high_photons}",
                ],
                height=430,
            )

            print("Gaussian RMSE:", round(rmse(gaussian_noisy, image), 4))
            print("Poisson low-light RMSE:", round(rmse(poisson_low, image), 4))
            print("Poisson higher-photon RMSE:", round(rmse(poisson_high, image), 4))
            """
        ),
        md(
            """
            ### Exercise 1

            Change `gaussian_sigma`, `poisson_low_photons`, and `poisson_high_photons`.

            Before rerunning, predict which image will become noisier and why.
            """
        ),
        code(
            """
            # TODO: change these values.
            gaussian_sigma = 0.04
            peak_photons = 35

            gaussian_experiment = add_gaussian_noise(image, gaussian_sigma, rng)
            poisson_experiment, poisson_experiment_counts = add_poisson_noise(image, peak_photons, rng)

            show_heatmaps(
                [image, gaussian_experiment, poisson_experiment],
                ["clean", f"Gaussian sigma={gaussian_sigma}", f"Poisson peak={peak_photons}"],
                height=430,
            )

            print("Gaussian RMSE:", round(rmse(gaussian_experiment, image), 4))
            print("Poisson RMSE:", round(rmse(poisson_experiment, image), 4))
            """
        ),
        md(
            """
            ### 2. Gaussian Noise Has Constant Variance

            In the additive Gaussian model,

            ```text
            y_i = x_i + eta_i,   eta_i ~ N(0, sigma^2)
            ```

            the variance of the residual does not depend on the true intensity.
            """
        ),
        code(
            """
            residual = gaussian_noisy - image
            dark_pixels = (image > 0.05) & (image < 0.20)
            bright_pixels = image > 0.75

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=residual.ravel(),
                    nbinsx=80,
                    histnorm="probability density",
                    marker_color="#24536b",
                    name="all residuals",
                )
            )
            fig.update_layout(
                title="Gaussian residual histogram",
                xaxis_title="observed - true intensity",
                yaxis_title="density",
                height=380,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()

            print("residual mean:", round(float(residual.mean()), 4))
            print("residual std, dark pixels:", round(float(residual[dark_pixels].std()), 4))
            print("residual std, bright pixels:", round(float(residual[bright_pixels].std()), 4))
            """
        ),
        md(
            """
            ### Exercise 2

            Why are the two standard deviations close? What would make them different in a real camera?
            """
        ),
        code(
            """
            answer = "TODO: write your explanation here."
            print(answer)
            """
        ),
        md(
            """
            ### 3. Poisson Noise Depends on the Signal

            For a photon count,

            ```text
            z_i ~ Poisson(lambda_i)
            ```

            the mean and variance are both `lambda_i`.
            """
        ),
        code(
            """
            true_intensities = np.array([0.15, 0.45, 0.80])
            peak_photons = 50
            count_samples = rng.poisson(peak_photons * true_intensities[:, None], size=(3, 8000))

            fig = go.Figure()
            colors = ["#24536b", "#9a5b2f", "#2f7a54"]
            for intensity, samples, color in zip(true_intensities, count_samples, colors):
                fig.add_trace(
                    go.Histogram(
                        x=samples,
                        nbinsx=45,
                        histnorm="probability",
                        opacity=0.55,
                        marker_color=color,
                        name=f"true intensity={intensity}",
                    )
                )
            fig.update_layout(
                title="Poisson count distributions",
                xaxis_title="photon count",
                yaxis_title="probability",
                barmode="overlay",
                height=390,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()

            for intensity, samples in zip(true_intensities, count_samples):
                print(
                    f"intensity={intensity:.2f}",
                    "sample mean=", round(float(samples.mean()), 2),
                    "sample variance=", round(float(samples.var()), 2),
                )
            """
        ),
        md(
            """
            ### Exercise 3

            Change `peak_photons`. What happens to the spread of the count distributions?
            """
        ),
        code(
            """
            # TODO: change this value.
            peak_photons = 12
            intensity = 0.50

            samples = rng.poisson(peak_photons * intensity, size=8000)

            fig = go.Figure(
                go.Histogram(
                    x=samples,
                    nbinsx=35,
                    histnorm="probability",
                    marker_color="#24536b",
                )
            )
            fig.update_layout(
                title=f"Poisson counts: intensity={intensity}, peak={peak_photons}",
                xaxis_title="count",
                yaxis_title="probability",
                height=360,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()

            print("sample mean:", round(float(samples.mean()), 3))
            print("sample variance:", round(float(samples.var()), 3))
            print("sample SNR:", round(float(samples.mean() / samples.std()), 3))
            """
        ),
        md(
            """
            ### 4. Signal-to-Noise Ratio

            Signal-to-noise ratio compares the signal size with the typical noise size.

            For Gaussian noise, a simple pixelwise SNR is `x / sigma`.

            For Poisson counts, the count SNR is approximately `sqrt(lambda)`.
            """
        ),
        code(
            """
            intensities = np.linspace(0.02, 1.0, 200)
            sigma = 0.08
            gaussian_snr = intensities / sigma
            photon_levels = [12, 40, 140]

            fig = make_subplots(rows=1, cols=2, subplot_titles=["Gaussian", "Poisson"])
            fig.add_trace(
                go.Scatter(x=intensities, y=gaussian_snr, mode="lines", name="sigma=0.08"),
                row=1,
                col=1,
            )
            for peak in photon_levels:
                fig.add_trace(
                    go.Scatter(
                        x=intensities,
                        y=np.sqrt(peak * intensities),
                        mode="lines",
                        name=f"peak={peak}",
                    ),
                    row=1,
                    col=2,
                )
            fig.update_xaxes(title_text="true intensity", row=1, col=1)
            fig.update_xaxes(title_text="true intensity", row=1, col=2)
            fig.update_yaxes(title_text="SNR", row=1, col=1)
            fig.update_yaxes(title_text="SNR", row=1, col=2)
            fig.update_layout(height=390, margin=dict(l=20, r=20, t=60, b=45))
            fig.show()
            """
        ),
        md(
            """
            ### Exercise 4

            Suppose a pixel has expected photon count `lambda=25`.

            Compute its standard deviation and SNR. Then repeat for `lambda=100`.
            """
        ),
        code(
            """
            for lam in [25, 100]:
                standard_deviation = np.sqrt(lam)
                snr = lam / standard_deviation
                print(f"lambda={lam}: std={standard_deviation:.2f}, SNR={snr:.2f}")
            """
        ),
        md(
            """
            ### 5. Likelihoods Become Data-Fidelity Terms

            A likelihood asks: if the true value were `x`, how plausible would the observation be?

            Minimizing a negative log-likelihood gives a data-fidelity term.
            """
        ),
        code(
            """
            candidate_x = np.linspace(0.01, 1.0, 400)

            observed_y = 0.57
            sigma = 0.09
            gaussian_cost = gaussian_nll(candidate_x, observed_y, sigma)
            gaussian_cost = gaussian_cost - gaussian_cost.min()

            observed_count = 20
            peak_photons = 35
            poisson_cost = poisson_nll(candidate_x, observed_count, peak_photons)
            poisson_cost = poisson_cost - poisson_cost.min()

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["Gaussian negative log-likelihood", "Poisson negative log-likelihood"],
            )
            fig.add_trace(go.Scatter(x=candidate_x, y=gaussian_cost, mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=candidate_x, y=poisson_cost, mode="lines"), row=1, col=2)
            fig.update_xaxes(title_text="candidate true intensity", row=1, col=1)
            fig.update_xaxes(title_text="candidate true intensity", row=1, col=2)
            fig.update_yaxes(title_text="relative cost", row=1, col=1)
            fig.update_yaxes(title_text="relative cost", row=1, col=2)
            fig.update_layout(height=390, showlegend=False, margin=dict(l=20, r=20, t=60, b=45))
            fig.show()

            print("Gaussian minimizer:", round(float(candidate_x[np.argmin(gaussian_cost)]), 3))
            print("observed_y:", observed_y)
            print("Poisson minimizer:", round(float(candidate_x[np.argmin(poisson_cost)]), 3))
            print("observed_count / peak_photons:", round(observed_count / peak_photons, 3))
            """
        ),
        md(
            """
            ### Exercise 5

            Change `observed_count` and `peak_photons`.

            What happens to the minimizer of the Poisson cost?
            """
        ),
        code(
            """
            # TODO: change these values.
            observed_count = 8
            peak_photons = 20

            poisson_cost = poisson_nll(candidate_x, observed_count, peak_photons)
            poisson_cost = poisson_cost - poisson_cost.min()

            fig = go.Figure(go.Scatter(x=candidate_x, y=poisson_cost, mode="lines"))
            fig.update_layout(
                title="Poisson negative log-likelihood",
                xaxis_title="candidate true intensity",
                yaxis_title="relative cost",
                height=360,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()

            print("minimizer:", round(float(candidate_x[np.argmin(poisson_cost)]), 3))
            print("count / peak:", round(observed_count / peak_photons, 3))
            """
        ),
        md(
            """
            ### 6. Averaging Repeated Gaussian Measurements

            If we have independent observations of the same image, averaging reduces Gaussian noise.

            The standard deviation decreases like `1 / sqrt(number of observations)`.
            """
        ),
        code(
            """
            sigma = 0.12
            repeated = np.array([add_gaussian_noise(image, sigma, rng) for _ in range(16)])
            counts = [1, 2, 4, 8, 16]
            averaged_images = [repeated[:count].mean(axis=0) for count in counts]
            errors = [rmse(average, image) for average in averaged_images]

            show_heatmaps(
                [image] + averaged_images[:4],
                ["clean"] + [f"average {count}" for count in counts[:4]],
                height=420,
            )

            fig = go.Figure(go.Scatter(x=counts, y=errors, mode="lines+markers"))
            fig.update_layout(
                title="RMSE after averaging repeated noisy measurements",
                xaxis_title="number of measurements",
                yaxis_title="RMSE",
                height=360,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()

            for count, error in zip(counts, errors):
                print(f"average {count:2d}: RMSE={error:.4f}")
            """
        ),
        md(
            """
            ### Exercise 6

            Averaging helps with random noise.

            Give one example of an imaging failure that averaging cannot fix.
            """
        ),
        code(
            """
            answer = "TODO: write your example here."
            print(answer)
            """
        ),
        md(
            """
            ## Checks

            Answer in your own words:

            1. Why does Gaussian noise lead to a squared-error data term?
            2. Why is Poisson noise signal-dependent?
            3. What happens to SNR when photon count increases?
            4. Why does regularization still matter after choosing a likelihood?
            """
        ),
        md(
            """
            ## Next Steps

            In the next class, we will combine data-fidelity terms with priors or regularizers:

            ```text
            minimize data_fit(y, f(x)) + lambda * regularizer(x)
            ```

            Week 4 gives the statistical meaning of the `data_fit` part.
            """
        ),
    ]


def week05_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Week 5 - Ill-Posed Inverse Problems

            This notebook accompanies the fifth MATH 435 slide deck.

            ## Goal

            By the end, you should be able to:

            - explain Hadamard's three conditions for well-posedness;
            - construct a simple non-uniqueness example;
            - inspect singular values of blur operators;
            - see how small singular values amplify noise;
            - experiment with truncated SVD as a first stabilization idea.
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
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots


            def normalize01(values):
                values = np.asarray(values, dtype=float)
                vmin = values.min()
                vmax = values.max()
                if vmax == vmin:
                    return np.zeros_like(values)
                return (values - vmin) / (vmax - vmin)


            def rmse(estimate, reference):
                return float(np.sqrt(np.mean((estimate - reference) ** 2)))


            def gaussian_kernel1d(radius, sigma):
                axis = np.arange(-radius, radius + 1)
                kernel = np.exp(-(axis**2) / (2 * sigma**2))
                return kernel / kernel.sum()


            def convolution_matrix(n, sigma, radius=None):
                if radius is None:
                    radius = max(3, int(np.ceil(4 * sigma)))
                kernel = gaussian_kernel1d(radius, sigma)
                matrix = np.zeros((n, n))
                for col in range(n):
                    impulse = np.zeros(n)
                    impulse[col] = 1.0
                    matrix[:, col] = np.convolve(impulse, kernel, mode="same")
                return matrix


            def test_signal(n):
                t = np.linspace(0, 1, n, endpoint=False)
                signal = (
                    0.55 * np.sin(2 * np.pi * 3 * t)
                    + 0.28 * np.sin(2 * np.pi * 13 * t)
                    + 0.22 * (t > 0.45)
                    - 0.18 * (t > 0.72)
                )
                return signal / np.max(np.abs(signal))


            def truncated_svd_solution(matrix, y, keep):
                u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
                coefficients = u.T @ y
                filtered = np.zeros_like(coefficients)
                filtered[:keep] = coefficients[:keep] / singular_values[:keep]
                return vh.T @ filtered


            def show_heatmaps(images, titles, colorscales=None, zmin=0, zmax=1, height=430):
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
            """
        ),
        md(
            """
            ## Steps

            ### 1. Hadamard's Checklist

            An inverse problem is well posed when it has:

            1. existence: at least one solution;
            2. uniqueness: at most one solution;
            3. stability: small data errors produce small solution errors.

            Imaging inverse problems often fail uniqueness or stability.
            """
        ),
        code(
            """
            hadamard = {
                "existence": "at least one solution matches the data",
                "uniqueness": "only one solution matches the data",
                "stability": "small data changes cause small solution changes",
            }

            for name, meaning in hadamard.items():
                print(f"{name:10s}: {meaning}")
            """
        ),
        md(
            """
            ### 2. Non-Uniqueness from Missing Pixels

            If an operator records only some pixels, different hidden regions can produce the same observed data.

            This is a linear projection of an already formed image. It is not the same as a full physical model of scene occlusion.
            """
        ),
        code(
            """
            image = data.camera().astype(float) / 255.0
            image = image[70:326, 100:356]

            texture = data.coins().astype(float) / 255.0
            texture = normalize01(texture[30:286, 30:286])

            mask = np.ones_like(image, dtype=bool)
            mask[80:176, 92:180] = False

            candidate_1 = image.copy()
            candidate_2 = image.copy()
            candidate_2[~mask] = texture[~mask]

            observation_1 = candidate_1[mask]
            observation_2 = candidate_2[mask]

            observed_display = image.copy()
            observed_display[~mask] = 1.0

            show_heatmaps(
                [candidate_1, candidate_2, observed_display, mask.astype(float)],
                ["candidate 1", "candidate 2", "observed pixels", "mask"],
                height=430,
            )

            print("observations equal:", np.array_equal(observation_1, observation_2))
            print("observed entries:", observation_1.size)
            print("missing entries:", np.size(mask) - observation_1.size)
            """
        ),
        md(
            """
            ### Exercise 1

            Change the missing region in the mask.

            Does the equality of the two observation vectors still hold?
            """
        ),
        code(
            """
            # TODO: change these slice coordinates.
            row_start = 70
            row_stop = 155
            col_start = 70
            col_stop = 165

            experiment_mask = np.ones_like(image, dtype=bool)
            experiment_mask[row_start:row_stop, col_start:col_stop] = False

            experiment_1 = candidate_1[experiment_mask]
            experiment_2 = candidate_2[experiment_mask]

            print("observations equal:", np.array_equal(experiment_1, experiment_2))
            print("observed fraction:", round(float(experiment_mask.mean()), 3))
            """
        ),
        md(
            """
            ### 3. One-Dimensional Stability Warning

            The scalar equation `y = epsilon * x` has inverse `x = y / epsilon`.

            If `epsilon` is tiny, data errors are magnified.
            """
        ),
        code(
            """
            epsilons = np.array([1, 0.1, 0.01, 0.001])
            data_error = 1e-4

            for epsilon in epsilons:
                solution_error = data_error / epsilon
                print(f"epsilon={epsilon:g}, data error={data_error:g}, solution error={solution_error:g}")
            """
        ),
        md(
            """
            ### Exercise 2

            Change `data_error`. Which values of `epsilon` make the inverse unreliable?
            """
        ),
        code(
            """
            # TODO: change this value.
            data_error = 5e-5

            fig = go.Figure(
                go.Scatter(
                    x=epsilons,
                    y=data_error / epsilons,
                    mode="lines+markers",
                )
            )
            fig.update_layout(
                title="Error magnification by division",
                xaxis_title="epsilon",
                yaxis_title="solution error",
                xaxis_type="log",
                yaxis_type="log",
                height=360,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()
            """
        ),
        md(
            """
            ### 4. Singular Values of Blur

            Singular values tell us which directions are strongly or weakly transmitted by a linear operator.

            For a blur matrix, stronger blur creates smaller singular values.
            """
        ),
        code(
            """
            n = 90
            blur_sigmas = [1.2, 2.4, 4.8]

            fig = go.Figure()
            for sigma in blur_sigmas:
                matrix = convolution_matrix(n, sigma)
                singular_values = np.linalg.svd(matrix, compute_uv=False)
                fig.add_trace(
                    go.Scatter(
                        y=singular_values,
                        mode="lines",
                        name=f"sigma={sigma}",
                    )
                )
                print(
                    f"sigma={sigma}",
                    "condition number=",
                    f"{singular_values[0] / singular_values[-1]:.3e}",
                )

            fig.update_layout(
                title="Singular values of blur matrices",
                xaxis_title="index",
                yaxis_title="singular value",
                yaxis_type="log",
                height=420,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()
            """
        ),
        md(
            """
            ### Exercise 3

            Add a new blur strength to `blur_sigmas`.

            What happens to the singular-value decay and the condition number?
            """
        ),
        code(
            """
            # TODO: change this value.
            sigma = 3.2

            matrix = convolution_matrix(n, sigma)
            singular_values = np.linalg.svd(matrix, compute_uv=False)

            print("largest singular value:", round(float(singular_values[0]), 6))
            print("smallest singular value:", f"{singular_values[-1]:.3e}")
            print("condition number:", f"{singular_values[0] / singular_values[-1]:.3e}")
            """
        ),
        md(
            """
            ### 5. Noise Amplification by an Inverse

            We now blur a signal, add a very small amount of noise, and compare direct inversion with truncated SVD.
            """
        ),
        code(
            """
            rng = np.random.default_rng(43505)
            n = 128
            clean = test_signal(n)

            blur_sigma = 2.4
            noise_level = 1e-5
            keep = 32

            matrix = convolution_matrix(n, blur_sigma)
            blurred = matrix @ clean
            noisy = blurred + noise_level * rng.standard_normal(n)

            unstable = np.linalg.pinv(matrix, rcond=1e-8) @ noisy
            truncated = truncated_svd_solution(matrix, noisy, keep=keep)

            x_axis = np.arange(n)
            fig = make_subplots(
                rows=1,
                cols=4,
                subplot_titles=["true signal", "blurred + noise", "unstable inverse", "truncated SVD"],
            )
            for index, values in enumerate([clean, noisy, unstable, truncated], start=1):
                fig.add_trace(go.Scatter(x=x_axis, y=values, mode="lines"), row=1, col=index)
                fig.update_xaxes(showticklabels=False, row=1, col=index)
            fig.update_yaxes(range=[-1.5, 1.5], row=1, col=1)
            fig.update_yaxes(range=[-1.5, 1.5], row=1, col=2)
            fig.update_yaxes(range=[-8, 8], row=1, col=3)
            fig.update_yaxes(range=[-1.5, 1.5], row=1, col=4)
            fig.update_layout(height=390, showlegend=False, margin=dict(l=20, r=20, t=60, b=35))
            fig.show()

            print("blurred RMSE:", round(rmse(blurred, clean), 4))
            print("unstable inverse RMSE:", round(rmse(unstable, clean), 4))
            print("truncated SVD RMSE:", round(rmse(truncated, clean), 4))
            """
        ),
        md(
            """
            ### Exercise 4

            Change `noise_level` and `keep`.

            What happens if you keep too few singular values? What happens if you keep too many?
            """
        ),
        code(
            """
            # TODO: change these values.
            noise_level = 1e-4
            keep_values = [12, 24, 36, 60]

            rng = np.random.default_rng(43505)
            noisy = blurred + noise_level * rng.standard_normal(n)

            for keep in keep_values:
                estimate = truncated_svd_solution(matrix, noisy, keep=keep)
                print(f"keep={keep:2d}, RMSE={rmse(estimate, clean):.4f}")
            """
        ),
        md(
            """
            ### 6. SVD Coefficients

            In SVD coordinates, inversion divides by singular values.

            Noise in a direction with a small singular value can become dominant.
            """
        ),
        code(
            """
            rng = np.random.default_rng(43505)
            matrix = convolution_matrix(128, 3.2)
            clean = test_signal(128)
            blurred = matrix @ clean
            noise = 0.012 * rng.standard_normal(128)
            noisy = blurred + noise

            u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
            clean_coefficients = np.abs(u.T @ blurred)
            noisy_coefficients = np.abs(u.T @ noisy)
            amplified_noise = np.abs(u.T @ noise) / singular_values

            fig = make_subplots(rows=1, cols=2, subplot_titles=["singular values", "coefficient magnitudes"])
            fig.add_trace(go.Scatter(y=singular_values, mode="lines", name="singular values"), row=1, col=1)
            fig.add_trace(go.Scatter(y=clean_coefficients + 1e-14, mode="lines", name="clean data"), row=1, col=2)
            fig.add_trace(go.Scatter(y=noisy_coefficients + 1e-14, mode="lines", name="noisy data"), row=1, col=2)
            fig.add_trace(go.Scatter(y=amplified_noise + 1e-14, mode="lines", name="noise divided by s_i"), row=1, col=2)
            fig.update_yaxes(type="log", row=1, col=1)
            fig.update_yaxes(type="log", row=1, col=2)
            fig.update_xaxes(title_text="index", row=1, col=1)
            fig.update_xaxes(title_text="index", row=1, col=2)
            fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=45))
            fig.show()
            """
        ),
        md(
            """
            ## Checks

            Answer in your own words:

            1. What are Hadamard's three conditions?
            2. What is a nullspace?
            3. Why do small singular values create unstable inverses?
            4. What is the tradeoff in truncated SVD?
            """
        ),
        md(
            """
            ## Next Steps

            Week 6 will replace the hard cutoff in truncated SVD with Tikhonov regularization:

            ```text
            minimize ||Ax - y||_2^2 + lambda ||x||_2^2
            ```

            The central question becomes how `lambda` controls the bias-stability tradeoff.
            """
        ),
    ]


def week06_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Week 6 - Tikhonov Regularization

            This notebook accompanies the sixth MATH 435 slide deck.

            ## Goal

            By the end, you should be able to:

            - solve a Tikhonov-regularized least-squares problem;
            - derive and use the normal equations;
            - interpret Tikhonov filter factors;
            - sweep the regularization parameter `lambda`;
            - explain the bias-stability tradeoff.
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


            def rmse(estimate, reference):
                return float(np.sqrt(np.mean((estimate - reference) ** 2)))


            def gaussian_kernel1d(radius, sigma):
                axis = np.arange(-radius, radius + 1)
                kernel = np.exp(-(axis**2) / (2 * sigma**2))
                return kernel / kernel.sum()


            def gaussian_kernel2d(size, sigma):
                axis = np.arange(-(size // 2), size // 2 + 1)
                xx, yy = np.meshgrid(axis, axis)
                kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                return kernel / kernel.sum()


            def convolution_matrix(n, sigma):
                radius = max(3, int(np.ceil(4 * sigma)))
                kernel = gaussian_kernel1d(radius, sigma)
                matrix = np.zeros((n, n))
                for col in range(n):
                    impulse = np.zeros(n)
                    impulse[col] = 1.0
                    matrix[:, col] = np.convolve(impulse, kernel, mode="same")
                return matrix


            def test_signal(n):
                t = np.linspace(0, 1, n, endpoint=False)
                signal = (
                    0.58 * np.sin(2 * np.pi * 3 * t)
                    + 0.24 * np.sin(2 * np.pi * 14 * t)
                    + 0.20 * (t > 0.43)
                    - 0.18 * (t > 0.72)
                )
                return signal / np.max(np.abs(signal))


            def tikhonov_solution(matrix, y, lam):
                n = matrix.shape[1]
                lhs = matrix.T @ matrix + lam * np.eye(n)
                rhs = matrix.T @ y
                return np.linalg.solve(lhs, rhs)


            def centered_kernel_fft(kernel, shape):
                padded = np.zeros(shape)
                kh, kw = kernel.shape
                padded[:kh, :kw] = kernel
                padded = np.roll(padded, -(kh // 2), axis=0)
                padded = np.roll(padded, -(kw // 2), axis=1)
                return np.fft.fft2(padded)


            def frequency_tikhonov(noisy, kernel, lam):
                h_fft = centered_kernel_fft(kernel, noisy.shape)
                y_fft = np.fft.fft2(noisy)
                estimate = np.real(np.fft.ifft2(np.conj(h_fft) * y_fft / (np.abs(h_fft) ** 2 + lam)))
                return np.clip(estimate, 0.0, 1.0)


            def show_heatmaps(images, titles, zmin=0, zmax=1, height=430):
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
            ## Steps

            ### 1. Build a Small Ill-Conditioned Deblurring Problem

            We begin with a 1D signal and a blur matrix. The matrix is deliberately ill-conditioned.
            """
        ),
        code(
            """
            rng = np.random.default_rng(43506)

            n = 128
            blur_sigma = 2.8
            noise_level = 0.018

            matrix = convolution_matrix(n, blur_sigma)
            clean = test_signal(n)
            blurred = matrix @ clean
            noisy = blurred + noise_level * rng.standard_normal(n)

            singular_values = np.linalg.svd(matrix, compute_uv=False)

            print("condition number of A:", f"{singular_values[0] / singular_values[-1]:.3e}")
            print("condition number of A.T @ A:", f"{(singular_values[0] ** 2) / (singular_values[-1] ** 2):.3e}")

            fig = make_subplots(rows=1, cols=2, subplot_titles=["true signal", "blurred + noise"])
            fig.add_trace(go.Scatter(y=clean, mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(y=noisy, mode="lines"), row=1, col=2)
            fig.update_layout(height=360, showlegend=False, margin=dict(l=20, r=20, t=60, b=35))
            fig.show()
            """
        ),
        md(
            """
            ### 2. Solve the Tikhonov Normal Equations

            Tikhonov regularization solves

            ```text
            minimize ||Ax - y||_2^2 + lambda ||x||_2^2
            ```

            The normal equations are

            ```text
            (A.T @ A + lambda * I) x = A.T @ y
            ```
            """
        ),
        code(
            """
            lambdas = [1e-6, 1e-3, 1e-1]
            estimates = [tikhonov_solution(matrix, noisy, lam) for lam in lambdas]

            fig = make_subplots(
                rows=1,
                cols=5,
                subplot_titles=["true", "data"] + [f"lambda={lam:g}" for lam in lambdas],
            )
            for index, values in enumerate([clean, noisy] + estimates, start=1):
                fig.add_trace(go.Scatter(y=values, mode="lines"), row=1, col=index)
                fig.update_yaxes(range=[-1.35, 1.35], row=1, col=index)
                fig.update_xaxes(showticklabels=False, row=1, col=index)
            fig.update_layout(height=380, showlegend=False, margin=dict(l=20, r=20, t=60, b=35))
            fig.show()

            for lam, estimate in zip(lambdas, estimates):
                print(f"lambda={lam:g}, RMSE={rmse(estimate, clean):.4f}")
            """
        ),
        md(
            """
            ### Exercise 1

            Change the three `lambda` values above. Which one gives the best visual reconstruction?
            """
        ),
        code(
            """
            # TODO: change these values.
            student_lambdas = [1e-5, 3e-3, 3e-1]

            for lam in student_lambdas:
                estimate = tikhonov_solution(matrix, noisy, lam)
                residual = np.linalg.norm(matrix @ estimate - noisy)
                norm = np.linalg.norm(estimate)
                error = rmse(estimate, clean)
                print(f"lambda={lam:g}, residual={residual:.4f}, norm={norm:.4f}, RMSE={error:.4f}")
            """
        ),
        md(
            """
            ### 3. Sweep Lambda

            In teaching examples we can measure error because we know the true signal. In real inverse problems, the true image is unknown.
            """
        ),
        code(
            """
            lambda_grid = np.logspace(-8, 0, 70)
            errors = []
            residuals = []
            solution_norms = []
            system_conditions = []

            for lam in lambda_grid:
                estimate = tikhonov_solution(matrix, noisy, lam)
                errors.append(rmse(estimate, clean))
                residuals.append(np.linalg.norm(matrix @ estimate - noisy))
                solution_norms.append(np.linalg.norm(estimate))
                system_conditions.append((singular_values[0] ** 2 + lam) / (singular_values[-1] ** 2 + lam))

            best_index = int(np.argmin(errors))
            best_lambda = lambda_grid[best_index]

            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=["RMSE", "L-curve", "condition number"],
            )
            fig.add_trace(go.Scatter(x=lambda_grid, y=errors, mode="lines"), row=1, col=1)
            fig.add_trace(go.Scatter(x=[best_lambda], y=[errors[best_index]], mode="markers"), row=1, col=1)
            fig.add_trace(go.Scatter(x=residuals, y=solution_norms, mode="lines"), row=1, col=2)
            fig.add_trace(go.Scatter(x=[residuals[best_index]], y=[solution_norms[best_index]], mode="markers"), row=1, col=2)
            fig.add_trace(go.Scatter(x=lambda_grid, y=system_conditions, mode="lines"), row=1, col=3)
            fig.update_xaxes(type="log", title_text="lambda", row=1, col=1)
            fig.update_yaxes(type="log", title_text="RMSE", row=1, col=1)
            fig.update_xaxes(type="log", title_text="residual norm", row=1, col=2)
            fig.update_yaxes(type="log", title_text="solution norm", row=1, col=2)
            fig.update_xaxes(type="log", title_text="lambda", row=1, col=3)
            fig.update_yaxes(type="log", title_text="condition", row=1, col=3)
            fig.update_layout(height=410, showlegend=False, margin=dict(l=20, r=20, t=60, b=45))
            fig.show()

            print("best lambda in this teaching example:", f"{best_lambda:.3e}")
            print("best RMSE:", round(errors[best_index], 4))
            """
        ),
        md(
            """
            ### Exercise 2

            Increase `noise_level` in Step 1 and rerun the notebook from there.

            Does the best `lambda` move larger or smaller?
            """
        ),
        code(
            """
            answer = "TODO: write your observation here after rerunning with a different noise level."
            print(answer)
            """
        ),
        md(
            """
            ### 4. Tikhonov Filter Factors

            Direct inversion multiplies by `1 / sigma_i`.

            Tikhonov multiplies by:

            ```text
            sigma_i / (sigma_i^2 + lambda)
            ```
            """
        ),
        code(
            """
            fig = make_subplots(rows=1, cols=2, subplot_titles=["singular values", "Tikhonov inverse factors"])
            fig.add_trace(go.Scatter(y=singular_values, mode="lines", name="singular values"), row=1, col=1)

            for lam in [1e-6, 1e-4, 1e-2, 1e-1]:
                factors = singular_values / (singular_values**2 + lam)
                fig.add_trace(go.Scatter(y=factors, mode="lines", name=f"lambda={lam:g}"), row=1, col=2)

            fig.update_yaxes(type="log", row=1, col=1)
            fig.update_yaxes(type="log", row=1, col=2)
            fig.update_xaxes(title_text="index", row=1, col=1)
            fig.update_xaxes(title_text="index", row=1, col=2)
            fig.update_layout(height=410, margin=dict(l=20, r=20, t=60, b=45))
            fig.show()
            """
        ),
        md(
            """
            ### Exercise 3

            For `sigma = 0.001`, compare direct inversion with Tikhonov for `lambda = 0.01`.
            """
        ),
        code(
            """
            sigma = 0.001
            lam = 0.01

            direct_factor = 1 / sigma
            tikhonov_factor = sigma / (sigma**2 + lam)

            print("direct inverse factor:", direct_factor)
            print("Tikhonov factor:", tikhonov_factor)
            """
        ),
        md(
            """
            ### 5. Bias-Variance Tradeoff by Simulation

            We repeat the noisy measurement many times and look at how the average reconstruction and the reconstruction variability depend on `lambda`.
            """
        ),
        code(
            """
            rng = np.random.default_rng(43506)
            small_n = 90
            small_matrix = convolution_matrix(small_n, sigma=2.6)
            small_clean = test_signal(small_n)
            small_blurred = small_matrix @ small_clean

            lambda_grid_small = np.logspace(-7, -0.2, 35)
            trials = 80
            simulation_noise = 0.02
            bias_squared = []
            variance = []

            for lam in lambda_grid_small:
                trial_estimates = []
                for _ in range(trials):
                    trial_noisy = small_blurred + simulation_noise * rng.standard_normal(small_n)
                    trial_estimates.append(tikhonov_solution(small_matrix, trial_noisy, lam))
                stack = np.asarray(trial_estimates)
                mean_estimate = stack.mean(axis=0)
                bias_squared.append(np.mean((mean_estimate - small_clean) ** 2))
                variance.append(np.mean(np.var(stack, axis=0)))

            total = np.asarray(bias_squared) + np.asarray(variance)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lambda_grid_small, y=bias_squared, mode="lines", name="bias^2"))
            fig.add_trace(go.Scatter(x=lambda_grid_small, y=variance, mode="lines", name="variance"))
            fig.add_trace(go.Scatter(x=lambda_grid_small, y=total, mode="lines", name="bias^2 + variance"))
            fig.update_layout(
                title="Bias-variance tradeoff",
                xaxis_title="lambda",
                yaxis_title="mean squared contribution",
                xaxis_type="log",
                yaxis_type="log",
                height=410,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()
            """
        ),
        md(
            """
            ### Exercise 4

            In your own words, why does variance decrease as `lambda` grows? Why does bias increase?
            """
        ),
        code(
            """
            answer = "TODO: write your bias-variance explanation here."
            print(answer)
            """
        ),
        md(
            """
            ### 6. Real Image Deblurring with Frequency-Domain Tikhonov

            For convolution with periodic boundary assumptions, Tikhonov deblurring can be written in the Fourier domain:

            ```text
            X_lambda = conj(H) * Y / (|H|^2 + lambda)
            ```
            """
        ),
        code(
            """
            rng = np.random.default_rng(43506)
            image = data.camera().astype(float) / 255.0
            image = image[120:376, 120:376]

            kernel = gaussian_kernel2d(25, 3.5)
            blurred_image = ndimage.convolve(image, kernel, mode="reflect")
            noisy_image = np.clip(blurred_image + 0.018 * rng.standard_normal(image.shape), 0, 1)

            image_lambdas = [1e-5, 5e-2, 1.0]
            image_estimates = [frequency_tikhonov(noisy_image, kernel, lam) for lam in image_lambdas]

            show_heatmaps(
                [image, noisy_image] + image_estimates,
                ["original", "blurred + noise"] + [f"lambda={lam:g}" for lam in image_lambdas],
                height=430,
            )

            for lam, estimate in zip(image_lambdas, image_estimates):
                print(f"lambda={lam:g}, RMSE={rmse(estimate, image):.4f}")
            """
        ),
        md(
            """
            ## Checks

            Answer in your own words:

            1. What objective does Tikhonov regularization minimize?
            2. What are the normal equations?
            3. How does `lambda` affect small singular-value directions?
            4. What is the bias-stability tradeoff?
            """
        ),
        md(
            """
            ## Next Steps

            Week 7 moves from this closed-form quadratic model to a broader variational view:

            ```text
            minimize energy(image)
            ```

            The same idea remains: choose an objective that balances data agreement and image structure.
            """
        ),
    ]


def week07_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Week 7 - Variational Formulation

            This notebook accompanies the seventh MATH 435 slide deck.

            ## Goal

            By the end, you should be able to:

            - write a reconstruction problem as energy minimization;
            - identify data fidelity and regularization terms;
            - interpret convex and nonconvex energies visually;
            - verify the Euler-Lagrange optimality condition for an `l2` smoothing model;
            - run gradient descent and compare it with the closed-form solution.
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
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots


            def rmse(estimate, reference):
                return float(np.sqrt(np.mean((estimate - reference) ** 2)))


            def periodic_laplacian_matrix_eigenvalues(shape):
                rows, cols = shape
                row_freq = 2 * np.pi * np.fft.fftfreq(rows)
                col_freq = 2 * np.pi * np.fft.fftfreq(cols)
                rr, cc = np.meshgrid(row_freq, col_freq, indexing="ij")
                return 4 - 2 * np.cos(rr) - 2 * np.cos(cc)


            def quadratic_denoise(noisy, lam):
                eig = periodic_laplacian_matrix_eigenvalues(noisy.shape)
                estimate = np.real(np.fft.ifft2(np.fft.fft2(noisy) / (1 + lam * eig)))
                return np.clip(estimate, 0, 1)


            def smoothness_energy(image):
                vertical = np.roll(image, -1, axis=0) - image
                horizontal = np.roll(image, -1, axis=1) - image
                return float(np.sum(vertical**2 + horizontal**2))


            def quadratic_energy(u, y, lam):
                return 0.5 * float(np.sum((u - y) ** 2)) + 0.5 * lam * smoothness_energy(u)


            def gradient(u, y, lam):
                laplacian_operator = (
                    4 * u
                    - np.roll(u, 1, axis=0)
                    - np.roll(u, -1, axis=0)
                    - np.roll(u, 1, axis=1)
                    - np.roll(u, -1, axis=1)
                )
                return (u - y) + lam * laplacian_operator


            def show_heatmaps(images, titles, zmin=0, zmax=1, height=430):
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
            ## Steps

            ### 1. Energy = Data Fit + Regularization

            A variational model chooses a reconstruction by minimizing an energy.

            For a scalar teaching example:

            ```text
            E(x) = 0.5 * (x - y)^2 + 0.5 * lambda * x^2
            ```
            """
        ),
        code(
            """
            x = np.linspace(-2.5, 2.5, 500)
            observed_y = 0.8
            lam = 0.55

            data_term = 0.5 * (x - observed_y) ** 2
            regularizer = 0.5 * lam * x**2
            energy = data_term + regularizer
            minimizer = observed_y / (1 + lam)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=data_term, mode="lines", name="data fit"))
            fig.add_trace(go.Scatter(x=x, y=regularizer, mode="lines", name="regularizer"))
            fig.add_trace(go.Scatter(x=x, y=energy, mode="lines", name="energy"))
            fig.add_trace(
                go.Scatter(
                    x=[minimizer],
                    y=[0.5 * (minimizer - observed_y) ** 2 + 0.5 * lam * minimizer**2],
                    mode="markers",
                    name="minimizer",
                    marker=dict(size=10),
                )
            )
            fig.update_layout(
                title="Energy terms",
                xaxis_title="candidate x",
                yaxis_title="cost",
                height=390,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()

            print("observed value:", observed_y)
            print("minimizer:", round(float(minimizer), 4))
            """
        ),
        md(
            """
            ### Exercise 1

            Change `lam`.

            What happens to the minimizer when regularization becomes stronger?
            """
        ),
        code(
            """
            # TODO: change this value.
            lam = 2.0

            minimizer = observed_y / (1 + lam)
            print("lambda:", lam)
            print("minimizer:", round(float(minimizer), 4))
            """
        ),
        md(
            """
            ### 2. Convexity as Geometry

            Convex energies have no deceptive local minima. Nonconvex energies can have several valleys.
            """
        ),
        code(
            """
            grid = np.linspace(-2.5, 2.5, 120)
            xx, yy = np.meshgrid(grid, grid)
            convex = 0.65 * (xx - 0.3) ** 2 + 1.2 * (yy + 0.2) ** 2 + 0.35 * xx * yy
            nonconvex = 0.15 * (xx**2 + yy**2) + 0.8 * np.sin(2.4 * xx) * np.cos(2.0 * yy)

            fig = make_subplots(rows=1, cols=2, subplot_titles=["convex energy", "nonconvex energy"])
            fig.add_trace(go.Contour(z=convex, x=grid, y=grid, colorscale="Viridis", showscale=False), row=1, col=1)
            fig.add_trace(go.Contour(z=nonconvex, x=grid, y=grid, colorscale="Viridis", showscale=False), row=1, col=2)
            fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=35))
            fig.show()
            """
        ),
        md(
            """
            ### Exercise 2

            In the nonconvex plot, choose two different valleys. Why might gradient descent end in different places depending on the initialization?
            """
        ),
        code(
            """
            answer = "TODO: write your explanation here."
            print(answer)
            """
        ),
        md(
            """
            ### 3. Quadratic Image Denoising

            We now use the variational model

            ```text
            E(u) = 0.5 ||u - y||_2^2 + 0.5 lambda ||grad u||_2^2
            ```

            The first term keeps the result close to the noisy image. The second term penalizes roughness.
            """
        ),
        code(
            """
            rng = np.random.default_rng(43507)

            image = data.camera().astype(float) / 255.0
            image = image[80:336, 90:346]
            noisy = np.clip(image + 0.10 * rng.standard_normal(image.shape), 0, 1)

            lambdas = [0.15, 1.2, 8.0]
            estimates = [quadratic_denoise(noisy, lam) for lam in lambdas]

            show_heatmaps(
                [image, noisy] + estimates,
                ["original", "noisy"] + [f"lambda={lam:g}" for lam in lambdas],
                height=430,
            )

            print("noisy RMSE:", round(rmse(noisy, image), 4))
            for lam, estimate in zip(lambdas, estimates):
                print(f"lambda={lam:g}, RMSE={rmse(estimate, image):.4f}, smoothness={smoothness_energy(estimate):.1f}")
            """
        ),
        md(
            """
            ### Exercise 3

            Try a larger lambda. What improves? What gets worse?
            """
        ),
        code(
            """
            # TODO: change this value.
            lam = 3.0

            estimate = quadratic_denoise(noisy, lam)
            show_heatmaps([noisy, estimate], ["noisy", f"lambda={lam:g}"], height=420)
            print("RMSE:", round(rmse(estimate, image), 4))
            print("smoothness energy:", round(smoothness_energy(estimate), 1))
            """
        ),
        md(
            """
            ### 4. Euler-Lagrange Residual

            The Euler-Lagrange equation for the quadratic denoising model is

            ```text
            (u - y) + lambda * L u = 0
            ```

            where `L` is a discrete negative Laplacian.
            """
        ),
        code(
            """
            lam = 1.2
            closed_form = quadratic_denoise(noisy, lam)
            residual = gradient(closed_form, noisy, lam)

            print("Euler-Lagrange residual norm:", f"{np.linalg.norm(residual):.3e}")
            print("maximum absolute residual:", f"{np.max(np.abs(residual)):.3e}")
            """
        ),
        md(
            """
            ### Exercise 4

            Compute the gradient norm at the noisy image itself. Is the noisy image already a minimizer of the variational energy?
            """
        ),
        code(
            """
            lam = 1.2
            residual_at_noisy = gradient(noisy, noisy, lam)

            print("gradient norm at noisy image:", f"{np.linalg.norm(residual_at_noisy):.3e}")
            print("gradient norm at minimizer:", f"{np.linalg.norm(gradient(closed_form, noisy, lam)):.3e}")
            """
        ),
        md(
            """
            ### 5. Gradient Descent

            Instead of solving the Euler-Lagrange equation directly, we can iterate:

            ```text
            u_{k+1} = u_k - step * grad E(u_k)
            ```
            """
        ),
        code(
            """
            lam = 1.5
            step = 0.08
            current = noisy.copy()
            energies = [quadratic_energy(current, noisy, lam)]
            errors = [rmse(current, image)]
            snapshots = [current.copy()]

            for iteration in range(80):
                current = current - step * gradient(current, noisy, lam)
                energies.append(quadratic_energy(current, noisy, lam))
                errors.append(rmse(current, image))
                if iteration in [2, 9, 39, 79]:
                    snapshots.append(current.copy())

            fig = make_subplots(rows=1, cols=2, subplot_titles=["energy", "RMSE"])
            fig.add_trace(go.Scatter(y=energies, mode="lines", name="energy"), row=1, col=1)
            fig.add_trace(go.Scatter(y=errors, mode="lines", name="RMSE"), row=1, col=2)
            fig.update_yaxes(type="log", row=1, col=1)
            fig.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=45))
            fig.show()

            show_heatmaps(snapshots, ["iter 0", "iter 3", "iter 10", "iter 40", "iter 80"], height=430)
            print("initial energy:", round(energies[0], 3))
            print("final energy:", round(energies[-1], 3))
            """
        ),
        md(
            """
            ### Exercise 5

            Change `step`.

            What happens if the step is too large? Try `step = 0.20` and rerun only the gradient-descent cell.
            """
        ),
        code(
            """
            answer = "TODO: write what you observe when changing the step size."
            print(answer)
            """
        ),
        md(
            """
            ## Checks

            Answer in your own words:

            1. What is a variational model?
            2. Why is convexity useful?
            3. What does an Euler-Lagrange equation express?
            4. Why does quadratic smoothness blur edges?
            """
        ),
        md(
            """
            ## Next Steps

            Week 8 studies Total Variation regularization. It modifies the smoothness term so that edges are penalized differently:

            ```text
            integral |grad u|
            ```
            """
        ),
    ]


def week08_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Week 8 - Total Variation Regularization

            This notebook accompanies the eighth MATH 435 slide deck.

            ## Goal

            By the end, you should be able to:

            - compute a discrete Total Variation (TV) quantity;
            - compare l2 smoothing and TV denoising;
            - explain why TV preserves strong edges;
            - inspect gradient magnitude maps;
            - identify the staircasing effect.
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
            from skimage.restoration import denoise_tv_chambolle
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots


            def rmse(estimate, reference):
                return float(np.sqrt(np.mean((estimate - reference) ** 2)))


            def periodic_laplacian_eigenvalues(shape):
                grids = np.meshgrid(
                    *[2 * np.pi * np.fft.fftfreq(size) for size in shape],
                    indexing="ij",
                )
                values = np.zeros(shape)
                for grid in grids:
                    values += 2 - 2 * np.cos(grid)
                return values


            def quadratic_smooth(noisy, lam):
                eig = periodic_laplacian_eigenvalues(noisy.shape)
                estimate = np.real(np.fft.ifftn(np.fft.fftn(noisy) / (1 + lam * eig)))
                return np.clip(estimate, 0.0, 1.0)


            def gradient_magnitude(image):
                if image.ndim == 1:
                    return np.abs(np.diff(image, append=image[-1]))
                vertical = np.diff(image, axis=0, append=image[-1:, :])
                horizontal = np.diff(image, axis=1, append=image[:, -1:])
                return np.sqrt(vertical**2 + horizontal**2)


            def total_variation(image):
                return float(np.sum(gradient_magnitude(image)))


            def smoothness_energy(image):
                gradient = gradient_magnitude(image)
                return float(np.sum(gradient**2))


            def show_heatmaps(images, titles, colorscales=None, zmin=0, zmax=1, height=430):
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
            """
        ),
        md(
            """
            ## Steps

            ### 1. Gradient Penalties

            Quadratic smoothing penalizes `0.5 * g^2`.

            Total Variation penalizes `abs(g)`.
            """
        ),
        code(
            """
            g = np.linspace(-3, 3, 600)
            l2_penalty = 0.5 * g**2
            tv_penalty = np.abs(g)

            fig = make_subplots(rows=1, cols=2, subplot_titles=["penalty", "derivative"])
            fig.add_trace(go.Scatter(x=g, y=l2_penalty, mode="lines", name="0.5 g^2"), row=1, col=1)
            fig.add_trace(go.Scatter(x=g, y=tv_penalty, mode="lines", name="|g|"), row=1, col=1)
            fig.add_trace(go.Scatter(x=g, y=g, mode="lines", name="l2 derivative"), row=1, col=2)
            fig.add_trace(go.Scatter(x=g, y=np.sign(g), mode="lines", name="TV subgradient"), row=1, col=2)
            fig.update_xaxes(title_text="gradient g", row=1, col=1)
            fig.update_xaxes(title_text="gradient g", row=1, col=2)
            fig.update_yaxes(title_text="cost", row=1, col=1)
            fig.update_yaxes(title_text="derivative", row=1, col=2)
            fig.update_layout(height=390, margin=dict(l=20, r=20, t=60, b=45))
            fig.show()
            """
        ),
        md(
            """
            ### Exercise 1

            Compute the two penalties for a gradient of size 4. Which model penalizes this sharp jump more?
            """
        ),
        code(
            """
            gradient_value = 4.0

            print("l2 penalty:", 0.5 * gradient_value**2)
            print("TV penalty:", abs(gradient_value))
            """
        ),
        md(
            """
            ### 2. Step Signal

            TV is well suited to piecewise-constant signals because it allows sharp jumps.
            """
        ),
        code(
            """
            rng = np.random.default_rng(43508)
            n = 220
            x = np.linspace(0, 1, n)
            clean = np.zeros(n)
            clean[x > 0.18] = 0.35
            clean[x > 0.45] = 0.9
            clean[x > 0.72] = 0.2
            noisy_signal = np.clip(clean + 0.13 * rng.standard_normal(n), 0, 1)

            l2_signal = quadratic_smooth(noisy_signal, lam=18.0)
            tv_signal = denoise_tv_chambolle(noisy_signal, weight=0.18)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=clean, mode="lines", name="clean"))
            fig.add_trace(go.Scatter(x=x, y=noisy_signal, mode="lines", name="noisy"))
            fig.add_trace(go.Scatter(x=x, y=l2_signal, mode="lines", name="l2 smoothing"))
            fig.add_trace(go.Scatter(x=x, y=tv_signal, mode="lines", name="TV"))
            fig.update_layout(
                title="Step signal denoising",
                xaxis_title="position",
                yaxis_title="intensity",
                height=410,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()

            print("noisy RMSE:", round(rmse(noisy_signal, clean), 4))
            print("l2 RMSE:", round(rmse(l2_signal, clean), 4))
            print("TV RMSE:", round(rmse(tv_signal, clean), 4))
            """
        ),
        md(
            """
            ### Exercise 2

            Change `weight` in the TV denoising line.

            What happens to noise, flat regions, and jump sharpness?
            """
        ),
        code(
            """
            # TODO: change this value.
            tv_weight = 0.35

            tv_experiment = denoise_tv_chambolle(noisy_signal, weight=tv_weight)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=clean, mode="lines", name="clean"))
            fig.add_trace(go.Scatter(x=x, y=tv_experiment, mode="lines", name=f"TV weight={tv_weight}"))
            fig.update_layout(height=360, margin=dict(l=20, r=20, t=55, b=45))
            fig.show()

            print("TV RMSE:", round(rmse(tv_experiment, clean), 4))
            print("TV norm:", round(total_variation(tv_experiment), 3))
            """
        ),
        md(
            """
            ### 3. Image Denoising: l2 Versus TV

            We now compare quadratic smoothing and TV denoising on a real image.
            """
        ),
        code(
            """
            rng = np.random.default_rng(43508)
            image = data.camera().astype(float) / 255.0
            image = image[80:336, 90:346]
            noisy = np.clip(image + 0.10 * rng.standard_normal(image.shape), 0, 1)

            l2_image = quadratic_smooth(noisy, lam=1.2)
            tv_image = denoise_tv_chambolle(noisy, weight=0.11)

            show_heatmaps(
                [image, noisy, l2_image, tv_image],
                ["original", "noisy", "l2 smoothing", "TV"],
                height=430,
            )

            for name, estimate in [("noisy", noisy), ("l2", l2_image), ("TV", tv_image)]:
                print(
                    f"{name:5s}",
                    "RMSE=", round(rmse(estimate, image), 4),
                    "TV=", round(total_variation(estimate), 1),
                    "l2-grad-energy=", round(smoothness_energy(estimate), 1),
                )
            """
        ),
        md(
            """
            ### 4. Gradient Magnitude Maps

            Gradient maps help us inspect edge preservation.
            """
        ),
        code(
            """
            maps = [gradient_magnitude(values) for values in [image, noisy, l2_image, tv_image]]
            vmax = np.percentile(maps[1], 99)

            show_heatmaps(
                maps,
                ["original gradients", "noisy gradients", "l2 gradients", "TV gradients"],
                colorscales=["Magma"] * 4,
                zmin=0,
                zmax=vmax,
                height=430,
            )
            """
        ),
        md(
            """
            ### Exercise 3

            Look at the gradient maps. Which method keeps the main contours more clearly?
            """
        ),
        code(
            """
            answer = "TODO: write your observation here."
            print(answer)
            """
        ),
        md(
            """
            ### 5. TV Parameter Sweep

            The TV weight controls the noise-edge tradeoff.
            """
        ),
        code(
            """
            weights = [0.03, 0.10, 0.28]
            estimates = [denoise_tv_chambolle(noisy, weight=weight) for weight in weights]

            show_heatmaps(
                [image, noisy] + estimates,
                ["original", "noisy"] + [f"weight={weight:g}" for weight in weights],
                height=430,
            )

            for weight, estimate in zip(weights, estimates):
                print(f"weight={weight:g}, RMSE={rmse(estimate, image):.4f}, TV={total_variation(estimate):.1f}")
            """
        ),
        md(
            """
            ### Exercise 4

            Add another TV weight to the list. Which value would you choose visually?
            """
        ),
        code(
            """
            # TODO: change this value.
            weight = 0.18

            estimate = denoise_tv_chambolle(noisy, weight=weight)
            show_heatmaps([noisy, estimate], ["noisy", f"TV weight={weight:g}"], height=420)
            print("RMSE:", round(rmse(estimate, image), 4))
            print("TV:", round(total_variation(estimate), 1))
            """
        ),
        md(
            """
            ### 6. Staircasing

            TV can turn smooth ramps into piecewise-flat plateaus. This is called staircasing.
            """
        ),
        code(
            """
            rng = np.random.default_rng(43508)
            n = 260
            x = np.linspace(0, 1, n)
            clean_ramp = 0.15 + 0.65 * x + 0.08 * np.sin(2 * np.pi * 3 * x)
            noisy_ramp = np.clip(clean_ramp + 0.045 * rng.standard_normal(n), 0, 1)
            l2_ramp = quadratic_smooth(noisy_ramp, lam=22.0)
            tv_ramp = denoise_tv_chambolle(noisy_ramp, weight=0.12)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=clean_ramp, mode="lines", name="clean smooth signal"))
            fig.add_trace(go.Scatter(x=x, y=noisy_ramp, mode="lines", name="noisy"))
            fig.add_trace(go.Scatter(x=x, y=l2_ramp, mode="lines", name="l2 smoothing"))
            fig.add_trace(go.Scatter(x=x, y=tv_ramp, mode="lines", name="TV"))
            fig.update_layout(
                title="Staircasing on a smooth signal",
                xaxis_title="position",
                yaxis_title="intensity",
                height=410,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()
            """
        ),
        md(
            """
            ## Checks

            Answer in your own words:

            1. What does Total Variation measure?
            2. Why does TV preserve edges better than l2 smoothing?
            3. What is staircasing?
            4. Why is TV optimization more delicate than quadratic smoothing?
            """
        ),
        md(
            """
            ## Next Steps

            Week 9 studies optimization methods. TV is convex but nonsmooth, so it motivates subgradient and proximal ideas.
            """
        ),
    ]


def main() -> None:
    write_notebook("week01_image_formation.ipynb", week01_cells())
    write_notebook("week02_convolution_blur.ipynb", week02_cells())
    write_notebook("week03_fourier_imaging.ipynb", week03_cells())
    write_notebook("week04_noise_likelihood.ipynb", week04_cells())
    write_notebook("week05_ill_posed_inverse_problems.ipynb", week05_cells())
    write_notebook("week06_tikhonov_regularization.ipynb", week06_cells())
    write_notebook("week07_variational_formulation.ipynb", week07_cells())
    write_notebook("week08_total_variation.ipynb", week08_cells())


if __name__ == "__main__":
    main()
