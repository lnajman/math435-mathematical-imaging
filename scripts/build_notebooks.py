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


def week09_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Week 9 - Optimization Methods

            This notebook accompanies the ninth MATH 435 slide deck.

            ## Goal

            By the end, you should be able to:

            - run gradient descent on a smooth convex energy;
            - explain how the step size changes convergence;
            - apply soft thresholding as a proximal operator;
            - run ISTA on a sparse inverse problem;
            - connect proximal steps to sparse imaging.
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
            from scipy.fft import dctn, idctn
            from scipy.ndimage import gaussian_filter
            from skimage import data
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots


            def soft_threshold(values, threshold):
                return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


            def rmse(estimate, reference):
                return float(np.sqrt(np.mean((estimate - reference) ** 2)))


            def quadratic_energy(point, matrix, center):
                diff = point - center
                return float(0.5 * diff @ matrix @ diff)


            def gradient_descent_path(start, matrix, center, step_size, iterations):
                point = start.astype(float).copy()
                points = [point.copy()]
                energies = [quadratic_energy(point, matrix, center)]
                for _ in range(iterations):
                    gradient = matrix @ (point - center)
                    point = point - step_size * gradient
                    points.append(point.copy())
                    energies.append(quadratic_energy(point, matrix, center))
                return np.array(points), np.array(energies)


            def show_image_grid(images, titles, colorscales=None, zmin=0, zmax=1, height=430):
                if colorscales is None:
                    colorscales = ["Gray"] * len(images)
                fig = make_subplots(rows=1, cols=len(images), subplot_titles=titles)
                for index, (image, colorscale) in enumerate(zip(images, colorscales), start=1):
                    fig.add_trace(
                        go.Heatmap(
                            z=image,
                            colorscale=colorscale,
                            zmin=zmin if colorscale == "Gray" else None,
                            zmax=zmax if colorscale == "Gray" else None,
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

            ### 1. Gradient Descent on a Convex Quadratic

            We minimize a simple two-variable energy:

            $$
            E(x)=\\frac12(x-x_*)^TQ(x-x_*).
            $$

            The minimizer is `center`.
            """
        ),
        code(
            """
            matrix = np.array([[1.0, 0.0], [0.0, 10.0]])
            center = np.array([0.9, -0.45])
            start = np.array([-1.7, 1.15])
            lipschitz = float(np.linalg.eigvalsh(matrix).max())

            x_values = np.linspace(-2.2, 1.5, 160)
            y_values = np.linspace(-1.2, 1.35, 160)
            xx, yy = np.meshgrid(x_values, y_values)
            energy = 0.5 * ((xx - center[0]) ** 2 + 10.0 * (yy - center[1]) ** 2)

            path_small, energies_small = gradient_descent_path(start, matrix, center, 0.055, 28)
            path_large, energies_large = gradient_descent_path(start, matrix, center, 0.175, 28)

            fig = go.Figure()
            fig.add_trace(go.Contour(x=x_values, y=y_values, z=energy, colorscale="Greys", showscale=False, contours=dict(showlabels=False)))
            fig.add_trace(go.Scatter(x=path_small[:, 0], y=path_small[:, 1], mode="lines+markers", name="small steps"))
            fig.add_trace(go.Scatter(x=path_large[:, 0], y=path_large[:, 1], mode="lines+markers", name="large steps"))
            fig.add_trace(go.Scatter(x=[center[0]], y=[center[1]], mode="markers", name="minimizer", marker=dict(size=13, symbol="star")))
            fig.update_layout(
                title="Gradient descent paths on a convex quadratic",
                xaxis_title="coordinate 1",
                yaxis_title="coordinate 2",
                height=520,
                margin=dict(l=20, r=20, t=60, b=45),
            )
            fig.show()

            print("Lipschitz constant L:", lipschitz)
            print("A common safe step size is at most 1/L =", round(1 / lipschitz, 3))
            """
        ),
        md(
            """
            ### 2. Step Size Experiment

            The step size controls stability. For this quadratic, the critical value is near `2 / L`.
            """
        ),
        code(
            """
            settings = [
                ("small", 0.4 / lipschitz),
                ("near 1/L", 1.0 / lipschitz),
                ("aggressive", 1.8 / lipschitz),
                ("too large", 2.1 / lipschitz),
            ]

            fig = go.Figure()
            for label, step_size in settings:
                _, energies = gradient_descent_path(start, matrix, center, step_size, 35)
                fig.add_trace(go.Scatter(y=energies, mode="lines+markers", name=f"{label}: tau={step_size:.3f}"))
                print(f"{label:10s} tau={step_size:.3f}: E0={energies[0]:.4f}, E35={energies[-1]:.4f}")

            fig.update_layout(
                title="Energy through gradient descent iterations",
                xaxis_title="iteration",
                yaxis_title="energy",
                yaxis_type="log",
                height=420,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()
            """
        ),
        md(
            """
            ### Exercise 1

            Change `manual_step_size` below. Try values below and above `2 / L`.

            What do you observe?
            """
        ),
        code(
            """
            # TODO: change this value.
            manual_step_size = 0.12

            path_manual, energies_manual = gradient_descent_path(start, matrix, center, manual_step_size, 35)

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=energies_manual, mode="lines+markers", name=f"tau={manual_step_size}"))
            fig.update_layout(
                title="Manual step-size experiment",
                xaxis_title="iteration",
                yaxis_title="energy",
                yaxis_type="log",
                height=360,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()

            print("first energy:", round(energies_manual[0], 4))
            print("last energy:", round(energies_manual[-1], 4))
            """
        ),
        md(
            """
            ### 3. Soft Thresholding

            Soft thresholding is the proximal operator for the l1 norm.

            $$
            S_\\alpha(v)=\\operatorname{sign}(v)\\max(|v|-\\alpha,0).
            $$
            """
        ),
        code(
            """
            values = np.linspace(-3, 3, 700)
            threshold = 0.8
            shrinked = soft_threshold(values, threshold)

            fig = make_subplots(rows=1, cols=2, subplot_titles=["absolute value", "soft threshold"])
            fig.add_trace(go.Scatter(x=values, y=np.abs(values), mode="lines", name="|x|"), row=1, col=1)
            fig.add_trace(go.Scatter(x=values, y=values, mode="lines", name="identity", line=dict(dash="dash")), row=1, col=2)
            fig.add_trace(go.Scatter(x=values, y=shrinked, mode="lines", name="soft threshold"), row=1, col=2)
            fig.update_xaxes(title_text="x", row=1, col=1)
            fig.update_xaxes(title_text="input", row=1, col=2)
            fig.update_yaxes(title_text="|x|", row=1, col=1)
            fig.update_yaxes(title_text="output", row=1, col=2)
            fig.update_layout(height=390, margin=dict(l=20, r=20, t=60, b=45))
            fig.show()
            """
        ),
        md(
            """
            ### Exercise 2

            Change the threshold. Which inputs become exactly zero?
            """
        ),
        code(
            """
            # TODO: change this value.
            threshold = 0.6
            test_values = np.array([-2.0, -0.7, -0.2, 0.0, 0.3, 0.9, 2.3])

            print("input: ", test_values)
            print("output:", soft_threshold(test_values, threshold))
            """
        ),
        md(
            """
            ### 4. ISTA for a Sparse Signal

            ISTA alternates:

            1. a gradient step for the data term;
            2. a soft-thresholding step for the l1 regularizer.
            """
        ),
        code(
            """
            def blur1d(values, sigma):
                return gaussian_filter(values, sigma=sigma, mode="wrap")


            def ista_sparse_signal(observed, lam, step_size, iterations, sigma):
                estimate = np.zeros_like(observed)
                energies = []
                active_counts = []
                for _ in range(iterations):
                    residual = blur1d(estimate, sigma) - observed
                    gradient = blur1d(residual, sigma)
                    estimate = soft_threshold(estimate - step_size * gradient, step_size * lam)
                    energy = 0.5 * float(np.sum((blur1d(estimate, sigma) - observed) ** 2)) + lam * float(np.sum(np.abs(estimate)))
                    energies.append(energy)
                    active_counts.append(int(np.count_nonzero(np.abs(estimate) > 1e-3)))
                return estimate, np.array(energies), np.array(active_counts)


            rng = np.random.default_rng(43509)
            n = 180
            positions = np.linspace(0, 1, n)
            sparse_signal = np.zeros(n)
            locations = np.array([18, 42, 75, 109, 143, 162])
            amplitudes = np.array([1.0, -0.75, 0.55, 0.9, -0.65, 0.5])
            sparse_signal[locations] = amplitudes
            observed_signal = blur1d(sparse_signal, sigma=3.0) + 0.035 * rng.standard_normal(n)

            ista_signal, ista_energies, active_counts = ista_sparse_signal(
                observed_signal,
                lam=0.035,
                step_size=0.95,
                iterations=80,
                sigma=3.0,
            )

            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["signal", "ISTA diagnostics"],
                specs=[[{}, {"secondary_y": True}]],
            )
            fig.add_trace(go.Scatter(x=positions, y=observed_signal, mode="lines", name="blurred noisy observation"), row=1, col=1)
            fig.add_trace(go.Scatter(x=positions, y=sparse_signal, mode="markers", name="true sparse signal"), row=1, col=1)
            fig.add_trace(go.Scatter(x=positions, y=ista_signal, mode="markers", name="ISTA estimate"), row=1, col=1)
            fig.add_trace(go.Scatter(y=ista_energies, mode="lines", name="energy"), row=1, col=2, secondary_y=False)
            fig.add_trace(go.Scatter(y=active_counts, mode="lines", name="active coefficients"), row=1, col=2, secondary_y=True)
            fig.update_yaxes(title_text="energy", row=1, col=2, secondary_y=False)
            fig.update_yaxes(title_text="active coefficients", row=1, col=2, secondary_y=True)
            fig.update_layout(height=430, margin=dict(l=20, r=20, t=60, b=45))
            fig.show()

            print("initial energy:", round(ista_energies[0], 4))
            print("final energy:", round(ista_energies[-1], 4))
            print("active coefficients:", active_counts[0], "->", active_counts[-1])
            """
        ),
        md(
            """
            ### Exercise 3

            Change `lam` below. How does it affect sparsity and data fit?
            """
        ),
        code(
            """
            # TODO: change this value.
            lam = 0.06

            estimate, energies, active = ista_sparse_signal(
                observed_signal,
                lam=lam,
                step_size=0.95,
                iterations=80,
                sigma=3.0,
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=positions, y=observed_signal, mode="lines", name="observation"))
            fig.add_trace(go.Scatter(x=positions, y=sparse_signal, mode="markers", name="true"))
            fig.add_trace(go.Scatter(x=positions, y=estimate, mode="markers", name=f"ISTA lam={lam}"))
            fig.update_layout(height=380, margin=dict(l=20, r=20, t=55, b=45))
            fig.show()

            print("final energy:", round(energies[-1], 4))
            print("active coefficients:", active[-1])
            """
        ),
        md(
            """
            ### 5. ISTA for Sparse Image Deblurring

            We now regularize DCT coefficients of an image. The unknown image is represented by transform coefficients, and the l1 penalty encourages many of them to become zero.
            """
        ),
        code(
            """
            def ista_dct_deblur(observed, lam, step_size, iterations, blur_sigma):
                coefficients = dctn(observed, norm="ortho")
                energies = []
                active_counts = []
                for _ in range(iterations):
                    image = idctn(coefficients, norm="ortho")
                    residual = gaussian_filter(image, sigma=blur_sigma, mode="reflect") - observed
                    gradient_image = gaussian_filter(residual, sigma=blur_sigma, mode="reflect")
                    gradient_coefficients = dctn(gradient_image, norm="ortho")
                    coefficients = soft_threshold(coefficients - step_size * gradient_coefficients, step_size * lam)
                    image = idctn(coefficients, norm="ortho")
                    energy = 0.5 * float(np.sum((gaussian_filter(image, sigma=blur_sigma, mode="reflect") - observed) ** 2)) + lam * float(np.sum(np.abs(coefficients)))
                    energies.append(energy)
                    active_counts.append(int(np.count_nonzero(np.abs(coefficients) > 1e-3)))
                return np.clip(idctn(coefficients, norm="ortho"), 0.0, 1.0), coefficients, np.array(energies), np.array(active_counts)


            rng = np.random.default_rng(43509)
            image = data.camera().astype(float)[96:224, 128:256] / 255.0
            observed = np.clip(gaussian_filter(image, sigma=1.1, mode="reflect") + 0.035 * rng.standard_normal(image.shape), 0, 1)
            ista_image, dct_coefficients, image_energies, image_active = ista_dct_deblur(
                observed,
                lam=0.0045,
                step_size=0.95,
                iterations=65,
                blur_sigma=1.1,
            )

            coefficient_view = np.log1p(np.abs(np.fft.fftshift(dct_coefficients)))
            show_image_grid(
                [image, observed, ista_image, coefficient_view],
                ["original", "blurred + noisy", "ISTA estimate", "log DCT coefficients"],
                colorscales=["Gray", "Gray", "Gray", "Magma"],
                height=430,
            )

            print("observed RMSE:", round(rmse(observed, image), 4))
            print("ISTA RMSE:", round(rmse(ista_image, image), 4))
            print("energy:", round(image_energies[0], 2), "->", round(image_energies[-1], 2))
            print("active DCT coefficients:", image_active[0], "->", image_active[-1])
            """
        ),
        md(
            """
            ### Exercise 4

            Change the regularization weight for the image. What improves and what gets worse?
            """
        ),
        code(
            """
            # TODO: change this value.
            image_lam = 0.009

            experiment, coefficients, energies, active = ista_dct_deblur(
                observed,
                lam=image_lam,
                step_size=0.95,
                iterations=65,
                blur_sigma=1.1,
            )

            show_image_grid([observed, experiment], ["observation", f"ISTA lam={image_lam:g}"], height=400)
            print("RMSE:", round(rmse(experiment, image), 4))
            print("final energy:", round(energies[-1], 2))
            print("active DCT coefficients:", active[-1])
            """
        ),
        md(
            """
            ## Checks

            Answer in your own words:

            1. Why does step size matter in gradient descent?
            2. What does soft thresholding do to small coefficients?
            3. Why is a proximal step useful for l1 or TV-like penalties?
            4. In ISTA, which part handles the data term and which part handles the prior?
            """
        ),
        md(
            """
            ## Next Steps

            Week 10 studies sparse reconstruction in more depth: why sparsity can make an inverse problem easier and how l1 differs from l2.
            """
        ),
    ]


def week10_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Week 10 - Sparse Reconstruction

            This notebook accompanies the tenth MATH 435 slide deck.

            ## Goal

            By the end, you should be able to:

            - explain sparsity in a chosen representation;
            - compare l1 and l2 regularization on an underdetermined problem;
            - run an l1 reconstruction with ISTA;
            - test how the number of measurements affects sparse recovery;
            - reconstruct an image from random pixels using a sparse DCT prior.
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
            from scipy.fft import dctn, idctn
            from skimage import data
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots


            def soft_threshold(values, threshold):
                return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


            def rmse(estimate, reference):
                return float(np.sqrt(np.mean((estimate - reference) ** 2)))


            def make_sparse_problem(measurements=32, unknowns=96, sparsity=6, seed=43510):
                rng_matrix = np.random.default_rng(seed)
                rng_signal = np.random.default_rng(seed + 1)
                full_rows = max(64, measurements)
                full_matrix = rng_matrix.standard_normal((full_rows, unknowns))
                matrix = full_matrix[:measurements] / np.sqrt(measurements)
                matrix = matrix / np.linalg.norm(matrix, axis=0, keepdims=True)

                support = np.sort(rng_signal.choice(unknowns, size=sparsity, replace=False))
                amplitudes = rng_signal.choice([-1.0, 1.0], size=sparsity) * rng_signal.uniform(0.65, 1.25, size=sparsity)
                true_signal = np.zeros(unknowns)
                true_signal[support] = amplitudes
                measurements_vector = matrix @ true_signal
                return matrix, true_signal, measurements_vector, support


            def minimum_l2_solution(matrix, measurements_vector):
                gram = matrix @ matrix.T
                return matrix.T @ np.linalg.solve(gram + 1e-10 * np.eye(gram.shape[0]), measurements_vector)


            def ista_l1(matrix, measurements_vector, lam, iterations=1200):
                lipschitz = float(np.linalg.norm(matrix, 2) ** 2)
                step_size = 0.95 / lipschitz
                estimate = np.zeros(matrix.shape[1])
                energies = []
                for _ in range(iterations):
                    residual = matrix @ estimate - measurements_vector
                    gradient = matrix.T @ residual
                    estimate = soft_threshold(estimate - step_size * gradient, step_size * lam)
                    energy = 0.5 * float(np.sum((matrix @ estimate - measurements_vector) ** 2)) + lam * float(np.sum(np.abs(estimate)))
                    energies.append(energy)
                return estimate, np.array(energies)


            def relative_error(estimate, reference):
                return float(np.linalg.norm(estimate - reference) / np.linalg.norm(reference))


            def active_count(values, tolerance=1e-2):
                return int(np.count_nonzero(np.abs(values) > tolerance))


            def show_signal_comparison(signals, titles, height=650):
                fig = make_subplots(rows=len(signals), cols=1, subplot_titles=titles, shared_xaxes=True)
                for row, (values, title) in enumerate(zip(signals, titles), start=1):
                    index = np.arange(len(values))
                    fig.add_trace(go.Scatter(x=index, y=values, mode="markers", name=title), row=row, col=1)
                    fig.add_trace(go.Scatter(x=index, y=np.zeros_like(values), mode="lines", line=dict(color="lightgray"), showlegend=False), row=row, col=1)
                    fig.update_yaxes(title_text="value", row=row, col=1)
                fig.update_xaxes(title_text="coefficient index", row=len(signals), col=1)
                fig.update_layout(height=height, margin=dict(l=20, r=20, t=70, b=45), showlegend=False)
                fig.show()


            def show_image_grid(images, titles, colorscales=None, zmin=0, zmax=1, height=430):
                if colorscales is None:
                    colorscales = ["Gray"] * len(images)
                fig = make_subplots(rows=1, cols=len(images), subplot_titles=titles)
                for index, (image, colorscale) in enumerate(zip(images, colorscales), start=1):
                    fig.add_trace(
                        go.Heatmap(
                            z=image,
                            colorscale=colorscale,
                            zmin=zmin if colorscale == "Gray" else None,
                            zmax=zmax if colorscale == "Gray" else None,
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

            ### 1. Sparse and Dense Coefficients

            Sparsity means that most coefficients are zero or negligible in a chosen representation.
            """
        ),
        code(
            """
            rng = np.random.default_rng(43510)
            n = 72
            sparse = np.zeros(n)
            sparse[[6, 18, 31, 44, 57, 66]] = [1.0, -0.8, 0.7, 1.15, -0.65, 0.9]
            dense = 0.18 * rng.standard_normal(n) + 0.25 * np.sin(np.linspace(0, 5 * np.pi, n))

            show_signal_comparison(
                [sparse, dense],
                [f"sparse vector: {active_count(sparse)} active entries", f"dense vector: {active_count(dense)} active entries"],
                height=500,
            )
            """
        ),
        md(
            """
            ### Exercise 1

            Change `tolerance`. When should a coefficient count as negligible rather than nonzero?
            """
        ),
        code(
            """
            # TODO: change this value.
            tolerance = 0.05

            print("sparse active count:", active_count(sparse, tolerance))
            print("dense active count:", active_count(dense, tolerance))
            """
        ),
        md(
            """
            ### 2. l1 Versus l2 Geometry

            l2 balls are round. l1 balls have corners on coordinate axes.
            Those corners are one reason l1 can produce exact zeros.
            """
        ),
        code(
            """
            x = np.linspace(-1.0, 1.35, 400)
            y = (1.0 - x) / 2.0
            l2_solution = np.array([0.2, 0.4])
            l1_solution = np.array([0.0, 0.5])

            theta = np.linspace(0, 2 * np.pi, 500)
            radius = np.linalg.norm(l2_solution)
            diamond_radius = np.sum(np.abs(l1_solution))
            diamond_x = [diamond_radius, 0, -diamond_radius, 0, diamond_radius]
            diamond_y = [0, diamond_radius, 0, -diamond_radius, 0]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="data constraint"))
            fig.add_trace(go.Scatter(x=radius * np.cos(theta), y=radius * np.sin(theta), mode="lines", name="l2 ball"))
            fig.add_trace(go.Scatter(x=diamond_x, y=diamond_y, mode="lines", name="l1 ball"))
            fig.add_trace(go.Scatter(x=[l2_solution[0]], y=[l2_solution[1]], mode="markers", name="l2 solution", marker=dict(size=11)))
            fig.add_trace(go.Scatter(x=[l1_solution[0]], y=[l1_solution[1]], mode="markers", name="l1 solution", marker=dict(size=11)))
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_layout(
                title="First contact with a data constraint",
                xaxis_title="x1",
                yaxis_title="x2",
                height=520,
                margin=dict(l=20, r=20, t=60, b=45),
            )
            fig.show()
            """
        ),
        md(
            """
            ### 3. Underdetermined Sparse Recovery

            We now create a problem with fewer measurements than unknowns.

            The true signal is sparse, but the reconstruction algorithm does not get told the support.
            """
        ),
        code(
            """
            matrix, true_signal, measurements_vector, support = make_sparse_problem(
                measurements=32,
                unknowns=96,
                sparsity=6,
            )

            l2_solution = minimum_l2_solution(matrix, measurements_vector)
            l1_solution, l1_energies = ista_l1(matrix, measurements_vector, lam=0.015, iterations=1200)

            show_signal_comparison(
                [true_signal, l2_solution, l1_solution],
                [
                    f"true sparse signal: support={support.tolist()}",
                    f"minimum l2-norm solution: active={active_count(l2_solution)}",
                    f"l1-regularized solution: active={active_count(l1_solution)}",
                ],
                height=720,
            )

            for name, estimate in [("l2", l2_solution), ("l1", l1_solution)]:
                print(
                    f"{name:3s}",
                    "relative error=", round(relative_error(estimate, true_signal), 4),
                    "residual=", round(float(np.linalg.norm(matrix @ estimate - measurements_vector)), 6),
                    "active=", active_count(estimate),
                )
            """
        ),
        md(
            """
            ### Reading the Result

            The l2 solution fits the measurements almost exactly, but it spreads coefficients across many entries.

            The l1 solution allows a small residual but is much closer to the sparse signal.
            """
        ),
        code(
            """
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=l1_energies, mode="lines", name="l1 objective"))
            fig.update_layout(
                title="l1 objective through ISTA iterations",
                xaxis_title="iteration",
                yaxis_title="objective value",
                height=360,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()
            """
        ),
        md(
            """
            ### Exercise 2

            Change the number of measurements and the l1 regularization weight.

            What changes first: residual, sparsity, or reconstruction error?
            """
        ),
        code(
            """
            # TODO: change these values.
            measurements = 24
            lam = 0.015

            matrix_exp, true_exp, y_exp, support_exp = make_sparse_problem(measurements=measurements, unknowns=96, sparsity=6)
            l2_exp = minimum_l2_solution(matrix_exp, y_exp)
            l1_exp, _ = ista_l1(matrix_exp, y_exp, lam=lam, iterations=1200)

            print("support:", support_exp.tolist())
            for name, estimate in [("l2", l2_exp), ("l1", l1_exp)]:
                print(
                    f"{name:3s}",
                    "relative error=", round(relative_error(estimate, true_exp), 4),
                    "residual=", round(float(np.linalg.norm(matrix_exp @ estimate - y_exp)), 6),
                    "active=", active_count(estimate),
                )
            """
        ),
        md(
            """
            ### 4. Measurement Sweep

            Compressed sensing intuition: if the signal is sparse and the measurements mix information well, recovery can work with far fewer measurements than unknowns.
            """
        ),
        code(
            """
            counts = [12, 16, 20, 24, 28, 32, 40, 56, 64]
            l2_errors = []
            l1_errors = []
            l1_active = []

            for count in counts:
                matrix_sweep, true_sweep, y_sweep, _ = make_sparse_problem(measurements=count, unknowns=96, sparsity=6)
                l2_sweep = minimum_l2_solution(matrix_sweep, y_sweep)
                l1_sweep, _ = ista_l1(matrix_sweep, y_sweep, lam=0.015, iterations=1500)
                l2_errors.append(relative_error(l2_sweep, true_sweep))
                l1_errors.append(relative_error(l1_sweep, true_sweep))
                l1_active.append(active_count(l1_sweep))

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=counts, y=l2_errors, mode="lines+markers", name="minimum l2-norm"), secondary_y=False)
            fig.add_trace(go.Scatter(x=counts, y=l1_errors, mode="lines+markers", name="l1 regularized"), secondary_y=False)
            fig.add_trace(go.Scatter(x=counts, y=l1_active, mode="lines+markers", name="active l1 entries"), secondary_y=True)
            fig.update_xaxes(title_text="number of measurements")
            fig.update_yaxes(title_text="relative error", secondary_y=False)
            fig.update_yaxes(title_text="active l1 entries", secondary_y=True)
            fig.update_layout(title="Sparse recovery improves when there are enough measurements", height=430, margin=dict(l=20, r=20, t=55, b=45))
            fig.show()
            """
        ),
        md(
            """
            ### Exercise 3

            Change `sparsity` below. How many measurements are needed before l1 recovery becomes reliable?
            """
        ),
        code(
            """
            # TODO: change this value.
            sparsity = 10

            for count in [24, 32, 40, 56, 64]:
                matrix_test, true_test, y_test, _ = make_sparse_problem(measurements=count, unknowns=96, sparsity=sparsity)
                l1_test, _ = ista_l1(matrix_test, y_test, lam=0.015, iterations=1500)
                print(
                    f"measurements={count:2d}",
                    "relative error=", round(relative_error(l1_test, true_test), 4),
                    "active=", active_count(l1_test),
                )
            """
        ),
        md(
            """
            ### 5. Image Sparsity in the DCT

            A natural-image patch is usually not sparse in pixels, but it may be approximately sparse in a transform domain.
            """
        ),
        code(
            """
            image = data.camera().astype(float)[80:208, 128:256] / 255.0
            coefficients = dctn(image, norm="ortho")
            sorted_magnitudes = np.sort(np.abs(coefficients).ravel())[::-1]

            reconstructions = []
            titles = []
            for fraction in [0.10, 0.03, 0.01]:
                keep = max(1, int(fraction * coefficients.size))
                threshold = sorted_magnitudes[keep - 1]
                sparse_coefficients = np.where(np.abs(coefficients) >= threshold, coefficients, 0.0)
                reconstruction = np.clip(idctn(sparse_coefficients, norm="ortho"), 0.0, 1.0)
                reconstructions.append(reconstruction)
                titles.append(f"{100*fraction:.0f}% kept, RMSE={rmse(reconstruction, image):.3f}")

            show_image_grid(
                [image, np.log1p(np.abs(np.fft.fftshift(coefficients)))] + reconstructions,
                ["original", "log DCT magnitude"] + titles,
                colorscales=["Gray", "Magma", "Gray", "Gray", "Gray"],
                height=430,
            )
            """
        ),
        md(
            """
            ### 6. Sparse DCT Reconstruction from Random Pixels

            Now we observe only a random subset of pixels and reconstruct DCT coefficients with an l1 penalty.
            """
        ),
        code(
            """
            def dct_sparse_inpaint(observed, mask, lam, iterations=180):
                coefficients = np.zeros_like(observed)
                energies = []
                for _ in range(iterations):
                    image_estimate = idctn(coefficients, norm="ortho")
                    residual = (image_estimate - observed) * mask
                    gradient = dctn(residual, norm="ortho")
                    coefficients = soft_threshold(coefficients - 0.95 * gradient, 0.95 * lam)
                    image_estimate = idctn(coefficients, norm="ortho")
                    energy = 0.5 * float(np.sum(((image_estimate - observed) * mask) ** 2)) + lam * float(np.sum(np.abs(coefficients)))
                    energies.append(energy)
                return np.clip(idctn(coefficients, norm="ortho"), 0.0, 1.0), coefficients, np.array(energies)


            rng = np.random.default_rng(43510)
            image_small = data.camera().astype(float)[96:160, 128:192] / 255.0
            mask = rng.random(image_small.shape) < 0.40
            observed = image_small * mask

            reconstruction, sparse_coefficients, inpaint_energies = dct_sparse_inpaint(
                observed,
                mask,
                lam=0.008,
                iterations=180,
            )

            show_image_grid(
                [
                    image_small,
                    observed,
                    reconstruction,
                    mask.astype(float),
                    np.log1p(np.abs(np.fft.fftshift(sparse_coefficients))),
                ],
                [
                    "original",
                    f"observed ({mask.mean():.0%} pixels)",
                    f"sparse DCT, RMSE={rmse(reconstruction, image_small):.3f}",
                    "mask",
                    "log DCT coefficients",
                ],
                colorscales=["Gray", "Gray", "Gray", "Gray", "Magma"],
                height=430,
            )

            print("zero-filled RMSE:", round(rmse(observed, image_small), 4))
            print("sparse DCT RMSE:", round(rmse(reconstruction, image_small), 4))
            print("active DCT coefficients:", active_count(sparse_coefficients, tolerance=1e-3))
            """
        ),
        md(
            """
            ### Exercise 4

            Change the observed-pixel fraction and l1 weight.

            What happens when the image is sampled less densely?
            """
        ),
        code(
            """
            # TODO: change these values.
            observed_fraction = 0.25
            image_lam = 0.008

            rng = np.random.default_rng(43511)
            test_mask = rng.random(image_small.shape) < observed_fraction
            test_observed = image_small * test_mask
            test_reconstruction, test_coefficients, _ = dct_sparse_inpaint(
                test_observed,
                test_mask,
                lam=image_lam,
                iterations=180,
            )

            show_image_grid(
                [image_small, test_observed, test_reconstruction],
                ["original", f"observed ({test_mask.mean():.0%} pixels)", f"reconstruction, lam={image_lam:g}"],
                height=400,
            )
            print("zero-filled RMSE:", round(rmse(test_observed, image_small), 4))
            print("sparse DCT RMSE:", round(rmse(test_reconstruction, image_small), 4))
            print("active DCT coefficients:", active_count(test_coefficients, tolerance=1e-3))
            """
        ),
        md(
            """
            ## Checks

            Answer in your own words:

            1. Why is sparsity representation-dependent?
            2. Why does l1 promote exact zeros more than l2?
            3. Why can an l2 solution fit the data but fail to recover the sparse signal?
            4. What assumptions are needed for compressed sensing-style reconstruction?
            """
        ),
        md(
            """
            ## Next Steps

            Week 11 studies wavelets and multiscale representations, which are a natural sparse representation for many images.
            """
        ),
    ]


def week11_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Week 11 - Wavelets and Multiscale Representation

            This notebook accompanies the eleventh MATH 435 slide deck.

            ## Goal

            By the end, you should be able to:

            - explain wavelets as localized multiscale patterns;
            - inspect 1D and 2D wavelet decompositions;
            - connect wavelet coefficients to sparsity;
            - compress an image by keeping large wavelet coefficients;
            - denoise an image by thresholding wavelet detail coefficients.
            """
        ),
        md(
            """
            ## Setup

            The notebook uses NumPy, SciPy, scikit-image, PyWavelets, and Plotly.

            If you run this outside Google Colab and a package is missing, install the course requirements from the repository root:

            ```bash
            python3 -m pip install -r requirements.txt
            ```
            """
        ),
        code(
            """
            import numpy as np
            import pywt
            from scipy.fft import dctn
            from scipy.ndimage import gaussian_filter
            from skimage import data
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots


            def rmse(estimate, reference):
                return float(np.sqrt(np.mean((estimate - reference) ** 2)))


            def show_image_grid(images, titles, colorscales=None, zmin=0, zmax=1, height=430):
                if colorscales is None:
                    colorscales = ["Gray"] * len(images)
                fig = make_subplots(rows=1, cols=len(images), subplot_titles=titles)
                for index, (image, colorscale) in enumerate(zip(images, colorscales), start=1):
                    fig.add_trace(
                        go.Heatmap(
                            z=image,
                            colorscale=colorscale,
                            zmin=zmin if colorscale == "Gray" else None,
                            zmax=zmax if colorscale == "Gray" else None,
                            showscale=index == len(images),
                        ),
                        row=1,
                        col=index,
                    )
                    fig.update_xaxes(showticklabels=False, row=1, col=index)
                    fig.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=index)
                fig.update_layout(height=height, margin=dict(l=20, r=20, t=60, b=20))
                fig.show()


            def wavelet_compress(image, fraction, wavelet="db2", level=3):
                coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level, mode="periodization")
                array, slices = pywt.coeffs_to_array(coeffs)
                keep = max(1, int(fraction * array.size))
                threshold = np.partition(np.abs(array).ravel(), -keep)[-keep]
                compressed_array = np.where(np.abs(array) >= threshold, array, 0.0)
                compressed_coeffs = pywt.array_to_coeffs(compressed_array, slices, output_format="wavedec2")
                reconstruction = pywt.waverec2(compressed_coeffs, wavelet=wavelet, mode="periodization")
                return np.clip(reconstruction[: image.shape[0], : image.shape[1]], 0.0, 1.0), keep


            def estimate_noise_sigma(coeffs):
                finest_diagonal = coeffs[-1][2]
                return float(np.median(np.abs(finest_diagonal)) / 0.6745)


            def wavelet_denoise(noisy, wavelet="db2", level=3, threshold_scale=1.0):
                coeffs = pywt.wavedec2(noisy, wavelet=wavelet, level=level, mode="periodization")
                sigma = estimate_noise_sigma(coeffs)
                threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(noisy.size))
                denoised_coeffs = [coeffs[0]]
                for horizontal, vertical, diagonal in coeffs[1:]:
                    denoised_coeffs.append(
                        tuple(pywt.threshold(detail, threshold, mode="soft") for detail in (horizontal, vertical, diagonal))
                    )
                reconstruction = pywt.waverec2(denoised_coeffs, wavelet=wavelet, mode="periodization")
                return np.clip(reconstruction[: noisy.shape[0], : noisy.shape[1]], 0.0, 1.0), threshold
            """
        ),
        md(
            """
            ## Steps

            ### 1. Haar Atoms

            The Haar wavelet starts with a local average and a local difference.
            """
        ),
        code(
            """
            x = np.linspace(0, 1, 800, endpoint=False)
            scaling = np.ones_like(x)
            haar = np.where(x < 0.5, 1.0, -1.0)
            shifted = np.zeros_like(x)
            mask = (x >= 0.25) & (x < 0.75)
            shifted[mask] = np.where(x[mask] < 0.5, 1.0, -1.0)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=scaling, mode="lines", name="scaling"))
            fig.add_trace(go.Scatter(x=x, y=haar, mode="lines", name="Haar wavelet"))
            fig.add_trace(go.Scatter(x=x, y=shifted, mode="lines", name="shifted and scaled"))
            fig.update_layout(
                title="Localized Haar patterns",
                xaxis_title="position",
                yaxis_title="amplitude",
                height=380,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()
            """
        ),
        md(
            """
            ### Exercise 1

            For two values `a` and `b`, compute the Haar average and detail.

            Change the values below. Which pair has the larger local change?
            """
        ),
        code(
            """
            # TODO: change these values.
            a = 8
            b = 6

            average = 0.5 * (a + b)
            detail = 0.5 * (a - b)

            print("average:", average)
            print("detail:", detail)
            """
        ),
        md(
            """
            ### 2. 1D Multiscale Decomposition

            A wavelet transform represents a signal as a coarse approximation plus details at several scales.
            """
        ),
        code(
            """
            def reconstruct_1d_component(coeffs, keep_index, wavelet, length):
                selected = []
                for index, coeff in enumerate(coeffs):
                    selected.append(coeff if index == keep_index else np.zeros_like(coeff))
                return pywt.waverec(selected, wavelet=wavelet, mode="periodization")[:length]


            n = 256
            x = np.linspace(0, 1, n, endpoint=False)
            signal = 0.35 + 0.45 * (x > 0.34) - 0.25 * (x > 0.68)
            signal += 0.15 * np.sin(2 * np.pi * 6 * x) * ((x > 0.48) & (x < 0.9))
            signal += 0.12 * np.exp(-((x - 0.16) / 0.035) ** 2)

            coeffs_1d = pywt.wavedec(signal, wavelet="haar", level=3, mode="periodization")
            approximation = reconstruct_1d_component(coeffs_1d, 0, "haar", n)
            detail3 = reconstruct_1d_component(coeffs_1d, 1, "haar", n)
            detail2 = reconstruct_1d_component(coeffs_1d, 2, "haar", n)
            detail1 = reconstruct_1d_component(coeffs_1d, 3, "haar", n)

            fig = make_subplots(rows=5, cols=1, shared_xaxes=True, subplot_titles=["signal", "A3 coarse approximation", "D3 detail", "D2 detail", "D1 detail"])
            for row, (values, name) in enumerate(
                [(signal, "signal"), (approximation, "A3"), (detail3, "D3"), (detail2, "D2"), (detail1, "D1")],
                start=1,
            ):
                fig.add_trace(go.Scatter(x=x, y=values, mode="lines", name=name), row=row, col=1)
            fig.update_layout(height=720, margin=dict(l=20, r=20, t=80, b=45), showlegend=False)
            fig.show()

            reconstruction = approximation + detail3 + detail2 + detail1
            print("reconstruction RMSE:", rmse(reconstruction, signal))
            """
        ),
        md(
            """
            ### Exercise 2

            Change the wavelet level. What changes when you use more or fewer scales?
            """
        ),
        code(
            """
            # TODO: change this value.
            level = 4

            coeffs_experiment = pywt.wavedec(signal, wavelet="haar", level=level, mode="periodization")
            reconstructed = pywt.waverec(coeffs_experiment, wavelet="haar", mode="periodization")[:n]

            print("number of coefficient groups:", len(coeffs_experiment))
            print("coarsest approximation length:", len(coeffs_experiment[0]))
            print("perfect reconstruction RMSE:", rmse(reconstructed, signal))
            """
        ),
        md(
            """
            ### 3. 2D Wavelet Subbands

            For images, wavelets produce a coarse approximation plus horizontal, vertical, and diagonal detail subbands.
            """
        ),
        code(
            """
            image = data.camera().astype(float)[80:208, 128:256] / 255.0
            coeffs_2d = pywt.wavedec2(image, wavelet="haar", level=2, mode="periodization")
            coeff_array, coeff_slices = pywt.coeffs_to_array(coeffs_2d)
            signed_log_coefficients = np.sign(coeff_array) * np.log1p(np.abs(coeff_array))

            show_image_grid(
                [image, signed_log_coefficients],
                ["image patch", "signed log wavelet coefficients"],
                colorscales=["Gray", "RdBu"],
                height=460,
            )

            print("image shape:", image.shape)
            print("coefficient array shape:", coeff_array.shape)
            print("coefficient groups:", len(coeffs_2d))
            """
        ),
        md(
            """
            ### 4. Coefficient Decay

            A sparse representation has a few large coefficients and many small ones.
            """
        ),
        code(
            """
            pixels = np.abs(image.ravel())
            dct_values = np.abs(dctn(image, norm="ortho").ravel())
            wavelet_values = np.abs(coeff_array.ravel())

            fig = go.Figure()
            for values, name in [(pixels, "pixels"), (dct_values, "DCT"), (wavelet_values, "wavelet")]:
                sorted_values = np.sort(values)[::-1]
                fig.add_trace(go.Scatter(y=sorted_values + 1e-8, mode="lines", name=name))
            fig.update_layout(
                title="Sorted coefficient magnitudes",
                xaxis_title="sorted coefficient index",
                yaxis_title="magnitude",
                yaxis_type="log",
                height=420,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()
            """
        ),
        md(
            """
            ### 5. Wavelet Compression

            We keep only the largest wavelet coefficients and set the rest to zero.
            """
        ),
        code(
            """
            fractions = [0.10, 0.03, 0.01]
            reconstructions = []
            titles = []
            for fraction in fractions:
                reconstruction, keep = wavelet_compress(image, fraction=fraction, wavelet="db2", level=3)
                reconstructions.append(reconstruction)
                titles.append(f"{100*fraction:.0f}% kept, RMSE={rmse(reconstruction, image):.3f}")

            show_image_grid(
                [image] + reconstructions,
                ["original"] + titles,
                height=430,
            )
            """
        ),
        md(
            """
            ### Exercise 3

            Change `keep_fraction` and `wavelet_name`.

            Try `haar`, `db2`, or `sym4`.
            """
        ),
        code(
            """
            # TODO: change these values.
            keep_fraction = 0.02
            wavelet_name = "db2"

            compressed, keep = wavelet_compress(image, fraction=keep_fraction, wavelet=wavelet_name, level=3)
            show_image_grid([image, compressed], ["original", f"{wavelet_name}, {100*keep_fraction:.1f}% kept"], height=400)

            print("kept coefficients:", keep)
            print("RMSE:", round(rmse(compressed, image), 4))
            """
        ),
        md(
            """
            ### 6. Wavelet Threshold Denoising

            Noise often creates many small detail coefficients. We threshold those details and reconstruct.
            """
        ),
        code(
            """
            rng = np.random.default_rng(43511)
            noisy = np.clip(image + 0.08 * rng.standard_normal(image.shape), 0.0, 1.0)
            gaussian = np.clip(gaussian_filter(noisy, sigma=1.0, mode="reflect"), 0.0, 1.0)
            wavelet_result, threshold = wavelet_denoise(noisy, wavelet="db2", level=3, threshold_scale=0.30)

            show_image_grid(
                [image, noisy, gaussian, wavelet_result],
                [
                    "original",
                    f"noisy, RMSE={rmse(noisy, image):.3f}",
                    f"Gaussian, RMSE={rmse(gaussian, image):.3f}",
                    f"wavelet, RMSE={rmse(wavelet_result, image):.3f}",
                ],
                height=430,
            )

            print("threshold:", round(threshold, 4))
            """
        ),
        md(
            """
            ### Exercise 4

            Change the threshold scale. What happens to noise and fine details?
            """
        ),
        code(
            """
            # TODO: change this value.
            threshold_scale = 0.6

            denoised, threshold = wavelet_denoise(noisy, wavelet="db2", level=3, threshold_scale=threshold_scale)

            show_image_grid(
                [noisy, denoised],
                ["noisy", f"threshold scale={threshold_scale:g}"],
                height=400,
            )
            print("threshold:", round(threshold, 4))
            print("RMSE:", round(rmse(denoised, image), 4))
            """
        ),
        md(
            """
            ### 7. Threshold Sweep

            The denoising threshold controls the noise-detail tradeoff.
            """
        ),
        code(
            """
            scales = [0.20, 0.30, 0.60]
            sweep_images = [noisy]
            sweep_titles = ["noisy"]
            for scale in scales:
                denoised, threshold = wavelet_denoise(noisy, wavelet="db2", level=3, threshold_scale=scale)
                sweep_images.append(denoised)
                sweep_titles.append(f"scale={scale:g}, RMSE={rmse(denoised, image):.3f}")

            show_image_grid(sweep_images, sweep_titles, height=430)
            """
        ),
        md(
            """
            ## Checks

            Answer in your own words:

            1. How is a wavelet different from a Fourier mode?
            2. What is the difference between an approximation coefficient and a detail coefficient?
            3. Why are wavelet coefficients often sparse for images?
            4. Why does denoising by thresholding involve a tradeoff?
            """
        ),
        md(
            """
            ## Next Steps

            Week 12 compares model-based and data-driven imaging: handcrafted priors such as TV and wavelets versus learned priors.
            """
        ),
    ]


def week12_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Week 12 - Model-Based vs Data-Driven Imaging

            This notebook accompanies the twelfth MATH 435 slide deck.

            ## Goal

            By the end, you should be able to:

            - distinguish model-based and data-driven reconstruction;
            - interpret a neural network as a parameterized operator;
            - learn a simple patch prior from clean image examples;
            - compare Gaussian, wavelet, and learned patch denoising;
            - explain why distribution shift matters.
            """
        ),
        md(
            """
            ## Setup

            The notebook uses NumPy, SciPy, scikit-image, PyWavelets, and Plotly.

            If you run this outside Google Colab and a package is missing, install the course requirements from the repository root:

            ```bash
            python3 -m pip install -r requirements.txt
            ```
            """
        ),
        code(
            """
            import numpy as np
            import pywt
            from scipy.ndimage import gaussian_filter
            from skimage import data
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots


            def rmse(estimate, reference):
                return float(np.sqrt(np.mean((estimate - reference) ** 2)))


            def show_image_grid(images, titles, colorscales=None, zmin=0, zmax=1, height=430):
                if colorscales is None:
                    colorscales = ["Gray"] * len(images)
                fig = make_subplots(rows=1, cols=len(images), subplot_titles=titles)
                for index, (image, colorscale) in enumerate(zip(images, colorscales), start=1):
                    fig.add_trace(
                        go.Heatmap(
                            z=image,
                            colorscale=colorscale,
                            zmin=zmin if colorscale == "Gray" else None,
                            zmax=zmax if colorscale == "Gray" else None,
                            showscale=index == len(images),
                        ),
                        row=1,
                        col=index,
                    )
                    fig.update_xaxes(showticklabels=False, row=1, col=index)
                    fig.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=index)
                fig.update_layout(height=height, margin=dict(l=20, r=20, t=60, b=20))
                fig.show()


            def extract_random_patches(image, patch_size, count, seed):
                rng = np.random.default_rng(seed)
                rows, cols = image.shape
                patches = np.empty((count, patch_size * patch_size))
                for index in range(count):
                    row = rng.integers(0, rows - patch_size + 1)
                    col = rng.integers(0, cols - patch_size + 1)
                    patches[index] = image[row : row + patch_size, col : col + patch_size].reshape(-1)
                return patches


            def fit_patch_pca(patches):
                mean = patches.mean(axis=0)
                _, singular_values, components = np.linalg.svd(patches - mean, full_matrices=False)
                return mean, components, singular_values


            def pca_patch_denoise(noisy, mean, components, patch_size, n_components):
                rows, cols = noisy.shape
                estimate = np.zeros_like(noisy)
                weights = np.zeros_like(noisy)
                basis = components[:n_components]
                for row in range(rows - patch_size + 1):
                    for col in range(cols - patch_size + 1):
                        patch = noisy[row : row + patch_size, col : col + patch_size].reshape(-1)
                        scores = (patch - mean) @ basis.T
                        reconstructed = mean + scores @ basis
                        estimate[row : row + patch_size, col : col + patch_size] += reconstructed.reshape(patch_size, patch_size)
                        weights[row : row + patch_size, col : col + patch_size] += 1.0
                return np.clip(estimate / np.maximum(weights, 1.0), 0.0, 1.0)


            def wavelet_denoise(noisy, threshold_scale=0.30):
                coeffs = pywt.wavedec2(noisy, wavelet="db2", level=3, mode="periodization")
                sigma = float(np.median(np.abs(coeffs[-1][2])) / 0.6745)
                threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(noisy.size))
                denoised_coeffs = [coeffs[0]]
                for horizontal, vertical, diagonal in coeffs[1:]:
                    denoised_coeffs.append(
                        tuple(pywt.threshold(detail, threshold, mode="soft") for detail in (horizontal, vertical, diagonal))
                    )
                reconstruction = pywt.waverec2(denoised_coeffs, wavelet="db2", mode="periodization")
                return np.clip(reconstruction[: noisy.shape[0], : noisy.shape[1]], 0.0, 1.0)
            """
        ),
        md(
            """
            ## Steps

            ### 1. Model-Based and Data-Driven Inputs

            A model-based method starts with an explicit forward model and prior.

            A data-driven method starts with examples and a loss.
            """
        ),
        code(
            """
            image = data.camera().astype(float) / 255.0
            train_image = image[:320, :320]
            test_image = image[352:448, 256:352]

            show_image_grid(
                [train_image, test_image],
                ["training image region", "held-out test crop"],
                height=430,
            )
            """
        ),
        md(
            """
            ### 2. A Model-Based Baseline

            We compare two handcrafted denoisers:

            - Gaussian smoothing;
            - wavelet thresholding.
            """
        ),
        code(
            """
            rng = np.random.default_rng(43512)
            noise_sigma = 0.08
            noisy = np.clip(test_image + noise_sigma * rng.standard_normal(test_image.shape), 0.0, 1.0)
            gaussian = np.clip(gaussian_filter(noisy, sigma=1.0, mode="reflect"), 0.0, 1.0)
            wavelet = wavelet_denoise(noisy, threshold_scale=0.30)

            show_image_grid(
                [test_image, noisy, gaussian, wavelet],
                [
                    "clean",
                    f"noisy, RMSE={rmse(noisy, test_image):.3f}",
                    f"Gaussian, RMSE={rmse(gaussian, test_image):.3f}",
                    f"wavelet, RMSE={rmse(wavelet, test_image):.3f}",
                ],
                height=430,
            )
            """
        ),
        md(
            """
            ### 3. Learn a Patch Prior

            We learn a low-dimensional patch model from clean training patches using SVD/PCA.

            This is data-driven, but not a neural network.
            """
        ),
        code(
            """
            patch_size = 7
            training_patches = extract_random_patches(
                train_image,
                patch_size=patch_size,
                count=6000,
                seed=43512,
            )
            patch_mean, patch_components, singular_values = fit_patch_pca(training_patches)

            print("patch dimension:", patch_size * patch_size)
            print("training patches:", training_patches.shape[0])
            print("learned components:", patch_components.shape[0])

            fig = go.Figure()
            fig.add_trace(go.Scatter(y=singular_values[:30], mode="lines+markers"))
            fig.update_layout(
                title="Learned patch spectrum",
                xaxis_title="component index",
                yaxis_title="singular value",
                height=360,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()
            """
        ),
        md(
            """
            ### Learned Patch Atoms

            The first PCA directions act like learned patch patterns.
            """
        ),
        code(
            """
            atoms = [patch_mean.reshape(patch_size, patch_size)]
            atoms += [patch_components[index].reshape(patch_size, patch_size) for index in range(9)]
            atom_titles = ["mean"] + [f"atom {index + 1}" for index in range(9)]

            show_image_grid(
                atoms,
                atom_titles,
                colorscales=["Gray"] * len(atoms),
                height=310,
            )
            """
        ),
        md(
            """
            ### 4. Learned Patch Denoising

            To denoise a patch, we project it onto the learned patch subspace and reconstruct it.
            """
        ),
        code(
            """
            learned = pca_patch_denoise(
                noisy,
                patch_mean,
                patch_components,
                patch_size=patch_size,
                n_components=16,
            )

            show_image_grid(
                [test_image, noisy, gaussian, wavelet, learned],
                [
                    "clean",
                    f"noisy, RMSE={rmse(noisy, test_image):.3f}",
                    f"Gaussian, RMSE={rmse(gaussian, test_image):.3f}",
                    f"wavelet, RMSE={rmse(wavelet, test_image):.3f}",
                    f"learned PCA, RMSE={rmse(learned, test_image):.3f}",
                ],
                height=430,
            )
            """
        ),
        md(
            """
            ### Exercise 1

            Change `n_components`. What happens when the learned model is too small or too large?
            """
        ),
        code(
            """
            # TODO: change this value.
            n_components = 8

            experiment = pca_patch_denoise(
                noisy,
                patch_mean,
                patch_components,
                patch_size=patch_size,
                n_components=n_components,
            )

            show_image_grid([noisy, experiment], ["noisy", f"PCA components={n_components}"], height=390)
            print("RMSE:", round(rmse(experiment, test_image), 4))
            """
        ),
        md(
            """
            ### 5. Capacity Curve

            The number of learned components controls the prior strength.
            """
        ),
        code(
            """
            component_counts = [2, 4, 8, 12, 16, 24, 32, 40]
            errors = []

            for count in component_counts:
                estimate = pca_patch_denoise(
                    noisy,
                    patch_mean,
                    patch_components,
                    patch_size=patch_size,
                    n_components=count,
                )
                errors.append(rmse(estimate, test_image))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=component_counts, y=errors, mode="lines+markers"))
            fig.update_layout(
                title="Denoising error versus learned model capacity",
                xaxis_title="PCA components",
                yaxis_title="RMSE",
                height=380,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()

            for count, error in zip(component_counts, errors):
                print(f"components={count:2d}, RMSE={error:.4f}")
            """
        ),
        md(
            """
            ### 6. Neural Network as a Parameterized Operator

            A neural network is also a parameterized map from input to output.

            Here is a tiny patch-to-patch architecture count. We do not train it; the point is to see where parameters live.
            """
        ),
        code(
            """
            input_dim = patch_size * patch_size
            hidden_units = 32
            output_dim = patch_size * patch_size

            parameter_count = input_dim * hidden_units + hidden_units + hidden_units * output_dim + output_dim

            print("input dimension:", input_dim)
            print("hidden units:", hidden_units)
            print("output dimension:", output_dim)
            print("trainable parameters:", parameter_count)
            """
        ),
        md(
            """
            ### 7. Distribution Shift

            A learned prior can work well when test data resembles training data.

            What happens when the test image changes?
            """
        ),
        code(
            """
            coins = data.coins().astype(float) / 255.0
            shifted_test = coins[40:136, 40:136]
            shifted_noisy = np.clip(shifted_test + noise_sigma * rng.standard_normal(shifted_test.shape), 0.0, 1.0)
            shifted_gaussian = np.clip(gaussian_filter(shifted_noisy, sigma=1.0, mode="reflect"), 0.0, 1.0)
            shifted_learned = pca_patch_denoise(
                shifted_noisy,
                patch_mean,
                patch_components,
                patch_size=patch_size,
                n_components=16,
            )

            show_image_grid(
                [shifted_test, shifted_noisy, shifted_gaussian, shifted_learned],
                [
                    "different test image",
                    f"noisy, RMSE={rmse(shifted_noisy, shifted_test):.3f}",
                    f"Gaussian, RMSE={rmse(shifted_gaussian, shifted_test):.3f}",
                    f"learned PCA, RMSE={rmse(shifted_learned, shifted_test):.3f}",
                ],
                height=430,
            )
            """
        ),
        md(
            """
            ### Exercise 2

            Change the training crop. Does the learned prior still help on the same test crop?
            """
        ),
        code(
            """
            # TODO: change these slices.
            alternative_train = image[0:192, 320:512]

            alternative_patches = extract_random_patches(alternative_train, patch_size=patch_size, count=4000, seed=43513)
            alternative_mean, alternative_components, _ = fit_patch_pca(alternative_patches)
            alternative_learned = pca_patch_denoise(
                noisy,
                alternative_mean,
                alternative_components,
                patch_size=patch_size,
                n_components=16,
            )

            show_image_grid(
                [alternative_train, test_image, alternative_learned],
                ["alternative training region", "test crop", f"learned result, RMSE={rmse(alternative_learned, test_image):.3f}"],
                height=430,
            )
            """
        ),
        md(
            """
            ## Checks

            Answer in your own words:

            1. What does a model-based method specify explicitly?
            2. What does a learned method get from training data?
            3. Why is a neural network a parameterized operator?
            4. Why can distribution shift be dangerous in imaging?
            """
        ),
        md(
            """
            ## Next Steps

            Week 13 studies plug-and-play and learned regularization: learned denoisers inside iterative reconstruction algorithms.
            """
        ),
    ]


def week13_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Week 13 - Plug-and-Play and Learned Regularization

            This notebook accompanies the thirteenth MATH 435 slide deck.

            ## Goal

            By the end, you should be able to:

            - implement a data-consistency step for deblurring;
            - replace a proximal step by a denoiser;
            - compare Tikhonov and plug-and-play reconstructions;
            - monitor fixed-point iterations;
            - test how learned denoisers depend on training data.
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
            from scipy.ndimage import gaussian_filter
            from skimage import data
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots


            def rmse(estimate, reference):
                return float(np.sqrt(np.mean((estimate - reference) ** 2)))


            def show_image_grid(images, titles, colorscales=None, zmin=0, zmax=1, height=430):
                if colorscales is None:
                    colorscales = ["Gray"] * len(images)
                fig = make_subplots(rows=1, cols=len(images), subplot_titles=titles)
                for index, (image, colorscale) in enumerate(zip(images, colorscales), start=1):
                    fig.add_trace(
                        go.Heatmap(
                            z=image,
                            colorscale=colorscale,
                            zmin=zmin if colorscale == "Gray" else None,
                            zmax=zmax if colorscale == "Gray" else None,
                            showscale=index == len(images),
                        ),
                        row=1,
                        col=index,
                    )
                    fig.update_xaxes(showticklabels=False, row=1, col=index)
                    fig.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=index)
                fig.update_layout(height=height, margin=dict(l=20, r=20, t=60, b=20))
                fig.show()


            def gaussian_kernel2d(size, sigma):
                axis = np.arange(-(size // 2), size // 2 + 1)
                xx, yy = np.meshgrid(axis, axis)
                kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
                return kernel / kernel.sum()


            def centered_kernel_fft(kernel, shape):
                padded = np.zeros(shape)
                kh, kw = kernel.shape
                padded[:kh, :kw] = kernel
                padded = np.roll(padded, -(kh // 2), axis=0)
                padded = np.roll(padded, -(kw // 2), axis=1)
                return np.fft.fft2(padded)


            def blur_periodic(image, kernel_fft):
                return np.real(np.fft.ifft2(np.fft.fft2(image) * kernel_fft))


            def adjoint_blur_periodic(image, kernel_fft):
                return np.real(np.fft.ifft2(np.fft.fft2(image) * np.conj(kernel_fft)))


            def data_gradient(estimate, observation, kernel_fft):
                residual = blur_periodic(estimate, kernel_fft) - observation
                return adjoint_blur_periodic(residual, kernel_fft)


            def data_residual(estimate, observation, kernel_fft):
                residual = blur_periodic(estimate, kernel_fft) - observation
                return float(np.linalg.norm(residual) / np.sqrt(residual.size))


            def tikhonov_deblur(observation, kernel_fft, lam):
                numerator = np.conj(kernel_fft) * np.fft.fft2(observation)
                denominator = np.abs(kernel_fft) ** 2 + lam
                return np.clip(np.real(np.fft.ifft2(numerator / denominator)), 0.0, 1.0)


            def pnp_gradient_descent(observation, kernel_fft, denoiser, iterations, step_size=1.0, clean=None):
                estimate = observation.copy()
                previous = estimate.copy()
                history = {"rmse": [], "residual": [], "change": []}

                for _ in range(iterations + 1):
                    if clean is not None:
                        history["rmse"].append(rmse(estimate, clean))
                    history["residual"].append(data_residual(estimate, observation, kernel_fft))
                    history["change"].append(float(np.linalg.norm(estimate - previous) / np.sqrt(estimate.size)))

                    previous = estimate.copy()
                    descent = estimate - step_size * data_gradient(estimate, observation, kernel_fft)
                    estimate = np.clip(denoiser(descent), 0.0, 1.0)

                return estimate, history


            def extract_random_patches(image, patch_size, count, seed):
                rng = np.random.default_rng(seed)
                rows, cols = image.shape
                patches = np.empty((count, patch_size * patch_size))
                for index in range(count):
                    row = rng.integers(0, rows - patch_size + 1)
                    col = rng.integers(0, cols - patch_size + 1)
                    patches[index] = image[row : row + patch_size, col : col + patch_size].reshape(-1)
                return patches


            def fit_patch_pca(patches):
                mean = patches.mean(axis=0)
                _, singular_values, components = np.linalg.svd(patches - mean, full_matrices=False)
                return mean, components, singular_values


            def pca_patch_denoise(noisy, mean, components, patch_size, n_components):
                rows, cols = noisy.shape
                estimate = np.zeros_like(noisy)
                weights = np.zeros_like(noisy)
                basis = components[:n_components]
                for row in range(rows - patch_size + 1):
                    for col in range(cols - patch_size + 1):
                        patch = noisy[row : row + patch_size, col : col + patch_size].reshape(-1)
                        scores = (patch - mean) @ basis.T
                        reconstructed = mean + scores @ basis
                        estimate[row : row + patch_size, col : col + patch_size] += reconstructed.reshape(patch_size, patch_size)
                        weights[row : row + patch_size, col : col + patch_size] += 1.0
                return np.clip(estimate / np.maximum(weights, 1.0), 0.0, 1.0)
            """
        ),
        md(
            """
            ## Steps

            ### 1. Build a Deblurring Problem

            We use a periodic convolution model:

            ```text
            y = A x + noise
            ```

            The functions above implement `A`, `A.T`, and the gradient of the data term.
            """
        ),
        code(
            """
            rng = np.random.default_rng(43513)

            image = data.camera().astype(float) / 255.0
            clean = image[132:228, 178:274]

            kernel = gaussian_kernel2d(size=21, sigma=2.4)
            kernel_fft = centered_kernel_fft(kernel, clean.shape)
            blurred = blur_periodic(clean, kernel_fft)
            observation = np.clip(blurred + 0.015 * rng.standard_normal(clean.shape), 0.0, 1.0)

            show_image_grid(
                [clean, kernel, observation],
                ["clean image crop", "blur kernel", f"blurred + noise, RMSE={rmse(observation, clean):.3f}"],
                colorscales=["Gray", "Viridis", "Gray"],
                height=420,
            )
            """
        ),
        md(
            """
            ### 2. Classical Baseline: Tikhonov

            Tikhonov deblurring solves a quadratic regularized problem. For periodic convolution, it has a Fourier-domain formula.
            """
        ),
        code(
            """
            tikhonov_lambda = 0.002
            tikhonov = tikhonov_deblur(observation, kernel_fft, lam=tikhonov_lambda)

            show_image_grid(
                [clean, observation, tikhonov],
                [
                    "clean",
                    f"observation, RMSE={rmse(observation, clean):.3f}",
                    f"Tikhonov, RMSE={rmse(tikhonov, clean):.3f}",
                ],
                height=420,
            )

            print("Tikhonov data residual:", round(data_residual(tikhonov, observation, kernel_fft), 4))
            """
        ),
        md(
            """
            ### 3. Plug-and-Play with a Gaussian Denoiser

            The PnP iteration alternates:

            ```text
            z = x - step_size * A.T(Ax - y)
            x = denoise(z)
            ```
            """
        ),
        code(
            """
            gaussian_sigma = 0.65
            gaussian_denoiser = lambda values: gaussian_filter(values, sigma=gaussian_sigma, mode="wrap")

            pnp_gaussian, gaussian_history = pnp_gradient_descent(
                observation,
                kernel_fft,
                gaussian_denoiser,
                iterations=24,
                step_size=1.0,
                clean=clean,
            )

            show_image_grid(
                [clean, observation, tikhonov, pnp_gaussian],
                [
                    "clean",
                    f"observation, RMSE={rmse(observation, clean):.3f}",
                    f"Tikhonov, RMSE={rmse(tikhonov, clean):.3f}",
                    f"PnP Gaussian, RMSE={rmse(pnp_gaussian, clean):.3f}",
                ],
                height=420,
            )

            print("PnP Gaussian data residual:", round(data_residual(pnp_gaussian, observation, kernel_fft), 4))
            """
        ),
        md(
            """
            ### Exercise 1

            Change `student_sigma`. What happens when denoising is too weak or too strong?
            """
        ),
        code(
            """
            # TODO: change this value.
            student_sigma = 0.35

            student_denoiser = lambda values: gaussian_filter(values, sigma=student_sigma, mode="wrap")
            student_pnp, student_history = pnp_gradient_descent(
                observation,
                kernel_fft,
                student_denoiser,
                iterations=24,
                step_size=1.0,
                clean=clean,
            )

            show_image_grid(
                [observation, student_pnp],
                ["observation", f"PnP sigma={student_sigma}, RMSE={rmse(student_pnp, clean):.3f}"],
                height=380,
            )
            print("data residual:", round(data_residual(student_pnp, observation, kernel_fft), 4))
            print("last fixed-point change:", round(student_history["change"][-1], 6))
            """
        ),
        md(
            """
            ### 4. Sweep the Denoiser Strength

            A stronger denoiser is not automatically better.
            """
        ),
        code(
            """
            sigmas = [0.0, 0.35, 0.65, 1.0, 1.45]
            sweep_rows = []

            for sigma in sigmas:
                if sigma == 0.0:
                    denoiser = lambda values: values
                else:
                    denoiser = lambda values, s=sigma: gaussian_filter(values, sigma=s, mode="wrap")

                estimate, history = pnp_gradient_descent(
                    observation,
                    kernel_fft,
                    denoiser,
                    iterations=24,
                    step_size=1.0,
                    clean=clean,
                )
                sweep_rows.append(
                    {
                        "sigma": sigma,
                        "rmse": rmse(estimate, clean),
                        "residual": data_residual(estimate, observation, kernel_fft),
                        "change": history["change"][-1],
                    }
                )

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=[row["sigma"] for row in sweep_rows], y=[row["rmse"] for row in sweep_rows], mode="lines+markers", name="RMSE"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=[row["sigma"] for row in sweep_rows], y=[row["residual"] for row in sweep_rows], mode="lines+markers", name="data residual"),
                secondary_y=True,
            )
            fig.update_xaxes(title_text="Gaussian denoiser sigma")
            fig.update_yaxes(title_text="RMSE", secondary_y=False)
            fig.update_yaxes(title_text="data residual", secondary_y=True)
            fig.update_layout(height=390, margin=dict(l=20, r=20, t=40, b=45))
            fig.show()

            for row in sweep_rows:
                print(row)
            """
        ),
        md(
            """
            ### 5. Plug-and-Play with a Learned Patch Denoiser

            Now we reuse the Week 12 idea: learn a PCA patch prior from clean camera patches.
            """
        ),
        code(
            """
            patch_size = 7
            train_image = image[:320, :320]
            training_patches = extract_random_patches(train_image, patch_size=patch_size, count=5000, seed=43513)
            patch_mean, patch_components, singular_values = fit_patch_pca(training_patches)

            learned_denoiser = lambda values: pca_patch_denoise(
                values,
                patch_mean,
                patch_components,
                patch_size=patch_size,
                n_components=18,
            )

            pnp_learned, learned_history = pnp_gradient_descent(
                observation,
                kernel_fft,
                learned_denoiser,
                iterations=24,
                step_size=1.0,
                clean=clean,
            )

            show_image_grid(
                [clean, observation, tikhonov, pnp_gaussian, pnp_learned],
                [
                    "clean",
                    f"observation, RMSE={rmse(observation, clean):.3f}",
                    f"Tikhonov, RMSE={rmse(tikhonov, clean):.3f}",
                    f"PnP Gaussian, RMSE={rmse(pnp_gaussian, clean):.3f}",
                    f"PnP learned, RMSE={rmse(pnp_learned, clean):.3f}",
                ],
                height=430,
            )
            """
        ),
        md(
            """
            ### 6. Monitor the Iteration

            A PnP algorithm may not minimize a known objective, so fixed-point change is useful to monitor.
            """
        ),
        code(
            """
            iterations = np.arange(len(gaussian_history["rmse"]))

            fig = make_subplots(rows=1, cols=2, subplot_titles=["RMSE", "fixed-point change"])
            fig.add_trace(go.Scatter(x=iterations, y=gaussian_history["rmse"], mode="lines+markers", name="Gaussian"), row=1, col=1)
            fig.add_trace(go.Scatter(x=iterations, y=learned_history["rmse"], mode="lines+markers", name="learned"), row=1, col=1)
            fig.add_trace(go.Scatter(x=iterations, y=gaussian_history["change"], mode="lines+markers", name="Gaussian change"), row=1, col=2)
            fig.add_trace(go.Scatter(x=iterations, y=learned_history["change"], mode="lines+markers", name="learned change"), row=1, col=2)
            fig.update_xaxes(title_text="iteration", row=1, col=1)
            fig.update_xaxes(title_text="iteration", row=1, col=2)
            fig.update_yaxes(title_text="RMSE", row=1, col=1)
            fig.update_yaxes(title_text="change", row=1, col=2)
            fig.update_layout(height=390, margin=dict(l=20, r=20, t=60, b=45))
            fig.show()
            """
        ),
        md(
            """
            ### Exercise 2

            Change `student_components`. Does the learned denoiser become too weak or too restrictive?
            """
        ),
        code(
            """
            # TODO: change this value.
            student_components = 10

            student_learned_denoiser = lambda values: pca_patch_denoise(
                values,
                patch_mean,
                patch_components,
                patch_size=patch_size,
                n_components=student_components,
            )
            student_learned_pnp, _ = pnp_gradient_descent(
                observation,
                kernel_fft,
                student_learned_denoiser,
                iterations=24,
                step_size=1.0,
                clean=clean,
            )

            show_image_grid(
                [pnp_learned, student_learned_pnp],
                ["18 components", f"{student_components} components"],
                height=380,
            )
            print("student RMSE:", round(rmse(student_learned_pnp, clean), 4))
            """
        ),
        md(
            """
            ### 7. Distribution Shift

            A learned denoiser trained on camera patches may not be ideal for a different image family.
            """
        ),
        code(
            """
            moon = data.moon().astype(float) / 255.0
            shifted_clean = moon[300:396, 100:196]
            shifted_kernel_fft = centered_kernel_fft(kernel, shifted_clean.shape)
            shifted_observation = np.clip(
                blur_periodic(shifted_clean, shifted_kernel_fft)
                + 0.015 * rng.standard_normal(shifted_clean.shape),
                0.0,
                1.0,
            )

            shifted_gaussian, _ = pnp_gradient_descent(
                shifted_observation,
                shifted_kernel_fft,
                gaussian_denoiser,
                iterations=24,
                step_size=1.0,
                clean=shifted_clean,
            )
            shifted_learned, _ = pnp_gradient_descent(
                shifted_observation,
                shifted_kernel_fft,
                learned_denoiser,
                iterations=24,
                step_size=1.0,
                clean=shifted_clean,
            )

            show_image_grid(
                [shifted_clean, shifted_observation, shifted_gaussian, shifted_learned],
                [
                    "moon crop",
                    f"observation, RMSE={rmse(shifted_observation, shifted_clean):.3f}",
                    f"PnP Gaussian, RMSE={rmse(shifted_gaussian, shifted_clean):.3f}",
                    f"camera-learned PnP, RMSE={rmse(shifted_learned, shifted_clean):.3f}",
                ],
                height=420,
            )
            """
        ),
        md(
            """
            ## Checks

            Answer in your own words:

            1. Which part of the PnP update uses the forward model?
            2. Which part acts like an implicit prior?
            3. Why might there be no explicit objective function?
            4. Why should we report data residuals and distribution-shift tests?
            """
        ),
        md(
            """
            ## Next Steps

            The remaining course work should connect these ideas to project design:

            - define a forward model;
            - choose a regularizer or denoiser;
            - decide which metrics matter;
            - document limitations and failure cases.
            """
        ),
    ]


def week14_cells() -> list[nbf.NotebookNode]:
    return [
        md(
            """
            # Week 14 - Stability, Robustness, and Ethics

            This notebook accompanies the fourteenth MATH 435 slide deck.

            ## Goal

            By the end, you should be able to:

            - run a parameter sensitivity test;
            - measure perturbation amplification;
            - check whether regularization suppresses a small feature;
            - test a learned prior across image families;
            - write a reliability checklist for an imaging project.
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
            from scipy.ndimage import gaussian_filter
            from skimage import data, draw
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots


            def rmse(estimate, reference):
                return float(np.sqrt(np.mean((estimate - reference) ** 2)))


            def relative_norm(values, reference):
                return float(np.linalg.norm(values) / max(np.linalg.norm(reference), 1e-12))


            def show_image_grid(images, titles, colorscales=None, zmin=0, zmax=1, height=430):
                if colorscales is None:
                    colorscales = ["Gray"] * len(images)
                fig = make_subplots(rows=1, cols=len(images), subplot_titles=titles)
                for index, (image, colorscale) in enumerate(zip(images, colorscales), start=1):
                    fig.add_trace(
                        go.Heatmap(
                            z=image,
                            colorscale=colorscale,
                            zmin=zmin if colorscale == "Gray" else None,
                            zmax=zmax if colorscale == "Gray" else None,
                            showscale=index == len(images),
                        ),
                        row=1,
                        col=index,
                    )
                    fig.update_xaxes(showticklabels=False, row=1, col=index)
                    fig.update_yaxes(autorange="reversed", showticklabels=False, row=1, col=index)
                fig.update_layout(height=height, margin=dict(l=20, r=20, t=60, b=20))
                fig.show()


            def gaussian_kernel2d(size, sigma):
                axis = np.arange(-(size // 2), size // 2 + 1)
                xx, yy = np.meshgrid(axis, axis)
                kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
                return kernel / kernel.sum()


            def centered_kernel_fft(kernel, shape):
                padded = np.zeros(shape)
                kh, kw = kernel.shape
                padded[:kh, :kw] = kernel
                padded = np.roll(padded, -(kh // 2), axis=0)
                padded = np.roll(padded, -(kw // 2), axis=1)
                return np.fft.fft2(padded)


            def blur_periodic(image, kernel_fft):
                return np.real(np.fft.ifft2(np.fft.fft2(image) * kernel_fft))


            def data_residual(estimate, observation, kernel_fft):
                residual = blur_periodic(estimate, kernel_fft) - observation
                return float(np.linalg.norm(residual) / np.sqrt(residual.size))


            def tikhonov_deblur_raw(observation, kernel_fft, lam):
                numerator = np.conj(kernel_fft) * np.fft.fft2(observation)
                denominator = np.abs(kernel_fft) ** 2 + lam
                return np.real(np.fft.ifft2(numerator / denominator))


            def tikhonov_deblur(observation, kernel_fft, lam):
                return np.clip(tikhonov_deblur_raw(observation, kernel_fft, lam), 0.0, 1.0)


            def local_contrast(image, center=(48, 48), radius=4):
                rr, cc = np.indices(image.shape)
                distance = np.sqrt((rr - center[0]) ** 2 + (cc - center[1]) ** 2)
                inner = image[distance <= radius]
                ring = image[(distance > radius + 3) & (distance <= radius + 10)]
                return float(inner.mean() - ring.mean())


            def extract_random_patches(image, patch_size, count, seed):
                rng = np.random.default_rng(seed)
                rows, cols = image.shape
                patches = np.empty((count, patch_size * patch_size))
                for index in range(count):
                    row = rng.integers(0, rows - patch_size + 1)
                    col = rng.integers(0, cols - patch_size + 1)
                    patches[index] = image[row : row + patch_size, col : col + patch_size].reshape(-1)
                return patches


            def fit_patch_pca(patches):
                mean = patches.mean(axis=0)
                _, singular_values, components = np.linalg.svd(patches - mean, full_matrices=False)
                return mean, components, singular_values


            def pca_patch_denoise(noisy, mean, components, patch_size, n_components):
                rows, cols = noisy.shape
                estimate = np.zeros_like(noisy)
                weights = np.zeros_like(noisy)
                basis = components[:n_components]
                for row in range(rows - patch_size + 1):
                    for col in range(cols - patch_size + 1):
                        patch = noisy[row : row + patch_size, col : col + patch_size].reshape(-1)
                        scores = (patch - mean) @ basis.T
                        reconstructed = mean + scores @ basis
                        estimate[row : row + patch_size, col : col + patch_size] += reconstructed.reshape(patch_size, patch_size)
                        weights[row : row + patch_size, col : col + patch_size] += 1.0
                return np.clip(estimate / np.maximum(weights, 1.0), 0.0, 1.0)
            """
        ),
        md(
            """
            ## Steps

            ### 1. Build a Deblurring Test Case

            We start with the same kind of inverse problem as earlier:

            ```text
            y = A x + noise
            ```

            The reliability question is not only "can we reconstruct?" but "when should we trust the reconstruction?"
            """
        ),
        code(
            """
            rng = np.random.default_rng(43514)

            image = data.camera().astype(float) / 255.0
            clean = image[132:228, 178:274]
            kernel = gaussian_kernel2d(size=21, sigma=2.4)
            kernel_fft = centered_kernel_fft(kernel, clean.shape)
            blurred = blur_periodic(clean, kernel_fft)
            observation = np.clip(blurred + 0.015 * rng.standard_normal(clean.shape), 0.0, 1.0)

            show_image_grid(
                [clean, kernel, observation],
                ["clean crop", "blur kernel", f"blurred + noise, RMSE={rmse(observation, clean):.3f}"],
                colorscales=["Gray", "Viridis", "Gray"],
                height=420,
            )
            """
        ),
        md(
            """
            ### 2. Parameter Sensitivity

            Sweep the Tikhonov parameter. Track both reconstruction error and data residual.
            """
        ),
        code(
            """
            lambda_grid = np.logspace(-5, -0.4, 18)
            sensitivity_rows = []

            for lam in lambda_grid:
                estimate = tikhonov_deblur(observation, kernel_fft, lam)
                sensitivity_rows.append(
                    {
                        "lambda": lam,
                        "rmse": rmse(estimate, clean),
                        "residual": data_residual(estimate, observation, kernel_fft),
                        "contrast": local_contrast(estimate),
                    }
                )

            best_row = min(sensitivity_rows, key=lambda row: row["rmse"])
            print("best lambda by RMSE:", f"{best_row['lambda']:.3e}")
            print("best RMSE:", round(best_row["rmse"], 4))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=lambda_grid, y=[row["rmse"] for row in sensitivity_rows], mode="lines+markers", name="RMSE"))
            fig.add_trace(go.Scatter(x=lambda_grid, y=[row["residual"] for row in sensitivity_rows], mode="lines+markers", name="data residual"))
            fig.add_vline(x=best_row["lambda"], line_dash="dash")
            fig.update_layout(
                title="Parameter sensitivity",
                xaxis_title="Tikhonov lambda",
                yaxis_title="score",
                xaxis_type="log",
                height=390,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()
            """
        ),
        md(
            """
            ### Exercise 1

            Choose three lambdas and compare the images. Which failure mode appears for weak and strong regularization?
            """
        ),
        code(
            """
            # TODO: change these values.
            student_lambdas = [1e-5, best_row["lambda"], 2e-1]

            student_images = [tikhonov_deblur(observation, kernel_fft, lam) for lam in student_lambdas]
            student_titles = [
                f"lambda={lam:.1e}, RMSE={rmse(estimate, clean):.3f}"
                for lam, estimate in zip(student_lambdas, student_images)
            ]

            show_image_grid(student_images, student_titles, height=410)
            """
        ),
        md(
            """
            ### 3. Perturbation Amplification

            Now add a tiny perturbation to the measured data in a vulnerable Fourier direction.

            We compare the relative change in the input data with the relative change in the reconstruction.
            """
        ),
        code(
            """
            rows, cols = clean.shape
            rr, cc = np.indices(clean.shape)
            target = np.sqrt(1e-5)
            magnitudes = np.abs(kernel_fft).copy()
            magnitudes[0, 0] = np.inf
            freq_row, freq_col = np.unravel_index(int(np.argmin(np.abs(magnitudes - target))), clean.shape)
            sinusoid = np.cos(2.0 * np.pi * (freq_row * rr / rows + freq_col * cc / cols))
            perturbation = 0.002 * sinusoid / np.sqrt(np.mean(sinusoid**2))
            perturbed_observation = observation + perturbation

            amplification_rows = []
            for lam in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
                base = tikhonov_deblur_raw(observation, kernel_fft, lam)
                perturbed = tikhonov_deblur_raw(perturbed_observation, kernel_fft, lam)
                input_change = relative_norm(perturbed_observation - observation, observation)
                output_change = relative_norm(perturbed - base, base)
                amplification_rows.append(
                    {
                        "lambda": lam,
                        "input_change": input_change,
                        "output_change": output_change,
                        "gain": output_change / input_change,
                    }
                )

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=[row["lambda"] for row in amplification_rows],
                    y=[row["gain"] for row in amplification_rows],
                    mode="lines+markers",
                )
            )
            fig.update_layout(
                title="Perturbation amplification",
                xaxis_title="Tikhonov lambda",
                yaxis_title="output-change / input-change",
                xaxis_type="log",
                yaxis_type="log",
                height=390,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()

            for row in amplification_rows:
                print(row)
            """
        ),
        md(
            """
            ### Exercise 2

            Change the perturbation amplitude from `0.002` to another value.

            Does the gain change? Why is that expected for a linear reconstruction formula?
            """
        ),
        code(
            """
            answer = "TODO: write your observation here after changing the perturbation amplitude."
            print(answer)
            """
        ),
        md(
            """
            ### 4. Small Feature Risk

            A reconstruction can look cleaner while weakening a small but important feature.
            """
        ),
        code(
            """
            feature_rng = np.random.default_rng(43514)
            background = gaussian_filter(feature_rng.random((96, 96)), sigma=7.0)
            background = (background - background.min()) / (background.max() - background.min())
            feature_clean = 0.30 + 0.35 * background
            disk_rows, disk_cols = draw.disk((48, 48), radius=4, shape=feature_clean.shape)
            feature_clean[disk_rows, disk_cols] = np.clip(feature_clean[disk_rows, disk_cols] + 0.32, 0.0, 1.0)

            feature_kernel = gaussian_kernel2d(size=19, sigma=2.1)
            feature_kernel_fft = centered_kernel_fft(feature_kernel, feature_clean.shape)
            feature_observation = np.clip(
                blur_periodic(feature_clean, feature_kernel_fft)
                + 0.012 * feature_rng.standard_normal(feature_clean.shape),
                0.0,
                1.0,
            )
            weak = tikhonov_deblur(feature_observation, feature_kernel_fft, lam=5e-4)
            strong = tikhonov_deblur(feature_observation, feature_kernel_fft, lam=5e-2)
            smoothed = gaussian_filter(strong, sigma=1.2, mode="wrap")

            feature_images = [feature_clean, feature_observation, weak, strong, smoothed]
            feature_titles = [
                f"clean, contrast={local_contrast(feature_clean):.3f}",
                "blurred + noise",
                f"weak, contrast={local_contrast(weak):.3f}",
                f"strong, contrast={local_contrast(strong):.3f}",
                f"smoothed, contrast={local_contrast(smoothed):.3f}",
            ]
            show_image_grid(feature_images, feature_titles, height=420)
            """
        ),
        md(
            """
            ### Exercise 3

            Change the disk radius or intensity. When does the feature become hard to distinguish from artifacts?
            """
        ),
        code(
            """
            observation = "TODO: write what changed when you modified the feature size or intensity."
            print(observation)
            """
        ),
        md(
            """
            ### 5. Learned Prior Across Image Families

            A learned prior may work well on data similar to its training examples and less well elsewhere.
            """
        ),
        code(
            """
            camera = data.camera().astype(float) / 255.0
            training = camera[:320, :320]
            patch_size = 7
            training_patches = extract_random_patches(training, patch_size=patch_size, count=4500, seed=43514)
            patch_mean, patch_components, _ = fit_patch_pca(training_patches)

            domain_cases = [
                ("camera", camera[352:448, 256:352]),
                ("moon", data.moon().astype(float)[300:396, 100:196] / 255.0),
                ("coins", data.coins().astype(float)[40:136, 40:136] / 255.0),
            ]

            domain_rows = []
            for name, domain_clean in domain_cases:
                domain_noisy = np.clip(domain_clean + 0.07 * rng.standard_normal(domain_clean.shape), 0.0, 1.0)
                domain_gaussian = np.clip(gaussian_filter(domain_noisy, sigma=1.0, mode="reflect"), 0.0, 1.0)
                domain_learned = pca_patch_denoise(
                    domain_noisy,
                    patch_mean,
                    patch_components,
                    patch_size=patch_size,
                    n_components=16,
                )
                domain_rows.append(
                    {
                        "domain": name,
                        "noisy": rmse(domain_noisy, domain_clean),
                        "Gaussian": rmse(domain_gaussian, domain_clean),
                        "camera-learned PCA": rmse(domain_learned, domain_clean),
                    }
                )

            domains = [row["domain"] for row in domain_rows]
            fig = go.Figure()
            for method in ["noisy", "Gaussian", "camera-learned PCA"]:
                fig.add_trace(go.Bar(x=domains, y=[row[method] for row in domain_rows], name=method))
            fig.update_layout(
                title="Robustness across image families",
                xaxis_title="test image family",
                yaxis_title="RMSE",
                barmode="group",
                height=400,
                margin=dict(l=20, r=20, t=55, b=45),
            )
            fig.show()

            for row in domain_rows:
                print(row)
            """
        ),
        md(
            """
            ### 6. Reliability Checklist

            A final reconstruction report should not only show the best image.

            It should say where the method is reliable and where it is not.
            """
        ),
        code(
            """
            project_checklist = {
                "forward model": "TODO: what is A, and when is it wrong?",
                "parameter sweep": "TODO: which parameter range did you test?",
                "robustness": "TODO: which noise/model/domain changes did you test?",
                "failure case": "TODO: show one honest failure or limitation",
                "ethics": "TODO: who could be affected by a wrong reconstruction?",
            }

            for item, answer in project_checklist.items():
                print(f"{item}: {answer}")
            """
        ),
        md(
            """
            ## Checks

            Answer in your own words:

            1. What does stability mean?
            2. Why can weak regularization amplify perturbations?
            3. Why is a smooth image not automatically reliable?
            4. What should a project report say about limitations?
            """
        ),
        md(
            """
            ## Next Steps

            Use the same reliability questions in your project report and oral defense:

            - What is your model?
            - What assumptions did you make?
            - What tests support your result?
            - Where does your method fail?
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
    write_notebook("week09_optimization_methods.ipynb", week09_cells())
    write_notebook("week10_sparse_reconstruction.ipynb", week10_cells())
    write_notebook("week11_wavelets.ipynb", week11_cells())
    write_notebook("week12_model_data.ipynb", week12_cells())
    write_notebook("week13_plug_and_play.ipynb", week13_cells())
    write_notebook("week14_stability_robustness.ipynb", week14_cells())


if __name__ == "__main__":
    main()
