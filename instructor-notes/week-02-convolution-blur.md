# Week 2 Instructor Notes: Convolution And Blur

Student-facing materials:

- Slides: `slides/week-02-convolution-blur.qmd`
- Book: `book/week-02-convolution-blur.qmd`
- Notebook: `notebooks/week02_convolution_blur.ipynb`
- Practice: `practice.qmd`, Week 2

## Week Purpose

Week 2 gives the first serious forward model:

```text
sharp image x -> convolution by a blur kernel -> blurred data y
```

The main point is not only how to compute convolution. The main point is that blur mixes nearby information and makes deblurring unstable.

## Weekly Two-Session Rhythm

| Moment | Instructor Focus |
|---|---|
| Session 1 | Recall $x$, $y$, and $A$; compute 1D convolution by hand; interpret kernels and PSFs; move to 2D blur on a real image. |
| Between sessions | Ask students to name one modeling assumption hidden inside a blur kernel. |
| Session 2 | Boundary conditions; matrix view of convolution; naive deblurring and instability; notebook blur-strength experiment. |
| After Session 2 | End with the question: what does blur preserve, mix, and weaken? |

## Discussion Pauses

After the 1D convolution hand calculation, ask:

> Did the peak disappear, or did it spread?

This distinction matters. Blur usually mixes information rather than deleting pixels directly.

After boundary conditions, ask:

> What should the algorithm assume beyond the edge of the image?

Students should notice that boundary handling is a modeling assumption, not a coding detail.

## Board Derivations

Do these slowly.

1. Write the 1D convolution rule:

   $$
   y_i = \frac14 x_{i-1}+\frac12 x_i+\frac14 x_{i+1}.
   $$

2. Compute one or two entries for a signal with a single spike.
3. Show that the kernel sums to 1 and preserves constants.
4. Write the linearity check:

   $$
   h*(a x+b z)=a(h*x)+b(h*z).
   $$

5. Draw a small Toeplitz-like matrix for 1D convolution. Do not dwell on matrix construction; the point is that fixed blur is linear.

Key board sentence:

```text
Blur is easy to apply because it averages; deblurring is hard because averaging weakens differences.
```

## Live Notebook Plan

Recommended live sequence:

1. Show 1D signal and kernel.
2. Display identity, box, Gaussian, and motion kernels.
3. Blur the real image with two different kernels.
4. Change Gaussian `sigma` and ask which structures disappear first.
5. Run the naive deblurring example only if time remains.

For the live demo, keep one parameter change prepared:

- increase `sigma`;
- change `kernel_size`;
- compare `boundary="symm"` with another boundary if the cell makes this convenient.

The best live moment is when students see that smooth images look "better" but contain less recoverable detail.

## Common Misconceptions

- Students may think blur is just aesthetic smoothing. Reframe it as a physical measurement model.
- Students may think the kernel is arbitrary. Explain that the kernel represents how a point spreads through the imaging system.
- Students may think normalized kernels are always correct. Clarify that normalization preserves average brightness, but the physical model decides the kernel.
- Students may think deblurring should be easy if the kernel is known. Emphasize that knowing the forward model does not mean the inverse is stable.

## Fallback Plan

If the notebook or internet fails:

- compute the 1D spike example on the board;
- draw three kernels: identity, box, motion;
- sketch what each would do to an edge;
- use the slide images for real blur;
- describe naive deblurring qualitatively as "dividing by small numbers."

The week can be taught without code if the students understand local averaging, boundary assumptions, and instability.

## End-of-Week Prompt

Ask:

```text
What assumption does convolution make about the blur, and what detail becomes weakly measured?
```

Expected idea: the same local blur rule is used everywhere; fine details and sharp transitions become weak.
