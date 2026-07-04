# Week 8 Instructor Notes: Total Variation Regularization

Student-facing materials:

- Slides: `slides/week-08-total-variation.qmd`
- Book: `book/week-08-total-variation.qmd`
- Notebook: `notebooks/week08_total_variation.ipynb`
- Practice: `practice.qmd`, Week 8

## Class Purpose

Week 8 introduces the first major nonquadratic prior:

```text
prefer images with sparse gradients, not necessarily small gradients
```

The class should make the difference between smoothing and edge preservation visible. TV is powerful, but not magic: it can preserve edges and also create staircasing.

## 75-Minute Teaching Rhythm

| Time | Instructor Focus |
|---:|---|
| 0-8 min | Recall variational form. Ask what quadratic smoothness does to edges. |
| 8-18 min | Image gradients and edge maps. |
| 18-30 min | Quadratic gradient penalty versus TV penalty. |
| 30-42 min | TV denoising model and edge-preserving intuition. |
| 42-52 min | Non-differentiability and why optimization becomes harder. |
| 52-64 min | Notebook comparison: Gaussian/quadratic smoothing versus TV. |
| 64-70 min | Artifacts: staircasing, contrast loss, texture loss. |
| 70-75 min | Exit question: when is TV's prior reasonable? |

## Discussion Pauses

After showing gradient magnitudes, ask:

> Should a good prior punish one large edge more than many small oscillations?

This helps students feel why $\ell_1$-type penalties behave differently from squared penalties.

After TV examples, ask:

> Did TV recover the truth, or did it impose a piecewise-smooth story?

The answer should be the second. It may be a good story, but it is still a story.

## Board Derivations

1. Write a discrete gradient:

   $$
   \nabla U[i,j] =
   (U[i+1,j]-U[i,j],\ U[i,j+1]-U[i,j]).
   $$

2. Compare penalties:

   $$
   \sum |\nabla U|^2
   \qquad \text{versus} \qquad
   \sum |\nabla U|.
   $$

3. State the ROF model:

   $$
   \hat{x}
   =
   \operatorname*{argmin}_x
   \frac12\|x-y\|_2^2+\lambda TV(x).
   $$

4. Draw a one-dimensional step edge and a smooth ramp.
5. Explain the artifact:

   ```text
   TV likes piecewise constant images, so smooth ramps can become steps.
   ```

## Live Notebook Plan

Recommended live sequence:

1. Display noisy image and gradient magnitude.
2. Run quadratic smoothing or Gaussian smoothing.
3. Run TV denoising for one moderate parameter.
4. Sweep the TV weight.
5. Zoom into an edge and into a smooth region.

Ask students:

- Which edge is preserved?
- Which texture is removed?
- Where do you see staircasing or cartooning?
- What would the result look like if the image were not piecewise smooth?

## Common Misconceptions

- Students may think TV preserves all important detail. It preserves some edges but can remove texture.
- Students may think TV is always better than quadratic smoothing. It depends on the image class and task.
- Students may think edge preservation proves data consistency. It does not; the residual must still be checked.
- Students may think non-differentiability is a minor technicality. It changes the optimization methods.

## Fallback Plan

If the notebook or internet fails:

- draw a step edge and a ramp;
- compare $\ell_2$ and $\ell_1$ penalties on a few gradient values;
- use slide images for TV versus smoothing;
- ask students to identify one success and one artifact.

## End-of-Class Prompt

Ask:

```text
TV is a good prior when ..., but it can fail when ...
```

Expected answer: good for piecewise-smooth images with sharp edges; can fail on smooth ramps, fine texture, or repeated patterns.
