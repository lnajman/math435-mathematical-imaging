# Week 10 Instructor Notes: Sparse Reconstruction

Student-facing materials:

- Slides: `slides/week-10-sparse-reconstruction.qmd`
- Book: `book/week-10-sparse-reconstruction.qmd`
- Notebook: `notebooks/week10_sparse_reconstruction.ipynb`
- Practice: `practice.qmd`, Week 10

## Week Purpose

Week 10 introduces sparsity as a prior:

```text
the unknown may be complicated in pixels but simple in the right representation
```

Students should understand both the promise and the danger. Sparse reconstruction works when the representation matches the image and the measurements preserve enough information.

## Weekly Two-Session Rhythm

| Moment | Instructor Focus |
|---|---|
| Session 1 | Recall soft thresholding; underdetermined systems; sparsity in a basis; $\ell_0$ idea and $\ell_1$ relaxation. |
| Between sessions | Ask students to name the representation where the unknown is sparse. |
| Session 2 | Geometry intuition; compressed sensing idea; notebook sampling experiment; representation mismatch and failure cases. |
| After Session 2 | End with the question: sparse where? |

## Discussion Pauses

After showing an underdetermined system, ask:

> If many images fit the data, why would the sparse one be preferred?

The answer must mention a prior, not the data alone.

After a successful sparse recovery, ask:

> What would make this success disappear?

Expected answers: wrong representation, too few measurements, coherent measurements, noise, or nonsparse image content.

## Board Derivations

1. Write the representation:

   $$
   x = \Psi \alpha.
   $$

2. Substitute into the forward model:

   $$
   y=A\Psi\alpha+\eta.
   $$

3. State the sparse ideal:

   $$
   \min_\alpha \|\alpha\|_0
   \quad \text{subject to} \quad
   A\Psi\alpha \approx y.
   $$

4. Write the convex relaxation:

   $$
   \min_\alpha
   \frac12\|A\Psi\alpha-y\|_2^2+\lambda\|\alpha\|_1.
   $$

5. Draw the $\ell_1$ diamond and a line of constraints if useful.

Key board sentence:

```text
Sparsity is not a property of an image alone; it is a property of an image in a representation.
```

## Live Notebook Plan

Recommended live sequence:

1. Show a sparse signal or coefficient vector.
2. Create undersampled measurements.
3. Compare least-squares or minimum-norm reconstruction with sparse reconstruction.
4. Change the number of measurements.
5. Change the representation or add mismatch if the notebook supports it.

Useful live questions:

- Where is the unknown sparse?
- Which coefficients are recovered first?
- What happens when the sampling density drops?
- What evidence would show that the sparse prior is inappropriate?

## Common Misconceptions

- Students may think sparsity means most pixels are zero. Usually sparsity is in transformed coefficients.
- Students may think $\ell_1$ always recovers the true sparse solution. Conditions matter.
- Students may think compressed sensing creates information from nothing. It selects among plausible solutions using a prior.
- Students may think a visually plausible sparse result is automatically data-consistent. Check the residual.

## Fallback Plan

If the notebook or internet fails:

- use a six-dimensional vector with two nonzero entries;
- draw several possible solutions to an underdetermined equation;
- show why choosing the sparse solution is an assumption;
- connect soft thresholding from Week 9 to $\ell_1$ sparsity.

## End-of-Week Prompt

Ask:

```text
Sparse reconstruction assumes the unknown is sparse in ..., and can fail when ...
```

Expected answer: sparse in a chosen basis or dictionary; fails with representation mismatch, insufficient measurements, or noise.
