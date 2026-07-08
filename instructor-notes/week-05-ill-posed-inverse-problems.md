# Week 5 Instructor Notes: Ill-Posed Inverse Problems

Student-facing materials:

- Slides: `slides/week-05-ill-posed-inverse-problems.qmd`
- Book: `book/week-05-ill-posed-inverse-problems.qmd`
- Notebook: `notebooks/week05_ill_posed_inverse_problems.ipynb`
- Practice: `practice.qmd`, Week 5

## Week Purpose

Week 5 names the core difficulty of the course. Inverse problems fail for structural reasons:

```text
missing directions, weak directions, noise, and non-uniqueness
```

The week should not feel like a list of pathologies. It should feel like a diagnostic framework students can reuse for every reconstruction method.

## Weekly Two-Session Rhythm

| Moment | Instructor Focus |
|---|---|
| Session 1 | Hadamard conditions; non-uniqueness and nullspace; stability; let students propose the naive inverse before showing why it is insufficient. |
| Between sessions | Ask students to classify one failure as non-existence, non-uniqueness, or instability. |
| Session 2 | Singular values as measurement strengths; noise amplification; truncated SVD; notebook SVD coefficient experiment. |
| After Session 2 | End with the question: what is one direction the data cannot defend? |

## Discussion Pauses

After writing $Ax=y$, ask:

> What would be the naive inverse-problem answer?

Let students say: solve the equation, invert $A$, or find $x$ such that $Ax=y$. Keep that naive answer visible. Then add the failure reasons one at a time:

- there may be no exact solution because of noise;
- there may be many solutions;
- the solution may be unstable;
- the correct-looking solution may depend on assumptions not present in the data.

After the nullspace example, ask:

> If $Az=0$, can the data ever tell whether $z$ was in the image?

The expected answer is no. That is the cleanest way to separate data from prior.

## Board Derivations

1. Write the nullspace fact:

   $$
   A(x+z)=Ax
   \qquad \text{if } Az=0.
   $$

2. State the consequence:

   ```text
   Data cannot distinguish x from x+z.
   ```

3. Write the SVD:

   $$
   A v_i = \sigma_i u_i.
   $$

4. Expand the inverse formally:

   $$
   x = \sum_i \frac{\langle y,u_i\rangle}{\sigma_i}v_i.
   $$

5. Circle the dangerous term $1/\sigma_i$.
6. Connect to Fourier deblurring: small frequency response and small singular values play the same role.

## Live Notebook Plan

Recommended live sequence:

1. Show a small matrix with a visible nullspace.
2. Perturb $y$ slightly and compare reconstructed $x$.
3. Display singular values for the toy operator or blur.
4. Show coefficients in stable and unstable directions.
5. Run the truncated or regularized inverse comparison if available.

Ask students to predict:

- Which singular directions will be amplified most?
- Does a small residual prove the image is correct?
- What changes when one singular value is zero?

## Common Misconceptions

- Students may think ill-posed means "hard to program." Clarify that it is mathematical, not only computational.
- Students may think more pixels always solve the problem. More data help only if they measure the missing directions.
- Students may think a low residual proves truth. Stress that many images can fit the same data.
- Students may think instability is a bug in the solver. Show it appears in the inverse formula itself.

## Fallback Plan

If the notebook or internet fails:

- use a $2\times 2$ or $2\times 3$ matrix on the board;
- draw two vectors mapped to the same measurement;
- show how dividing by $0.01$ turns small noise into a large coefficient;
- return to Fourier inverse filtering as a familiar example.

The essential week can be taught with only nullspace and singular values.

## End-of-Week Prompt

Ask:

```text
Name one reason an inverse problem can fail even when the forward model is known.
```

Good answers include non-uniqueness, invisible directions, small singular values, and noise amplification.
