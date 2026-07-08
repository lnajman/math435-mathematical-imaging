# Week 6 Instructor Notes: Tikhonov Regularization

Student-facing materials:

- Slides: `slides/week-06-tikhonov-regularization.qmd`
- Book: `book/week-06-tikhonov-regularization.qmd`
- Notebook: `notebooks/week06_tikhonov_regularization.ipynb`
- Practice: `practice.qmd`, Week 6
- Project: `project/index.qmd`

## Week Purpose

Week 6 gives the first complete regularized reconstruction model:

```text
fit the data, but control unstable or implausible image directions
```

It also anchors the project proposal moment. Keep the grading boundary clear: the assessed project requirements are the ones in the syllabus and project page.

## Weekly Two-Session Rhythm

| Moment | Instructor Focus |
|---|---|
| Session 1 | Recall small singular values; introduce Tikhonov as data fit plus penalty; derive normal equations and closed form. |
| Between sessions | Ask students to state what prior Tikhonov adds before seeing the SVD view. |
| Session 2 | SVD filter factors; choosing $\lambda$; notebook sweep; bias-stability tradeoff; project proposal bridge. |
| After Session 2 | End with the question: what prior does Tikhonov add, and what does it trade away? |

## Discussion Pauses

After writing the Tikhonov objective, ask:

> Which term is allowed to disagree with the data, and why?

Students should see that regularization deliberately permits some residual to avoid unstable reconstructions.

Before the project discussion, ask:

> For your project, what is $x$, what is $y$, and what forward process connects them?

This keeps projects inside the inverse-problem identity of the course.

## Board Derivations

1. Write the basic Tikhonov model:

   $$
   \hat{x}
   =
   \operatorname*{argmin}_x
   \|Ax-y\|_2^2+\lambda\|x\|_2^2.
   $$

2. Derive the normal equations:

   $$
   (A^\top A+\lambda I)\hat{x}=A^\top y.
   $$

3. If students are ready, state the generalized version:

   $$
   \|Ax-y\|_2^2+\lambda\|Lx\|_2^2.
   $$

4. Write the SVD filter factor:

   $$
   \frac{\sigma_i}{\sigma_i^2+\lambda}.
   $$

5. Interpret it:

   ```text
   Tikhonov replaces dangerous division by small singular values with controlled amplification.
   ```

## Live Notebook Plan

Recommended live sequence:

1. Show noisy blurred data.
2. Run a naive reconstruction or weakly regularized reconstruction.
3. Sweep $\lambda$ across too small, reasonable, and too large.
4. Display residual norm and solution norm.
5. If available, show the L-curve or parameter curve.

Good live questions:

- Which result has the lowest residual?
- Which result would you trust if the true image were hidden?
- What evidence would you report besides the picture?

## Project Proposal Moment

Use the last ten minutes to remind students that a project proposal should identify:

- team and topic;
- data source or synthetic data plan;
- unknown image or object $x$;
- observed data $y$;
- first forward process or degradation model;
- baseline method;
- second method for comparison;
- one reliability or failure question.

Do not add extra grading rules live. Point to the syllabus and project page as the authority.

## Common Misconceptions

- Students may think regularization is just smoothing. Clarify that it is a prior, and the prior may penalize different features depending on $L$.
- Students may think the best $\lambda$ is the one with the smallest residual. Explain the residual-prior tradeoff.
- Students may think Tikhonov restores lost information. It does not restore it from data; it chooses a plausible solution using a prior.
- Students may think the project must use advanced methods. Emphasize clear modeling, comparison, and evidence.

## Fallback Plan

If the notebook or internet fails:

- derive the normal equations on the board;
- draw a one-dimensional tradeoff curve with residual versus solution norm;
- use the SVD filter factor to explain the role of $\lambda$;
- discuss project proposals orally with the $x,y,A$ template.

## End-of-Week Prompt

Ask:

```text
Tikhonov regularization changes the inverse problem by ...
```

Expected answer: adding a prior or penalty that stabilizes weak data directions at the cost of bias.
