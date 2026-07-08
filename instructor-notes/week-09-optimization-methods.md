# Week 9 Instructor Notes: Optimization Methods

Student-facing materials:

- Slides: `slides/week-09-optimization-methods.qmd`
- Book: `book/week-09-optimization-methods.qmd`
- Notebook: `notebooks/week09_optimization_methods.ipynb`
- Practice: `practice.qmd`, Week 9

## Week Purpose

Week 9 separates the model from the algorithm:

```text
the energy says what we want; the optimization method says how we compute it
```

Students should learn to report convergence evidence, not only final images.

## Weekly Two-Session Rhythm

| Moment | Instructor Focus |
|---|---|
| Session 1 | Recall variational models; convexity; gradient descent for smooth energies; step size and failure from too-large steps. |
| Between sessions | Ask students to identify what should be monitored besides the final image. |
| Session 2 | Nonsmooth penalties; proximal step idea; soft thresholding; ISTA; notebook convergence curves. |
| After Session 2 | End with the question: what evidence says an iteration has stabilized? |

## Discussion Pauses

After gradient descent, ask:

> If each step is in a descent direction, why can the method still fail?

Expected answer: the step can be too large, or the problem may not satisfy the assumptions used by the method.

After showing an iteration curve, ask:

> Is a visually good image enough evidence that the algorithm converged?

Push for quantitative diagnostics: energy, residual, fixed-point change, and parameter sensitivity.

## Board Derivations

1. Write gradient descent:

   $$
   x^{k+1}=x^k-\tau \nabla E(x^k).
   $$

2. Draw small, reasonable, and too-large steps on a one-dimensional curve.
3. Introduce proximal splitting:

   $$
   x^{k+1}
   =
   \operatorname{prox}_{\tau\lambda R}
   \left(x^k-\tau\nabla D(x^k)\right).
   $$

4. For $\ell_1$, write soft thresholding:

   $$
   S_\alpha(t)=\operatorname{sign}(t)\max(|t|-\alpha,0).
   $$

5. State:

   ```text
   Proximal methods separate a data step from a prior step.
   ```

This sentence prepares Week 13 plug-and-play.

## Live Notebook Plan

Recommended live sequence:

1. Run gradient descent with a stable step size.
2. Increase step size until oscillation or divergence appears.
3. Show energy versus iteration.
4. Run ISTA or a proximal example.
5. Compare final image with convergence diagnostics.

Ask students to identify:

- the model being minimized;
- the update rule;
- the stopping criterion;
- the diagnostic they would report.

## Common Misconceptions

- Students may think optimization failure means the model is wrong. Sometimes the algorithm or step size is the problem.
- Students may think a stable-looking image means convergence. Check change or energy.
- Students may think all iterative methods minimize a known objective. This will become false or unclear for plug-and-play.
- Students may confuse residual decrease with total energy decrease.

## Fallback Plan

If the notebook or internet fails:

- sketch a one-dimensional quadratic and gradient descent steps;
- compute two soft-thresholding examples by hand;
- draw an energy curve that decreases and one that oscillates;
- connect proximal splitting to data/prior alternation.

## End-of-Week Prompt

Ask:

```text
In an iterative reconstruction, I would report convergence using ...
```

Expected answers include energy, residual, iterate change, active coefficients, or parameter sensitivity.
