# Week 7 Instructor Notes: Variational Formulation

Student-facing materials:

- Slides: `slides/week-07-variational-formulation.qmd`
- Book: `book/week-07-variational-formulation.qmd`
- Notebook: `notebooks/week07_variational_formulation.ipynb`
- Practice: `practice.qmd`, Week 7

## Week Purpose

Week 7 turns reconstruction into a general modeling language:

```text
reconstruction = minimizer of data term + prior term
```

Students should see an energy as an audit trail. Every term says what evidence or assumption is being used.

## Weekly Two-Session Rhythm

| Moment | Instructor Focus |
|---|---|
| Session 1 | General variational form; data terms from likelihood; regularizers from priors; first-order optimality. |
| Between sessions | Ask students to label data term, prior term, and unknown for one previous method. |
| Session 2 | Energy landscape; gradient descent intuition; notebook energy tracking; compare same data under different regularizers. |
| After Session 2 | End with the question: what does each term in the energy claim? |

## Discussion Pauses

After writing the general energy, ask:

> If two methods give different images, where in the energy did they disagree?

Push students to answer in terms of data term, prior, and parameter, not just "the algorithm."

After optimality conditions, ask:

> Does reaching a minimum prove the model was correct?

Expected answer: no. It proves the computation solved the chosen model, not that the chosen assumptions were true.

## Board Derivations

1. Write:

   $$
   E(x)=D(Ax,y)+\lambda R(x).
   $$

2. Label each part:

   ```text
   D: how data should be fit
   R: which images are preferred
   lambda: tradeoff strength
   ```

3. For differentiable terms, write:

   $$
   \nabla E(\hat{x})=0.
   $$

4. Show the Tikhonov example as a special case.
5. Connect to MAP estimation:

   ```text
   likelihood + prior -> posterior objective
   ```

Do not overdo probability notation unless students are comfortable. The central point is that modeling choices become mathematical terms.

## Live Notebook Plan

Recommended live sequence:

1. Show a reconstruction with a data term only.
2. Add a regularizer and compare the image.
3. Plot residual, prior value, and total energy if available.
4. Change $\lambda$ and ask students which curve should move.
5. Compare two priors on the same data.

Useful live questions:

- What does the residual say?
- What does the regularizer say?
- Which quantity is optimized, and which quantity is merely diagnostic?

## Common Misconceptions

- Students may think the energy is objective truth. Clarify that it is a chosen model.
- Students may think optimization and modeling are the same. Separate "what we minimize" from "how we minimize."
- Students may think a smaller energy always means a better reconstruction across different models. Energies from different models are not automatically comparable.
- Students may forget the role of $\lambda$. Keep returning to the tradeoff.

## Fallback Plan

If the notebook or internet fails:

- draw a two-term energy on the board;
- reuse Tikhonov as the concrete example;
- sketch residual decreasing while prior penalty increases;
- discuss two possible priors for the same noisy image.

## End-of-Week Prompt

Ask:

```text
For one reconstruction method today, identify D, R, and lambda.
```

This reinforces the notation students will need for TV, sparse priors, and learned priors.
