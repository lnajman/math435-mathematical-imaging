# Week 12 Instructor Notes: Model-Based Vs Data-Driven Imaging

Student-facing materials:

- Slides: `slides/week-12-model-data.qmd`
- Book: `book/week-12-model-data.qmd`
- Notebook: `notebooks/week12_model_data.ipynb`
- Practice: `practice.qmd`, Week 12

## Class Purpose

Week 12 is the explicit turn toward neural networks and learned methods. Keep the central identity visible:

```text
neural imaging = inverse problems + learned image models
```

Students should understand that learned methods do not remove assumptions. They move many assumptions into training data, architecture, loss, and optimization.

## 75-Minute Teaching Rhythm

| Time | Instructor Focus |
|---:|---|
| 0-8 min | Recall wavelets as handcrafted representations. Ask what could be learned instead. |
| 8-20 min | Model-based reconstruction: explicit forward model, data term, regularizer. |
| 20-32 min | Data-driven reconstruction: learned map $f_\theta(y)$. |
| 32-42 min | Training data as a prior. Architecture and loss as assumptions. |
| 42-54 min | Patch PCA as a small learned prior. |
| 54-64 min | Capacity and generalization. Too little, too much, and distribution shift. |
| 64-72 min | Notebook distribution-shift experiment. |
| 72-75 min | Exit question: what did the training set teach the method to expect? |

## Discussion Pauses

After writing $\hat{x}=f_\theta(y)$, ask:

> Where did the prior go?

Expected answer: into the training set, architecture, loss, and learned parameters.

After the distribution-shift example, ask:

> If a method works on one image family and fails on another, was it wrong or was it specialized?

The point is to make domain assumptions explicit, not to dismiss learned methods.

## Board Derivations

1. Write the classical model:

   $$
   \hat{x}
   =
   \operatorname*{argmin}_x
   D(Ax,y)+\lambda R(x).
   $$

2. Write the learned inverse map:

   $$
   \hat{x}=f_\theta(y).
   $$

3. Write supervised training:

   $$
   \min_\theta \sum_i \ell(f_\theta(y_i),x_i).
   $$

4. Add the audit check:

   $$
   A f_\theta(y) \approx y.
   $$

5. State:

   ```text
   A plausible image is not automatically a faithful reconstruction.
   ```

## Live Notebook Plan

Recommended live sequence:

1. Show patch extraction and learned PCA components.
2. Apply the learned patch prior to denoising.
3. Change the number of components.
4. Compare training-like and shifted test images.
5. Ask students to write a claim-evidence-limit note for one result.

Useful live questions:

- What does the learned prior know?
- What does it not know?
- What changes when the test image is from another family?
- What would count as evidence beyond visual quality?

## Common Misconceptions

- Students may think data-driven means assumption-free. Emphasize that assumptions are learned or embedded.
- Students may think neural networks are separate from inverse problems. Keep returning to data, forward model, and prior.
- Students may think training accuracy proves reconstruction reliability. Generalization and data consistency must be tested.
- Students may think capacity is always good. Too much capacity can preserve noise or overfit training expectations.

## Fallback Plan

If the notebook or internet fails:

- draw a training table of pairs $(y_i,x_i)$;
- sketch learned components as directions in patch space;
- compare model-based and data-driven assumptions in a two-column table;
- discuss one distribution-shift example verbally, such as natural photos versus microscopy.

## End-of-Class Prompt

Ask:

```text
In a learned reconstruction method, the prior is hidden in ...
```

Expected answer: training data, architecture, loss, optimizer, and learned parameters.
