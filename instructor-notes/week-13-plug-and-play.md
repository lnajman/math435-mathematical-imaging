# Week 13 Instructor Notes: Plug-And-Play And Learned Regularization

Student-facing materials:

- Slides: `slides/week-13-plug-and-play.qmd`
- Book: `book/week-13-plug-and-play.qmd`
- Notebook: `notebooks/week13_plug_and_play.ipynb`
- Practice: `practice.qmd`, Week 13

## Class Purpose

Week 13 shows how learned or handcrafted denoisers can enter an inverse-problem algorithm:

```text
data-consistency step + denoiser step -> plug-and-play reconstruction
```

This is one of the clearest places where neural imaging becomes a continuation of inverse problems rather than a separate topic.

## 75-Minute Teaching Rhythm

| Time | Instructor Focus |
|---:|---|
| 0-8 min | Recall proximal methods: data step followed by prior step. |
| 8-20 min | Proximal gradient and why a proximal map behaves like a denoiser. |
| 20-32 min | Replace the proximal map by a denoiser $D_\sigma$. |
| 32-44 min | Data consistency: why the forward model remains inside the loop. |
| 44-54 min | Fixed points and convergence diagnostics. |
| 54-64 min | Notebook deblurring comparison: Tikhonov, Gaussian PnP, learned PnP. |
| 64-72 min | Strength tradeoff and failure modes. |
| 72-75 min | Exit question: what does convergence prove and not prove? |

## Discussion Pauses

After writing the PnP update, ask:

> Which part uses the measured data, and which part uses prior image knowledge?

Students should identify the gradient/data-consistency step and the denoiser step.

After fixed-point convergence, ask:

> If the iteration stops changing, does that mean the image is true?

Expected answer: no. It means the data step and denoiser have reached a balance.

## Board Derivations

1. Start from proximal gradient:

   $$
   z^k=x^k-\tau A^\top(Ax^k-y),
   \qquad
   x^{k+1}=\operatorname{prox}_{\tau\lambda R}(z^k).
   $$

2. Replace the proximal map:

   $$
   x^{k+1}=D_\sigma(z^k).
   $$

3. Write the fixed-point equation:

   $$
   x^\star =
   D_\sigma
   \left(x^\star-\tau A^\top(Ax^\star-y)\right).
   $$

4. Circle two diagnostics:

   ```text
   data residual: ||Ax-y||
   fixed-point change: ||x^{k+1}-x^k||
   ```

5. State:

   ```text
   PnP may converge without minimizing a known explicit objective.
   ```

## Live Notebook Plan

Recommended live sequence:

1. Show the degraded observation and baseline reconstruction.
2. Run Gaussian PnP and learned PnP.
3. Compare RMSE where ground truth is available.
4. Compare data residual and fixed-point change.
5. Sweep denoiser strength.

Useful live questions:

- Which method best fits the measured data?
- Which method looks most plausible?
- Do those agree?
- What happens when denoising is too strong?

## Common Misconceptions

- Students may think PnP is denoising after reconstruction. It is denoising inside the iterative reconstruction loop.
- Students may think a stronger denoiser is always better. It can move the result away from the data.
- Students may think convergence proves correctness. It proves only a fixed point of the chosen update.
- Students may think learned denoisers remove the need for a forward model. In PnP, the forward model is the data-consistency anchor.

## Fallback Plan

If the notebook or internet fails:

- draw the PnP loop on the board;
- reuse proximal splitting from Week 9;
- compare two imaginary denoisers: weak and strong;
- discuss data residual versus visual plausibility.

## End-of-Class Prompt

Ask:

```text
In plug-and-play, the denoiser contributes ..., while the data step contributes ...
```

Expected answer: prior image preference; consistency with the measured data through the forward model.
