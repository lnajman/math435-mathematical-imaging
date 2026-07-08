# Week 4 Instructor Notes: Noise Models And Likelihoods

Student-facing materials:

- Slides: `slides/week-04-noise-likelihood.qmd`
- Book: `book/week-04-noise-likelihood.qmd`
- Notebook: `notebooks/week04_noise_likelihood.ipynb`
- Practice: `practice.qmd`, Week 4

## Week Purpose

Week 4 changes the meaning of "fit the data." The same residual can be normal, suspicious, or impossible depending on the noise model.

Students should leave with this habit:

```text
forward model + noise model -> data-fidelity term
```

This is the bridge from $y=Ax+\eta$ to optimization. Before choosing a reconstruction, we must decide what kind of mismatch between $Ax$ and $y$ is statistically plausible.

## Weekly Two-Session Rhythm

| Moment | Instructor Focus |
|---|---|
| Session 1 | Recall blur/Fourier instability; introduce observation models; additive Gaussian noise; Poisson photon counting. |
| Between sessions | Ask students to compare one residual under Gaussian and Poisson assumptions. |
| Session 2 | Likelihood as data fidelity; averaging and SNR limits; notebook noise comparison; reserve final 20 minutes for Quiz 1. |
| After Session 2 | End with the question: what does the noise model authorize us to ignore? |

## Discussion Pauses

After showing Gaussian residuals, ask:

> If two pixels have the same residual, should we always judge them equally?

This sets up Poisson noise, where the variability depends on intensity.

After the photon-counting example, ask:

> Is a dark noisy region failed reconstruction, or weak measurement?

The goal is to stop students from treating all noise as a display artifact.

## Board Derivations

Keep the derivations short and tied to interpretation.

1. Write the Gaussian observation model:

   $$
   y_i = (Ax)_i + \eta_i,
   \qquad
   \eta_i \sim \mathcal{N}(0,\sigma^2).
   $$

2. Derive the negative log-likelihood up to constants:

   $$
   -\log p(y\mid x)
   =
   \frac{1}{2\sigma^2}\|Ax-y\|_2^2 + C.
   $$

3. Write the Poisson model:

   $$
   y_i \sim \operatorname{Poisson}((Ax)_i).
   $$

4. Explain that the Poisson variance equals the mean. The noise level is part of the signal level.
5. State the modeling consequence:

   ```text
   The data term is not only algebra; it is a statistical claim about the measurement.
   ```

## Live Notebook Plan

Recommended live sequence:

1. Show a clean image and additive Gaussian noise.
2. Change the Gaussian noise standard deviation and ask what changes in the residual histogram.
3. Show Poisson noise at high and low photon counts.
4. Compare averaging repeated measurements with increasing exposure.
5. Run the likelihood or residual comparison section.

Useful live questions:

- Which image regions look most affected by Poisson noise?
- What would least squares over- or under-emphasize?
- If the true clean image is unknown, what diagnostic can still be checked?

## Common Misconceptions

- Students may think noise is always additive. Emphasize that photon noise is signal-dependent.
- Students may think more smoothing is the same as a better noise model. Smoothing is a reconstruction choice, not the observation model.
- Students may think the likelihood gives the true image. It only says which images make the data plausible under a model.
- Students may think Gaussian noise is "wrong" after seeing Poisson. Say it is a useful approximation in some regimes.

## Fallback Plan

If the notebook or internet fails:

- draw residual histograms on the board;
- compare two made-up pixels with the same absolute residual but different expected intensities;
- write Gaussian and Poisson models side by side;
- use the slide figures for noisy images and photon counts.

The week can be taught without computation if students understand that different physical measurements produce different data terms.

## End-of-Week Prompt

Ask students to complete:

```text
A data-fidelity term is a mathematical way to say ...
```

Expected idea: which mismatches between prediction and observation are plausible under the noise model.
