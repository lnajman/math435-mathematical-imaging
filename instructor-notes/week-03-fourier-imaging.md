# Week 3 Instructor Notes: Fourier Transform In Imaging

Student-facing materials:

- Slides: `slides/week-03-fourier-imaging.qmd`
- Book: `book/week-03-fourier-imaging.qmd`
- Notebook: `notebooks/week03_fourier_imaging.ipynb`
- Practice: `practice.qmd`, Week 3

## Week Purpose

Week 3 explains why deblurring is unstable. Fourier analysis turns the vague sentence "blur removes detail" into a precise statement:

```text
blur weakens some frequency directions, and inversion amplifies the weak directions
```

Students should also understand the contrast with wavelets later: Fourier atoms are global, while wavelets will give localized multiscale directions.

## Weekly Two-Session Rhythm

| Moment | Instructor Focus |
|---|---|
| Session 1 | Fourier as projection; Parseval/Pythagoras; global support of Fourier atoms; 2D magnitude spectra. |
| Between sessions | Ask students why one local edge affects many Fourier coefficients. |
| Session 2 | Magnitude versus phase; convolution theorem; frequency filtering; inverse filtering and stabilization; notebook cutoff or regularization experiment. |
| After Session 2 | End with why the naive Fourier inverse fails when frequencies are weak or noisy. |

## Discussion Pauses

After Fourier as projection, ask:

> If the Fourier basis is orthogonal, what does a coefficient measure?

Expected idea: how much of the signal lies in that oscillatory direction.

After global support, ask:

> If I change one Fourier coefficient, where does the image change?

Expected answer: everywhere, because Fourier atoms have global support. This prepares the later wavelet comparison.

Before inverse filtering, ask:

> If blur nearly removes a frequency, what must inverse filtering do to recover it?

Expected answer: divide by a small number, which amplifies noise.

## Board Derivations

Keep the formulas short and interpretive.

1. Write the DFT:

   $$
   \widehat{x}[k]=\sum_{n=0}^{N-1}x[n]e^{-2\pi i kn/N}.
   $$

2. Say: this is an inner product with a global oscillation.
3. Draw two orthogonal vectors and write:

   $$
   \|x\|^2 = \sum_k |\widehat{x}[k]|^2
   $$

   after normalization.

4. State the convolution theorem:

   $$
   \widehat{h*x}=\widehat{h}\,\widehat{x}.
   $$

5. Write the inverse filtering danger:

   $$
   \widehat{x}=\frac{\widehat{y}}{\widehat{h}}.
   $$

   If $|\widehat{h}|$ is small, noise becomes large.

Key board sentence:

```text
Fourier gives orthogonal global directions; blur weakens some directions; inversion amplifies weak directions.
```

## Live Notebook Plan

Recommended live sequence:

1. Show low and high 1D sine waves and their Fourier magnitudes.
2. Change one frequency value and rerun.
3. Show the 2D Fourier magnitude of the image.
4. Show low-pass and high-pass masks.
5. Run the deblurring example comparing naive and regularized inverse.

The strongest live parameter change is the regularization parameter in the deblurring section. Ask students to identify the tradeoff:

- too small: sharper but noisy or unstable;
- too large: stable but oversmoothed.

## Common Misconceptions

- Students may think Fourier coefficients correspond to local image regions. Emphasize global support.
- Students may think high frequencies are "bad." Clarify that high frequencies include edges and fine detail, not only noise.
- Students may think magnitude contains everything. Use the magnitude/phase slide to show phase carries geometry and alignment.
- Students may think inverse filtering fails because the code is crude. Emphasize the mathematical reason: division by small frequency response values.

## Fallback Plan

If the notebook or internet fails:

- draw slow and fast sine waves;
- draw a rough spectrum with peaks;
- sketch blur as suppressing high frequencies;
- write $\widehat{y}=\widehat{h}\widehat{x}$ and $\widehat{x}=\widehat{y}/\widehat{h}$;
- use the board to show why small $\widehat{h}$ is dangerous.

This week can be taught from the board because the central mechanism is algebraic.

## End-of-Week Prompt

Ask students to complete:

```text
Fourier helps explain deblurring because ...
```

Expected answer: it shows which frequency components blur weakens, and why trying to restore them amplifies noise.
