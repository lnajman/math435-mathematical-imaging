# Week 11 Instructor Notes: Wavelets And Multiscale Representation

Student-facing materials:

- Slides: `slides/week-11-wavelets.qmd`
- Book: `book/week-11-wavelets.qmd`
- Notebook: `notebooks/week11_wavelets.ipynb`
- Practice: `practice.qmd`, Week 11

## Week Purpose

Week 11 gives the missing counterpart to Fourier:

```text
Fourier uses global oscillations; wavelets use localized multiscale atoms
```

Students should connect wavelets to Pythagorean energy, sparsity, and compact support. This is the natural bridge from sparse reconstruction to learned representations.

## Weekly Two-Session Rhythm

| Moment | Instructor Focus |
|---|---|
| Session 1 | Recall Fourier as global projection; Haar averages and differences; compact support; multiscale localization; energy preservation. |
| Between sessions | Ask students to explain how Fourier and wavelets both use Pythagoras but differ in support. |
| Session 2 | 2D subbands; sparse wavelet coefficients; notebook thresholding and reconstruction quality; compare Fourier and wavelet failure modes. |
| After Session 2 | End with the question: what does localization buy us? |

## Discussion Pauses

After Fourier recall, ask:

> If a Fourier coefficient changes the whole image, how can we represent a local edge?

Expected idea: many global coefficients are needed. This motivates localized atoms.

After showing wavelet coefficients, ask:

> Is the energy gone after thresholding, or did we decide some coefficients were less important?

This ties thresholding to prior assumptions.

## Board Derivations

1. Use a four-sample signal and compute Haar averages and differences:

   ```text
   [a b c d] -> averages and details
   ```

2. Write the orthonormal transform idea:

   $$
   \alpha = W x,
   \qquad
   x = W^\top \alpha.
   $$

3. State the Pythagorean/Parseval property:

   $$
   \|x\|_2^2 = \|\alpha\|_2^2
   $$

   for an orthonormal wavelet transform.

4. Compare supports:

   ```text
   Fourier atom: global support
   wavelet atom: localized support and scale
   ```

5. Write a thresholding prior:

   $$
   \min_\alpha
   \frac12\|W^\top\alpha-y\|_2^2+\lambda\|\alpha\|_1.
   $$

## Live Notebook Plan

Recommended live sequence:

1. Show the Haar or wavelet decomposition of a simple signal.
2. Display wavelet coefficient subbands for an image.
3. Threshold small coefficients and reconstruct.
4. Change the threshold.
5. Compare wavelet family or decomposition level if time permits.

Useful live questions:

- Which coefficients represent coarse structure?
- Which coefficients represent local edges or texture?
- What disappears first under thresholding?
- Where do artifacts appear?

## Common Misconceptions

- Students may think wavelets are simply "better Fourier." Clarify that they trade perfect global frequency localization for spatial localization.
- Students may think compact support means no global consequences. Reconstruction still combines many coefficients.
- Students may think thresholding small coefficients is harmless. It is a prior and can erase weak but real structures.
- Students may think all wavelets behave the same. Support, smoothness, and boundary handling matter.

## Fallback Plan

If the notebook or internet fails:

- compute a Haar transform on four numbers;
- draw a global sine and a localized wavelet;
- sketch subbands: coarse, horizontal, vertical, diagonal;
- discuss thresholding as keeping only large coefficients.

## End-of-Week Prompt

Ask:

```text
Wavelets help sparse imaging because ...
```

Expected answer: they represent local multiscale structure with relatively few significant coefficients.
