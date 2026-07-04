# Week 11 Instructor Notes: Wavelets And Multiscale Representation

Student-facing materials:

- Slides: `slides/week-11-wavelets.qmd`
- Book: `book/week-11-wavelets.qmd`
- Notebook: `notebooks/week11_wavelets.ipynb`
- Practice: `practice.qmd`, Week 11

## Class Purpose

Week 11 gives the missing counterpart to Fourier:

```text
Fourier uses global oscillations; wavelets use localized multiscale atoms
```

Students should connect wavelets to Pythagorean energy, sparsity, and compact support. This is the natural bridge from sparse reconstruction to learned representations.

## 75-Minute Teaching Rhythm

| Time | Instructor Focus |
|---:|---|
| 0-8 min | Recall Fourier as global projection. Ask what is missing from that view. |
| 8-20 min | Haar averages and differences by hand. |
| 20-32 min | Compact support and multiscale localization. |
| 32-42 min | Orthonormal wavelet transform and energy preservation. |
| 42-54 min | Sparse wavelet coefficients for natural images. |
| 54-66 min | Notebook thresholding and reconstruction quality. |
| 66-72 min | Compare Fourier and wavelet failure modes. |
| 72-75 min | Exit question: what does localization buy us? |

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

## End-of-Class Prompt

Ask:

```text
Wavelets help sparse imaging because ...
```

Expected answer: they represent local multiscale structure with relatively few significant coefficients.
