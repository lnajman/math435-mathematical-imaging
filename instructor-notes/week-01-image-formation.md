# Week 1 Instructor Notes: Images As Mathematical Objects

Student-facing materials:

- Slides: `slides/week-01-image-formation.qmd`
- Book: `book/week-01-image-formation.qmd`
- Notebook: `notebooks/week01_image_formation.ipynb`
- Practice: `practice.qmd`, Week 1

## Week Purpose

The goal is to establish the central language of the course before any technical machinery appears:

```text
unknown image x -> forward process -> observed data y
```

Students should leave understanding that an image is not only a picture. It is a function, an array, and often a vector. They should also understand that missing recorded pixels are not the same as hidden scene content.

## Weekly Two-Session Rhythm

| Moment | Instructor Focus |
|---|---|
| Session 1 | Opening question; images as functions, arrays, and vectors; vectorization by hand; real image as data. |
| Between sessions | Ask students to identify $x$, $y$, and a possible forward process in one imaging example. |
| Session 2 | Forward model $y=Ax+\eta$; masking matrix; masking versus occlusion; five modeling questions; notebook mask parameter change. |
| After Session 2 | End with the question: what can never be recovered from the data alone? |

## Discussion Pauses

Pause after the first slide question and collect several answers:

- an image is a picture;
- an image is an array;
- an image is a measurement;
- an image is information about a scene.

Then say: all are useful, but algorithms need a mathematical object.

Pause again after masking:

> If a person is hidden behind a tree, is that the same as a pixel being missing from a recorded image?

This is where students should see the imaging difference from audio source separation. If information was never measured, no algorithm can reconstruct it without adding assumptions.

## Board Derivations

Do these by hand rather than only through slides.

1. Draw a small $3\times 4$ array and define a vectorization convention.
2. Write $x=\operatorname{vec}(U)$ and identify its length.
3. Draw a binary mask on a six-pixel vector:

   ```text
   x = [x1 x2 x3 x4 x5 x6]^T
   y = [x1 x3 x6]^T
   ```

4. Write the corresponding masking matrix with one `1` per observed row.
5. Ask what happens to $x_2$, $x_4$, and $x_5$.

Key board sentence:

```text
A records some information and discards other information.
```

## Live Notebook Plan

Run only the sections that make the concept visible. Do not try to execute every cell live.

Recommended live sequence:

1. Show the small synthetic image array.
2. Show `shape`, indexing, and one cropped patch.
3. Run the real grayscale image display.
4. Run the masked observation.
5. Change `observed_every` in the toy mask experiment.

Ask students to predict before running:

- What will happen to the number of observations?
- Which pixels are impossible to determine from `y` alone?
- If missing entries are set to zero, is zero a measurement or a display convention?

## Common Misconceptions

- Students may think vectorization destroys the image. Clarify that it changes representation, not information.
- Students may think a zero in the masked image means the true pixel is black. Clarify that zero is a placeholder for missing recorded data.
- Students may think the forward model must always be linear. Say that linear models are the first mathematical language, but not all imaging processes are linear.
- Students may think the inverse problem is simply "undo $A$." Stress that some information may not exist in the data.

## Fallback Plan

If the notebook or internet fails:

- draw the small array and masking matrix on the board;
- use screenshots from the slides showing real image, crop, and mask;
- ask students to work in pairs on the six-pixel masking example;
- end with the five modeling questions.

No live code is essential for Week 1. The core concept is representational: image, vector, data, forward process, missing information.

## End-of-Week Prompt

Ask students to write one sentence:

```text
In this week's example, x is ..., y is ..., and A does ...
```

This is not collected as a graded item. It sets the habit they will reuse every week.
