# Instructor Notes

These notes are for teaching preparation. They are not part of the student-facing Quarto website.

Important visibility note: the course repository is public on GitHub. These files are hidden from the rendered website because they are not listed in `_quarto.yml`, but they are not private if someone browses the source repository. Do not put confidential student information, exam solutions, or anything that must remain private here.

## How To Use These Notes

Each week includes:

- a 75-minute teaching rhythm;
- suggested pauses for student discussion;
- board derivations to do by hand;
- notebook cells or sections to run live;
- common misconceptions to watch for;
- a fallback plan if the notebook, Colab, or internet fails.

The notes are meant to support class flow, not to create additional graded work. The syllabus remains the authority for assessment.

## Current Files

- `week-01-image-formation.md`
- `week-02-convolution-blur.md`
- `week-03-fourier-imaging.md`

## Teaching Spine

The first three weeks should establish the language of the course:

1. The unknown image is a mathematical object: array, vector, or function.
2. The observed data are produced by a forward process.
3. Reconstruction requires deciding what is measured, what is missing, and what assumptions enter.
4. Fourier analysis explains why some directions are well measured and others are fragile.

Keep returning to the same habit:

```text
unknown x -> forward process -> observed data y -> reconstruction with assumptions
```

