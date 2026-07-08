# Instructor Notes

These notes are for teaching preparation. They are not part of the student-facing Quarto website.

Important visibility note: the course repository is public on GitHub. These files are hidden from the rendered website because they are not listed in `_quarto.yml`, but they are not private if someone browses the source repository. Do not put confidential student information, exam solutions, or anything that must remain private here.

## How To Use These Notes

Each teaching note includes:

- a weekly two-session teaching rhythm;
- suggested pauses for student discussion;
- board derivations to do by hand;
- notebook cells or sections to run live;
- common misconceptions to watch for;
- a fallback plan if the notebook, Colab, or internet fails.

The notes are meant to support class flow, not to create additional graded work. The syllabus remains the authority for assessment.

## Practical Pacing Rules

Treat each teaching note as a menu, not a checklist to exhaust. Each weekly note assumes two 75-minute meetings. For most weeks, protect three things:

1. the central modeling sentence for the week;
2. one board derivation that students can follow by hand;
3. one live notebook experiment with a parameter change.

If time is short, cut extra live-code cells before cutting discussion of assumptions, failure modes, or evidence. The recurring goal is not to show every figure; it is to help students ask what the data support, what the model assumes, and where the reconstruction can fail.

The notebook is part of the weekly class activity, not an extra afterthought. When judging pacing, count notebook and code-reference slides inside the weekly slide budget.

Useful slide-count targets:

| Week Type | Target Total Slides |
|---|---:|
| Normal two-session week | 38-48 |
| Quiz week | 34-40 |
| Heavy derivation week | 32-42 |
| Project-focused week | 30-40 |

For Week 4 and Week 12, reserve the final 20 minutes of Session 2 for the quiz.

## Current Files

- `week-01-image-formation.md`
- `week-02-convolution-blur.md`
- `week-03-fourier-imaging.md`
- `week-04-noise-likelihood.md`
- `week-05-ill-posed-inverse-problems.md`
- `week-06-tikhonov-regularization.md`
- `week-07-variational-formulation.md`
- `week-08-total-variation.md`
- `week-09-optimization-methods.md`
- `week-10-sparse-reconstruction.md`
- `week-11-wavelets.md`
- `week-12-model-data.md`
- `week-13-plug-and-play.md`
- `week-14-stability-robustness-ethics.md`
- `week-15-project-presentations.md`

## Teaching Spine

The notes follow the same arc as the student-facing course materials.

| Weeks | Teaching Role |
|---|---|
| 1-3 | Establish images as mathematical objects, forward models, convolution, and Fourier instability. |
| 4-6 | Turn noise, likelihood, ill-posedness, and Tikhonov regularization into the first full inverse-problem toolkit. |
| 7-10 | Move from variational modeling to optimization and sparse reconstruction. |
| 11-13 | Connect Fourier, wavelets, learned priors, and plug-and-play reconstruction. |
| 14-15 | Shift from reconstruction methods to reliability, project evidence, and oral defense. |

Keep returning to the same habit:

```text
unknown x -> forward process -> observed data y -> reconstruction with assumptions -> evidence and limits
```

## Grading Boundary

These notes may include suggested prompts, discussion pauses, and project-preparation reminders. They do not create additional graded work. The syllabus and instructor announcements remain authoritative for assessment.
