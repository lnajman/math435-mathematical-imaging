# MATH 435: Mathematical Imaging

Course materials for MATH 435, including Quarto Reveal.js slides, Python examples, assignments, project material, and an online book planned for the next phase.

## Local Quarto

This workspace includes a project-local Quarto CLI installed under `.tools/`, ignored by Git:

```bash
.tools/quarto/1.9.38/bin/quarto --version
```

Use the helper script for local rendering:

```bash
./scripts/quarto render
./scripts/quarto preview
```

To preview the Week 1 slides specifically:

```bash
./scripts/quarto preview slides/week-01-image-formation.qmd
```

The slide code blocks are currently displayed rather than executed during render, so lecture demos should be run from the terminal or a notebook:

```bash
python3 examples/week01_image_arrays.py
python3 examples/make_week01_figures.py
```

## Repository Layout

- `slides/`: lecture slides, one Quarto Reveal.js deck per lecture or week.
- `examples/`: reusable Python examples for demonstrations and assignments.
- `assignments/`: homework and quiz-facing material.
- `project/`: modeling project description, datasets, rubrics, and milestones.
- `assets/`: shared styling and static assets.

## Publishing

The repository is prepared for GitHub Pages with Quarto. The GitHub Action in `.github/workflows/publish.yml` installs Quarto in CI, renders the site, and publishes it to the `gh-pages` branch.
