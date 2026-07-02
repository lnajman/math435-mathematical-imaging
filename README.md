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

## Repository Layout

- `slides/`: lecture slides, one Quarto Reveal.js deck per lecture or week.
- `examples/`: reusable Python examples for demonstrations and assignments.
- `assignments/`: homework and quiz-facing material.
- `project/`: modeling project description, datasets, rubrics, and milestones.
- `assets/`: shared styling and static assets.

## Publishing

The repository is prepared for GitHub Pages with Quarto. The GitHub Action in `.github/workflows/publish.yml` installs Quarto in CI, renders the site, and publishes it to the `gh-pages` branch.

