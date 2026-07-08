# Week 14 Instructor Notes: Stability, Robustness, And Ethics

Student-facing materials:

- Slides: `slides/week-14-stability-robustness-ethics.qmd`
- Book: `book/week-14-stability-robustness-ethics.qmd`
- Notebook: `notebooks/week14_stability_robustness.ipynb`
- Practice: `practice.qmd`, Week 14
- Project: `project/index.qmd`

## Week Purpose

Week 14 closes the technical arc by asking when a reconstruction should be trusted.

Students should leave with the final course habit:

```text
claim -> evidence -> limitation -> consequence
```

This week also prepares the project report and oral defense without adding new grading categories beyond the syllabus.

## Weekly Two-Session Rhythm

| Moment | Instructor Focus |
|---|---|
| Session 1 | Recall learned reconstruction and PnP; stability; perturbation amplification; robustness axes and domain shift. |
| Between sessions | Ask students to write one project claim and one possible failure case. |
| Session 2 | Hallucination and unsupported detail; ethics; notebook reliability checks; project report preparation and oral-defense evidence. |
| After Session 2 | End with the question: what would make your project claim weaker? |

## Discussion Pauses

After showing a visually plausible reconstruction, ask:

> What evidence would convince you this structure was measured and not invented?

This is the central reliability question.

Before the project discussion, ask:

> What is one failure case your project should be able to explain?

Students should see failure cases as evidence of understanding, not embarrassment.

## Board Derivations

1. Write a stability ratio:

   $$
   \frac{\|\hat{x}(y+\delta)-\hat{x}(y)\|}{\|\delta\|}.
   $$

2. Interpret it in words:

   ```text
   small data changes should not cause unexplained large reconstruction changes
   ```

3. Write the claim-evidence-limit table:

   | Claim | Evidence | Limitation |
   |---|---|---|
   | method improves X | metric/figure/residual | tested only under Y |

4. Write the project defense triangle:

   ```text
   model + code + evidence
   ```

5. Add:

   ```text
   LLM use must be disclosed, but ownership is shown by explanation.
   ```

## Live Notebook Plan

Recommended live sequence:

1. Show a baseline reconstruction and a perturbed-data reconstruction.
2. Change noise level or forward-model mismatch.
3. Compare metrics and visual changes.
4. Fill one claim-evidence-limit row live.
5. Ask students to propose a robustness test for their project.

Useful live questions:

- What changed in the data?
- What changed in the reconstruction?
- Is the change expected from the model?
- Which claim becomes weaker after this test?

## Common Misconceptions

- Students may think reliability means the image looks good. Reliability requires evidence tied to the data and model.
- Students may think failure cases are optional. They are part of responsible reconstruction.
- Students may think an LLM-use statement is the same as understanding. The oral defense tests ownership.
- Students may think ethics is separate from mathematics. In imaging, mathematical uncertainty can have real consequences.

## Fallback Plan

If the notebook or internet fails:

- use a board table for claim, evidence, and limitation;
- discuss one medical or scientific hallucination scenario;
- ask students to write one project claim and one failure case;
- rehearse an oral-defense question: "How do you know the result is supported by the data?"

## End-of-Week Prompt

Ask:

```text
My strongest project claim is ..., and its most important limitation is ...
```

This can feed directly into final report revision.
