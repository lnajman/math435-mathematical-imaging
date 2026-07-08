# Week 15 Instructor Notes: Project Presentations And Oral Defenses

Student-facing materials:

- Project: `project/index.qmd`
- Syllabus: `syllabus.qmd`
- Weekly roadmap: `classes.qmd`, Week 15

## Week Purpose

Week 15 is not a new content chapter. It is the ownership week.

Students present and defend their work. The oral defense is the place where students show that they understand the model, code, evidence, limitations, and LLM use behind the submitted project.

Keep the structure aligned with the syllabus and project page. Do not introduce new grading criteria in the room.

## Weekly Two-Session Rhythm

Adjust the timing to the number of teams. A useful default is:

| Moment | Instructor Focus |
|---|---|
| Session 1 | Set expectations; run team presentations; keep time firmly; record claim, evidence, limitation, and follow-up question for each team. |
| Between sessions | Identify students who need a targeted oral-defense question about model, code, evidence, or LLM use. |
| Session 2 | Individual oral-defense questions, remaining presentations if needed, and course synthesis on reconstruction and trust. |
| After Session 2 | Record any follow-up needed under the syllabus and project-page assessment rules. |

If there are many teams, move oral defenses into scheduled individual slots outside the shared presentation block, following the syllabus constraints.

## Timing Variants

Use one of these formats depending on enrollment and room constraints:

| Format | Best Use | Instructor Move |
|---|---|---|
| Short talks plus live questions | Small class or few teams | Keep a visible timer and ask one model question plus one evidence question per team. |
| Shared presentations, defenses scheduled separately | Many teams | Use shared session time for project results and reserve individual ownership checks for scheduled slots. |
| Poster or table rotation | Very large or time-constrained group | Ask each student one targeted oral-defense question at the project station. |

Do not let the format change the assessment boundary. The syllabus and project page define what is assessed; the format only changes how the defense is organized.

## Presentation Listening Targets

For each project, listen for:

- clear definition of $x$, $y$, and the forward process;
- baseline method and second method compared on the same data;
- metric or qualitative evidence tied to the claim;
- at least one failure case or limitation;
- explicit LLM-use statement;
- whether each student can explain the code and decisions.

## Oral Defense Question Bank

Use a few questions per student. The goal is not to trap students; it is to verify ownership.

Model questions:

- What is the unknown $x$ in your project?
- What is the measured or degraded data $y$?
- What is your forward model or degradation process?
- Which part of the method is a prior?
- What assumption would make your method fail?

Code questions:

- Which function or notebook cell creates the degraded data?
- Where is the baseline method implemented?
- Which parameter had the largest effect?
- How did you check that two methods were compared fairly?

Evidence questions:

- What is your strongest result?
- What metric supports it?
- What visual result supports it?
- What result weakened your original expectation?
- What would you test next with more time?

LLM-use questions:

- Where did an LLM help in the project?
- What did you personally verify?
- Which part could not be delegated to the LLM?
- What did you change after checking the LLM output?

## Board Structure

Keep a simple table visible while teams present:

| Team | Claim | Evidence | Limitation | Oral follow-up |
|---|---|---|---|---|

This helps keep comments concrete and prevents presentations from becoming only visual tours.

## Common Issues To Watch For

- A team shows only the best image and no failure case.
- A method comparison changes several variables at once.
- Students report RMSE when ground truth exists but ignore data residual.
- Students show plausible visual improvement without explaining measurement support.
- The LLM-use statement is present but vague.
- One team member cannot explain a central method or code path.

## Fallback Plan

If projection, Colab, or a notebook fails:

- ask the team to present from exported figures or screenshots;
- have them define $x$, $y$, method, evidence, and limitation verbally;
- use oral-defense questions to evaluate understanding;
- do not let a technical display failure erase the defense of the model.

## Closing Prompt

End by returning to the course identity:

```text
inverse problem + prior + computation + evidence -> reconstruction that can be defended
```

Ask students to say, in one sentence, what they will remember about trusting reconstructed images.
