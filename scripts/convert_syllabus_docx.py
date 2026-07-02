#!/usr/bin/env python3
"""Convert the official Word syllabus into the Quarto syllabus page."""

from __future__ import annotations

import re
import subprocess
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCX = Path("/Users/laurentnajman/Downloads/Math435-Course Syllabus Template July 2025 copie.docx")
OUTPUT = ROOT / "syllabus.qmd"
QUARTO = ROOT / "scripts" / "quarto"


SECTION_HEADINGS = {
    "Prerequisites": "Prerequisites",
    "Corequisites": "Corequisites",
    "Course Catalog Description": "Course Catalog Description",
    "Textbook": "Textbook",
    "Reference Materials": "Reference Materials",
    "Course Structure & Learning Methodology": "Course Structure & Learning Methodology",
    "Course Topics": "Course Topics",
    "Laboratory and/or Computing/Digital Resources": "Laboratory and/or Computing/Digital Resources",
    "Course Learning Outcomes (CLO) and Contributions to BSc Applied Math, Statistics, and Data Science Program Learning Outcomes (PLO)<sup>\\*</sup>:": "Course Learning Outcomes (CLO) and Program Learning Outcomes (PLO)",
    "Assessment": "Assessment",
    "Grading Scheme": "Grading Scheme",
    "Academic Integrity Statement": "Academic Integrity Statement",
    "Copyright and Plagiarism": "Copyright and Plagiarism",
    "Instructor Policy on Late Submission of Assignments": "Instructor Policy on Late Submission of Assignments",
    "Additional Information:": "Additional Information",
    "Weekly plan (Tentative)": "Weekly Plan (Tentative)",
}


def pandoc_markdown(docx_path: Path) -> str:
    result = subprocess.run(
        [
            str(QUARTO),
            "pandoc",
            str(docx_path),
            "-t",
            "gfm",
            "--wrap=none",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def clean_markdown(markdown: str) -> str:
    markdown = markdown.replace("\r\n", "\n")
    markdown = re.sub(r"^##\s*$\n+", "", markdown, flags=re.MULTILINE)
    markdown = markdown.replace("**  **", "")
    markdown = markdown.replace(
        "*<span class=\"mark\">This syllabus must be augmented by a syllabus supplement for students</span>*",
        "::: {.callout-note}\nThis syllabus must be augmented by a syllabus supplement for students.\n:::\n\n## Syllabus Supplement",
    )
    markdown = re.sub(r"``` math\n(.*?)\n```", r"$$\n\1\n$$", markdown, flags=re.DOTALL)
    markdown = markdown.replace("<blockquote>\n", "").replace("</blockquote>", "")

    lines = markdown.splitlines()
    cleaned: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        match = re.fullmatch(r"\*\*(.+?)\*\*", stripped)
        if index == 0 and match:
            cleaned.append("## Course Information")
            cleaned.append("")
            cleaned.append(stripped)
            continue
        if match and match.group(1) in SECTION_HEADINGS:
            cleaned.append(f"## {SECTION_HEADINGS[match.group(1)]}")
            continue
        cleaned.append(line)

    markdown = "\n".join(cleaned)
    markdown = flatten_policy_sections(markdown)
    markdown = re.sub(r"(\$\$\n.*?\n\$\$)\n(?=\*\*)", r"\1\n\n", markdown, flags=re.DOTALL)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip() + "\n"


def replace_between(markdown: str, start: str, end: str, body: str) -> str:
    pattern = rf"({re.escape(start)}\n\n).*?(?=\n{re.escape(end)})"
    return re.sub(pattern, rf"\1{body.strip()}\n", markdown, flags=re.DOTALL)


def flatten_policy_sections(markdown: str) -> str:
    late_policy = """
### Quizzes

There will be no make-up quizzes. If the student fails to attend a scheduled quiz **without** an approved absence from the Student Success Department, then a grade of zero for that quiz will be assigned.

If the student fails to attend a quiz and the **Student Success Department confirms** that the student's absence from that quiz is approved, then the grade that will be assigned to that quiz is:

- the grade of the Semester examination if that quiz took place before the Semester examination;
- the grade of the Final Examination if that quiz took place after the Semester examination.

### Semester Examination

There will be no make-up Semester Examination. If the student fails to attend the scheduled Semester examination **without** an approved absence from the Student Success Department, then a grade of zero will be assigned for the Semester examination.

If the student fails to attend the Semester examination and the **Student Success Department confirms** that the absence from the Semester examination is approved, then the grade for the Final Examination will also be assigned as the grade for the Semester examination.

**A maximum of 40% of the course grade can be compensated through the above policy.**
"""

    additional_information = """
### Assessment

Late submission of the group project will not be accepted.

### Project

The project will be assigned in Week 6 of the semester. Students will work in pairs (teams of two) to formulate, analyze, and solve a mathematical imaging reconstruction problem. Each team will develop an appropriate mathematical model, implement computational reconstruction methods, analyze the results, and evaluate model performance and parameter sensitivity.

Students are required to use Python for all computational implementations and analyses.

A written report and the corresponding code must be submitted in Week 13.

In Week 14, each team will participate in an individual oral defense. Each student will be examined separately on the modeling decisions, implementation, and interpretation of results. Both students are expected to demonstrate full understanding of the entire project.

Failure to participate in the oral defense will result in a maximum of 50% of the project grade.

### Calculator Policy

You will be permitted to only use a basic calculator in the Quizzes, Semester examination and the Final Examination.

### Classroom Policies

Discipline is one of the main components of your academic and professional success.

Please arrive at the class periods in a timely fashion. Students arriving late to class are a distraction to both the instructor and the other students in the classroom.

Please ensure that you sign your name on the Attendance register. If you do not sign the Attendance register, then you will be marked "Absent" in the Banner.

**If you are absent for any lecture, it is your responsibility to cover the missed material.**
"""

    markdown = replace_between(
        markdown,
        "## Instructor Policy on Late Submission of Assignments",
        "## Additional Information",
        late_policy,
    )
    markdown = replace_between(
        markdown,
        "## Additional Information",
        "## Weekly Plan (Tentative)",
        additional_information,
    )
    return markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "docx",
        nargs="?",
        default=str(DEFAULT_DOCX),
        help="Path to the Word syllabus document.",
    )
    args = parser.parse_args()

    docx_path = Path(args.docx).expanduser()
    if not docx_path.exists():
        raise SystemExit(f"Could not find syllabus Word document: {docx_path}")

    body = clean_markdown(pandoc_markdown(docx_path))
    page = (
        "---\n"
        "title: \"Syllabus\"\n"
        "toc: true\n"
        "toc-depth: 3\n"
        "---\n\n"
        "<!-- This page is generated from the official Word syllabus by scripts/convert_syllabus_docx.py. -->\n\n"
        f"{body}"
    )
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)} from {docx_path}")


if __name__ == "__main__":
    main()
