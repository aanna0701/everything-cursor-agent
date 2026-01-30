---
name: paper-review
description: Analyzes papers by type (Survey, Modeling, Dataset, Metric) with type-specific checklists and a 4-step workflow. Use when the user provides a paper (PDF, text, or URL) and wants structured review notes, taxonomy extraction, critical questioning, or feasibility assessment.
---

# Paper review by type: Survey, Modeling, Dataset, Metric

Review papers by **type** (assign primary type or mix %), **deconstruct** against that type’s checklist, then **critical questioning**. Output: structured notes only; no code or experiments unless asked. Scope: AI/ML papers; user-provided text, PDF, or URL. Out of scope: full implementation, math correctness guarantee, legal advice.

**Input**: Paper content; optional type hint or “analyze all types.” **Output**: Type + weight, checklist answers (outline form), 4-step notes, critical questions and caveats.

---

## Paper types and checklists

### 1. Survey — Knowledge-graph mode

**Goal**: Field “map” into the reader’s mind.

- **Hierarchy**: Can the taxonomy be drawn as a mind map? Extract and structure it.
- **Milestones**: 3–5 “S-tier” papers that changed the field?
  - (1) List each
  - (2) Briefly justify
- **Paradigm shift**: Why did past mainstream methods give way to current ones? (History of limitations.)
- **Future road**: ≥3 open challenges with short context?
  - (1)(2)(3) each with short context

### 2. Modeling — Logical design verification mode

**Goal**: Core “cheat key” + feasibility of porting to user’s code.

- **Problem–fix pair**: (1) Bottleneck in existing models? (2) Which module addresses it?
- **Mathematical logic**: In key equations (Loss, attention, etc.), physical/geometric meaning of each variable?
- **Ablation evidence**: Which component contributes most? (Cite table/section.)
- **Efficiency check**: Params/FLOPs increase vs. performance gain—reasonable?

### 3. Dataset — Reliability and usability

**Goal**: Can the data be trusted to train the user’s model?

- **Pipeline**: Human involvement in Collection → Cleaning → Annotation?
  - (1) Collection (2) Cleaning (3) Annotation
- **Diversity metrics**: Balanced? Imbalance or long-tail?
- **Gold standard**: (1) Reference for label accuracy? (2) Inter-annotator agreement (e.g. Cohen’s κ)?
- **License & ethics**: (1) Commercial use allowed? (2) Privacy/ethical filtering (PII, consent)?

### 4. Metric — Evaluation validity

**Goal**: Is evaluating the user’s model with this metric logically justified?

- **Human alignment**: (1) What “human judgment” do existing metrics miss? (2) How does this metric quantify it?
- **Sensitivity**: Captures small improvements/degradations? (Evidence or lack thereof.)
- **Complexity**: Cost (time/compute) practical?
- **Case study**: Failure cases (high on other metrics, bad outputs) that this metric catches?

---

## 4-step workflow

1. **Contextualize**: Primary type (or weights if mixed). State primary for main checklist.
2. **Deconstruct**: Fill primary type’s checklist from the paper only. Short, scannable bullets; cite §/Fig/Table. If “full” or “all types,” fill all four where the paper has info.
3. **Critical questioning**: (1) Omission check—omitted or downplayed results that could change interpretation? (2) Transfer check—same on user’s dataset/environment? Note assumptions.
4. **Summary**: One short paragraph (takeaway, strongest evidence, main caveat). Optional: 2–3 next steps.

---

## Rules

1. Use only provided paper content; do not invent.
2. Run Steps 1→2→3 in order; always include both critical questions (and type-specific ones when obvious).
3. **Output format**: Outline form—bullet lists; multiple entries as `(1)` newline `(2)` newline `(3)`. Headings per step/type. Math in LaTeX where relevant (§/Table/Fig for citations).
4. **Missing info**: State “Not reported” or “Unclear”; do not guess. Optionally say what would be needed.
5. **Numbers**: Only values from the paper; cite Table/Fig/§. No inventing, approximating, or rounding. Unreported → “Not reported.”

---

## Checklist (skill application)

- [ ] Type(s) and weights stated; primary checklist filled from paper.
- [ ] Critical questioning (omission + transfer) present; summary (± next steps) provided.
- [ ] No invented content; missing info and numbers marked; sources cited.
- [ ] Outline-form bullets; (1)/(2)/(3) for multiple entries; math notation where relevant.
