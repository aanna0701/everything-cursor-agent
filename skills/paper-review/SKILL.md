---
name: paper-review
description: Analyzes papers by type (Survey, Modeling, Dataset, Metric) with type-specific checklists and a 4-step workflow. Use when the user provides a paper (PDF, text, or URL) and wants structured review notes, taxonomy extraction, critical questioning, or feasibility assessment.
---

# Paper review by type: Survey, Modeling, Dataset, Metric

Review papers by **type** (assign primary type or mix %), **deconstruct** against that type's checklist, then **critical questioning**. Output: structured notes only; no code or experiments unless asked. Scope: AI/ML papers; user-provided text, PDF, or URL. Out of scope: full implementation, math correctness guarantee, legal advice.

**Input**: Paper content; optional type hint or "analyze all types." **Output**: Type + weight, checklist answers (outline form), 4-step notes, critical questions and caveats.

---

## Paper types and checklists

### 1. Survey — Knowledge-graph mode

**Goal**: Field "map" into the reader's mind. Fill every sub-item from the paper; mark "Not reported" when missing.

- **Target audience & motivation**: (1) Who is it for (entry-level, practitioner, expert)? (2) Stated motivation or gap the survey fills. (3) Prerequisites (e.g. assumed background). Cite §.
- **Scope & coverage**: (1) Time span (e.g. 2015–2024) and domains/topics covered. (2) Explicit exclusions or boundaries (what is out of scope). (3) Comparison tables (if any)—dimensions (method, task, metric, etc.). Cite §/Table.
- **Search & inclusion criteria**: (1) How papers were selected (databases, keywords, snowball). (2) Inclusion/exclusion criteria (language, venue, date). (3) Final count of papers cited (total and per category if reported). Cite §.
- **Taxonomy & hierarchy**: (1) Main axes (e.g. by method family, by task, by application). (2) Number of levels and branch structure; extract as mind map. (3) Key terms and definitions (consistency across sections). Cite §/Fig.
- **Per-topic coverage**: (1) Which topics get most space (paragraph/section count or page share). (2) Which are briefly mentioned; any imbalance? (3) Cross-links between topics (e.g. "see §X for datasets"). Cite §.
- **Milestones**: 3–5 "S-tier" papers that changed the field?
  - (1) List each with year, title, one-line contribution
  - (2) Briefly justify why S-tier (impact, citations, paradigm shift)
- **Paradigm shift**: (1) Why did past mainstream methods give way to current ones? (2) Key turning points (years, events, papers). (3) History of limitations (what failed, what replaced it). Cite §.
- **Future road**: ≥3 open challenges with short context?
  - (1)(2)(3) each with short context and (if stated) suggested directions
- **Figures & tables inventory**: (1) Timeline or evolution figure (if any). (2) Comparison tables (methods vs tasks vs metrics). (3) Statistics (citation counts, paper counts per category—if reported). Cite Fig/Table.
- **Benchmarks & datasets**: (1) Key benchmarks/datasets mentioned and how they are used in the narrative. (2) Cross-reference to "recommended" benchmarks for each sub-area. Cite §/Table.
- **Gaps & limitations**: (1) Sub-areas under-covered or missing. (2) Author-stated limitations of the survey. (3) Bias risk (venue bias, language bias, recency). Cite §.
- **Reproducibility**: (1) Are paper lists/tables machine-readable or in appendix? (2) Citation style (full list, BibTeX, link). (3) How to verify or extend the survey. Cite §.

### 2. Modeling — Logical design verification mode

**Goal**: Core "cheat key" + feasibility of porting to user's code. Fill every sub-item from the paper; mark "Not reported" when missing.

- **Architecture**: (1) Modular (separate components, pluggable) vs end-to-end (single pipeline). (2) Diagram reference (§/Fig). (3) Main blocks and data flow in one sentence.
- **Backbone**: (1) Encoder: name, variant, pretrained source (e.g. ImageNet, in-domain). (2) Decoder / head: name and output format. (3) Frozen vs fine-tuned layers (if stated).
- **Data environment**: (1) Simulation / synthetic vs real-world. (2) Domain: general, medical, robotics, autonomous driving, etc. (3) Single domain or multi-domain; cross-domain evaluation (if any).
- **Input modality**: (1) Primary: camera/video, LiDAR, radar, text, audio, multi-modal list. (2) Secondary modalities and how they are fused (early/late). (3) Any modality-specific preprocessing (e.g. projection, tokenization).
- **Input resolution / format**: (1) Spatial resolution (e.g. 224×224, 1920×1080). (2) Temporal: frame rate, clip length, sequence length. (3) Any resolution ablations reported.
- **Data augmentation & preprocessing**: (1) Training-time augmentation (list; cite §). (2) Preprocessing (normalization, cropping, resizing). (3) Test-time augmentation (if any).
- **Learning algorithm**: (1) Paradigm: supervised (SL), self-supervised (SSL), reinforcement learning (RL), semi-supervised, etc. (2) Loss function(s) and key terms (cite equation). (3) Training objective in one sentence.
- **Metrics**: (1) Primary metric(s) and definition. (2) Secondary metrics. (3) Dataset-specific metrics (e.g. mAP, BLEU, task-specific). Cite Table/§.
- **Data splits**: (1) Train / val / test counts (samples or sequences). (2) Split protocol: random, temporal, identity-based, official split. (3) Cross-dataset or cross-domain evaluation (if any).
- **Training hyperparameters**: (1) Optimizer, learning rate, schedule. (2) Batch size, epochs or steps. (3) Key regularizers (weight decay, dropout, etc.). (4) Hardware (GPU type × count) and training time—if reported. Cite §/Table.
- **Novelty**: (1) One-sentence claim of main contribution. (2) What is new vs prior work (module, objective, or setup). (3) Claimed advantages (e.g. efficiency, accuracy, generalization).
- **Problem–fix pair**: (1) Bottleneck in existing models? (2) Which module or design choice addresses it?
- **Mathematical logic**: In key equations (loss, attention, etc.), physical/geometric meaning of each variable? (Cite equation numbers.)
- **Ablation evidence**: Which component contributes most? (Cite table/section; list key ablations.)
- **Efficiency check**: Params, FLOPs, latency vs baseline; increase vs performance gain—reasonable? (Cite numbers.)
- **Validation experiments**: (1) What experiments were performed to validate the method? (e.g. main results, ablations, sensitivity, robustness, generalization). (2) Experimental design: baselines, datasets, protocols per experiment. (3) Summary of each experiment type and what it is intended to show. Cite §/Table/Fig.

### 3. Dataset — Reliability and usability

**Goal**: Can the data be trusted to train the user's model? Fill every sub-item from the paper; mark "Not reported" when missing.

- **Task & output format**: (1) Task type (classification, detection, segmentation, QA, generation, etc.). (2) Label types and vocabulary (classes, attributes, relations; hierarchy if any). (3) Input/output schema (per sample: fields, format). Cite §/Table.
- **Schema & specification**: (1) File format and directory structure (e.g. COCO, custom JSON). (2) Version or release date; changelog or errata if any. (3) Optional vs required fields; null/missing handling. Cite §.
- **Modality & sensors**: (1) Modalities (image, video, text, audio, LiDAR, etc.). (2) Resolution, frame rate, sequence length (if applicable). (3) Raw vs preprocessed (e.g. features only). Cite §/Table.
- **Collection setup**: (1) Source (in-the-wild, lab, web, synthetic). (2) Environment (location, duration, devices, conditions). (3) Consent and recruitment (if human subjects). Cite §.
- **Pipeline — Collection → Cleaning → Annotation**:
  - (1) **Collection**: protocol, sampling strategy, exclusion criteria.
  - (2) **Cleaning**: dedup, filtering, quality checks, outlier removal.
  - (3) **Annotation**: protocol, annotation manual (if cited), tools, who (experts, crowd), training for annotators. Cite §.
- **Size & splits**: (1) Total samples; per-split counts (train/val/test)—exact numbers. (2) Split protocol: random, temporal, identity-based, official split, stratification. (3) Per-class or per-domain counts; holdout policy. Cite §/Table.
- **Diversity & balance**: (1) Balanced vs imbalanced; long-tail distribution (cite stats if any). (2) Diversity metrics (if any). (3) Geographic, demographic, domain, or style coverage. Cite §/Table.
- **Quality control**: (1) Spot checks, adjudication, revision rounds. (2) Re-annotation or consistency checks. (3) Reported error rates or correction process. Cite §.
- **Gold standard**: (1) Reference for label accuracy (expert, benchmark, prior dataset). (2) Inter-annotator agreement (e.g. Cohen's κ, Krippendorff's α); sample size and subset for agreement. (3) Conflict resolution protocol. Cite §/Table.
- **License & ethics**: (1) License: research-only vs commercial use; redistribution and derivative works. (2) Privacy/ethical filtering (PII, consent, de-identification). (3) Compliance (e.g. HIPAA, GDPR) if stated. Cite §.
- **Baselines & benchmarks**: (1) Reported baseline results in the paper (model, metric, score). (2) SOTA or leaderboard at release (if any). (3) Recommended evaluation protocol. Cite §/Table.
- **Versioning & updates**: (1) v1/v2 or update history; what changed. (2) Deprecation or superseding dataset (if any). (3) How to cite (citation format). Cite §.
- **Usability**: (1) Download and access (URL, API, request form, approval). (2) Documentation and scripts (loading, splits, examples). (3) Known issues, errata, or limitations. Cite §.

### 4. Metric — Evaluation validity

**Goal**: Is evaluating the user's model with this metric logically justified? Fill every sub-item from the paper; mark "Not reported" when missing.

- **Definition & formula**: (1) Formal definition or equation (cite equation number). (2) Scale/range (bounded or not; theoretical min/max). (3) Higher = better or lower = better; interpretation in one sentence.
- **Input/output**: (1) What the metric takes as input (e.g. hypothesis, reference(s), model output format). (2) What it outputs (single score, per-segment, distribution). (3) Unit and aggregation (e.g. average over samples). Cite §.
- **Assumptions & preconditions**: (1) Required format (tokenized, aligned, normalized). (2) Single vs multiple references; how references are used. (3) Domain or task assumptions (e.g. English only, length range). Cite §.
- **Computation steps**: (1) Pipeline (e.g. normalize → compare → aggregate). (2) Key sub-steps and formulas. (3) Optional components (e.g. external embedder, reranker). Cite §/Fig.
- **Human alignment**: (1) What "human judgment" do existing metrics miss? (2) How does this metric quantify it? (3) Human study: design, sample size, correlation coefficient (e.g. Pearson, Spearman)—if reported. Cite §/Table.
- **Sensitivity**: (1) Captures small improvements/degradations? (2) Evidence (experiments or lack thereof). (3) Known insensitivities (e.g. length, word order, paraphrasing) or saturation. Cite §.
- **Robustness**: (1) Sensitivity to length, order, or paraphrasing (if discussed). (2) Adversarial or failure inputs (if studied). (3) Variance across runs or annotators (if reported). Cite §.
- **Complexity**: (1) Cost: time per sample, compute (GPU/memory), scalability. (2) Dependencies (external models, APIs, libraries). (3) Practical for large-scale or online eval? Cite §.
- **Baselines & comparison**: (1) Compared to which existing metrics? (2) Correlation or disagreement reported (tables, scatter plots). (3) When to prefer this metric over others (author claim). Cite §/Table.
- **Applicability**: (1) Tasks/domains where the metric is valid or recommended. (2) When not to use (author-stated limitations). (3) Extension to other languages or modalities (if discussed). Cite §.
- **Failure cases**: (1) Cases where this metric disagrees with others (high on others, bad output—does this metric catch it?). (2) Failure modes where this metric is misleading (if discussed). Cite §.
- **Implementation & availability**: (1) Code, API, or toolkit (URL, license). (2) Reproducibility (random seed, version). (3) Standardized usage (e.g. default config, preprocessing). Cite §.

---

## 4-step workflow

1. **Contextualize**: Primary type (or weights if mixed). State primary for main checklist.
2. **Deconstruct**: Fill primary type's checklist from the paper only. Short, scannable bullets; cite §/Fig/Table. If "full" or "all types," fill all four where the paper has info.
3. **Critical questioning**: (1) Omission check—omitted or downplayed results that could change interpretation? (2) Transfer check—same on user's dataset/environment? Note assumptions.
4. **Summary**: One short paragraph (takeaway, strongest evidence, main caveat). Optional: 2–3 next steps.

---

## Rules

1. Use only provided paper content; do not invent.
2. Run Steps 1→2→3 in order; always include both critical questions (and type-specific ones when obvious).
3. **Output format**: Outline form—bullet lists; multiple entries as `(1)` newline `(2)` newline `(3)`. Headings per step/type. Math in LaTeX where relevant (§/Table/Fig for citations).
4. **Missing info**: State "Not reported" or "Unclear"; do not guess. Optionally say what would be needed.
5. **Numbers**: Only values from the paper; cite Table/Fig/§. No inventing, approximating, or rounding. Unreported → "Not reported."

---

## Checklist (skill application)

- [ ] Type(s) and weights stated; primary checklist filled from paper (all sub-items where applicable).
- [ ] Critical questioning (omission + transfer) present; summary (± next steps) provided.
- [ ] No invented content; missing info and numbers marked; sources cited.
- [ ] Outline-form bullets; (1)/(2)/(3) for multiple entries; math notation where relevant.
