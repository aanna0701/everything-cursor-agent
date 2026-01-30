---
name: create-ai-instruction-skills
description: Converts user requirements into Cursor Agent Skills (SKILL.md only). Use when the user provides requirement lines, asks for a skill/agent instruction, wants a draft for SKILL.md, or needs executable directives for the Cursor AI as a skill.
---

# Create Skill for Cursor

When the user gives requirements in natural language, produce a **Cursor Agent Skill** as a single **SKILL.md** file that the agent can discover and execute without ambiguity. This skill creates **skills only**; for rules (`.mdc`), use the create-rule skill or Cursor rules documentation.

---

## When to Use

- User provides **one or more requirement lines** (e.g. "do X", "create a skill for Y", "summarize what I want the AI to do as a skill")
- Requirements must be turned into a **clear SKILL.md** the Cursor agent can invoke when relevant
- A **draft or final skill** is needed for `~/.cursor/skills/<name>/SKILL.md` or `.cursor/skills/<name>/SKILL.md`

---

## Output Principles

1. **Actionable**: Phrase as behavior: "Do X", "When Y, do Z"—write in terms of **concrete actions**.
2. **Role and scope**: In one or two paragraphs, state the AI’s **role** (e.g. code reviewer, document writer) and **scope** (what is in/out of scope) when following this instruction.
3. **Consistent structure**: Use **MD structure**—headings (H1/H2/H3), lists, code blocks, checklists—so the instruction is easy to scan.
4. **Remove ambiguity**: Replace vague wording ("if possible", "as appropriate") with **conditions, exceptions, and priorities**.
5. **Language**: Match the user’s language (e.g. Korean → Korean, English → English). If mixing, specify per section.

---

## Output: SKILL.md Structure (Template)

Produce a **single SKILL.md** with YAML frontmatter and body. Use the structure below as a base; add, remove, or merge sections to fit the requirements.

```markdown
---
name: short-skill-name
description: Brief description of what this skill does and when the agent should use it (include trigger terms)
---

# [Skill title – one-line summary]

[1–2 paragraphs: purpose of this skill, and in what situation Cursor should apply it]

---

## Role and scope

- **Role**: (e.g. code reviewer, document writer, refactoring executor)
- **Scope**: (e.g. only `src/` in this repo, API design only, English docs only)
- **Out of scope**: (optional) behaviors explicitly excluded

---

## Input / output (when applicable)

- **Input**: What the user provides (files, message, selection, etc.)
- **Output**: What is produced after following the skill (patch, summary, checklist, etc.)

---

## Rules / procedure

1. (Rule 1 – concrete action)
2. (Rule 2)
3. (Exception: when X, do Y)

---

## Format / style (when applicable)

- (e.g. code style, document tone, file naming rules)

---

## Checklist (optional)

- [ ] (Completion condition 1)
- [ ] (Completion condition 2)
```

- **Frontmatter**: `name` (lowercase, hyphenated, max 64 chars), `description` (specific, include WHEN to use; max 1024 chars). Write description in **third person** for agent discovery.

---

## Writing Guidelines

| Item        | Recommended                                      | Avoid                                                |
|------------|---------------------------------------------------|------------------------------------------------------|
| Wording    | "When X, always do Y"                             | "It would be nice to Y", "Y is okay" (ambiguous)     |
| Examples   | Do/Don’t, code snippets, input/output samples     | Abstract description only, no examples               |
| Length     | Only as long as needed; split into sections       | One long block of text                               |
| Terminology| Use project/domain terms consistently             | Mixing different words for the same concept          |

---

## Example (requirement → skill draft)

**User message (requirement):**  
"Before opening a PR, check that our repo’s code style is followed and only suggest fixes. Do not apply changes."

**Output skill (summary):**

```markdown
---
name: pre-pr-style-check
description: Run code style checks before a PR and output fix suggestions only; do not modify files. Use when the user asks for pre-PR style check or suggestions-only lint.
---
# Pre-PR code style check and fix suggestions only (no edits)

This skill runs **only a code style check** before a PR is opened (or right before opening one locally) and **outputs only suggestions** for violations. **It does not modify code or files.**

---

## Role and scope

- **Role**: Code style checker; output suggestions only
- **Scope**: This repo’s code style rules (e.g. Ruff, Black, project rules)
- **Out of scope**: Saving files, auto-fixing, commit/push

---

## Rules

1. Run checks against the project’s defined linter/formatter rules.
2. For each violation, output **file:line**, **rule ID/name**, and **suggested fix** in text only.
3. Suggest fixes at a "you can change it like this" level; do not apply changes (no file writes).
4. Show suggested changes in code blocks with clear before/after.

---

## Checklist

- [ ] Style check run on all changed/added code
- [ ] Only suggestions were output; no files were modified
```

---

## Summary

- **Input**: User message (requirements); optionally the user’s Cursor root path.
- **Output**: A **single SKILL.md** (frontmatter + body) following the principles and structure above; the skill file is created in the chosen location.
- **Goal**: The agent creates the skill file under the user’s Cursor root (if provided or after asking) or in the same root as this instruction; no paths are hardcoded.

---

## Where to create the skill

Do **not** hardcode Cursor root paths (paths vary by OS and environment). Use only the user-provided path or the path of this instruction’s directory.

### 1. Cursor root (optional — ask if not given)

- **If the user provides a Cursor root path** in the request, use it: create the skill at `<cursor-root>/skills/<skill-name>/SKILL.md`. Create the folder `skills/<skill-name>` under that root if needed, and write the SKILL.md content there.
- **If the user does not provide a Cursor root path**, ask once: e.g. “What is your Cursor root path? (The directory that contains your `skills` folder.)” Use the path they give and create the skill at `<cursor-root>/skills/<skill-name>/SKILL.md`.
- Use the path exactly as provided; do not substitute or assume a fixed path.

### 2. Same root as this instruction (always available)

- Create the skill in the **same `skills` directory** that contains this instruction (`create-ai-instruction-skills`).
- Path: `<skills-root>/<skill-name>/SKILL.md`, where `<skills-root>` is the parent of the directory containing this SKILL.md.
- Use this when the user prefers the skill to live in the project, or in addition to creating under Cursor root if both are requested.

### 3. Naming and scope

- Use a short, **lowercase, hyphenated** name (e.g. `pre-pr-style-check`) in the frontmatter `name`. If the user specifies a name, use it; otherwise derive one from the requirement.
- **Do not** create or reference `.cursor/rules/` or `.mdc` in the produced skill; that is out of scope.

---

## Skill Creation Checklist (when applying this skill)

Before finalizing a skill produced by this skill, verify:

- [ ] YAML frontmatter has `name` and `description`; description is third-person and includes trigger terms (when to use)
- [ ] Role and scope are stated; out-of-scope is explicit when needed
- [ ] Rules/procedure are concrete actions, not vague ("when X, do Y")
- [ ] Format/style and checklist are included when applicable
- [ ] Language matches user preference (or is specified per section)
- [ ] No ambiguous wording ("if possible", "as appropriate") without conditions
- [ ] No hardcoded Cursor root path used; only user-provided path or same root as this instruction
- [ ] If Cursor root was not provided, user was asked for it; skill created at `<cursor-root>/skills/<skill-name>/SKILL.md` or at `<skills-root>/<skill-name>/SKILL.md` as chosen
