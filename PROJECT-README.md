# Project-Level Cursor Setup (Python ML/DL)

Use this when adding Cursor rules and skills to a **single ML/DL project**. For a global instructions library, see the main [README.md](README.md).

---

## Quick setup

### 1. Copy rules into the project

```powershell
# From your project root
mkdir -Force .cursor/rules
Copy-Item -Path "<path-to>/everything-cursor-agent/rules/*" -Destination .cursor/rules/ -Recurse
```

Optionally add YAML frontmatter and save as `.mdc` (see Cursor docs for rule format).

### 2. Copy skills (project-level)

```powershell
mkdir -Force .cursor/skills
Copy-Item -Path "<path-to>/everything-cursor-agent/skills/*" -Destination .cursor/skills/ -Recurse
```

Or install skills globally: copy into `~/.cursor/skills/` so they apply to all projects.

### 3. Main instructions

Merge [examples/USER-CURSOR.md](examples/USER-CURSOR.md) into your project’s `cursor_rules.md` or a rule in `.cursor/rules/` so the agent always sees your preferences.

---

## What you get

- **Rules**: Python style, security, testing, git, patterns, performance, when-to-use-skills.
- **Skills**: coding-standards, ml-training-patterns, ml-inference-patterns, ml-deployment-patterns, ml-experiment-reproducibility, tdd-workflow, security-review, project-guidelines-example, create-ai-instruction-skills.

See [rules/when-to-use-skills.md](rules/when-to-use-skills.md) for when to use each skill.

---

## Project layout (ML/DL)

Use [examples/project-template/](examples/project-template/) as a skeleton: `src/`, `tests/`, `configs/`, `scripts/`, and standard ML/DL entrypoints (train, eval, inference, serve).

---

## Updates

Re-copy `rules/` and `skills/` from this repo when you want to pull the latest instructions.
