# Examples

Example instructions and a project template for Cursor + Python ML/DL.

---

## Contents

| Item | Description |
|------|--------------|
| **USER-CURSOR.md** | Example main instructions for Cursor (merge into `cursor_rules.md` or `.cursor/rules/`) |
| **project-template/** | ML/DL project skeleton (src, tests, configs, scripts) |

---

## USER-CURSOR.md

Reference for your project’s main Cursor instructions. Copy or merge into:

- Project root: `cursor_rules.md`, or
- `.cursor/rules/` as a rule file (e.g. with frontmatter for Cursor).

Adjust to your preferences (language, style, tooling).

---

## project-template/

Skeleton for a new ML/DL project:

```
project-template/
├── configs/     # YAML/TOML configs
├── data/        # Data layout (raw, processed)
├── models/      # Checkpoints, exported models
├── notebooks/    # Exploration, analysis
└── src/         # train.py, eval.py, inference.py, etc.
```

**Usage:**

```powershell
Copy-Item -Path "everything-cursor-agent/examples/project-template" -Destination "C:\path\to\new-project" -Recurse
cd C:\path\to\new-project
# Then copy rules/skills from everything-cursor-agent and add your code
```

---

## See also

- [../README.md](../README.md) — Library overview and install
- [../PROJECT-README.md](../PROJECT-README.md) — Project-level Cursor setup
