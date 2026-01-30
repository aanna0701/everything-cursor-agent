# Main Cursor Instructions (Example)

Use this as a reference for your project’s main instructions (e.g. in `cursor_rules.md` or a rule in `.cursor/rules/`). Adjust to your preferences.

---

## General

- All file contents (code, comments, docs) must be in **English only**.
- When implementing features: write in **Python**; use **uv** for dependencies (e.g. `uv add <pkg>`); provide run instructions (`uv run python ...`) and a minimal example; clean up temporary artifacts.
- When drafting docs: extract from code comments and README; include Overview, Install, Usage, Structure, Examples; English only.
- When refactoring: make comments English; add/normalize type hints; preserve behavior unless asked otherwise; summarize changes.

---

## Rules and skills

- **Rules** in this project live in `.cursor/rules/` (or are referenced from this library). Follow them for style, security, testing, and git.
- **Skills** are in `~/.cursor/skills/` or `.cursor/skills/`. Use the **when-to-use-skills** rule to decide which skill to apply (e.g. tdd-workflow for new features, security-review for auth/APIs).

---

## Code style

- Type hints on all functions; **pathlib** for paths; **logging** instead of print.
- Prefer immutability; many small files (e.g. 200–400 lines, max ~600).
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`.
- TDD when appropriate: write tests first, then implement; aim for 80%+ coverage on data/preprocessing.

---

## Success criteria

- Tests pass; no security issues; code is readable and matches requirements.
