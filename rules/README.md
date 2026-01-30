# Rules (Cursor)

This folder contains Cursor [Rules](https://cursor.com/docs/context/rules) in `.mdc` format. Rules are applied in Agent (Chat) to give the model consistent, reusable context.

## Rule Types (Cursor)

| Rule type | Description |
|-----------|-------------|
| **Always Apply** | Applied to every chat session. |
| **Apply Intelligently** | Agent applies when it judges the rule relevant based on the `description`. |
| **Apply to Specific Files** | Applied when the file you're editing matches the `globs` pattern. |
| **Apply Manually** | Applied when you @mention the rule in chat (e.g. `@always-apply`, `@testing`). Any rule can be applied this way. |

## How Our Rules Map to These Types

| File | Primary type | Secondary |
|------|--------------|------------|
| `always-apply.mdc` | **Always Apply** — every chat | — |
| `git-workflow.mdc` | **Apply Intelligently** — when committing, PRs, feature flow | @git-workflow |
| `patterns.mdc` | **Apply Intelligently** + **Apply to Specific Files** (`**/*.py`) | @patterns |
| `performance.mdc` | **Apply Intelligently** — when planning agent usage, task breakdown | @performance |
| `python-coding-style.mdc` | **Apply Intelligently** + **Apply to Specific Files** (`**/*.py`) | @python-coding-style |
| `security.mdc` | **Apply Intelligently** — when auth, user input, secrets, deploying ML | @security |
| `testing.mdc` | **Apply Intelligently** + **Apply to Specific Files** (`**/test_*.py`, `**/tests/**`) | @testing |

**Apply Manually**: In any chat you can type `@` and pick a rule (e.g. `@testing`) to force that rule to be applied for that conversation.

## Files Overview (by description)

| File | When applied (description) |
|------|-----------------------------|
| `always-apply.mdc` | Always — global defaults: English-only, Python/uv, code style (type hints, pathlib, logging, file size, conventional commits), docs/refactor/new-project, skills reference, verify with pytest/ruff/pyright. |
| `git-workflow.mdc` | PR workflow, feature implementation flow (commit format is in always-apply). |
| `patterns.mdc` | Implementing ML/DL code (config, dataset, training, inference, device, reproducibility, batching). |
| `performance.mdc` | Planning agent usage, context scope, task breakdown, build/test failure handling. |
| `python-coding-style.mdc` | Python ML/DL only: device placement and reproducibility (general style is in always-apply). |
| `security.mdc` | Auth, user input, secrets, deploying ML. |
| `testing.mdc` | Writing or running tests (coverage, TDD, ML/DL focus, numeric comparisons; test commands in always-apply). |

## Apply to Specific Files (globs)

- `patterns.mdc`, `python-coding-style.mdc` — applied when editing files matching `**/*.py`
- `testing.mdc` — applied when editing `**/test_*.py` or `**/tests/**`

## Managing rules

- **Add a rule**: Create a new `.mdc` with frontmatter: `description`, `alwaysApply` (true/false), and optional `globs`.
- **Always Apply**: Set `alwaysApply: true`; only one file in this folder uses it (`always-apply.mdc`). Keep under ~500 lines.
- **Apply Intelligently**: Set `alwaysApply: false` and write a clear `description` so the agent can decide when to load it.
- **Apply to Specific Files**: Add `globs` (e.g. `["**/*.py"]`) so the rule is applied when the edited file matches.
- **Apply Manually**: Any rule can be applied by @mentioning it in chat (e.g. `@security`).
