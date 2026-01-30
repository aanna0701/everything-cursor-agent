# Contributing to Everything Cursor Agent

Thanks for contributing. This repo is a Cursor-only instructions library (rules and skills) for Python ML/DL.

## What we accept

### Rules

Guidelines the agent should follow (style, security, testing, git, patterns):

- Place in `rules/` as `.md` (or `.mdc` with frontmatter if you follow Cursor’s rule format).
- Keep focused and short; one concern per file.

### Skills

Task-specific workflows (how to train, deploy, test, etc.):

- Place in `skills/<skill-name>/SKILL.md`.
- Each skill needs YAML frontmatter: `name`, `description`.
- Description should be specific and include when Cursor should use it (see Cursor skill docs).
- Keep SKILL.md under ~500 lines; use separate reference files if needed.

## How to contribute

1. Fork the repo and create a branch (e.g. `add-my-rule`).
2. Add or edit files under `rules/` or `skills/<name>/`.
3. Follow existing patterns (see [create-skill](https://cursor.com/docs) / create-rule docs).
4. Open a PR with a short description of what you added and why.

## Guidelines

- Do: Keep rules/skills focused, clear, and in English.
- Do: Test that your rule/skill works with Cursor before submitting.
- Don’t: Include secrets, API keys, or machine-specific paths.
- Don’t: Add Claude Code–specific content (agents, commands, hooks); this repo is Cursor-only.

## File naming

- Rules: lowercase with hyphens, e.g. `python-coding-style.md`.
- Skills: folder name = skill name, e.g. `skills/tdd-workflow/SKILL.md`.
