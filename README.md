# Everything Cursor Agent (Python ML/DL)

**Cursor-only instructions library for Python ML/DL workflows.**

This repo provides **rules** and **skills** optimized for Cursor: copy rules into `.cursor/rules/`, skills into `~/.cursor/skills/` or `.cursor/skills/`, and use them with the Cursor AI agent.

---

## What's Inside

```
everything-cursor-agent/
├── rules/                    # Cursor rules (copy to .cursor/rules/)
│   ├── python-coding-style.md
│   ├── security.md
│   ├── testing.md
│   ├── git-workflow.md
│   ├── patterns.md
│   ├── performance.md
│   └── when-to-use-skills.md
├── skills/                   # Agent skills (copy to ~/.cursor/skills/ or .cursor/skills/)
│   ├── coding-standards/
│   ├── ml-training-patterns/
│   ├── ml-inference-patterns/
│   ├── ml-deployment-patterns/
│   ├── ml-experiment-reproducibility/
│   ├── tdd-workflow/
│   ├── security-review/
│   ├── project-guidelines-example/
│   └── create-ai-instruction-skills/
├── examples/
│   ├── project-template/     # ML/DL project skeleton
│   └── USER-CURSOR.md        # Example main instructions for Cursor
├── PROJECT-README.md        # Project-level Cursor setup
└── CONTRIBUTING.md
```

---

## Quick Start

### 1. Install rules (project or global)

**Project-level** (recommended for ML/DL projects):

```powershell
# From repo root
mkdir -p .cursor/rules
Copy-Item -Recurse everything-cursor-agent/rules/* .cursor/rules/
```

**Or** copy rule files into your project’s `.cursor/rules/` and add frontmatter if you use `.mdc` (see Cursor docs).

### 2. Install skills

**Personal** (all projects):

```powershell
Copy-Item -Recurse everything-cursor-agent/skills/* $env:USERPROFILE\.cursor\skills\
```

**Project-level** (share with repo):

```powershell
mkdir -p .cursor/skills
Copy-Item -Recurse everything-cursor-agent/skills/* .cursor/skills/
```

### 3. Main instructions

Use `examples/USER-CURSOR.md` as a reference, or merge its content into your project’s `cursor_rules.md` or `.cursor/rules/` so the agent always sees your preferences.

---

## Tooling Stack

| Tool   | Purpose                |
|--------|------------------------|
| **uv** | Package management     |
| **ruff** | Lint / format       |
| **pyright** | Type checking    |
| **pytest** | Tests              |
| **FastAPI** | Inference serving |

---

## Rules vs Skills

| Aspect   | Rules                         | Skills                          |
|----------|-------------------------------|---------------------------------|
| Purpose  | Always-follow guidelines      | Task-specific workflows         |
| Scope    | Project or global             | Invoked when relevant           |
| Content  | “Must do” (style, security)   | “How to” (training, deployment) |
| Location | `.cursor/rules/`              | `~/.cursor/skills/` or `.cursor/skills/` |

---

## MCP (Optional)

Cursor supports MCP. Configure servers in Cursor settings or project config. Do not enable too many MCPs at once to preserve context; keep the number of active tools reasonable.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — use freely, modify as needed.
