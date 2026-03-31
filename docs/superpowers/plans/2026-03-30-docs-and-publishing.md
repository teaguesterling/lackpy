# Docs & Publishing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add mkdocs-material documentation (tutorial, API reference, extension guide) with ReadTheDocs deployment, plus a GitHub Actions workflow for PyPI publishing via trusted publishing.

**Architecture:** mkdocs-material with mkdocstrings-python for auto-generated API docs. ReadTheDocs builds from `.readthedocs.yaml`. PyPI publishing via GitHub Actions on tag push using trusted publishing (OIDC, no tokens). Docs live in `docs/` at the repo root.

**Tech Stack:** mkdocs-material, mkdocstrings[python], readthedocs, GitHub Actions, hatch

---

## File Structure

```
docs/
├── index.md                    # Landing page / overview
├── getting-started.md          # Installation + first use
├── tutorial.md                 # Walkthrough: CLI and programmatic usage
├── concepts/
│   ├── architecture.md         # Pipeline overview, module roles
│   ├── language-spec.md        # The lackpy restricted Python subset
│   ├── kits.md                 # Kits, toolbox, providers
│   └── inference.md            # Inference pipeline, tiers, providers
├── extending/
│   ├── tool-providers.md       # Writing custom tool providers
│   ├── inference-providers.md  # Writing custom inference providers
│   └── custom-rules.md        # Writing custom validation rules
├── reference/
│   ├── api.md                  # Auto-generated from docstrings (mkdocstrings)
│   └── cli.md                  # CLI command reference
├── css/
│   └── extra.css               # Minor style tweaks
mkdocs.yml                      # mkdocs configuration
.readthedocs.yaml               # ReadTheDocs build config
.github/
└── workflows/
    └── publish.yml             # PyPI publish on tag push
```

---

### Task 1: mkdocs Configuration and Dependencies

**Files:**
- Create: `mkdocs.yml`
- Modify: `pyproject.toml`
- Create: `.readthedocs.yaml`
- Create: `docs/css/extra.css`

- [ ] **Step 1: Add docs dependencies to pyproject.toml**

Add a `docs` optional dependency group after the existing `dev` group:

```toml
docs = [
    "mkdocs-material>=9",
    "mkdocstrings[python]>=0.24",
    "mkdocs-gen-files>=0.5",
    "mkdocs-literate-nav>=0.6",
]
```

Also add project URLs:

```toml
[project.urls]
Documentation = "https://lackpy.readthedocs.io"
Repository = "https://github.com/teague/lackpy"
```

- [ ] **Step 2: Create mkdocs.yml**

```yaml
site_name: lackpy
site_description: Python that lacks most of Python. Restricted program generation and execution for tool composition.
site_url: https://lackpy.readthedocs.io

repo_url: https://github.com/teague/lackpy
repo_name: teague/lackpy

theme:
  name: material
  palette:
    - scheme: default
      primary: deep purple
      accent: amber
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: deep purple
      accent: amber
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - content.code.copy
    - content.code.annotate
    - navigation.sections
    - navigation.expand
    - navigation.top
    - toc.follow

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.inlinehilite
  - pymdownx.tabbed:
      alternate_style: true
  - toc:
      permalink: true
  - attr_list
  - def_list

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src]
          options:
            show_source: false
            show_root_heading: true
            show_root_full_path: false
            heading_level: 3
            members_order: source
            docstring_style: google
            merge_init_into_class: true
            show_if_no_docstring: false

nav:
  - Home: index.md
  - Getting Started: getting-started.md
  - Tutorial: tutorial.md
  - Concepts:
    - Architecture: concepts/architecture.md
    - Language Spec: concepts/language-spec.md
    - Kits & Toolbox: concepts/kits.md
    - Inference Pipeline: concepts/inference.md
  - Extending:
    - Tool Providers: extending/tool-providers.md
    - Inference Providers: extending/inference-providers.md
    - Custom Rules: extending/custom-rules.md
  - Reference:
    - Python API: reference/api.md
    - CLI: reference/cli.md

extra_css:
  - css/extra.css
```

- [ ] **Step 3: Create .readthedocs.yaml**

```yaml
version: 2

build:
  os: ubuntu-22.04
  tools:
    python: "3.12"

mkdocs:
  configuration: mkdocs.yml

python:
  install:
    - method: pip
      path: .
      extra_requirements:
        - docs
```

- [ ] **Step 4: Create docs/css/extra.css**

```css
/* Slightly tighter code blocks */
.md-typeset code {
    font-size: 0.85em;
}
```

- [ ] **Step 5: Install docs deps and verify mkdocs builds**

Run: `pip install -e ".[docs]" && mkdocs build --strict 2>&1 | tail -5`
Expected: Build succeeds (with warnings about missing doc files — that is fine, we create them next)

- [ ] **Step 6: Commit**

```bash
git add mkdocs.yml .readthedocs.yaml pyproject.toml docs/css/
git commit -m "chore: add mkdocs configuration, ReadTheDocs config, docs dependencies"
```

---

### Task 2: Landing Page and Getting Started

**Files:**
- Create: `docs/index.md`
- Create: `docs/getting-started.md`

- [ ] **Step 1: Create docs/index.md**

The landing page should cover:
- One-line description and what lackpy does
- Why lackpy (zero-dep core, AST security boundary, plugin-based, untrusted model, ratchet)
- Quick example: CLI usage and Python API (using LackpyService, ToolSpec, ArgSpec)
- Part of the Rigged Suite mention
- Next steps links to getting-started, tutorial, architecture

- [ ] **Step 2: Create docs/getting-started.md**

Should cover:
- Installation (`pip install lackpy`, `lackpy[ollama]`, `lackpy[full]`)
- Optional dependencies table (ollama, anthropic, sandbox, mcp)
- Initialize a workspace (`lackpy init`)
- First use: CLI (`lackpy delegate`)
- First use: Python API (async example with LackpyService)
- Configuration pointer
- Next steps links

- [ ] **Step 3: Verify mkdocs builds**

Run: `mkdocs build --strict 2>&1 | tail -3`

- [ ] **Step 4: Commit**

```bash
git add docs/index.md docs/getting-started.md
git commit -m "docs: add landing page and getting started guide"
```

---

### Task 3: Tutorial

**Files:**
- Create: `docs/tutorial.md`

- [ ] **Step 1: Create docs/tutorial.md**

Step-by-step walkthrough covering:
- Setup (install + init)
- Understanding the pipeline (diagram: Intent -> Kit Resolution -> Inference -> Validation -> Execution -> Trace)
- Validating programs (valid/invalid examples, what gets rejected table)
- Working with kits (CLI: list, info, create; Python API: register tools, kit_info)
- Generating programs (generate without executing, full delegate pipeline, tier explanation)
- Running programs directly (skip inference with `lackpy run` and `service.run_program`)
- Using parameters (params dict, how inferencer sees metadata not values)
- Creating templates (CLI and Python API, the ratchet concept)
- Custom validation rules (no_loops, max_calls examples)
- The trace (what delegate returns, accessing program, tier, grade, output, steps)
- Next steps links

- [ ] **Step 2: Verify mkdocs builds**

Run: `mkdocs build --strict 2>&1 | tail -3`

- [ ] **Step 3: Commit**

```bash
git add docs/tutorial.md
git commit -m "docs: add tutorial — walkthrough of CLI and Python API"
```

---

### Task 4: Concepts Pages

**Files:**
- Create: `docs/concepts/architecture.md`
- Create: `docs/concepts/language-spec.md`
- Create: `docs/concepts/kits.md`
- Create: `docs/concepts/inference.md`

- [ ] **Step 1: Create docs/concepts/architecture.md**

Cover:
- Pipeline diagram (text art: Kit Resolution -> Inference -> Validation -> Execution -> Trace)
- Modules table (module, responsibility, dependencies) — emphasize lang/ is stdlib only
- Service layer role (both CLI and MCP are thin adapters)
- Security model (primary: AST validation, secondary: restricted exec, v2: nsjail)
- Grade system (grade_w, effects_ceiling, join of tool grades)

- [ ] **Step 2: Create docs/concepts/language-spec.md**

Cover:
- Design philosophy (compose tools, cannot escape, auditable, grade-decidable)
- Allowed constructs (structural, expressions, builtins list)
- Forbidden constructs table (construct + why)
- Forbidden names (with explanation)
- Additional checks (namespace, for-loop, string dunder)
- Custom rules pointer

- [ ] **Step 3: Create docs/concepts/kits.md**

Cover:
- Toolbox vs Kits distinction
- ToolSpec fields (name, provider, description, args, returns, grades)
- Registering tools (Python example)
- Tool providers table (builtin, python, v2: mcp-local, mcp-system, inline)
- Kit parameter forms table (str, list, dict, None)
- Predefined kit file format (frontmatter + tool list)
- Managing kits (CLI commands)
- Grade computation example

- [ ] **Step 4: Create docs/concepts/inference.md**

Cover:
- Tiers table (templates, rules, ollama, anthropic — cost and when)
- Each tier explained (templates with pattern example, rules with mappings, ollama with model config, anthropic as fallback)
- Dispatch flow (available -> generate -> sanitize -> validate -> retry once -> next)
- Configuration (config.toml example)
- The ratchet (explicit create + promotion from traces)
- Custom providers pointer

- [ ] **Step 5: Verify mkdocs builds**

Run: `mkdocs build --strict 2>&1 | tail -3`

- [ ] **Step 6: Commit**

```bash
git add docs/concepts/
git commit -m "docs: add concepts — architecture, language spec, kits, inference"
```

---

### Task 5: Extension Guides

**Files:**
- Create: `docs/extending/tool-providers.md`
- Create: `docs/extending/inference-providers.md`
- Create: `docs/extending/custom-rules.md`

- [ ] **Step 1: Create docs/extending/tool-providers.md**

Cover:
- The protocol (name property, available(), resolve(tool_spec) -> Callable)
- Example: REST API provider (complete working code)
- Registering a custom provider
- Built-in providers reference (builtin: read/glob/write/edit, python: import-based)

- [ ] **Step 2: Create docs/extending/inference-providers.md**

Cover:
- The protocol (name property, available(), async generate() -> str|None)
- Example: cache provider (complete working code)
- Integration (config.toml, service init)
- Prompt construction helper (build_system_prompt)
- Output sanitization (automatic in dispatch)

- [ ] **Step 3: Create docs/extending/custom-rules.md**

Cover:
- The interface (callable: ast.Module -> list[str])
- Built-in rules (no_loops, max_depth, max_calls, no_nested_calls) with examples
- Writing your own (complete example: no_string_literals)
- Parameterized rules (factory function pattern: max_string_length)

- [ ] **Step 4: Verify mkdocs builds**

Run: `mkdocs build --strict 2>&1 | tail -3`

- [ ] **Step 5: Commit**

```bash
git add docs/extending/
git commit -m "docs: add extension guides — tool providers, inference providers, custom rules"
```

---

### Task 6: API Reference and CLI Reference

**Files:**
- Create: `docs/reference/api.md`
- Create: `docs/reference/cli.md`

- [ ] **Step 1: Create docs/reference/api.md**

Use mkdocstrings `::: module.Class` directives for auto-generated API docs. Sections:
- Service Layer (LackpyService with specific members listed)
- Validation (validate function, ValidationResult)
- Grading (Grade, compute_grade)
- Grammar Constants (ALLOWED_NODES, FORBIDDEN_NODES, FORBIDDEN_NAMES, ALLOWED_BUILTINS)
- Custom Rules (no_loops, max_depth, max_calls, no_nested_calls)
- Kit (Toolbox, ToolSpec, ArgSpec, resolve_kit, ResolvedKit)
- Execution (RestrictedRunner, ExecutionResult, Trace, TraceEntry)
- Inference (InferenceDispatcher, GenerationResult, build_system_prompt, format_params_description, sanitize_output)

- [ ] **Step 2: Create docs/reference/cli.md**

Manual reference for each CLI command:
- `lackpy delegate` — usage, arguments table, example
- `lackpy generate` — usage, example
- `lackpy run` — usage, example
- `lackpy create` — usage, arguments table, example
- `lackpy validate` — usage
- `lackpy spec` — usage
- `lackpy status` — usage
- `lackpy kit` (list, info, create) — usage and examples
- `lackpy toolbox` (list, show) — usage
- `lackpy template` (list, test) — usage
- `lackpy init` — usage

- [ ] **Step 3: Verify mkdocs builds and API reference renders**

Run: `mkdocs build --strict 2>&1 | tail -5`
Expected: Build succeeds. Check `site/reference/api/index.html` exists.

- [ ] **Step 4: Commit**

```bash
git add docs/reference/
git commit -m "docs: add API reference (mkdocstrings) and CLI reference"
```

---

### Task 7: GitHub Actions — PyPI Publishing

**Files:**
- Create: `.github/workflows/publish.yml`

- [ ] **Step 1: Create .github/workflows/publish.yml**

The workflow should have three jobs:

**test** — runs on push to main and on PRs:
- Matrix: Python 3.10, 3.11, 3.12
- Install with dev deps
- Run pytest

**build** — runs on tag push (v*):
- Install hatch
- Build with `hatch build`
- Upload dist/ as artifact

**publish** — runs after build, on tag push only:
- Download artifact
- Publish to PyPI using `pypa/gh-action-pypi-publish@release/v1`
- Uses `environment: pypi` and `permissions: id-token: write` for trusted publishing

```yaml
name: CI

on:
  push:
    branches: [main]
    tags: ["v*"]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install
        run: pip install -e ".[dev]"
      - name: Run tests
        run: pytest tests/ -v

  build:
    if: startsWith(github.ref, 'refs/tags/v')
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install build tools
        run: pip install hatch
      - name: Build package
        run: hatch build
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write
    steps:
      - name: Download artifacts
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

- [ ] **Step 2: Commit**

```bash
git add .github/
git commit -m "ci: add GitHub Actions workflow — CI tests and PyPI publishing"
```

---

### Task 8: Final Build Verification

**Files:** None created — verification only.

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All 132 tests pass

- [ ] **Step 2: Build docs**

Run: `mkdocs build --strict`
Expected: Clean build, no errors

- [ ] **Step 3: Build package**

Run: `hatch build`
Expected: Produces `dist/lackpy-0.1.0.tar.gz` and `dist/lackpy-0.1.0-py3-none-any.whl`

- [ ] **Step 4: Verify package contents**

Run: `tar tzf dist/lackpy-0.1.0.tar.gz | head -20`
Expected: Contains `src/lackpy/` files, `pyproject.toml`, no `docs/` or `tests/`

- [ ] **Step 5: Commit any final adjustments**

```bash
git add -A
git commit -m "chore: final verification — docs build, package build, tests pass"
```
