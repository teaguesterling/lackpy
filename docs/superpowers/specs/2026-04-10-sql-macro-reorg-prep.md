# SQL Macro Reorganization Prep

*Planning doc for lackpy's interpreter work anticipating a larger reorg
where SQL macros move to fledgling and pluckit stops defining them inline.
Nothing in this doc is implementation work — it's a checklist of things
to do (and not do) in the current branch so the eventual reorg is cheap.*

## The target layering

```
Layer 4: Consumers (lackpy interpreters, kibitzer, agent-riggs, etc.)
          ↓
Layer 3: pluckit — Python fluent API, plugin architecture
          ↓ depends on
Layer 2: fledgling-python — thin pip package that bundles fledgling's
          SQL files and loads them into a DuckDB connection from Python.
          Python-specific convenience layer. In a parallel ecosystem you
          could also have fledgling-rust, fledgling-go, etc. — each
          delivering the same SQL in a different language's package
          format.
          ↓ bundles
Layer 1: fledgling — the core SQL macro library. Language-agnostic by
          design: it's just SQL files that load into a DuckDB connection.
          Nothing in fledgling's core is Python-specific.
          ↓ uses
Layer 0: DuckDB extensions (sitting_duck, duck_block_utils, markdown,
          webbed, read_lines, ...) — native C++ extensions providing
          the primitives.
```

**The key correction to an earlier version of this doc**: fledgling is
language-agnostic at its core. fledgling-python is a Python-side
delivery mechanism that sits *above* fledgling, not beside it. Other
languages would have their own delivery packages (fledgling-rust,
fledgling-go) bundling the same SQL.

**The organizing rule**: if it's composition of primitives via SQL, it's
a fledgling macro (layer 1, language-agnostic). If it requires C++ to
be efficient or correct, it stays in a DuckDB extension (layer 0).
Python code that wraps a SQL macro for Python consumers belongs in
fledgling-python or pluckit (layer 2 or 3), not in fledgling itself.

## What stays where

### sitting_duck (layer 0, C++ extension, unchanged)

- `ast_select(source, selector)` — the tree-sitter-backed selector evaluator
- `ast_get_source(file, start, end)` — source text extraction from the AST
- `read_ast(source)` — the base AST iterator
- Any function that requires C++ for correctness or performance

### fledgling (layer 1, language-agnostic SQL macros — target for new SQL work)

- `ast_definitions`, `ast_class_members`, `ast_dead_code`, and other
  composed macros (many are already SQL-defined and can move here from
  wherever they currently live)
- **`pss_render(source, sheet)`** — the pss-as-macro experiment I started;
  should be written as a fledgling SQL macro from day one, not embedded
  in pluckit or lackpy
- **`ast_select_render(source, selector)`** — ast-select's custom
  rendering (selector as H1, qualified_name + file:range as H2,
  language-tagged code block) — currently done in Python inside lackpy's
  AstSelectInterpreter; should become a fledgling SQL macro
- Cross-cutting views that mix code, git, docs, and chat sources

All of these are pure SQL. No Python dependency. A non-Python consumer
(a Rust CLI tool, a shell script using `duckdb` on the command line)
could load fledgling's `.sql` files directly and use the macros.

### fledgling-python (layer 2, Python delivery wrapper)

- Pip package that bundles fledgling's `.sql` files as package resources
- Helper to load those macros into a `duckdb.DuckDBPyConnection`
- Python-side convenience imports (`from fledgling import register_macros`)
- Any Python helpers that call the macros and return Python-native
  results (pandas DataFrames, dataclasses, etc.)

The `fledgling-python` package is *how fledgling reaches Python consumers*,
not *where fledgling's logic lives*. When fledgling's core SQL changes,
fledgling-python just re-bundles it and releases.

### pluckit (layer 3, Python API — unchanged public surface)

- `Plucker` class, plugin architecture, Selection methods
- `Plucker.view()` — keeps its method signature but internally calls
  fledgling macros (via fledgling-python) instead of hand-building SQL
  strings
- `AstViewer` plugin — ditto
- `fledgling` plugin (future) — pluckit plugin that auto-loads
  fledgling's macros and exposes them via pluckit's chain API, becoming
  the natural Python-level bridge between the layers

### lackpy (layer 4, consumer — unchanged interpreter API)

- `PythonInterpreter`, `AstSelectInterpreter`, `PssInterpreter`, etc.
- These interpreters call pluckit, which calls fledgling-python, which
  calls fledgling's SQL macros, which call the DuckDB extensions.
  Lackpy doesn't care which layer owns what; it just needs the public
  pluckit API to keep working.

## What to avoid in the current branch

1. **Don't embed SQL strings in lackpy Python code.** If a SQL macro is
   needed, write it as a standalone SQL file under `docs/superpowers/sql-drafts/`
   (or similar) with a note that it should move to fledgling. Lackpy's
   Python code calls pluckit's public methods and doesn't construct SQL
   directly.

2. **Don't duplicate formatting logic that will later live in fledgling.**
   The AstSelectInterpreter's current markdown rendering (selector
   heading, qualified_name per match, dedented code block) is a good
   candidate for a fledgling macro `ast_select_render`. The Python
   implementation in the current branch is acceptable as a prototype but
   should be marked with a `# TODO: move to fledgling ast_select_render`
   comment so we remember.

3. **Don't add SQL macros to pluckit directly.** If you find yourself
   wanting to write SQL inside pluckit's Python code, that's a signal
   the logic should go to fledgling.

## The pss-render macro (draft, blocked)

When sitting_duck releases with `ast_select`, the pss-as-macro experiment
can be finished. The target shape is a fledgling macro:

```sql
-- Fledgling macro: render a selector sheet as markdown code regions.
-- Depends on: sitting_duck (ast_select, ast_get_source),
--             duck_block_utils (db_heading, db_code, db_assemble),
--             markdown (duck_blocks_to_md).
CREATE OR REPLACE MACRO pss_render(source, selector) AS (
    duck_blocks_to_md(
        (SELECT db_assemble(list(blocks))
         FROM (
             SELECT list_concat(
                 db_heading(1, file_path || ':' || start_line || '-' || end_line),
                 db_code(language, ast_get_source(file_path, start_line, end_line))
             ) AS blocks
             FROM ast_select(source, selector)
         ))
    )
);
```

(Grammar is not tested — ast_select isn't available yet. This is a
sketch to bookmark the approach, not production code.)

A multi-rule sheet version would accept a table of (selector, show_mode)
rows and iterate, using `db_blocks_merge` to combine the per-rule
outputs. That table could come from a parsing macro that uses regex to
split the sheet string into rules.

## The ast_select_render macro (draft, blocked)

```sql
-- Fledgling macro: render ast-select results in the heading-per-selector
-- format that lackpy's AstSelectInterpreter currently produces in Python.
-- Depends on: same as pss_render.
CREATE OR REPLACE MACRO ast_select_render(source, selector) AS (
    duck_blocks_to_md(
        list_concat(
            db_heading(1, '`' || selector || '`'),
            (SELECT list_concat(list(blocks))
             FROM (
                 SELECT list_concat(
                     db_heading(2, coalesce(qualified_name, name) ||
                                    ' — ' || file_path || ':' ||
                                    start_line || '-' || end_line),
                     db_code(language, ast_get_source(file_path,
                                                      start_line,
                                                      end_line))
                 ) AS blocks
                 FROM ast_select(source, selector)
             ))
        )
    )
);
```

When this macro exists, the AstSelectInterpreter can shrink from ~300
lines to ~50: parse the selector, call the macro, return the result.
The custom Python formatting goes away.

## The webbed-to-HTML angle (confirmed working)

Experiment: render duck_blocks to HTML via the `webbed` extension's
`duck_blocks_to_html` function, in parallel to `duck_blocks_to_md`.

Result: **both work for the core case**. The same blocks built via
`db_assemble([db_heading(...), db_paragraph(...), db_code(...)])` serialize
to both formats through a single function call. HTML escaping is
automatic. The HTML output includes semantic tags and language classes
on code blocks (`<pre><code class="language-python">`) which a browser
syntax highlighter can pick up for free.

What works:

- Headings (`<h1>`, `<h2>`, ...)
- Paragraphs (`<p>`)
- Code blocks with language tags (`<pre><code class="language-X">`)
- HTML escaping of content

What doesn't work yet:

- Lists (`db_list` + `db_list_item` composition produces empty `<ul></ul>`
  in HTML and spacing issues in markdown). Needs a different composition
  pattern than the flat list I tried. Not a blocker for the view use
  case — headings + paragraphs + code blocks cover it.

What this enables:

```sql
-- Fledgling macro that takes a format parameter
CREATE OR REPLACE MACRO pss_render(
    source,
    selector,
    format := 'markdown'
) AS (
    CASE format
        WHEN 'html'     THEN duck_blocks_to_html(<blocks>)
        WHEN 'markdown' THEN duck_blocks_to_md(<blocks>)
        ELSE                  duck_blocks_to_md(<blocks>)
    END
);
```

Same block construction, two render targets. Adding more render formats
(e.g. plain text via `db_blocks_to_text`, or pandoc AST via
`duck_blocks_to_pandoc_ast`) is a matter of adding a WHEN clause.

The lens/umwelt future work — an interactive HTML viewer for views — is
one macro parameter away once the blocks are being built correctly.

Both `duck_blocks_to_md` and `duck_blocks_to_html` are layer-0 extension
functions (markdown extension and webbed extension respectively). They
stay in their extensions. The fledgling layer composes them into
higher-level macros like `pss_render` that take a format parameter and
dispatch to the right serializer.

## Migration checklist (for when the reorg happens)

1. Release sitting_duck with `ast_select` exposed
2. Publish fledgling-python package with current fledgling's SQL macros
3. Write `pss_render`, `ast_select_render`, and any new macros in
   fledgling
4. Update pluckit's `AstViewer` plugin to call fledgling macros instead
   of building SQL strings in Python
5. Update lackpy's `AstSelectInterpreter` to call pluckit's updated
   viewer (which now delegates to fledgling under the hood) and drop
   the custom Python rendering
6. Delete duplicated SQL across repos; keep one copy in fledgling
7. Version-bump each package that changed public surface
