"""Extract distinct heredoc Python bodies from Claude Code history into a parquet snapshot.

SQL is Appendix A of docs/superpowers/specs/2026-10-18-lackpy-ste-design.md.
Usage: python scripts/corpus/extract.py [--snapshot PATH] [--calls PARQUET] [--force]

Claude Code prunes old session files, so the live history yields fewer calls than it did on
2026-10-18. --calls rebuilds the snapshot from a saved copy of the `calls` table.
"""
import argparse
import sys
from pathlib import Path

import duckdb

DEFAULT = Path("~/.local/share/lackpy/corpus/heredocs-2026-10-18.parquet").expanduser()
HISTORY = str(Path("~/.claude/projects/*/*.jsonl").expanduser())

CALLS = f"""
CREATE TABLE calls AS
WITH m AS (
  SELECT filename, unnest(json_extract(message, '$.content[*]')) AS c
  FROM read_json('{HISTORY}', format='newline_delimited',
       columns={{type:'VARCHAR', message:'JSON'}}, filename=true, ignore_errors=true)
  WHERE type='assistant')
SELECT filename, json_extract_string(c,'$.input.command') AS cmd,
       json_extract_string(c,'$.input.description') AS descr
FROM m
WHERE json_extract_string(c,'$.type')='tool_use'
  AND json_extract_string(c,'$.name')='Bash'
  AND regexp_matches(json_extract_string(c,'$.input.command'), 'python3? (- )?<<')
"""

BODIES = r"""
CREATE TABLE bodies AS
SELECT *, regexp_extract(cmd,
  '(?s)python3? (?:- )?<<-?[''"]?(PY|EOF|PYEOF|PYTHON|PYCODE|END)[''"]?\n(.*?)\n(PY|EOF|PYEOF|PYTHON|PYCODE|END)\b', 2) AS body
FROM calls
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=DEFAULT)
    ap.add_argument("--calls", type=Path,
                    help="build from a saved calls parquet (filename, cmd, descr) instead of the live history")
    ap.add_argument("--force", action="store_true", help="overwrite an existing snapshot")
    args = ap.parse_args()
    out = args.snapshot.expanduser()
    if out.exists() and not args.force:
        print(f"snapshot exists: {out} (use --force to overwrite)", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    if args.calls:
        con.execute(f"CREATE TABLE calls AS SELECT * FROM read_parquet('{args.calls.expanduser()}')")
    else:
        con.execute(CALLS)
    con.execute(BODIES)
    con.execute(
        "COPY (SELECT any_value(filename) AS session_file, any_value(descr) AS description, body "
        f"FROM bodies WHERE body <> '' GROUP BY body ORDER BY body) TO '{out}' (FORMAT parquet)"
    )
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out}')").fetchone()[0]
    print(f"wrote {out}: {n} distinct non-empty bodies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
