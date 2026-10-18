# STE usability eval: findings log

Findings in the order they were found. Each line says what was measured.

1. 2026-10-18, while writing references (Claude Opus 5.5): the natural idiom
   `for p in sorted(Path("src").glob("*.cpp")):` fails `style.io-position`
   ("I/O call .glob() must not be an argument"). It hit 4 of 30 references
   (patch-03, scan-01, scan-05, def-01) plus 1 with `iterdir()` (scan-02).
   References were rewritten as `paths = Path("src").glob(...)` /
   `for p in sorted(paths):`. Teague's post-Plan-1 ruling (design section 8)
   already relaxes this; the eval scores the language as built at 3bfb23f.
2. Same session: two more references failed. `total += len(p.read_text().splitlines())`
   fails `style.io-position` (an I/O call in the receiver chain of an argument), and
   `print(len(load_rows(text)))` fails `style.call-nesting` (depth 3). So 7 of 30
   first-draft references by Claude broke a shape rule, and all 7 were
   `io-position` or `call-nesting`. That is the pair Teague relaxed after Plan 1.
3. Same session, scoring haiku: 2 of haiku's 30 programs failed the strict
   check only because Claude's task text was underspecified. scan-05 omitted
   the final newline of report.txt, and def-01 printed `a.cpp: 2` where the
   reference prints `a.cpp 2`; the task never fixed either detail. The scorer
   now reports `correct` (byte-exact) and `correct_loose` (same tokens,
   ignoring `:`, `,` and whitespace). Both are recorded; the task text is not
   changed mid-run, so both models see identical prompts.
4. Same session, Qwen3-30B-A3B-Instruct round 1 (local Tiiny device; model
   inferred from `ttt`'s default route, because the device reports only
   `model.q8_0.gguf`): 8/30 valid, 26/30 correct by the loose check, so 14
   programs were correct but invalid. Its Python habits override the guide in
   context. Violations across the 30 programs: `except Exception` 11,
   `exit()` 6, `raise` 4 ("fail if ..." became `raise RuntimeError` instead of
   the guide's `assert`), `open()` 2, I/O call as an argument 5. Haiku, given
   the same prompt, reached for none of these.
5. Method correction, same session. The first local chain's results are
   quarantined in runs/2026-10-18/_unverified/ and are not scored.
   - Qwen3-30B rounds 1 and 2 went through `ttt`'s `default` route while
     gpt-oss-20b was also loaded, and the device reports no model name. So
     their model is unverified.
   - Qwen3.6 round 1 overlapped Claude's own smoke tests. Those tests requested
     Qwen3-30B on the same device, and the device log shows the two models
     swapping roughly every 30 s (22:13:50–22:15:30 UTC). Claude first misread
     this as another client using the device. It was Claude's own test.
   The rerun (chain-explicit.sh) uses explicit model routes only. After every
   call, run_local.py records any device-log launch of another chat model in
   `<id>.SWAP.txt`.
6. Same session. The pre-prompt (`format_preprompt()`, committed at 827aca0,
   before any of its results existed) gave haiku 30/30 valid in round 1,
   against 24/30 with the guide alone. By the loose check, 30/30 were correct
   in both cases. The pre-prompt removed the need for a repair round. Haiku's
   pre-prompt r2 holds only copies of r1, because no program needed repair.
7. Same session, Qwen3-30B-A3B-Instruct on its explicit route (no swaps
   recorded). With the guide alone, 12/30 programs were valid in round 1 and
   11 were valid and correct (loose). With the pre-prompt, 27/30 were valid
   and 25 valid and correct. A repair round with the guide alone raised
   validity (12 → 17) but lowered loose correctness (27 → 23): fixing the
   flagged constructs broke working code. With the pre-prompt, repair moved
   valid-and-correct from 25 to 26.
8. Harness defects found while scoring finding 7. Both are fixed, and every
   program is now re-derived from its saved reply before scoring (reextract.py).
   - extract() needed a closing fence. One reply with an opening fence and no
     close kept the fence line in the program.
   - A transport failure (HTTP 503) was saved as the program. A stream that
     stopped mid-program (scan-05, 271 bytes) was too.
   run_local.py now retries both up to 3 times and records `finish_reason`
   in <id>.meta.json. One reply (qwen3-30b r2 patch-05, 7.6 KB) is a real
   model failure, not a harness one: Qwen3-30B repeated one comment line
   until it hit the token limit. The two transport failures in qwen3-30b r2
   are re-asked once the Qwen3.6 runs no longer hold the device.
9. Final scores, 2026-10-18, from runs/2026-10-18/results.json. Every reply was
   re-extracted with the fixed extract(). No swaps and no empty replies are
   recorded. The two qwen3-30b r2 transport failures were re-asked; the
   originals are kept as *.transport-failure.txt. `both~` = valid and correct
   by the loose check. The last column counts programs whose only violations
   are style.io-position or style.call-nesting (ruled to relax; Plan 2 Task 1)
   and which are already correct: a static estimate, not a run.

   | writer | prompt | r1 valid | r1 both~ | r2 both~ | r1 both~ after relax |
   |---|---|---|---|---|---|
   | haiku | guide | 24 | 24 | 30 | 27 |
   | haiku | pre | 30 | 30 | 30 | 30 |
   | Qwen3-30B-A3B-Instruct | guide | 12 | 11 | 16 | 16 |
   | Qwen3-30B-A3B-Instruct | pre | 27 | 25 | 26 | 26 |
   | Qwen3.6-35B-A3B (thinking off) | guide | 14 | 13 | 16 | 17 |
   | Qwen3.6-35B-A3B (thinking off) | pre | 24 | 22 | 25 | 28 |

   - The pre-prompt is the largest lever for every writer: Qwen3-30B r1 goes
     from 11 to 25, and Qwen3.6 from 13 to 22.
   - Repair rounds help little. For the local models they trade correctness
     for validity, so the pre-prompt beats the repair loop.
   - Qwen3.6 without the pre-prompt breaks style.def-length (6 programs):
     it writes long functions.
