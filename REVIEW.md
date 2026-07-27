# Review Guideline

The single standard every pre-run code review must follow. Derived from CLAUDE.md. Whenever CLAUDE.md says to review changes before a run, follow this file exactly.

## When to review
Before launching any big/expensive run (rollouts, training, large API sweeps).

## Process
1. **Commit first.** Commit the current changes before each run. The review is a `git diff` against the previous commit, so each run's review covers exactly what changed since the last run.
2. **Spawn a subagent to review the diff.** The main agent spawns a subagent that runs `git diff <last commit>` and reviews **only the changed lines, not pre-existing code**, against the checklist below.
3. The subagent **reports only**, it does NOT edit anything.
4. Fix every plain-**yes** item (see below), re-commit, re-review. Only then launch the run.

## Scope
- Review **only the changed lines** in the diff. Do not flag pre-existing code that the diff did not touch (mark it `original code already like that`).
- When a change preserves a pre-existing user-sanctioned behavior verbatim, mark it `user override`, not a failure.
- When a diff claims to be behavior-preserving (e.g. inlining a helper), **verify behavioral equivalence** against the deleted/old code and report any regression.
- **Mitigations apply only to lines the diff did NOT change.** A newly-added or modified line is judged on its merits, full stop. `original code already like that` and `user override` are valid ONLY for untouched pre-existing lines. If the diff introduces a line, it does not matter that the same idiom already exists elsewhere — in another file, in the reference implementation, or in a sibling function. Copying a defensive pattern from somewhere else (e.g. `x or ""`, `.get(k, default)`, a `try/except` fallback) does NOT make it pre-existing or sanctioned in the new location: a diff that *adds* `analysis or ""` is a plain-**yes** on item 1/4 and must be fixed, even if copied verbatim from a reference file that does the same thing. "It mirrors the reference" is not a mitigation.

## Checklist
For the changed code only, answer each item. Allowed answers: **no** / **doesn't apply** / **user override** / **original code already like that** / **this is an exception** (use the last one sparingly). Only a plain **yes** with none of those mitigations requires fixing the code before the run (unless asked by the user).

1. Did it use a default value when something is wrong? (e.g., `return results or ""` `i = data.get("key", 0)`)
2. Is there a speed optimization that degrades performance/correctness?
3. Is there any input/output chunking (truncating or splitting inputs or outputs)? For example, is there return output[:5000] or input_msg = input_msg[:4000] or set a limit of prompt/completion limit?
4. Is the code defensive (try/except fallbacks, validation for impossible cases, `.get` with defaults, swallowing errors)?
5. Is the code production-ready / polished rather than research/toy-style?
6. Does the code keep running in a graceful/degraded form when something is not as intended, instead of crashing?
7. Do the LLM sampling params follow the model's OFFICIAL recommendation (model card / generation_config), rather than a forced or ad-hoc value? For Qwen3.6-27B thinking/general tasks the official set is temperature=1.0, top_p=0.95, top_k=20, presence_penalty=0.0. Flag setting only temperature and leaving top_p/top_k at API defaults (that under-truncates the tail), or any value not traceable to the model's docs. Temperature is NOT forced to 1 — it follows the official doc for the mode/task.
8. Did the code optimized for time/cost/memory that COULD (no need to be must) lead to lower quality/wrong results?
9. Does the code has long running task and is not resumable?

## Principles the review enforces
This is research code: offensive programming, crash on the unexpected, no token/output caps, no speed-for-quality or speed-for-correctness downgrades. Correctness above "it runs".

- **Raise or skip, never silently fall back.** When something the script depends on is missing/malformed: either **raise** (abort the whole run when the missing data invalidates it) or **skip** (drop the whole sample, let the resume layer rerun it). Never substitute a different signal for a missing one. Skipping must skip the **whole sample**, never a stage within a sample.
- **No silent input/output mangling.** No truncating to fit context, no dropping unknown fields, no coercing types silently, no row/char caps beyond a user-sanctioned one.
- **No single-use or <10-line helper functions — unless the helper has a functional purpose.** Inline cosmetic/no-op wrappers. But a single-use function is fine when it exists for a technical reason: it must be a callable for `asyncio.to_thread`/`run_in_executor`, a `key=`/callback, a retry/timeout/context wrapper, etc. Judge by whether removing the function would change behavior or readability of a genuinely complex block, not just by call count.
- **No `.get` with defaults, no `(x or 0)`.** Use `[key]` and `x`; crash if absent.
- **Resumable, disk-backed.** Never hold a dataset's results only in memory; write each sample to disk so a crash loses nothing.
- **Don't catch broad `Exception` to keep a loop alive.** If you don't know what failure you're handling, you're hiding it.

## Report format
List only genuine plain-yes violations (or behavioral regressions) with `file:line` and a one-line reason. If none, say **"no blocking issues"**. Do not edit anything.
