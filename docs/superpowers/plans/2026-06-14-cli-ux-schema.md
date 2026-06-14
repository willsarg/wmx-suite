# CLI UX Schema + Port Implementation Plan

> **For agentic workers:** implement task-by-task; steps use checkbox (`- [ ]`) syntax. TDD: write the failing test first, then the code.

**Goal:** Replace the ~201 ad-hoc `print()` calls in `cli.py` with **one shared rendering schema** so every command speaks the same visual language: glossed human-default output, a `--verbose` power-user layer, `next` blocks on success and `why`/`try` blocks on failure, TTY-aware color, and clean plain text when piped. Visual source of truth: `docs/mockups/cli-output-mockup.html` (approved: inline-gloss default + Normal/Verbose).

**Architecture:** A new `wmx_suite/ui.py` module owns *all* presentation. It exposes a `Console` (holds `color`/`verbose`/stream) plus a small set of layout **primitives** (`field`, `section`, `table`, `next_block`, `guidance`, `status_line`, `glyph`, `raw`) built on **semantic roles** (label/value/gloss/header/accent/good/warn/bad/metric/dim) rather than raw ANSI. Each command is refactored into **gather → render**: a data step (DB/system reads) returns a plain dict, and a pure render step turns `(Console, data)` into output. Render steps are what tests assert on (golden snapshots in plain mode). Color is on iff `stream.isatty() and not NO_COLOR and not --no-color`.

**Tech Stack:** Python (stdlib only — no new deps), pytest.

**Spec/visual reference:** `docs/mockups/cli-output-mockup.html`

**Conventions (all tasks):** `uv` only; tests hardware-free (no model load, no live memory probing, no prod DB — `monkeypatch.setattr(db, "DB_PATH", tmp_path/"suite.db")`); never weaken RULE #1; commit to `main`; trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Before declaring done: `uv run pytest -q` and `uv run python -m compileall -q wmx_suite tests`.

---

## The schema (what `ui.py` provides)

**Roles → ANSI (only emitted when `color` is true):**
| role | use | color |
|------|-----|-------|
| `label` | field labels | dim |
| `value` | field values | default/bold |
| `gloss` | inline explanations | dimmer |
| `header` | section titles, table headers | cyan |
| `accent` | commands a user can type | magenta |
| `good` / `warn` / `bad` | ✓ / ⚠ / ✗ status | green / yellow / red |
| `metric` | headroom & key numbers | cyan |
| `dim` | secondary text | dim |

**Primitives:**
- `field(label, value, gloss=None, *, indent=0, value_role="value")` → `"label : value   gloss"` (gloss only when not piping noise; always shown — gloss is the default per the approved design).
- `section(title)` → styled header line.
- `table(columns, rows)` → `columns=[(header, align, role)]`; auto-widths; styled header row; per-cell role overrides. The single source of all column alignment.
- `glyph(status)` → `✓`/`✗`/`⚠` in the right role color.
- `next_block(items)` → `next` header + aligned `command  why` rows (accent + gloss).
- `guidance(headline, why, tries)` → the failure pattern: bad headline, `why` lines, `try` options.
- `status_line(parts)` → the machine one-liner (device · free · models · calibrated).
- `raw(title, pairs|table)` → **no-op unless `console.verbose`**; renders the power-user appendix.

**Color/verbose policy (the testable contract):**
- Not a TTY (piped/redirected/captured) → zero ANSI, stable plain text. This is what makes goldens and agents work.
- `NO_COLOR` env set, or `--no-color` → no ANSI even on a TTY.
- `--verbose`/`-v` → `raw(...)` appendices appear; nothing is removed from the default view.

---

## Global flags (plumbing)

Add `--verbose/-v` and `--no-color` as **global** options available to every subcommand (argparse parent parser), parsed into one `Console` passed to each `cmd_*`.
- `run` is intercepted before argparse (`main()`), so strip `--verbose/-v/--no-color` out of its argv there and build its `Console` the same way.
- Front door: top-level no-args **or** bare `-h/--help` renders the grouped landing (set subparsers `required=False`; if `args.cmd is None` → landing). Keep argparse's per-subcommand `--help` working.

---

## File structure
- **New** `wmx_suite/ui.py` — Console + primitives + roles.
- **New** `tests/test_ui.py` — primitive behavior, color/verbose policy, alignment.
- **New** `wmx_suite/views.py` — pure render functions per command (`render_system`, `render_health`, `render_landing`, `render_scan`, `render_show`, `render_list`, `render_run_refusal`, error renderers). (Keeps `cli.py` thin: gather + call view.)
- **New** `tests/test_views.py` — golden/plain-mode assertions per command with fixture data.
- **Modify** `wmx_suite/cli.py` — global flags, front-door intercept, replace each command body with gather→view; route `run`'s `[run]` lines through views.

---

## Phase 1 — `ui.py` schema + global flags (foundation)

### Task 1.1: Console + roles + color policy
- [ ] Test (`tests/test_ui.py`): `Console(color=False)` emits no ANSI; `Console(color=True)` wraps with expected codes; `from_stream` is False when `NO_COLOR` set or `isatty()` False (use a fake stream). 
- [ ] Implement `Console`, `style(role, text)`, `Console.from_args(stream, no_color, verbose)`.

### Task 1.2: layout primitives
- [ ] Tests: `field` alignment + gloss; `table` auto-width + header + alignment + per-cell role; `glyph`; `next_block`; `guidance`; `status_line`; `raw` is empty when `verbose=False` and present when True.
- [ ] Implement all primitives (plain-text correctness is the contract; color is additive).

### Task 1.3: global flags + Console wiring
- [ ] Test: parser exposes `--verbose/--no-color` on subcommands; `main()` builds a `Console`; `run` argv has the flags stripped.
- [ ] Implement parent parser + `Console` construction; thread into `args` (e.g. `args.console`).

---

## Phase 2 — front door + system + health (highest traffic; worked example)

### Task 2.1: front door (`wmx-suite` no-args / `-h`)
- [ ] Test (`tests/test_views.py`): `render_landing(console, data)` with fixture machine data → contains the grouped sections, status line, NEW-HERE path; benchmarks expand only when `verbose`; plain mode has no ANSI.
- [ ] Implement `render_landing`; intercept no-args/help in `main()`; gather status (device, free, models-ready count, calibrated bool).

### Task 2.2: `system`
- [ ] Test: `render_system(console, data)` → budget cascade (RAM→wall→safe→wired→free) with gloss; `raw` block (wall source/bytes, margin source, calibration ISO, sampling) only under verbose; `next` block; plain mode stable.
- [ ] Refactor `cmd_system` to gather a dict and call `render_system`.

### Task 2.3: `health`
- [ ] Test: `render_health(console, data)` → glossed budget block + model table (loads at / spare room / safe context, ✓/✗) + legend + `next`; verbose adds per-model raw table (base/slope/cap) + margin/baseline; the `✗` row reads "over budget — won't load"; **RULE #1 unaffected** (render is display-only; gating stays in `launcher.predict`).
- [ ] Refactor `cmd_health` to gather (reuse existing `launcher.predict` calls) → dict → `render_health`.

---

## Phase 3 — scan + show + list (same pattern)
- [ ] **3.1 scan** — `render_scan`: ✓ rows, quantizable/fp16 translation + gloss, `registered N`, `next`; verbose appends `[cache_type, exact GB]`.
- [ ] **3.2 show** — `render_show`: "what it is" + "how its memory behaves" groups with translation; verbose appends the full raw architecture dump (the current 14 fields); `next`. Test the data/render split with fixture `models.describe` output (no cache read).
- [ ] **3.3 list** — `render_list`: human table (loads/safe/speed/fit + ⚠ tight); fit = good/ok/poor from R²; crash-context hidden in default; verbose appends raw-fit table (slope/R²/crash ctx/runs).

## Phase 4 — run messages + errors (safety-meets-UX)
- [ ] **4.1 run refusal/plan** — route the `[run]` lines through `render_run_*`: refusal uses `guidance(headline, why, tries)`; verbose appends the raw `live_base+model=total · slope · wall/threshold` math. **Do not change refusal logic** — only its presentation. Regression test: a refused config still exits non-zero / does not launch.
- [ ] **4.2 errors** — consistent `guidance` for: model-not-in-cache (with searched path under verbose), no-characterized-models, invalid `--margin`. Each ends with a `try` next step.

## Phase 5 — benchmarks (FULL treatment)
The 7 `benchmark-*` commands get the complete schema like the core commands:
- [ ] **5.1** Each benchmark's setup/result output routed through `ui` primitives (`section`, `table`, `status_line`, `glyph`), with glossed column headers and human-readable metric names (RTF, chars/sec, TTFA, peak GB, etc.).
- [ ] **5.2** A `next` block on each (e.g. point to related benchmark, or `web` for charts) and a `guidance` block for their pre-flight refusals / safeguard-triggered stops (these are RULE #1 messages — presentation only, gating untouched).
- [ ] **5.3** `--verbose` appendix per benchmark exposing the raw per-rung JSON-equivalent numbers (repeats, medians, per-trial spread) the workers emit.
- [ ] **5.4** gather→render split + golden tests for each, same as core commands.

## Phase 6 — consistency sweep + lock-in
- [ ] Golden snapshot test per command (plain mode, fixture data) to prevent future drift.
- [ ] Grep `cli.py` for stray `print(`/`\033[` that bypass `ui`; fold them in.
- [ ] Document `--verbose`/`--no-color`/`NO_COLOR` and the schema in `AGENTS.md` (conventions) + a short `docs/` note pointing at the mockup as the visual source of truth.
- [ ] `uv run pytest -q` + `uv run python -m compileall -q wmx_suite tests` green.

---

## Testing strategy
- **Pure render functions** + a `Console(color=False)` make every screen deterministic → golden assertions on exact plain-text output. No DB, no hardware, no MLX.
- Color tested separately (role→code) so goldens stay plain.
- Safety: RULE #1 logic is untouched; this is a presentation refactor. Refusal/gating tests from the existing suite must stay green; add display-only tests for the new messages.

## Risks / watch-items
- **`run` interception** — flags must be stripped without disturbing passthrough to `mlx_lm.generate`. Cover with a test.
- **Front-door `--help` override** — keep per-subcommand `--help` intact.
- **Scope creep on benchmarks** — gate behind the scope decision; the core 7 user commands are the priority.
- **No behavior change to safety** — reviewers should confirm only presentation moved.
