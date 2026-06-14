# ModernBERT Embeddings Memory Benchmark — Design Spec

**Issue:** #14 — Benchmark ModernBERT (Embeddings) batch memory scaling
**Date:** 2026-06-14
**Status:** Approved design, pending implementation plan

## 1. Goal

Add a new **embeddings** benchmark category to wmx-suite that maps the **OS-wired
memory surface** of ModernBERT across a 2D **batch × sequence-length** grid, plus
throughput and latency — following the existing benchmark conventions and the project's
RULE #1: **never run a configuration predicted to breach the crash wall.**

This is the suite's first *encoder-only* (non-causal) model category. It deliberately does
**not** reuse the causal-LM probe path (`characterize`, `estimated_slope_gb_per_k`,
`is_causal`), because those are specific to autoregressive KV-cache growth and would refuse
a non-causal model.

## 2. Runtime & dependency

- **Library:** `mlx-embeddings>=0.1.0` (Blaizzy/mlx-embeddings; floor pin to match the
  repo's `>=` convention). Native ModernBERT support, pure-MLX inference (no torch). Pulls
  `transformers>=5.0` and `mlx-vlm>=0.4.0` as transitive deps — accepted as the cost of a
  **core dependency** (added to `pyproject.toml [project] dependencies` alongside
  `kokoro-mlx`). Verify `uv sync` resolves cleanly against existing pins (`mlx-lm>=0.31`,
  `mlx 0.31.2`) after adding.
- **Default model:** `mlx-community/nomicai-modernbert-embed-base-bf16` — an actual
  embedding model (`feature-extraction`), bf16 (matches the "measure at production
  settings" rule), the most-downloaded ModernBERT in `mlx-community`.
- **Verified model facts:** 22 hidden layers (8 global / 14 local, sliding window 128),
  hidden_size 768, 12 heads (MHA, head_dim 64), vocab 50368, `max_position_embeddings`
  8192. Attention is dense `(batch, 1, seq, seq)` — **no unpadding/flash**, so peak
  activation memory genuinely scales with `batch × seq²`.

## 3. Architecture overview

Three pieces, mirroring the **LLM probe** layering (`probe.py` + `probe_worker.py` + CLI),
NOT the single-process Kokoro layering:

```
cli.py  benchmark-embeddings
   │  resolves margin + mlx_version, db.start_embeddings_run(), prints grid table
   ▼
embeddings_probe.py   (NEW orchestrator — analog of probe.py)
   │  per-row ramp, predictive-skip gate, incremental fit, DB persistence
   │  spawns ONE isolated subprocess per (batch, seq) cell
   ▼
probe_worker_embeddings.py   (NEW worker — analog of probe_worker.py)
      measures ONE cell, prints one JSON line, exits
```

### Why per-cell subprocess isolation (the critical correction)

The Kokoro workers sweep all rungs in **one** long-lived process; wired residue from a
large cell would inflate the next cell's reading and could accumulate toward the wall.
The LLM probe instead spawns a **fresh subprocess per measurement** and samples a settled
baseline between them. Embeddings cells allocate large `seq²` attention buffers, so we
adopt the isolated-subprocess model: **one subprocess per grid cell**, `repeats` forward
passes *inside* that process (model loaded once per cell), `sample_settled_baseline()`
called by the orchestrator between cells.

## 4. The worker — `wmx_suite/probe_worker_embeddings.py`

Measures a **single** `(batch, seq)` cell and prints exactly one terminal JSON line.

**Args:** `--model`, `--batch` (int), `--seq` (int), `--repeats` (int, default 3),
`--margin` (float).

**Import convention (required for testability):** the worker does
`import mlx.core as mx`, `import mlx_embeddings`, and `from . import system` at module
top — NOT `from mlx_embeddings import load` / `from .system import wired_gb`. Tests then
patch `mlx_embeddings.load`, `mx.*`, and `system.wired_gb`/`system.read_limits` on those
module objects, which are shared with the worker. (A `from x import y` binding would dodge
the patch.) The `mlx_embeddings` import is wrapped so `ImportError` is reportable.

**Flow:**
1. **Pre-flight gate (defense in depth):** `limits = system.read_limits()`;
   `threshold = limits.safe_threshold_gb(margin)`; abort (print
   `{"status":"error","note":"Pre-flight aborted: ..."}`, `sys.exit(0)`, model **never
   loaded**) if `limits.wired_now_gb + MODEL_WEIGHT_EST_GB >= threshold`. The
   `MODEL_WEIGHT_EST_GB` headroom term (a small constant, e.g. 0.6 GB, covering a
   ModernBERT-base bf16 weight load) ensures the worker never loads weights it can't
   afford even if the orchestrator gate were wrong. This is the kernel-panic backstop:
   subprocess isolation prevents *residue accumulation* but does NOT prevent a single
   child's allocation from panicking the host, so both gates must hold.
2. **Load** model+tokenizer via `mlx_embeddings.load(args.model)`. On `ImportError`, print
   `{"status":"error","note":"... add mlx-embeddings ..."}` and `sys.exit(1)`.
   (Tokenizer is loaded but unused — synthetic inputs are built directly.)
3. **Build synthetic input** of exact shape:
   `input_ids = mx.zeros((batch, seq), dtype=mx.int32)` (id 0 is in-vocab),
   `attention_mask = mx.ones((batch, seq), dtype=embed_dtype)` where
   `embed_dtype = model.model.embeddings.tok_embeddings.weight.dtype` (bf16 for the
   default model). Passing `attention_mask=None` is equivalent (the model builds the same
   all-ones mask); we pass it explicitly for clarity.
4. **Background OS-wired sampler (the safety-critical correction):** before the timed work,
   start a daemon thread that records `max(system.wired_gb())` every ~50 ms (identical to
   the LLM `probe_worker.py` sampler). MLX may free per-layer attention buffers mid-forward,
   so a single post-eval read can miss the true high-water — and the fit that gates *larger*
   cells is built from these readings, so an undercount propagates into an under-prediction.
   The sampler's high-water is the authoritative `os_wired_gb`.
5. **Warmup:** one forward pass `model(input_ids, attention_mask=attention_mask)`, then
   `mx.eval(output.last_hidden_state)` (NOT `mx.eval(output)` — the return is a
   `BaseModelOutput` dataclass; eval the array field to force the graph) and
   `mx.clear_cache()`.
6. **Measure `repeats` times:** for each repeat — `mx.clear_cache()`,
   `mx.reset_peak_memory()`, `t0=perf_counter()`, forward, `mx.eval(output.last_hidden_state)`,
   record `compute_time`, and `mx.get_peak_memory()/1e9`. Stop the sampler after the last
   repeat.
7. **Aggregate (conservative for memory, central for timing):**
   - `os_wired_gb` = the sampler's high-water **max** over the whole measurement window.
   - `peak_gb` = **max** of `mx.get_peak_memory()/1e9` across repeats.
   - `compute_time` = **median** across repeats; `throughput_tps = batch*seq/compute_time`;
     `latency_ms = compute_time*1000`.
   Max (not median) for the memory metrics keeps the gate-feeding data on the safe side.
8. **Emit** one line:
   ```json
   {"status":"rung_done","batch":<int>,"seq":<int>,
    "os_wired_gb":<f>,"peak_gb":<f>,"compute_time":<f>,
    "throughput_tps":<f>,"latency_ms":<f>}
   ```

**Status vocabulary:** `rung_done`, `error`. (No `safeguard_triggered`/`row_skipped` in the
worker — the predictive skip is the orchestrator's job, since it owns the cross-cell view.)

## 5. The orchestrator — `wmx_suite/embeddings_probe.py` (NEW)

Analog of `probe.py`. Owns the per-row ramp, the predictive-skip safety gate, the
incremental fit, and DB persistence. Hardware-free-testable (subprocess spawn is injected /
mockable).

**Entry point:**
```python
def sweep(con, run_id, model, batches, seqs, repeats, margin_gb=None, *, on_event=None) -> dict
```
- `run_id` is created by the CLI via `db.start_embeddings_run` and passed in; the
  orchestrator persists each cell's measurement under it. (The orchestrator does NOT call
  `db.connect()` — the CLI owns the connection, unlike `probe.characterize`.)
- `batches` default `[1, 2, 4, 8, 16, 32]`; `seqs` default
  `[128, 256, 512, 1024, 2048, 4096, 8192]` (both overridable from the CLI).
- **`on_event` callback (a new convention this spec introduces — probe.py uses an internal
  `verbose` closure, not an injected callback).** Called once per orchestrator event with a
  dict: `{"event": "cell_done", "batch", "seq", "os_wired_gb", "peak_gb",
  "throughput_tps", "latency_ms"}`, `{"event": "row_skipped", "batch", "seq",
  "predicted_gb"}`, or `{"event": "preflight_abort", "note"}`. The CLI passes a callback
  that renders the grid table; tests pass a collector. Defaults to a no-op.
- Returns a summary dict (`model`, `n_cells_measured`, `n_cells_skipped`, `run_id`).

**Memory model — real-fit-governed gate, analytic prior for cold start only.**

This mirrors `probe.py`: the gate trusts a fit built from **real measured high-water peaks**
(from the worker's background sampler), extrapolated **one rung at a time** with a safety
factor; the analytic prior only seeds the gate before enough real data exists. (An earlier
draft used `max(analytic, fitted)` permanently — rejected, because the conservative
all-layers-global prior predicts ~35 GB at seq 8192 and would *permanently* skip the entire
high-seq region even where real memory is safe, defeating the benchmark.)

Predicted absolute OS-wired peak for a candidate cell:
```
predicted = live_base + model_base + a*(batch*seq) + b*(batch*seq^2)
```
- `live_base` = `system.sample_settled_baseline()` sampled fresh **between cells** (after the
  previous subprocess exits and the IOGPU floor settles).
- `model_base` = weights-resident floor. **Seeded non-zero** from a weight-size estimate
  (mirroring `probe.estimate_base_gb`: `weights_bytes * RESIDENT_FACTOR + overhead`) so the
  cold-start gate and the pre-flight account for the cost of *loading* the model — never 0.
  Replaced by the measured smallest-cell delta once available.
- `a`, `b` = linear and quadratic (`seq²`) coefficients.

**Gate logic — two distinct constants, by phase (this resolves the over-estimate
contradiction):**
- **Cold start** (`< MIN_FIT_POINTS`, default 3, measured cells): gate with the
  **sum-over-all-layers OVER-estimate** — a safe upper bound when we have no data:
  `A_COLD = num_layers * hidden_size * 2 / 1e9` (= 22*768*2/1e9 ≈ 3.38e-5 GB per
  `batch*seq`) and `B_COLD = num_layers * num_heads * 2 / 1e9` (= 22*12*2/1e9 ≈ 5.28e-7 GB
  per `batch*seq²`). Because the sweep ramps seq from the smallest value, cold start only
  ever gates the first one or two tiny cells (e.g. `(1,128)` → `B_COLD*16384 ≈ 0.009 GB`),
  so the over-estimate is harmless here; it is NOT used as a permanent floor.
- **Fitted** (`>= MIN_FIT_POINTS`): least-squares fit (through the origin, two features) of
  `delta = os_wired_gb - cell_baseline` on `(batch*seq, batch*seq²)` across all measured
  cells → `(a_fit, b_fit)`. The fit — built from **real sampler high-water data** — governs.
  Clamp each coefficient to a **one-layer PHYSICAL FLOOR** (a true lower bound: at least one
  global layer's attention matrix and one residual-stream copy must be resident at peak),
  NOT the cold over-estimate:
  `A_FLOOR = hidden_size * 2 / 1e9` (≈ 1.54e-6), `B_FLOOR = num_heads * 2 / 1e9` (≈ 2.4e-8).
  `a = max(A_FLOOR, max(0.0, a_fit))`, `b = max(B_FLOOR, max(0.0, b_fit))`. The floor + the
  `max(0,·)` clamp stop a noisy/near-zero fit from under-predicting below the physical
  minimum, while letting the fit climb to the real (possibly multi-layer) value. (Using the
  cold sum-over-layers as the fitted floor was rejected: `B_COLD` predicts ~35 GB at
  seq 8192 and would permanently skip the high-seq region.)
- Predictions carry a safety factor `PRED_SAFETY = 1.25` on the model terms, atop the
  always-present `wall − margin` threshold. The per-row ramp refines the quadratic `b` from
  progressively larger *safe* cells before extrapolating one doubling-step further, so the
  fit's `b` reflects real `seq²` growth rather than being guessed from tiny cells.
- Each prediction extrapolates only to the **next** rung from all prior data (seqs double,
  so the `seq²` term quadruples per step); the row stops at the first predicted breach, so
  we never extrapolate far. Residual risk (a regime change between the last safe rung and
  the skipped one) is bounded by `PRED_SAFETY` + the 2 GB margin — the same accepted risk
  model as `probe.characterize`.

**Traversal — per-row ramp + predictive skip (the approved strategy):**
```
preflight: live = sample_settled_baseline()
           if live + model_base(seed) >= threshold: emit preflight_abort; return
for batch in batches (ascending):
    for seq in seqs (ascending):
        if (batch, seq) known-unsafe by monotonic pruning: emit row_skipped; continue/break
        live_base = sample_settled_baseline()
        predicted = live_base + model_base + PRED_SAFETY*(a*batch*seq + b*batch*seq^2)
        if predicted >= threshold:
            emit row_skipped(batch, seq, predicted); record smallest-unsafe-seq[batch]; break
        result = run_cell_subprocess(model, batch, seq, repeats, margin)   # parent gates BEFORE spawn
        if result.status == "error": emit + abort sweep (exit nonzero)
        else: persist measurement; emit cell_done; refit
```
- **Skip semantics:** when a cell is predicted to breach, the rest of that row (larger seqs)
  is skipped; continue with the next batch. Memory is monotonic in batch at fixed seq
  (dense attention makes `batch*seq²` an exact factorization here), so once `(batch, seq*)`
  is unsafe, `(batch', seq*)` for `batch' > batch` is too — track the smallest unsafe seq
  per batch and prune those cells in later rows without spawning. Pruning only avoids wasted
  work; safety comes from the per-cell gate.
- **The gate runs in the PARENT (orchestrator) before any subprocess is spawned** — this is
  the primary safety line. The worker's own pre-flight (Section 4.1) is the backstop.

**Cell subprocess invocation:** `[sys.executable, "-m",
"wmx_suite.probe_worker_embeddings", "--model", m, "--batch", b, "--seq", s, "--repeats",
r, "--margin", margin]` via `subprocess.run(...)` (capture stdout), parse the single JSON
line. (Per-cell, `_stream_worker`'s line-streaming is unnecessary; a single
`subprocess.run` + parse mirrors `probe._run_worker`.)

## 6. Database — `wmx_suite/db.py`

Two new tables appended to `SCHEMA` (kokoro-family pattern, `ON DELETE CASCADE`):

```sql
CREATE TABLE IF NOT EXISTS embeddings_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,
    mlx_version TEXT,
    created_at  TEXT
);
CREATE TABLE IF NOT EXISTS embeddings_measurements (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL,
    batch_size     INTEGER NOT NULL,
    seq_len        INTEGER NOT NULL,
    os_wired_gb    REAL,
    peak_gb        REAL,
    throughput_tps REAL,
    latency_ms     REAL,
    FOREIGN KEY (run_id) REFERENCES embeddings_runs(id) ON DELETE CASCADE
);
```

Functions (exact kokoro signatures/pattern):
- `start_embeddings_run(con, model_id, mlx_version) -> int`
- `add_embeddings_measurement(con, run_id, batch_size, seq_len, os_wired_gb, peak_gb, throughput_tps, latency_ms) -> None`
- `get_all_embeddings_runs(con) -> list[dict]`
- `get_embeddings_measurements(con, run_id) -> list[dict]`
- `get_latest_embeddings_run(con) -> dict | None`

No migration block needed (new tables created by `executescript`); `connect()`'s additive
migration is only for columns added to pre-existing tables.

## 7. CLI — `wmx_suite/cli.py`

- Subparser `benchmark-embeddings` → `cmd_benchmark_embeddings`. Args: `--model`
  (default `mlx-community/nomicai-modernbert-embed-base-bf16`), `--batches`
  (default `1,2,4,8,16,32`), `--seqs` (default `128,256,512,1024,2048,4096,8192`),
  `--repeats` (default 3), `--margin` (default None → `_configured_margin`).
- `cmd_benchmark_embeddings`: resolve margin; `mlx_version = mx.__version__`;
  `run_id = db.start_embeddings_run(con, args.model, mlx_version)`; call
  `embeddings_probe.sweep(con, run_id, ..., on_event=render)` where `render` consumes the
  event dicts (Section 5) to print a **grid table** (rows = batch, columns = seq; each cell
  shows `os_wired_gb` / `throughput_tps`; `row_skipped` cells marked `SKIP`; not-reached
  cells blank). On `preflight_abort` or a worker error event, print the note and
  `sys.exit(1)`.

## 8. Web — `wmx_suite/web/`

- `app.py`: `@app.route("/embeddings")` → `embeddings_dashboard()` (lists
  `get_all_embeddings_runs`, decorated with summary stats: max safe seq per batch, peak
  throughput); `@app.route("/embeddings/run/<int:run_id>")` → `embeddings_run_detail()`
  (renders the grid from `get_embeddings_measurements`).
- Templates `embeddings_dashboard.html` + `embeddings_run.html`, extending `base.html`.
- Add one `<li>` nav entry in `base.html` (static nav list) linking
  `url_for('embeddings_dashboard')`.

## 9. Tests — `tests/test_embeddings_benchmark.py` (hardware-free)

All tests avoid real model loads, live memory probing, and the production DB
(`monkeypatch.setattr(db, "DB_PATH", tmp_path/"suite.db")`).

**Patch-target rule (important):** because the worker imports modules (`import
mlx_embeddings`, `import mlx.core as mx`, `from . import system`), patch attributes **on
those shared module objects**, NOT on locally-bound names:
- `monkeypatch.setattr(mlx_embeddings, "load", fake_load)`
- `monkeypatch.setattr(mx, "get_peak_memory", ...)` (and `reset_peak_memory`,
  `clear_cache`) — `mx is mlx.core`, so this is visible to the worker.
- `monkeypatch.setattr(system, "wired_gb", ...)`, `setattr(system, "read_limits", ...)`,
  `setattr(system, "sample_settled_baseline", ...)`.
A `from mlx_embeddings import load` style in the worker would make `load` un-patchable —
the worker MUST keep module-level imports (Section 4).

1. **DB lifecycle:** `start_embeddings_run` / `add_embeddings_measurement` /
   `get_all_embeddings_runs` / `get_embeddings_measurements` / `get_latest_embeddings_run`
   round-trip, including FK cascade delete.
2. **Worker — happy path:** patch as above; `fake_load` returns a mock model whose
   `__call__` returns an object with a `.last_hidden_state` mx.array and whose
   `model.embeddings.tok_embeddings.weight.dtype` is set; run
   `probe_worker_embeddings.main()` with patched argv; assert the `rung_done` JSON fields
   and `throughput_tps`/`latency_ms` math, and that `os_wired_gb`/`peak_gb` are the **max**
   (not median) of the patched readings.
3. **Worker — pre-flight refusal (RULE #1 guard):** patch `read_limits` so
   `wired_now_gb + MODEL_WEIGHT_EST_GB >= safe_threshold_gb()`; run `main()`; assert
   `mlx_embeddings.load` was **never called**, an `error` line was emitted, exit 0.
   Mutation check: removing the `+ MODEL_WEIGHT_EST_GB` headroom must not make a
   marginal-pressure case wrongly pass (include a case that is safe without the term but
   unsafe with it, asserting the abort).
4. **Orchestrator — predictive skip (RULE #1 guard):** inject a fake cell-runner; feed
   measured points (≥ MIN_FIT_POINTS) that make the fit predict the next cell `>= threshold`;
   assert the cell-runner is **not invoked** for the predicted-unsafe cell, a `row_skipped`
   event fires, and the sweep advances to the next batch. Mutation check: removing the gate
   must make this test fail.
5. **Orchestrator — cold-start gate uses non-zero model_base:** with zero measured cells and
   a high `live_base`, assert the first cell is gated using the non-zero weight seed (a
   `model_base=0` implementation would wrongly spawn it). Guards blocker #1.
6. **Orchestrator — monotonic pruning:** once `(batch, seq*)` is unsafe, assert
   `(batch', seq*)` with `batch' > batch` is not spawned.
7. **CLI — parse → DB:** drive `cmd_benchmark_embeddings` with the orchestrator mocked to
   emit canned `cell_done`/`row_skipped` events; assert the grid table prints and
   measurements land in the DB.

Run `uv run pytest -q` and `uv run python -m compileall -q wmx_suite tests` before
claiming completion.

## 10. Out of scope (YAGNI)

- No solving for a single "safe ceiling" scalar (the deliverable is the surface + the safe
  frontier implied by skipped cells).
- No quantized-model sweep matrix (default bf16 only; `--model` allows manual override).
- No analytic flash-attention modeling — source confirms dense attention; the conservative
  prior + incremental fit cover it.
- No Whisper (#12); that reuses this category's scaffolding in a follow-up.

## 11. Key risks resolved

Verified by two rounds of independent sub-agent audits (assumptions, then this document):

- **mlx-embeddings API + direct `(batch, seq)` control** — verified against installed
  source: `model(input_ids, attention_mask=...)`, `mx.eval(output.last_hidden_state)`,
  embed dtype via `model.model.embeddings.tok_embeddings.weight.dtype`.
- **Subprocess isolation** — corrected to per-cell (LLM-probe pattern), not single-process
  (Kokoro). Isolation prevents residue accumulation but NOT a single child's kernel panic,
  so the parent gate is primary and the worker pre-flight is the backstop.
- **`model_base` cold-start prior** — made **non-zero** (weight-size estimate), used from
  cell 0 and in pre-flight, so the gate accounts for model-load cost. (Audit blocker #1.)
- **Transient-peak measurement** — worker uses a 50 ms **background OS-wired sampler**
  (high-water), not a single post-eval read, so the gate-feeding fit isn't built on
  undercounted data. Memory metrics aggregated as **max** across repeats. (Audit blocker.)
- **Gate governance** — fit from real measured peaks governs (extrapolated one rung at a
  time × `PRED_SAFETY`); the conservative analytic prior is **cold-start only**, with an
  `a/b = max(analytic, max(0,fit))` clamp. Avoids the self-defeating permanent
  `max(analytic,fitted)` that would skip the whole high-seq region. (Audit blocker #3.)
- **Encoder memory model** — dense `batch×seq²` confirmed from source (no unpadding/flash).
- **Non-causal models** bypass `is_causal`/`characterize` — dedicated worker + orchestrator
  never touch the causal path.
- **Test patch targets** — module-level imports + patch on shared module objects, so mocks
  actually intercept (`from x import y` style explicitly forbidden in the worker).
