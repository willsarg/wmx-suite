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

- **Library:** `mlx-embeddings==0.1.0` (Blaizzy/mlx-embeddings). Native ModernBERT support,
  pure-MLX inference (no torch). Pulls `transformers>=5.0` and `mlx-vlm>=0.4.0` as
  transitive deps — accepted as the cost of a **core dependency** (added to
  `pyproject.toml [project] dependencies` alongside `kokoro-mlx`).
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

**Flow:**
1. **Pre-flight gate (defense in depth):** `limits = read_limits()`;
   `threshold = limits.safe_threshold_gb(margin)`; if `limits.wired_now_gb >= threshold`,
   print `{"status":"error","note":"Pre-flight aborted: ..."}` and `sys.exit(0)`. The
   model is **never loaded** in this branch.
2. **Load** model+tokenizer via `mlx_embeddings.load(args.model)`. On `ImportError`, print
   `{"status":"error","note":"... pip install '.[...]' / mlx-embeddings ..."}` and exit 1.
   (Tokenizer is loaded but unused — synthetic inputs are built directly.)
3. **Build synthetic input** of exact shape:
   `input_ids = mx.zeros((batch, seq), dtype=mx.int32)`,
   `attention_mask = mx.ones((batch, seq), dtype=<model embed dtype, bf16>)`.
4. **Warmup:** one forward pass `model(input_ids, attention_mask=attention_mask)`, then
   `mx.eval(...)` the output and `mx.clear_cache()`. (No JSON emitted for warmup at the
   cell level; warmup status is an orchestrator concern.)
5. **Measure `repeats` times:** for each repeat — `mx.clear_cache()`,
   `mx.reset_peak_memory()`, time the forward pass (`mx.eval` to force completion),
   record `compute_time`, `mx.get_peak_memory()/1e9` as `peak_gb`, and `wired_gb()` as
   `os_wired_gb`. Use the **median** across repeats for each metric.
6. **Emit** one line:
   ```json
   {"status":"rung_done","batch":<int>,"seq":<int>,
    "os_wired_gb":<f>,"peak_gb":<f>,"compute_time":<f>,
    "throughput_tps":<f>,"latency_ms":<f>}
   ```
   where `throughput_tps = batch*seq/compute_time`, `latency_ms = compute_time*1000`.

**Status vocabulary:** `rung_done`, `error`. (No `safeguard_triggered`/`row_skipped` in the
worker — the predictive skip is the orchestrator's job, since it owns the cross-cell view.)

**Memory measurement note:** like the Kokoro workers, `os_wired_gb` is read once after the
forward completes. The forward is a single synchronous op (no incremental decode), so a
post-eval read captures the settled high-water for that cell; the orchestrator's
between-cell settled baseline plus the conservative predictive gate provide the safety
margin against transient spikes.

## 5. The orchestrator — `wmx_suite/embeddings_probe.py` (NEW)

Analog of `probe.py`. Owns the per-row ramp, the predictive-skip safety gate, the
incremental fit, and DB persistence. Hardware-free-testable (subprocess spawn is injected /
mockable).

**Entry point (shape mirrors probe.py conventions):**
```python
def sweep(con, run_id, model, batches, seqs, repeats, margin_gb=None, *, log=_default_log) -> dict
```
- `run_id` is created by the CLI via `db.start_embeddings_run` and passed in; the
  orchestrator persists each cell's measurement under it.
- `batches` default `[1, 2, 4, 8, 16, 32]`; `seqs` default
  `[128, 256, 512, 1024, 2048, 4096, 8192]` (both overridable from the CLI).
- Returns a summary dict (model, n_cells_measured, n_cells_skipped, run_id).

**Memory model (encoder-specific analytic prior + incremental fit):**

Predicted absolute OS-wired peak for a cell:
```
predicted = live_base + model_base + a*(batch*seq) + b*(batch*seq^2)
```
- `live_base` = `sample_settled_baseline()` sampled fresh **between cells** (after the
  previous subprocess exits).
- `model_base` = weights-resident floor; seeded from a conservative constant prior and
  replaced by the measured smallest-cell delta once available.
- `a` (linear term) absorbs residual-stream + FFN + local-attention activations.
- `b` (quadratic term) is the global-attention `seq²` cost.

**Analytic priors (conservative = over-estimate, the safe direction for a never-crash
gate):**
- `b_analytic` treats **all 22 layers as global** (true count is 8) — a ~2.75× over-estimate
  of the `seq²` term:
  `b_analytic = num_layers * num_heads * 2 bytes / 1e9` GB per `(batch*seq^2)` unit
  (= 22 * 12 * 2 / 1e9). Plus the residual-stream term in `a_analytic`.
- These priors are used **only** until ≥2 real cells exist to fit against; the first cells
  ramped (small batch, small seq) are trivially safe.

**Fit:** least-squares fit of `delta = os_wired_gb - cell_baseline` against features
`(batch*seq, batch*seq^2)` across all measured cells → refined `(a, b)`; `model_base`
from the smallest measured cell. The **gate uses `max(analytic, fitted)`** for each term so
a lagging fit can never under-predict.

**Traversal — per-row ramp + predictive skip (the approved strategy):**
```
for batch in batches (ascending):
    for seq in seqs (ascending):
        live_base = sample_settled_baseline()
        predicted = live_base + model_base + max-of(analytic,fitted) terms
        if predicted >= threshold:
            log+emit row_skipped(batch, seq, predicted); break   # skip rest of row
        result = run_cell_subprocess(model, batch, seq, repeats, margin)
        if result.status == "error": handle (abort or skip per note below)
        else: persist measurement; update fit
```
- **Skip semantics:** when a cell is predicted to breach, the *rest of that row* (larger
  seqs) is skipped and we continue with the next (larger) batch — whose smaller seqs may
  still be safe and worth measuring. Memory is monotonically increasing in batch at fixed
  seq, so once `(batch, seq*)` is unsafe, `(batch', seq*)` for `batch' > batch` is also
  unsafe. Optimization: track the
  smallest unsafe seq seen and prune known-unsafe cells in later rows without spawning.
  (Safety is unaffected; this only avoids wasted safe-but-pointless work. Implementation
  may start simple — gate every cell — and add pruning if useful.)
- **Pre-flight at orchestrator start:** if the very first cell is already predicted unsafe
  (host pressure too high), abort the whole sweep with a clear message, like
  `probe.characterize`'s pre-flight.

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
  `embeddings_probe.sweep(con, run_id, ...)` with a `log` callback that renders a **grid
  table** (rows = batch, columns = seq; each cell shows `os_wired_gb` / `throughput_tps`;
  predicted-skipped cells marked e.g. `SKIP`; not-reached cells blank). On orchestrator
  error/abort, print the message and `sys.exit(1)`.

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

1. **DB lifecycle:** `start_embeddings_run` / `add_embeddings_measurement` /
   `get_all_embeddings_runs` / `get_embeddings_measurements` / `get_latest_embeddings_run`
   round-trip, including FK cascade delete.
2. **Worker — happy path:** patch `mlx_embeddings.load` (returns mock model+tokenizer),
   `mlx.core.get_peak_memory`/`reset_peak_memory`/`clear_cache`, `wmx_suite.system.wired_gb`
   and `read_limits`; run `probe_worker_embeddings.main()` with patched argv; assert the
   `rung_done` JSON has correct fields and `throughput_tps`/`latency_ms` math.
3. **Worker — pre-flight refusal (RULE #1 guard):** patch `read_limits` so
   `wired_now_gb >= safe_threshold_gb()`; run `main()`; assert `mlx_embeddings.load` was
   **never called** and an `error` line was emitted with exit 0. (This test does not exist
   for Kokoro — written fresh here.)
4. **Orchestrator — predictive skip (RULE #1 guard):** inject a fake cell-runner; feed
   measured points that make the fit predict the next cell `>= threshold`; assert the
   cell-runner is **not invoked** for the predicted-unsafe cell, a `row_skipped` event is
   emitted, and the sweep advances to the next batch. Mutation check during implementation:
   removing the gate must make this test fail.
5. **Orchestrator — monotonic pruning:** once `(batch, seq*)` is unsafe, assert
   `(batch', seq*)` with `batch' > batch` is not spawned.
6. **CLI — parse → DB:** drive `cmd_benchmark_embeddings` with the orchestrator/subprocess
   mocked to emit canned events; assert grid table prints and measurements land in the DB.

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

- mlx-embeddings API + direct `(batch, seq)` control — **verified against installed
  source** (`model(input_ids, attention_mask=...)`).
- Subprocess isolation — **corrected** to per-cell (LLM-probe pattern), not single-process
  (Kokoro pattern).
- Encoder memory model — dense `batch×seq²` confirmed from source; conservative
  all-layers-global prior is a safe over-estimate; incremental fit refines it.
- Non-causal models bypass `is_causal`/`characterize` — handled by a dedicated worker +
  orchestrator that never touch the causal path.
