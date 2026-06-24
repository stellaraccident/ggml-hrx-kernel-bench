# GGML HRX Kernel Bench

Standalone Python project for developing, verifying, benchmarking, and tuning
GGML HRX Loom kernels outside llama.cpp.

This project is intentionally path-neutral. It does not assume a workspace
layout, build directory, cache directory, ROCm installation, or llama.cpp
checkout. Tool paths, kernel paths, and output directories are supplied by CLI
flags or config files.

## Status

This project now contains the imported HRX2 Loom corpus and a first usable
bench pipeline:

- HRX2 kernel and catalog import/reporting,
- route-backed candidate planning,
- explicit per-family config binding specs,
- observed shape metadata ingestion for live-profiled llama.cpp runs,
- NumPy fixture and golden generation for pilot families,
- focused Loom link/compile/run commands,
- JSONL ledgers plus preserved evidence directories,
- catalog candidate summary generation.

Unsupported or broken kernels are represented as ledger rows instead of being
excluded. Imported HRX2 kernels should remain source-faithful; target-specific
experiments belong in run metadata and temporary evidence, not kernel rewrites.

## Layout

```text
ggml-hrx-kernel-bench/
  pyproject.toml
  README.md
  src/ggml_hrx_kernel_bench/
  specs/
  kernels/
    hrx2/
  catalog/
    hrx2/
  schemas/
```

`kernels/hrx2/` contains the bench-owned HRX2 Loom corpus. `catalog/hrx2/`
contains the route/source metadata imported from the HRX2 handoff.

Family-specific route binding policy lives in
`src/ggml_hrx_kernel_bench/family_specs.py`. Add or correct kernel shape/config
requirements there instead of adding global string-matching fallbacks.

Observed live shapes live in `catalog/hrx2/observed_shapes.json`. Synthetic
search schedules are only bootstrap probes; real tuning sweeps should converge
on observed shapes collected from llama.cpp runs.

## Install

From this directory:

```bash
python3 -m pip install -e .
```

For fixture and golden generation:

```bash
python3 -m pip install -e ".[numpy]"
```

## Basic Usage

Generate an HRX2 import manifest:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/hrx2-import \
  --original-hrx2-root /path/to/llama.cpp-ref/ggml/src/ggml-hrx2 \
  import-hrx2
```

Plan the full corpus without compiling:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/hrx2-plan \
  plan
```

Plan using accumulated live-profiled shapes:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/hrx2-observed-plan \
  --sweep observed \
  plan
```

If no observed shape matches a route, `--sweep observed` falls back to that
route's minimal smoke shape so the route remains visible in ledgers.

Merge extracted llama.cpp shape traces into the bench metadata:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/shape-accumulate \
  --shape-trace runs/llama-shapes.jsonl \
  accumulate-shapes
```

The intended trace row shape is JSONL, one observed dispatch/route shape per
line:

```json
{
  "family": "mul_mat_q4_k_f32",
  "source_id": "mul_mat_q4_k_f32",
  "route_id": "mul_mat_q4_k_f32_wmma64x64_f16acc_k256_8192_r64_32768_c64_wg128",
  "root_symbol": "@hrx2_mul_mat_q4_k_f32_wmma64x64_f16acc",
  "shape": {"k": 4096, "rows": 4096, "cols": 64},
  "count": 37,
  "tags": ["llama-bench", "prefill"],
  "source": {
    "program": "llama-bench",
    "model": "qwen2.5-7b-q4_k_m",
    "args": "-p 512 -b 512 -ub 512"
  }
}
```

Rows may also be copied from bench ledgers: if a row contains a `candidate`
object, `accumulate-shapes` reads `candidate.family`, `candidate.route_id`,
`candidate.root_symbol`, `candidate.source_id`, and `candidate.shape`.

Plan or run a pilot family slice:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/pilot-fixtures \
  --family mul_mat_q4_k_f32,rms_norm_f32,copy_f32_f16,cont_f32 \
  --limit 8 \
  fixtures
```

Compile a focused candidate set:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/copy-compile \
  --loom-link /path/to/loom-link \
  --loom-compile /path/to/loom-compile \
  --rocm-path /path/to/rocm \
  --family copy_f32_f16 \
  --limit 1 \
  compile
```

Run a generated correctness-gated benchmark:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/copy-run \
  --loom-link /path/to/loom-link \
  --iree-benchmark-loom /path/to/iree-benchmark-loom \
  --rocm-path /path/to/rocm \
  --family copy_f32_f16 \
  --limit 1 \
  run
```

## Full Supported-Family Sweep

Use `sweep-supported` for a single top-level workflow that generates ledgers for
every currently ABI-backed family. It writes stage ledgers under the selected
run directory:

```text
<output-dir>/plan/ledger.jsonl
<output-dir>/fixtures/ledger.jsonl
<output-dir>/link/ledger.jsonl
<output-dir>/compile/ledger.jsonl   # only when --loom-compile is provided
<output-dir>/run/ledger.jsonl
<output-dir>/report.md
```

The command requires `--loom-link` and `--iree-benchmark-loom`. Provide
`--loom-compile` as well when you want standalone compile report summaries in
the Markdown report. Unsupported or failing candidates remain visible as ledger
rows and are promoted to the top of the report.

Example:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/supported-gfx1100 \
  --target gfx1100 \
  --rocm-path /path/to/rocm \
  --loom-link /path/to/loom-link \
  --loom-compile /path/to/loom-compile \
  --iree-benchmark-loom /path/to/iree-benchmark-loom \
  --iterations 1 \
  --warmup-iterations 0 \
  --max-batches 1 \
  sweep-supported
```

Use `--family` to smoke a subset, `--limit` to cap candidates during harness
development, and `--sweep edge` or `--sweep observed` when you want broader
coverage than the default minimal smoke shape per route.

On a `gfx1151` card running the HRX2 kernels as `gfx1100`, set the ROCr override
in the environment:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 \
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/supported-gfx1151-as-gfx1100 \
  --target gfx1100 \
  --rocm-path /path/to/rocm \
  --loom-link /path/to/loom-link \
  --loom-compile /path/to/loom-compile \
  --iree-benchmark-loom /path/to/iree-benchmark-loom \
  --iterations 1 \
  --warmup-iterations 0 \
  --max-batches 1 \
  sweep-supported
```

## Markdown Reports

`sweep-supported` writes `<output-dir>/report.md` automatically. To regenerate a
report from existing stage ledgers:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/supported-gfx1100 \
  report
```

To summarize arbitrary ledgers:

```bash
python3 -m ggml_hrx_kernel_bench \
  --output-dir runs/report-only \
  --ledger runs/supported-gfx1100/run/ledger.jsonl \
  --ledger runs/supported-gfx1100/compile/ledger.jsonl \
  --report-output runs/report-only/report.md \
  report
```

Reports put failures, unsupported workbenches, rejected correctness checks, and
tool errors first. They then summarize status counts, per-family timing winners,
candidate shapes/config bindings/dispatch metadata, and standalone compile
statistics when available.

Legacy single-spec mode is still available:

```bash
python3 -m ggml_hrx_kernel_bench \
  --spec specs/mul_mat_q4_k_f32.json \
  --output-dir runs/q4k-plan \
  plan
```

Compile commands require explicit Loom tool paths:

```bash
python3 -m ggml_hrx_kernel_bench \
  --spec specs/mul_mat_q4_k_f32.json \
  --kernel-source /path/to/mul_mat_q4_k_f32.loom \
  --loom-compile /path/to/loom-compile \
  --target gfx1100 \
  --output-dir runs/q4k-compile \
  compile
```

When this project is promoted to its own repository, CI and local examples
should provide a small synthetic Loom source rather than depending on the HRX
workspace checkout.

## Benchmark Module Materialization

Run-mode should benchmark a focused benchmark module, not an entire multi-kernel
source catalog. The intended pipeline is:

```text
kernel library + benchmark wrapper + concrete config
  -> selective materialization
  -> correctness-gated benchmark
```

Today that materialization is done with `loom-link --mode=link --root=...`
before invoking `iree-benchmark-loom`. The linked module is kept under the
candidate evidence directory so compiler reports, workbenches, fixtures, and
failure logs can be inspected together.

## Outputs

Commands write to the caller-provided `--output-dir`. Generated files should be
treated as run artifacts. Promote only intentional specs, kernels, concise
summaries, or catalog rows into source control.

The primary evidence file is:

```text
<output-dir>/ledger.jsonl
```

Rows are append-only JSON objects carrying spec identity, config, tool paths,
shape parameters, and command results.

Current smoke evidence from this workspace is under:

```text
cache/ggml-hrx-kernel-bench/import-smoke/
cache/ggml-hrx-kernel-bench/plan-family-spec-smoke/
cache/ggml-hrx-kernel-bench/fixtures-pilot/
cache/ggml-hrx-kernel-bench/compile-copy-smoke/
cache/ggml-hrx-kernel-bench/compile-rms-smoke2/
cache/ggml-hrx-kernel-bench/compile-q4-smoke/
cache/ggml-hrx-kernel-bench/run-copy-smoke2/
```
