from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUCCESS_STATUSES = {"ok", "planned", "fixtures_ready", "linked", "compiled", "ran"}


@dataclass(frozen=True)
class ReportOptions:
    max_issues: int = 50
    top_per_family: int = 5


def write_markdown_report(ledger_paths: list[Path], output_path: Path, options: ReportOptions) -> dict[str, Any]:
    rows = read_ledgers(ledger_paths)
    lines = render_markdown(rows, ledger_paths, options)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {
        "report_path": str(output_path),
        "ledger_paths": [str(path) for path in ledger_paths],
        "row_count": len(rows),
        "issue_count": sum(1 for row in rows if is_issue(row)),
    }


def read_ledgers(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            rows.append({"action": "report-input", "status": "missing_ledger", "message": str(path)})
            continue
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    rows.append(
                        {
                            "action": "report-input",
                            "status": "invalid_ledger_json",
                            "message": f"{path}:{line_number}: {exc}",
                        }
                    )
                    continue
                row["_ledger_path"] = str(path)
                row["_ledger_line"] = line_number
                rows.append(row)
    return rows


def render_markdown(rows: list[dict[str, Any]], ledger_paths: list[Path], options: ReportOptions) -> list[str]:
    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    action_counts = Counter(str(row.get("action", "unknown")) for row in rows)
    family_rows = rows_by_family(rows)
    issues = [row for row in rows if is_issue(row)]

    lines: list[str] = [
        "# GGML HRX Kernel Bench Report",
        "",
        "## Issues",
        "",
    ]
    if issues:
        lines.extend(
            [
                f"Found **{len(issues)}** issue rows. The first {min(len(issues), options.max_issues)} are listed below.",
                "",
                "| Status | Action | Family | Candidate | Message | Evidence |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in issues[: options.max_issues]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_cell(str(row.get("status", "unknown"))),
                        md_cell(str(row.get("action", "unknown"))),
                        md_cell(family_of(row)),
                        md_cell(candidate_id_of(row)),
                        md_cell(issue_message(row)),
                        md_cell(evidence_link(row)),
                    ]
                )
                + " |"
            )
        if len(issues) > options.max_issues:
            lines.append("")
            lines.append(f"Additional issue rows omitted: {len(issues) - options.max_issues}.")
    else:
        lines.append("No issue rows found.")

    lines.extend(
        [
            "",
            "## Overview",
            "",
            f"- Ledger rows: **{len(rows)}**",
            f"- Families: **{len([family for family in family_rows if family != 'unknown'])}**",
            f"- Ledgers: **{len(ledger_paths)}**",
            "",
            "### Status Counts",
            "",
            "| Status | Rows |",
            "| --- | ---: |",
        ]
    )
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {md_cell(status)} | {count} |")

    lines.extend(["", "### Action Counts", "", "| Action | Rows |", "| --- | ---: |"])
    for action, count in sorted(action_counts.items()):
        lines.append(f"| {md_cell(action)} | {count} |")

    lines.extend(
        [
            "",
            "## Per-Family Summary",
            "",
            "| Family | Rows | Ran | Compiled | Linked | Fixtures | Issues | Best mean op ns | Best candidate | Best shape |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for family in sorted(family_rows):
        if family == "unknown":
            continue
        entries = family_rows[family]
        counts = Counter(str(row.get("status", "unknown")) for row in entries)
        best = best_timed_row(entries)
        lines.append(
            "| "
            + " | ".join(
                [
                    md_cell(family),
                    str(len(entries)),
                    str(counts.get("ran", 0)),
                    str(counts.get("compiled", 0)),
                    str(counts.get("linked", 0)),
                    str(counts.get("fixtures_ready", 0)),
                    str(sum(1 for row in entries if is_issue(row))),
                    format_ns(timing_ns(best)),
                    md_cell(candidate_id_of(best) if best else ""),
                    md_cell(short_json(candidate_shape(best)) if best else ""),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Best Timed Candidates", ""])
    timed_any = False
    for family in sorted(family_rows):
        timed = sorted(
            [row for row in family_rows[family] if timing_ns(row) is not None],
            key=lambda row: timing_ns(row) or 0,
        )
        if not timed:
            continue
        timed_any = True
        lines.extend(
            [
                f"### {family}",
                "",
                "| Mean op ns | Status | Candidate | Shape | Config | Dispatch | Correctness |",
                "| ---: | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in timed[: options.top_per_family]:
            lines.append(
                "| "
                + " | ".join(
                    [
                        format_ns(timing_ns(row)),
                        md_cell(str(row.get("status", ""))),
                        md_cell(candidate_id_of(row)),
                        md_cell(short_json(candidate_shape(row))),
                        md_cell(short_json(candidate_config(row))),
                        md_cell(short_json(candidate_dispatch(row))),
                        md_cell(short_json(benchmark_summary(row).get("correctness"))),
                    ]
                )
                + " |"
            )
        lines.append("")
    if not timed_any:
        lines.append("No timing rows found. Run with `--iree-benchmark-loom` to collect benchmark timings.")

    lines.extend(["", "## Compile Summaries", ""])
    compiled = [row for row in rows if compile_summary(row)]
    if compiled:
        lines.extend(
            [
                "| Family | Candidate | Status | Code bytes | Spills | Local bytes | Report |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in compiled:
            summary = compile_summary(row)
            compile_row = row.get("compile") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        md_cell(family_of(row)),
                        md_cell(candidate_id_of(row)),
                        md_cell(str(row.get("status", ""))),
                        md_cell(format_number(summary.get("emission_code_byte_count"))),
                        md_cell(format_number(summary.get("allocation_spill_count"))),
                        md_cell(format_number(summary.get("memory_local_bytes"))),
                        md_cell(str(compile_row.get("report") or "")),
                    ]
                )
                + " |"
            )
    else:
        lines.append("No standalone compile summaries found. Provide `--loom-compile` to `sweep-supported` to collect them.")

    lines.extend(["", "## Ledger Inputs", ""])
    for path in ledger_paths:
        lines.append(f"- `{path}`")
    return lines


def rows_by_family(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[family_of(row)].append(row)
    return out


def is_issue(row: dict[str, Any]) -> bool:
    status = str(row.get("status", "unknown"))
    if status not in SUCCESS_STATUSES:
        return True
    for key in ("link", "compile", "benchmark"):
        evidence = row.get(key)
        if isinstance(evidence, dict) and evidence.get("returncode") not in (None, 0):
            return True
    summary = benchmark_summary(row)
    if summary.get("failure"):
        return True
    correctness = summary.get("correctness")
    return not correctness_ok(correctness)


def family_of(row: dict[str, Any]) -> str:
    candidate = row.get("candidate") or {}
    spec = row.get("spec") or {}
    return str(candidate.get("family") or spec.get("op") or row.get("family") or "unknown")


def candidate_id_of(row: dict[str, Any] | None) -> str:
    if not row:
        return ""
    candidate = row.get("candidate") or {}
    spec = row.get("spec") or {}
    return str(candidate.get("candidate_id") or spec.get("id") or row.get("candidate_id") or "")


def candidate_shape(row: dict[str, Any] | None) -> Any:
    if not row:
        return None
    candidate = row.get("candidate") or {}
    return candidate.get("shape") or row.get("values")


def candidate_config(row: dict[str, Any] | None) -> Any:
    if not row:
        return None
    candidate = row.get("candidate") or {}
    return candidate.get("config_bindings") or row.get("config_bindings")


def candidate_dispatch(row: dict[str, Any] | None) -> Any:
    if not row:
        return None
    candidate = row.get("candidate") or {}
    return candidate.get("dispatch")


def benchmark_summary(row: dict[str, Any]) -> dict[str, Any]:
    benchmark = row.get("benchmark")
    if not isinstance(benchmark, dict):
        return {}
    summary = benchmark.get("summary")
    return summary if isinstance(summary, dict) else {}


def compile_summary(row: dict[str, Any]) -> dict[str, Any]:
    compile_row = row.get("compile")
    if not isinstance(compile_row, dict):
        return {}
    summary = compile_row.get("report_summary")
    return summary if isinstance(summary, dict) else {}


def timing_ns(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    timing = benchmark_summary(row).get("operation_timing_ns")
    if isinstance(timing, dict):
        value = timing.get("mean")
    else:
        value = timing
    if value is None:
        value = benchmark_summary(row).get("mean_physical_dispatch_duration_ns")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_timed_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    timed = [row for row in rows if timing_ns(row) is not None]
    return min(timed, key=lambda row: timing_ns(row) or 0) if timed else None


def correctness_ok(value: Any) -> bool:
    if value in (None, "", "passed", "ok", True):
        return True
    if isinstance(value, dict):
        failed = value.get("failed_sample_count")
        if failed is not None:
            try:
                return int(failed) == 0
            except (TypeError, ValueError):
                return False
        state = value.get("state")
        if state is not None:
            return state in ("passed", "ok")
    return False


def issue_message(row: dict[str, Any]) -> str:
    for key in ("message", "error"):
        if row.get(key):
            return str(row[key])
    oracle = row.get("oracle")
    if isinstance(oracle, dict) and oracle.get("message"):
        return str(oracle["message"])
    workbench = row.get("workbench")
    if isinstance(workbench, dict) and workbench.get("message"):
        return str(workbench["message"])
    summary = benchmark_summary(row)
    if summary.get("failure"):
        return short_json(summary["failure"], limit=180)
    for key in ("benchmark", "compile", "link"):
        evidence = row.get(key)
        if isinstance(evidence, dict) and evidence.get("stderr_path"):
            return str(evidence["stderr_path"])
    return ""


def evidence_link(row: dict[str, Any]) -> str:
    for key in ("benchmark", "compile", "link"):
        evidence = row.get(key)
        if isinstance(evidence, dict):
            path = evidence.get("stderr_path") or evidence.get("stdout_path")
            if path:
                return str(path)
    return str(row.get("_ledger_path") or "")


def format_ns(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.0f}"


def format_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.0f}"
    return str(value)


def short_json(value: Any, *, limit: int = 120) -> str:
    if value in (None, {}, []):
        return ""
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if len(text) > limit:
        return text[: limit - 1] + "..."
    return text


def md_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()
