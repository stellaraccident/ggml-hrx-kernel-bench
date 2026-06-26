from __future__ import annotations

from ggml_hrx_kernel_bench.route_schedules import (
    concrete_shapes_for_route,
    schedule_points_for_route,
    select_test_route,
)
from ggml_hrx_kernel_bench.llama_catalog import _test_schedule_for_family


def test_edge_schedule_is_bounded_and_annotated() -> None:
    route = {
        "id": "q4",
        "family": "mul_mat_q4_k_f32",
        "source_id": "mul_mat_q4_k_f32",
        "root_symbol": "@q4",
        "shape_domain": {"k_min": 256, "k_max": 8192, "rows_min": 1, "rows_max": 4096, "cols_min": 1, "cols_max": 512},
        "shape_guards": {"k_multiple_of": 256},
    }

    points = schedule_points_for_route(route, sweep="edge")

    assert 1 < len(points) <= 6
    assert all(point.source in {"llm-atlas", "atlas-upper-bound"} for point in points)
    assert all(point.route_id == "q4" for point in points)
    assert all(point.facts["k"] % 256 == 0 for point in points)


def test_observed_schedule_uses_matching_observations_or_smoke_fallback() -> None:
    route = {
        "id": "copy",
        "family": "copy_f32_f16",
        "shape_domain": {"ncols_min": 1, "ncols_max": 4096, "nrows_min": 1, "nrows_max": 128},
    }

    observed = concrete_shapes_for_route(
        route,
        sweep="observed",
        observed_shapes=[{"ncols": 257, "nrows": 17}, {"ncols": 8192, "nrows": 1}],
    )
    fallback = concrete_shapes_for_route(route, sweep="observed", observed_shapes=[])

    assert observed == [{"ncols": 257, "nrows": 17, "cols": 257, "rows": 17}]
    assert fallback == [{"ncols": 64, "nrows": 1, "cols": 64, "rows": 1}]


def test_test_route_selection_prefers_specific_domains() -> None:
    routes = [
        {"id": "wide", "family": "add_f32", "shape_domain": {"ncols_min": 1, "ncols_max": 4096}},
        {"id": "exact", "family": "add_f32", "shape_domain": {"ncols_min": 128, "ncols_max": 128}},
    ]

    selected = select_test_route("add_f32", routes)

    assert selected is not None
    assert selected["id"] == "exact"


def test_q4_test_route_selection_preserves_direct_preference() -> None:
    routes = [
        {
            "id": "mul_mat_q4_k_f32_wmma64x64_f16acc_k256_8192_r64_32768_c64_wg128",
            "family": "mul_mat_q4_k_f32",
            "shape_domain": {"k_min": 256, "k_max": 8192, "rows_min": 64, "rows_max": 32768, "cols_min": 64, "cols_max": 64},
        },
        {
            "id": "mul_mat_q4_k_f32_direct_k256_32768_r1_32768_c1_wg256",
            "family": "mul_mat_q4_k_f32",
            "shape_domain": {"k_min": 256, "k_max": 32768, "rows_min": 1, "rows_max": 32768, "cols_min": 1, "cols_max": 1},
        },
    ]

    selected = select_test_route("mul_mat_q4_k_f32", routes)

    assert selected is not None
    assert "_direct_" in selected["id"]


def test_llama_edge_test_schedule_exports_multiple_route_cases() -> None:
    routes = [
        {
            "id": "mul_mat_q4_k_f32_direct_k256_32768_r1_32768_c1_wg256",
            "family": "mul_mat_q4_k_f32",
            "op": "MUL_MAT",
            "source_id": "mul_mat_q4_k_f32",
            "root_symbol": "@direct",
            "shape_domain": {"k_min": 256, "k_max": 32768, "rows_min": 1, "rows_max": 32768, "cols_min": 1, "cols_max": 1},
            "shape_guards": {"k_multiple_of": 256},
        },
        {
            "id": "mul_mat_q4_k_f32_wmma64x64_f16acc_k256_8192_r64_32768_c64_wg128",
            "family": "mul_mat_q4_k_f32",
            "op": "MUL_MAT",
            "source_id": "mul_mat_q4_k_f32",
            "root_symbol": "@wmma",
            "shape_domain": {"k_min": 256, "k_max": 8192, "rows_min": 64, "rows_max": 32768, "cols_min": 64, "cols_max": 64},
            "shape_guards": {"k_multiple_of": 256},
        },
    ]

    minimal = _test_schedule_for_family(target_key="gfx1100", family_id="mul_mat_q4_k_f32", routes=routes, sweep="minimal")
    edge = _test_schedule_for_family(target_key="gfx1100", family_id="mul_mat_q4_k_f32", routes=routes, sweep="edge")

    assert minimal is not None
    assert edge is not None
    assert len(minimal["cases"]) == 1
    assert len(edge["cases"]) > 1
    assert {case["expected_route_id"] for case in edge["cases"]} == {route["id"] for route in routes}
