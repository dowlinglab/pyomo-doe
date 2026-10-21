"""Check true family witnesses and nuisance-reoptimized threshold resolution."""
import json
from pathlib import Path

from data import load
from profile_refinement import ProfileRefiner
from run_profile_refinement import selected_names

ROOT = Path(__file__).parent/"results"


def test_targeted_refinement_selection_is_complete_and_unique():
    names = selected_names()
    assert len(names) == len(set(names)) == 18
    catalog = json.loads((ROOT/"profile_index.json").read_text())["profiles"]
    assert all(name in catalog for name in names)
    assert sum("original" in name for name in names) == 9


def read_case(case, parameter):
    record = json.loads((ROOT/"fits"/(case+".json")).read_text())
    profile = json.loads((ROOT/"profiles"/(case+"__"+parameter+".json")).read_text())["profile"]
    return record, profile, [load(name) for name in record["datasets"]]


def test_original_family_extrema_become_verified_zero_lr_witnesses():
    record, profile, data = read_case("joint__original_n0_physics_cold_linear_exact", "CpH")
    refiner = ProfileRefiner(record, profile, data, starts=2)
    for endpoint in (refiner.family["minimum"], refiner.family["maximum"]):
        point = min(refiner.cache.values(), key=lambda row: abs(row["parameter_value"]-endpoint["value"]))
        assert abs(point["parameter_value"]-endpoint["value"]) < 1e-12
        assert abs(point["sse"]-record["fit"]["sse"]) < 1e-7
        assert any(attempt.get("kind") == "verified_equivalence_witness" for attempt in point["attempts"])


def test_refining_both_one_stage_modes_resolves_each_observed_threshold_crossing():
    record, profile, data = read_case("joint__chain_n1_io_cold_linear_exact", "theta")
    result = ProfileRefiner(record, profile, data, starts=2, max_nfev=300).refine(statistic_tolerance=.005)
    assert not result["refinement_requires_revisit"] and not result["baseline_requires_revisit"]
    assert len(result["refinement_crossings"]) == 8
    assert all(row["statistic_error"] < .005 for row in result["refinement_crossings"])
    assert len(result["intervals_unknown_sigma_iid"]) == 2
    for value in (record["fit"]["parameters"]["theta"], record["fit"]["parameters"]["tau"]):
        assert any(component["lower"] <= value <= component["upper"] for component in result["intervals_unknown_sigma_iid"])
