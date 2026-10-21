"""Explicit selection must preserve first-pass artifacts and audited provenance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from profile_catalog import ProfileCatalog

HERE = Path(__file__).parent
RESULTS = HERE/"results"
JOINT_THETA = "joint__chain_n1_io_cold_linear_exact__theta.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+"\n")


def synthetic_refined_catalog(root):
    # This is a provenance-parser fixture, not a numerical refinement or audit.
    first = {"profiles":{f"profile_{i}.json":{"path":f"profiles/profile_{i}.json","sha256":"unused",
                      "analytic_family_inclusion_failed":i<18} for i in range(300)}}
    write(root/"profile_index.json",first)
    selected = deepcopy(first)
    names = list(first["profiles"])[:18]
    for name in names:
        old = first["profiles"][name]
        source = root/"refined_retry"/name
        write(source,{"fixture_only":True})
        selected["profiles"][name].update(path=str(source.relative_to(root)),sha256=sha(source),refined=True,
                         analytic_family_inclusion_failed=False,source_first_pass_path=old["path"])
    manifest = {"status":"complete","failed":{},"completed":{n:{} for n in names},
                "identity":{"source_catalog_sha256":sha(root/"profile_index.json")}}
    write(root/"refined_retry/manifest.json",manifest)
    audit = {"status":"passed","refined_profiles":18,"profiles":[{"profile":n} for n in names]}
    write(root/"profile_refinement_audit.json",audit)
    selected.update(refinement_manifest_sha256=sha(root/"refined_retry/manifest.json"),
                    refinement_manifest_path="refined_retry/manifest.json",refinement_audit_path="profile_refinement_audit.json",
                    refinement_audit_sha256=sha(root/"profile_refinement_audit.json"))
    write(root/"selected_profile_index.json",selected)
    return names


def test_default_catalog_preserves_verified_source_and_legacy_artifacts():
    catalog = ProfileCatalog(RESULTS)
    assert catalog.kind == "preserved_first_pass" and len(catalog.entries) == 300
    path = catalog.profile_path(JOINT_THETA)
    assert sha(path) == catalog.entries[JOINT_THETA]["sha256"]
    catalog.verify_manifest({"input_sha256":{str(catalog.path):sha(catalog.path)}})


def test_flagged_coarse_physical_interval_cannot_be_consumed_as_interpretable():
    catalog = ProfileCatalog(RESULTS)
    with pytest.raises(RuntimeError,match="excludes exact equivalent"):
        catalog.profile_path("joint__original_n0_physics_cold_linear_exact__CpH.json")


def test_profile_content_hash_and_catalog_completeness_are_checked(tmp_path):
    data = json.loads((RESULTS/"profile_index.json").read_text())
    data["profiles"][JOINT_THETA]["sha256"] = "0"*64
    write(tmp_path/"modified.json",data)
    catalog = ProfileCatalog(RESULTS,tmp_path/"modified.json")
    with pytest.raises(RuntimeError,match="file path/bytes"):
        catalog.profile_path(JOINT_THETA)
    data["profiles"].pop(JOINT_THETA)
    write(tmp_path/"modified.json",data)
    with pytest.raises(RuntimeError,match="all 300"):
        ProfileCatalog(RESULTS,tmp_path/"modified.json")


def test_selected_catalog_requires_complete_unchanged_audit_and_supports_retry_path(tmp_path):
    names = synthetic_refined_catalog(tmp_path)
    catalog = ProfileCatalog(tmp_path,"selected_profile_index.json")
    assert catalog.kind == "audited_selected_refinement" and catalog.selection["refined_profiles"] == 18
    assert catalog.profile_path(names[0]) == tmp_path/"refined_retry"/names[0]
    write(tmp_path/"profile_refinement_audit.json",{"status":"passed","changed_after_audit":True})
    with pytest.raises(RuntimeError,match="receipt bytes changed"):
        ProfileCatalog(tmp_path,"selected_profile_index.json")


def test_artifact_cannot_silently_switch_catalog(tmp_path):
    synthetic_refined_catalog(tmp_path)
    catalog = ProfileCatalog(tmp_path,"selected_profile_index.json")
    catalog.verify_manifest({"profile_catalog":catalog.selection})
    changed = dict(catalog.selection,sha256="0"*64)
    with pytest.raises(RuntimeError,match="another explicit profile catalog"):
        catalog.verify_manifest({"profile_catalog":changed})
    with pytest.raises(RuntimeError,match="Legacy artifact"):
        catalog.verify_manifest({"input_sha256":{"profile_index.json":"0"*64}})


def test_table_generation_owns_only_five_outputs_and_preserves_first_pass_cells(tmp_path,monkeypatch):
    import render_tables
    import audit_tables
    foreign = tmp_path/"foreign.tex"
    foreign.write_text("Another producer owns this file.\n")
    monkeypatch.setattr(sys,"argv",["render_tables.py","--results",str(RESULTS),"--output",str(tmp_path),
                                   "--profile-index","profile_index.json"])
    render_tables.main()
    manifest = json.loads((tmp_path/"table_manifest.json").read_text())
    expected = {"joint_parameters.tex","calibration_rmse.tex","coordinate_metrics.tex","delay_uncertainty.tex","ou_diagnostics.tex"}
    assert set(manifest["outputs"]) == expected and foreign.read_text() == "Another producer owns this file.\n"
    default = tmp_path/"default_first_pass"
    monkeypatch.setattr(sys,"argv",["render_tables.py","--results",str(RESULTS),"--output",str(default)])
    render_tables.main()
    for name in expected:
        assert (tmp_path/name).read_bytes() == (default/name).read_bytes()
    monkeypatch.setattr(sys,"argv",["audit_tables.py","--results",str(RESULTS),"--tables",str(tmp_path)])
    audit_tables.main()
    audit = json.loads((tmp_path/"table_numeric_audit.json").read_text())
    assert audit["numeric_cells_checked"] == 150 and set(audit["table_sha256"]) == expected
    path = tmp_path/"calibration_rmse.tex"
    path.write_text("\n".join(line for line in path.read_text().splitlines() if not line.startswith("$n=1$"))+"\n")
    manifest["outputs"][path.name] = sha(path)
    write(tmp_path/"table_manifest.json",manifest)
    with pytest.raises(RuntimeError,match="Incomplete table model rows"):
        audit_tables.main()
