"""Index the completed study without replacing phase-specific run identities."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    phases = {}
    for name, relative, expected in [
        ("baseline_fits", "results/manifest.json", 99),
        ("bootstrap", "results/bootstrap/manifest.json", 240),
        ("correlated_likelihood", "results/ou/manifest.json", 48),
        ("coverage_chunks", "results/coverage/manifest.json", 360),
        ("assumption_sensitivity", "results/sensitivity/manifest.json", 241),
        ("prediction_uncertainty", "results/prediction_uncertainty/manifest.json", 131),
        ("profile_refinement", "results/profile_refinement/manifest.json", 18),
        ("parameter_design", "results/design_final/manifest.json", 112),
        ("discrimination_design", "results/discrimination_designs/manifest.json", 12),
        ("discrimination_refits", "results/discrimination_refits/manifest.json", 72),
    ]:
        data = read(relative)
        assert data["status"] == "complete", relative
        assert len(data["completed"]) == expected, relative
        assert not data.get("failed"), relative
        phases[name] = {"manifest": relative, "completed_jobs": expected,
                        "sha256": sha(ROOT / relative)}

    baseline = read("results/manifest.json")
    raw_data = read("data_audit.json")["files"]
    for record in raw_data:
        assert sha(REPO / record["file"]) == record["sha256"], record["file"]
    assert baseline["config"]["orders"] == [1, 2, 3, 4, 6, 8, 12]
    assert set(baseline["config"]["coordinates"]) == {"io", "physics"}
    assert set(baseline["config"]["initializations"]) == {"cold", "input_equilibrated"}
    assert set(baseline["config"]["calibration_sets"]) == {"sine", "step", "joint"}
    boot = read("results/bootstrap/manifest.json")
    assert sum(x["successful"] for x in boot["completed"].values()) == 96000
    cov = read("results/coverage/manifest.json")
    assert sum(x["complete"] for x in cov["completed"].values()) == 3600
    assert sum(x["failed"] for x in cov["completed"].values()) == 0
    profiles = read("results/selected_profile_index.json")
    assert len(profiles["profiles"]) == 300
    for directory in ["structural", "family_uncertainty"]:
        relative = f"results/{directory}/manifest.json"
        data = read(relative)
        assert data["status"] == "complete"
        for name, digest in data["output_sha256"].items():
            assert sha(ROOT / "results" / directory / name) == digest, name
        phases[directory] = {"manifest": relative, "sha256": sha(ROOT / relative),
                             "scope": data["notice"]}
    assert len(read("results/onset_limit/onset.json")["rows"]) == 18
    for relative in [
        "results/profile_refinement_audit.json", "results/coverage_summary/audit.json",
        "results/ou/summary_audit.json", "results/sensitivity/audit.json",
        "results/prediction_uncertainty/audit.json", "results/design_final/audit.json",
        "results/discrimination_designs/audit.json", "results/discrimination_refits/audit.json",
        "manuscript/figures/coverage_numeric_audit.json",
        "manuscript/figures/design_report_numeric_audit.json",
        "manuscript/figures/followup_numeric_audit.json",
        "manuscript/figures/uncertainty_numeric_audit.json",
        "manuscript/tables/table_numeric_audit.json",
    ]:
        assert read(relative)["status"] == "passed", relative
    migration = read("notebook_scripts/manifest.json")
    assert len(migration["scripts"]) == 13
    for row in migration["scripts"]:
        assert row["scientific_ast_preserved"]
        assert sha(REPO / row["notebook"]) == row["notebook_sha256"]
        assert sha(REPO / row["script"]) == row["script_sha256"]

    visual = read("manuscript/final_visual_review.json")
    assert visual["pages_reviewed"] == list(range(1, 49))
    pdf = ROOT / "manuscript/TC_Lab_delay_study_draft0.pdf"
    assert sha(pdf) == visual["pdf_sha256"]
    for path, digest in visual["source_tex_sha256"].items():
        assert sha(ROOT / "manuscript" / path) == digest, path
    for field in ["figure_pdf_sha256", "table_tex_sha256"]:
        for path, digest in visual[field].items():
            assert sha(ROOT / "manuscript" / path) == digest, path
    preflight = read("literature/manuscript_audit/preflight.json")
    assert preflight["draft_sha256"] == visual["draft_sha256"]
    assert sha(ROOT / "manuscript/bibliography.bib") == preflight["bibliography_sha256"]
    assert sha(ROOT / "literature/manuscript_audit/draft.tex") == visual["draft_sha256"]
    assert len(preflight["citations"]) == 24
    literature = read("literature/manuscript_audit/final_packet_receipt.json")
    assert literature["draft_sha256"] == visual["draft_sha256"]
    assert literature["citation_occurrences"] == 24
    assert literature["unique_cited_works"] == 7

    reproduction = read("results/final_reproduction/audit.json")
    assert reproduction["status"] == "passed"
    for case in reproduction["cases"]:
        assert case["successful_starts"] == 24
        assert case["absolute_sse_difference_C2"] == 0
        assert case["maximum_prediction_difference_C"] == 0
    suites = ET.parse(ROOT / "results/final_tests.xml").getroot()
    suite_rows = [suites] if suites.tag == "testsuite" else list(suites)
    tests = {key: sum(int(s.get(key, 0)) for s in suite_rows)
             for key in ["tests", "failures", "errors", "skipped"]}
    assert tests == {"tests": 141, "failures": 0, "errors": 0, "skipped": 0}
    started = datetime(2026, 10, 21, 1, 39, 3, tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    assert (now - started).total_seconds() >= 43200
    merge = "02ab3d979c7787a6866562a6ae9b0004a662e023"
    subprocess.run(["git", "merge-base", "--is-ancestor", merge, "HEAD"],
                   cwd=REPO, check=True)
    branch = subprocess.check_output(["git", "branch", "--show-current"],
                                     cwd=REPO, text=True).strip()
    assert branch == "reformulation-exploration"

    # Nested manifests retain measured source commits. This index hashes their
    # current receipts, not a claim that every phase used today's source tree.
    files = set()
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(x in path.parts for x in ["_private", "_runs", ".mplconfig", "__pycache__"]):
            continue
        if path.name in {"study_manifest.json", "PROJECT.md"}:
            continue
        if (path.parent == ROOT or path.name == "manifest.json"
                or "audit" in path.name or "review" in path.name
                or "source_snapshot" in path.name or path == pdf
                or path.name in {"source_review.json", "acquired_sources.json", "final_tests.xml", "onset.json"}):
            if path.suffix in {".json", ".py", ".md", ".txt", ".pdf", ".xml"}:
                files.add(path)
    manifest = {
        "schema": 1, "status": "research_artifacts_verified_delivery_separate",
        "indexed_utc": now.isoformat(), "research_started_utc": started.isoformat(),
        "elapsed_hours_at_index": (now - started).total_seconds() / 3600,
        "branch": branch, "merged_main_commit": merge,
        "index_parent_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "phases": phases,
        "additional_counts": {"migrated_notebooks": 13, "selected_profiles": 300,
                              "bootstrap_draws": 96000, "coverage_datasets": 3600,
                              "unchanged_raw_csvs": len(raw_data)},
        "tests": tests, "representative_fresh_reproduction": reproduction,
        "report": {"path": str(pdf.relative_to(ROOT)), "sha256": sha(pdf),
                   "pages": 48, "draft_sha256": visual["draft_sha256"],
                   "review": "Agent editorial and all-page visual inspection; human scientific review pending"},
        "literature": literature,
        "receipt_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
        "scope": "Index of phase authorities and final artifacts. Preserved failed trials are not selected results. Numerical audits retain their individual scopes; no global optimizer, hardware or real-data coverage certificate. Push and Gmail delivery are recorded separately in delivery_receipt.json.",
    }
    target = ROOT / "results/study_manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(target), "indexed_receipts": len(files),
                      "elapsed_hours": manifest["elapsed_hours_at_index"], "status": manifest["status"]}))


if __name__ == "__main__":
    main()
