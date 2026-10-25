"""Inspect cached upstream source as text; do not import hardware library."""
import ast
import base64
import csv
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
source = root.parent / "_private/upstream/tclab_source.py"
data = source.read_bytes()
tree = ast.parse(data.decode("utf-8"))
rows = []
for cls in tree.body:
    if isinstance(cls, ast.ClassDef) and cls.name == "TCLab":
        for method in cls.body:
            if isinstance(method, ast.FunctionDef) and method.name in {
                "__init__", "T1", "T2", "scan", "send_and_receive"
            }:
                text = ast.get_source_segment(data.decode("utf-8"), method)
                rows.append({"class": cls.name, "method": method.name,
                             "line": method.lineno,
                             "end_line": method.end_lineno,
                             "method_sha256": hashlib.sha256(text.encode()).hexdigest(),
                             "signature_arguments": [a.arg for a in method.args.args]})
out = {"source_path": str(source.relative_to(root.parent)),
       "source_sha256": hashlib.sha256(data).hexdigest(),
       "source_kind": "cached upstream source, not verified pip wheel",
       "methods": rows, "hardware_imported": False,
       "package_installed": False}
(root / "source_inspection.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))

installed = Path("/Users/adowling/opt/anaconda3/envs/controls3/lib/python3.10/site-packages")
dist = installed / "tclab-1.0.0.dist-info"
metadata = (dist / "METADATA").read_text()
record_rows = list(csv.reader((dist / "RECORD").read_text().splitlines()))
record = {row[0]: row for row in record_rows}
files = []
for rel in ["tclab/tclab.py", "tclab/historian.py", "tclab/version.py",
            "tclab-1.0.0.dist-info/METADATA", "tclab-1.0.0.dist-info/INSTALLER",
            "tclab-1.0.0.dist-info/WHEEL", "tclab-1.0.0.dist-info/RECORD"]:
    payload = (installed / rel).read_bytes()
    digest = hashlib.sha256(payload).digest()
    record_hash = record[rel][1]
    encoded = "sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("=")
    match = encoded == record_hash if record_hash else None
    assert match is not False, rel
    files.append({"path": rel, "bytes": len(payload), "sha256": digest.hex(),
                  "record_hash": record_hash, "record_hash_matches": match})
installed_tree = ast.parse((installed / "tclab/historian.py").read_text())
csv_methods = [{"class": cls.name, "method": f.name, "line": f.lineno,
                "end_line": f.end_lineno}
               for cls in installed_tree.body if isinstance(cls, ast.ClassDef)
               for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "to_csv"]
installed_out = {"site_packages_path": str(installed), "package": "tclab",
                 "version": "1.0.0", "installer": (dist / "INSTALLER").read_text().strip(),
                 "metadata_version_verified": "Version: 1.0.0\n" in metadata,
                 "upstream_tclab_source_identical": (installed / "tclab/tclab.py").read_bytes() == data,
                 "files": files, "historian_csv_method": csv_methods,
                 "package_imported": False, "environment_modified": False,
                 "wheel_archive_verified": False,
                 "limit": "Existing installed package; RECORD consistency is not independent wheel provenance or acquisition version"}
assert installed_out["metadata_version_verified"]
(root / "installed_package_inspection.json").write_text(json.dumps(installed_out, indent=2) + "\n")
print(json.dumps(installed_out, indent=2))
