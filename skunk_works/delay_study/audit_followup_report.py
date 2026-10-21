"""Check displayed follow-up scalars against independently audited saved records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--figures", type=Path, default=here/"manuscript/figures")
    parser.add_argument("--tables", type=Path, default=here/"manuscript/tables")
    args = parser.parse_args()
    manifest_path = args.figures/"followup_figure_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data_path = args.figures/"followup_plot_data.json"
    if sha(data_path) != manifest["data_sha256"]:
        raise RuntimeError("Display data changed")
    for name, digest in manifest["input_sha256"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError("Input changed: "+name)
    data = json.loads(data_path.read_text())
    cache, count, maximum = {}, 0, 0.
    for key, reference in data["references"].items():
        path = Path(reference["path"])
        if str(path) not in manifest["input_sha256"]:
            raise RuntimeError("Scalar source is not hash-bound")
        if path not in cache:
            if path.suffix == ".csv":
                with path.open() as handle:
                    cache[path] = list(csv.DictReader(handle))
            else:
                cache[path] = json.loads(path.read_text())
        if "pointer" in reference:
            expected = cache[path]
            for part in reference["pointer"]:
                expected = expected[part]
        else:
            expected = float(cache[path][reference["csv_row"]][reference["column"]])
        error = abs(expected-reference["value"])
        maximum = max(maximum, error)
        if error > 1e-12:
            raise RuntimeError("Displayed scalar differs from saved result: "+key)
        count += 1
    for stem, record in data["nuisance"].items():
        for metric, values in record.items():
            expected = [data["references"][stem+"/"+mode+"/"+metric]["value"]
                        for mode in ["baseline", "history", "ambient_cold", "ambient_history"]]
            if values != expected:
                raise RuntimeError("Nuisance plot array differs from checked scalars")
    for stem, record in data["reconstruction"].items():
        prefix = stem.removesuffix("linear_exact")
        for metric, values in record.items():
            suffixes = ["linear_exact", "left_exact", "right_exact"]
            if "pure_delay" not in stem:
                suffixes.append("right_backward_euler")
            expected = [data["references"][prefix+suffix+"/"+metric]["value"] for suffix in suffixes]
            if values != expected:
                raise RuntimeError("Reconstruction plot array differs from checked scalars")
    table_count = 0
    for name, rows in data["table_cells"].items():
        path = args.tables/(name+".tex")
        if sha(path) != manifest["table_sha256"][path.name]:
            raise RuntimeError("Table changed")
        text = path.read_text()
        for row in rows:
            tokens = [format(data["references"][key]["value"], fmt) for key, fmt in row["entries"]]
            if tokens != row["tokens"] or row["title"]+" & "+" & ".join(tokens)+r" \\" not in text:
                raise RuntimeError("Printed row differs from checked scalar formats")
            table_count += len(tokens)
    receipt = {"status": "passed", "scalar_values_checked": count,
               "table_cells_checked": table_count, "maximum_absolute_difference": maximum,
               "display_manifest_sha256": sha(manifest_path), "display_data_sha256": sha(data_path),
               "auditor_sha256": sha(Path(__file__)),
               "scope": "Saved-source hashes, plotted scalar arrays and table formatting; numerical analyses rely on their separate full phase audits. Rendered color/grayscale and manuscript pages require visual review. No coverage or physical-mechanism validation."}
    (args.figures/"followup_numeric_audit.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
