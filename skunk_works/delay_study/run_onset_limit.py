"""Small deterministic onset calculation, separate from saved structural outputs."""
import argparse
import hashlib
import json
from pathlib import Path

from onset_limit import step_at_mean_delay, clt_coefficient
from run_fits import save_json

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE/"results/onset_limit")
    args = parser.parse_args()
    parameters = {"gain": .78, "thermal_time": 150., "delay": 18., "command": 50.}
    coefficient = clt_coefficient(**parameters)
    orders = [1, 2, 3, 4, 6, 8, 12, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536]
    rows = [{"order": order, "rise_at_mean_delay_C": step_at_mean_delay(order, **parameters),
             "sqrt_n_times_rise_C": order**.5*step_at_mean_delay(order, **parameters),
             "relative_to_CLT_limit": order**.5*step_at_mean_delay(order, **parameters)/coefficient}
            for order in orders]
    identity = {name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                for name in ("onset_limit.py", "run_onset_limit.py", "run_fits.py")}
    path = args.output/"onset.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["source_sha256"] != identity:
            raise RuntimeError("Preserve onset results from the earlier producer")
    save_json(path, {"parameters": parameters, "CLT_coefficient_C": coefficient, "rows": rows,
                     "source_sha256": identity,
                     "notice": "Cold constant step, fixed parameters, exact tilted Erlang CDF. Pure-delay output at the nominal onset is zero; finite chains respond before that time. The CLT coefficient is a derived asymptotic result, not a fitted regression rate or a refitted-model error."})
    print(json.dumps({"rows": len(rows), "coefficient_C": coefficient, "last_ratio": rows[-1]["relative_to_CLT_limit"]}))


if __name__ == "__main__":
    main()
