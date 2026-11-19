#!/usr/bin/env python3
"""Verify every Sinh output against an independent mathematical float64 oracle."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

NUMEL = 16384
BYTE_SIZE = NUMEL * 4
# Fixed before device execution; all elements must satisfy this bound.
RTOL = 1e-5
ATOL = 1e-6


def read_tensor(path):
    path = Path(path)
    size = path.stat().st_size
    if size != BYTE_SIZE:
        raise ValueError(f"{path}: expected {BYTE_SIZE} bytes, got {size}")
    values = np.fromfile(path, dtype="<f4")
    if values.size != NUMEL:
        raise ValueError(f"{path}: expected {NUMEL} float32 elements")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: contains NaN or infinity")
    return values


def verify(output_path, golden_path, input_x_path):
    actual = read_tensor(output_path)
    golden = read_tensor(golden_path)
    x = read_tensor(input_x_path)
    if not ((x >= 1) & (x <= 10)).all():
        raise ValueError("Input values fall outside the required [1,10] range")
    independent = np.sinh(x.astype(np.float64)).astype(np.float32)
    golden_matches = np.array_equal(golden, independent)
    reference64 = independent.astype(np.float64)
    absolute_error = np.abs(actual.astype(np.float64) - reference64)
    relative_error = absolute_error / np.abs(reference64)
    # Explicit elementwise comparison; no fractional error allowance.
    close = absolute_error <= ATOL + RTOL * np.abs(reference64)
    positive = actual > 0
    good = close & positive
    failed_indexes = np.flatnonzero(~good)
    return {
        "passed": bool(golden_matches and good.all()),
        "dtype": "float32", "shape": [8, 2048], "numel": NUMEL,
        "bytes_per_tensor": BYTE_SIZE, "rtol": RTOL, "atol": ATOL,
        "reference": "float32(sinh(float64(x)))",
        "golden_matches_independent_reference": bool(golden_matches),
        "golden_failed_elements": int(np.count_nonzero(golden != independent)),
        "all_finite": True, "all_positive": bool(positive.all()),
        "input_min": float(x.min()), "input_max": float(x.max()),
        "failed_elements": int(failed_indexes.size),
        "max_abs_error": float(absolute_error.max()),
        "mean_abs_error": float(absolute_error.mean()),
        "max_rel_error": float(relative_error.max()),
        "core_failed_counts": [
            int(np.count_nonzero(~good[core * 2048:(core + 1) * 2048]))
            for core in range(8)
        ],
        "sample_elements": [
            {"index": i, "x": float(x[i]), "golden": float(golden[i]),
             "reference": float(independent[i]), "actual": float(actual[i]),
             "abs_error": float(absolute_error[i]),
             "rel_error": float(relative_error[i])}
            for i in range(8)
        ],
        "first_failed_indexes": failed_indexes[:100].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="output/output_z.bin")
    parser.add_argument("golden", nargs="?", default="output/golden.bin")
    parser.add_argument("--input-x", default="input/input_x.bin")
    parser.add_argument("--json", default="output/verification.json", dest="json_path")
    args = parser.parse_args()
    try:
        result = verify(args.output, args.golden, args.input_x)
    except (OSError, ValueError) as error:
        result = {"passed": False, "error": str(error), "numel": NUMEL,
                  "dtype": "float32", "rtol": RTOL, "atol": ATOL}
    destination = Path(args.json_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if "error" in result:
        print("[FAIL] " + result["error"])
    else:
        print(f"[VERIFY] elements={NUMEL} failed={result['failed_elements']} "
              f"max_abs_error={result['max_abs_error']:.9g} "
              f"max_rel_error={result['max_rel_error']:.9g} rtol={RTOL:g} atol={ATOL:g}")
        print("[VERIFY] golden_matches_independent_reference="
              + str(result['golden_matches_independent_reference']))
        print("[VERIFY] core_failed_counts=" + json.dumps(result["core_failed_counts"]))
        print("[VERIFY] first 8 elements: index x expected actual abs_error rel_error")
        for s in result["sample_elements"]:
            print(f"{s['index']:5d} {s['x']:.9g} {s['reference']:.9g} "
                  f"{s['actual']:.9g} {s['abs_error']:.9g} {s['rel_error']:.9g}")
    print("test pass" if result["passed"] else "test fail")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
