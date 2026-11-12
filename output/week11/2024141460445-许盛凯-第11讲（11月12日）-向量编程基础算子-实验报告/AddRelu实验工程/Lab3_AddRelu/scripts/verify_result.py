#!/usr/bin/env python3
"""Strictly verify all 16384 float32 AddRelu output elements."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


NUMEL = 16384
BYTE_SIZE = NUMEL * 4
RTOL = 1e-6
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


def verify(output_path, golden_path, input_x_path, input_y_path):
    actual = read_tensor(output_path)
    golden = read_tensor(golden_path)
    x = read_tensor(input_x_path)
    y = read_tensor(input_y_path)
    if not (((x >= -10) & (x <= 10)).all() and ((y >= -10) & (y <= 10)).all()):
        raise ValueError("Input values fall outside the required [-10,10] range")

    # Independent reference: every binary32 input is exact in binary64; add in
    # binary64 and round once to binary32 before applying ReLU.
    independent_sum = (x.astype(np.float64) + y.astype(np.float64)).astype(np.float32)
    independent = np.maximum(independent_sum, np.float32(0.0))
    golden_matches = np.array_equal(golden, independent)
    close = np.isclose(actual, independent, rtol=RTOL, atol=ATOL, equal_nan=False)
    nonnegative = actual >= 0
    good = close & nonnegative
    failed_indexes = np.flatnonzero(~good)
    absolute_error = np.abs(actual.astype(np.float64) - independent.astype(np.float64))
    result = {
        "passed": bool(golden_matches and good.all()),
        "dtype": "float32",
        "shape": [8, 2048],
        "numel": NUMEL,
        "bytes_per_tensor": BYTE_SIZE,
        "rtol": RTOL,
        "atol": ATOL,
        "golden_matches_independent_reference": bool(golden_matches),
        "all_finite": True,
        "all_nonnegative": bool(nonnegative.all()),
        "failed_elements": int(failed_indexes.size),
        "max_abs_error": float(absolute_error.max()),
        "mean_abs_error": float(absolute_error.mean()),
        "sum_counts": {
            "negative": int(np.count_nonzero(independent_sum < 0)),
            "zero": int(np.count_nonzero(independent_sum == 0)),
            "positive": int(np.count_nonzero(independent_sum > 0)),
        },
        "core_failed_counts": [
            int(np.count_nonzero(~good[core * 2048:(core + 1) * 2048]))
            for core in range(8)
        ],
        "sample_elements": [
            {
                "index": i,
                "x": float(x[i]),
                "y": float(y[i]),
                "sum": float(independent_sum[i]),
                "golden": float(golden[i]),
                "reference": float(independent[i]),
                "actual": float(actual[i]),
                "abs_error": float(absolute_error[i]),
            }
            for i in range(12)
        ],
        "first_failed_indexes": failed_indexes[:100].tolist(),
    }
    if not golden_matches:
        result["golden_failed_elements"] = int(np.count_nonzero(golden != independent))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", default="output/output_z.bin")
    parser.add_argument("golden", nargs="?", default="output/golden.bin")
    parser.add_argument("--json", default="output/verification.json", dest="json_path")
    parser.add_argument("--input-x")
    parser.add_argument("--input-y")
    args = parser.parse_args()
    lab_root = Path(args.golden).resolve().parent.parent
    input_x = args.input_x or lab_root / "input/input_x.bin"
    input_y = args.input_y or lab_root / "input/input_y.bin"
    try:
        result = verify(args.output, args.golden, input_x, input_y)
    except (OSError, ValueError) as error:
        result = {"passed": False, "error": str(error), "numel": NUMEL, "dtype": "float32"}
    destination = Path(args.json_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if "error" in result:
        print("[FAIL] " + result["error"])
    else:
        print(
            f"[VERIFY] elements={NUMEL} failed={result['failed_elements']} "
            f"max_abs_error={result['max_abs_error']:.9g} "
            f"rtol={RTOL:g} atol={ATOL:g}"
        )
        print("[VERIFY] sum_counts=" + json.dumps(result["sum_counts"], sort_keys=True))
        print("[VERIFY] core_failed_counts=" + json.dumps(result["core_failed_counts"]))
        print("[VERIFY] first 12 elements: index x y sum expected actual")
        for sample in result["sample_elements"]:
            print(
                f"{sample['index']:5d} {sample['x']:+.9g} {sample['y']:+.9g} "
                f"{sample['sum']:+.9g} {sample['reference']:+.9g} {sample['actual']:+.9g}"
            )
    print("test pass" if result["passed"] else "test fail")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
