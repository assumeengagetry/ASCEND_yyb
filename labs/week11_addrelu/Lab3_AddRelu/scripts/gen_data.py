#!/usr/bin/env python3
"""Generate the fixed-size float32 inputs required by the AddRelu lab."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SHAPE = (8, 2048)
NUMEL = 8 * 2048
TILE_LENGTH = 128
CORE_LENGTH = 2048
CASES = ("random", "boundary", "all_negative", "all_positive", "cancellation")


def generate(case, seed):
    rng = np.random.default_rng(seed)
    if case == "all_negative":
        x = rng.uniform(-10.0, -0.01, NUMEL).astype(np.float32)
        y = rng.uniform(-10.0, -0.01, NUMEL).astype(np.float32)
    elif case == "all_positive":
        x = rng.uniform(0.01, 10.0, NUMEL).astype(np.float32)
        y = rng.uniform(0.01, 10.0, NUMEL).astype(np.float32)
    else:
        x = rng.uniform(-10.0, 10.0, NUMEL).astype(np.float32)
        y = rng.uniform(-10.0, 10.0, NUMEL).astype(np.float32)
        if case == "cancellation":
            y = -x

    boundary_indexes = []
    if case == "boundary":
        one_up = np.nextafter(np.float32(1.0), np.float32(2.0))
        one_down = np.nextafter(np.float32(1.0), np.float32(0.0))
        pairs = np.array([
            (-10.0, -10.0), (10.0, 10.0), (-10.0, 10.0),
            (10.0, -10.0), (0.0, 0.0), (-3.0, 1.0),
            (3.0, -1.0), (1.0, -3.0), (-1.0, 3.0),
            (one_up, -1.0), (one_down, -1.0), (1.0, -one_down),
        ], dtype=np.float32)
        # Exercise each 128-element tile edge and all eight core regions.
        indexes = set(range(len(pairs)))
        for edge in range(TILE_LENGTH, NUMEL, TILE_LENGTH):
            indexes.update((edge - 1, edge, edge + 1))
        indexes.update((NUMEL - 2, NUMEL - 1))
        boundary_indexes = sorted(indexes)
        for order, index in enumerate(boundary_indexes):
            x[index], y[index] = pairs[order % len(pairs)]

    assert x.dtype == y.dtype == np.float32
    assert x.size == y.size == NUMEL
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert ((x >= -10) & (x <= 10)).all()
    assert ((y >= -10) & (y <= 10)).all()
    sums = np.add(x, y, dtype=np.float32)
    golden = np.maximum(sums, np.float32(0.0)).astype(np.float32)
    return x.reshape(SHAPE), y.reshape(SHAPE), golden.reshape(SHAPE), boundary_indexes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="random")
    parser.add_argument("--seed", type=int, default=20261112)
    args = parser.parse_args()
    Path("input").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    x, y, golden, boundary_indexes = generate(args.case, args.seed)
    files = {
        "input/input_x.bin": x,
        "input/input_y.bin": y,
        "output/golden.bin": golden,
    }
    for filename, values in files.items():
        values.astype("<f4", copy=False).tofile(filename)
        assert Path(filename).stat().st_size == NUMEL * 4
    sums = np.add(x, y, dtype=np.float32)
    metadata = {
        "case": args.case,
        "seed": args.seed,
        "shape": list(SHAPE),
        "numel": NUMEL,
        "dtype": "float32",
        "byte_order": "little",
        "bytes_per_tensor": NUMEL * 4,
        "input_range": [-10.0, 10.0],
        "core_count": 8,
        "elements_per_core": CORE_LENGTH,
        "tile_length": TILE_LENGTH,
        "formula": "maximum(add(x, y, dtype=float32), float32(0))",
        "sum_counts": {
            "negative": int(np.count_nonzero(sums < 0)),
            "zero": int(np.count_nonzero(sums == 0)),
            "positive": int(np.count_nonzero(sums > 0)),
        },
        "boundary_indexes": boundary_indexes,
        "sha256": {
            filename: hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            for filename in files
        },
    }
    Path("case_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"[DATA] case={args.case} seed={args.seed} dtype=float32 "
        f"shape=8x2048 elements={NUMEL} bytes={NUMEL * 4}"
    )
    print("[DATA] sum_counts=" + json.dumps(metadata["sum_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
