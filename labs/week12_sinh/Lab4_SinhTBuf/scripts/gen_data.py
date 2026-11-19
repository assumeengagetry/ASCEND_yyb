#!/usr/bin/env python3
"""Generate reproducible float32 Sinh inputs in [1,10] and a float64 oracle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SHAPE = (8, 2048)
NUMEL = 16384
TILE_LENGTH = 128
CORE_LENGTH = 2048
CASES = ("random", "boundary", "ramp", "ones", "tens")


def generate(case, seed=20261119):
    rng = np.random.default_rng(seed)
    boundary_indexes = []
    if case == "ramp":
        x = np.linspace(1.0, 10.0, NUMEL, dtype=np.float32)
    elif case == "ones":
        x = np.ones(NUMEL, dtype=np.float32)
    elif case == "tens":
        x = np.full(NUMEL, 10.0, dtype=np.float32)
    elif case in ("random", "boundary"):
        x = rng.uniform(1.0, 10.0, NUMEL).astype(np.float32)
    else:
        raise ValueError(f"Unknown case: {case}")
    if case == "boundary":
        values = np.array([
            1.0, np.nextafter(np.float32(1.0), np.float32(10.0)),
            np.nextafter(np.float32(10.0), np.float32(1.0)), 10.0,
            2.0, 3.0, 5.0, 7.5,
        ], dtype=np.float32)
        indexes = set(range(len(values)))
        for edge in range(TILE_LENGTH, NUMEL, TILE_LENGTH):
            indexes.update((edge - 1, edge, edge + 1))
        indexes.update((NUMEL - 2, NUMEL - 1))
        boundary_indexes = sorted(indexes)
        for order, index in enumerate(boundary_indexes):
            x[index] = values[order % len(values)]
    assert x.dtype == np.float32 and x.size == NUMEL
    assert np.isfinite(x).all() and ((x >= 1) & (x <= 10)).all()
    # Use the mathematical Sinh oracle, not the kernel's basic-API exponential sequence.
    golden = np.sinh(x.astype(np.float64)).astype(np.float32)
    return x.reshape(SHAPE), golden.reshape(SHAPE), boundary_indexes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="random")
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()
    Path("input").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    x, golden, boundary_indexes = generate(args.case, args.seed)
    files = {"input/input_x.bin": x, "output/golden.bin": golden}
    for filename, values in files.items():
        values.astype("<f4", copy=False).tofile(filename)
        assert Path(filename).stat().st_size == NUMEL * 4
    metadata = {
        "case": args.case, "seed": args.seed, "shape": list(SHAPE),
        "numel": NUMEL, "dtype": "float32", "byte_order": "little",
        "bytes_per_tensor": NUMEL * 4, "input_range": [1.0, 10.0],
        "actual_input_min": float(x.min()), "actual_input_max": float(x.max()),
        "core_count": 8, "elements_per_core": CORE_LENGTH,
        "tile_length": TILE_LENGTH, "tiles_per_core": 16,
        "formula": "float32(sinh(float64(x)))",
        "boundary_indexes": boundary_indexes,
        "sha256": {
            filename: hashlib.sha256(Path(filename).read_bytes()).hexdigest()
            for filename in files
        },
    }
    Path("case_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[DATA] case={args.case} seed={args.seed} dtype=float32 "
          f"shape=8x2048 elements={NUMEL} bytes={NUMEL * 4}")
    print(f"[DATA] input_min={float(x.min()):.9g} input_max={float(x.max()):.9g} "
          "oracle=float32(sinh(float64(x)))")


if __name__ == "__main__":
    main()
