#!/usr/bin/env python3
"""Strictly verify all float32 Matmul outputs against the shared CPU reference.

Usage: python3 scripts/verify_result.py OUTPUT GOLDEN --json verification.json
The threshold is fixed before device runs; any failed element fails the case.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

M, K, N = 4096, 4096, 5120
NUMEL = M * N
BYTE_SIZE = NUMEL * 4
RTOL = 1e-4
ATOL = 0.02
CHUNK = 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_tensor(path):
    path = Path(path)
    size = path.stat().st_size
    if size != BYTE_SIZE:
        raise ValueError(f'{path}: expected {BYTE_SIZE} bytes, got {size}')
    return np.memmap(path, dtype='<f4', mode='r', shape=(NUMEL,))


def verify(output_path, golden_path):
    actual = read_tensor(output_path)
    golden = read_tensor(golden_path)
    failed = nonfinite_actual = nonfinite_golden = nonpositive_actual = nonpositive_golden = 0
    max_abs_error = max_rel_error = sum_squared_error = 0.0
    first_failed_indexes = []
    row_failed_counts = np.zeros(M, dtype=np.int64)
    for start in range(0, NUMEL, CHUNK):
        end = min(start + CHUNK, NUMEL)
        x = np.asarray(actual[start:end], dtype=np.float64)
        y = np.asarray(golden[start:end], dtype=np.float64)
        finite_x, finite_y = np.isfinite(x), np.isfinite(y)
        finite = finite_x & finite_y
        positive_x, positive_y = x > 0, y > 0
        nonfinite_actual += int(np.count_nonzero(~finite_x))
        nonfinite_golden += int(np.count_nonzero(~finite_y))
        nonpositive_actual += int(np.count_nonzero(~positive_x))
        nonpositive_golden += int(np.count_nonzero(~positive_y))
        # Invalid entries never contribute NaN to metrics, but still fail explicitly.
        errors = np.zeros(end - start, dtype=np.float64)
        errors[finite] = np.abs(x[finite] - y[finite])
        relative = np.zeros_like(errors)
        safe = finite & positive_y
        relative[safe] = errors[safe] / np.abs(y[safe])
        valid = finite & positive_x & positive_y & (errors <= ATOL + RTOL * np.abs(y))
        bad = np.flatnonzero(~valid) + start
        failed += int(bad.size)
        if len(first_failed_indexes) < 100:
            first_failed_indexes.extend(int(i) for i in bad[:100-len(first_failed_indexes)])
        row_failed_counts += np.bincount(bad // N, minlength=M)
        max_abs_error = max(max_abs_error, float(errors.max()))
        max_rel_error = max(max_rel_error, float(relative.max()))
        sum_squared_error += float(np.dot(errors, errors))
    sample_indexes = sorted(set([0, 1, 2, 127, 128, N-1, N, N+1,
                                 (M//2)*N + N//2, NUMEL-2, NUMEL-1]))
    samples = []
    for index in sample_indexes:
        x, y = float(actual[index]), float(golden[index])
        samples.append({'index': index, 'row': index // N, 'column': index % N,
                        'actual': x if np.isfinite(x) else None,
                        'golden': y if np.isfinite(y) else None,
                        'abs_error': abs(x-y) if np.isfinite(x) and np.isfinite(y) else None})
    return {
        'passed': failed == 0, 'shape': [M, N], 'M': M, 'K': K, 'N': N,
        'dtype': 'float32', 'format': 'ND', 'numel': NUMEL, 'bytes_per_tensor': BYTE_SIZE,
        'rtol': RTOL, 'atol': ATOL, 'failed_elements': failed,
        'all_finite': nonfinite_actual == nonfinite_golden == 0,
        'nonfinite_actual': nonfinite_actual, 'nonfinite_golden': nonfinite_golden,
        'nonpositive_actual': nonpositive_actual, 'nonpositive_golden': nonpositive_golden,
        'max_abs_error': max_abs_error, 'max_rel_error': max_rel_error,
        'root_mean_square_error': (sum_squared_error / NUMEL) ** 0.5,
        'error_metrics_exclude_nonfinite_pairs': bool(nonfinite_actual or nonfinite_golden),
        'first_failed_indexes': first_failed_indexes,
        'rows_with_failures': [int(i) for i in np.flatnonzero(row_failed_counts)],
        'sample_elements': samples,
        'output_sha256': sha256_file(output_path), 'golden_sha256': sha256_file(golden_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('golden')
    parser.add_argument('--json', default='verification.json', dest='json_path')
    args = parser.parse_args()
    try:
        result = verify(args.output, args.golden)
    except (OSError, ValueError) as error:
        result = {'passed': False, 'error': str(error), 'shape': [M, N],
                  'numel': NUMEL, 'dtype': 'float32', 'rtol': RTOL, 'atol': ATOL}
    destination = Path(args.json_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if 'error' in result:
        print('[FAIL] ' + result['error'])
    else:
        print(f'[VERIFY] shape={M}x{N} elements={NUMEL} failed={result["failed_elements"]} '
              f'rtol={RTOL:g} atol={ATOL:g}')
        print(f'[VERIFY] finite={result["all_finite"]} '
              f'max_abs_error={result["max_abs_error"]:.9g} '
              f'max_rel_error={result["max_rel_error"]:.9g} '
              f'rmse={result["root_mean_square_error"]:.9g}')
        print('[SAMPLES] index row column actual golden abs_error')
        for s in result['sample_elements']:
            print(s['index'], s['row'], s['column'], s['actual'], s['golden'], s['abs_error'])
    print('test pass' if result['passed'] else 'test fail')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
