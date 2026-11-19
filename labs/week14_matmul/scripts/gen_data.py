#!/usr/bin/env python3
"""Generate shared Matmul inputs and an independent CPU NumPy reference.

Usage: python3 scripts/gen_data.py --output-dir data
All variants must reuse this directory without regenerating different inputs.
"""
import os
# Set before importing NumPy so a large CPU reference does not oversubscribe.
for _variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                  'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_variable] = '8'

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

M, K, N = 4096, 4096, 5120
SEED = 20261119


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def generate(output_dir, seed=SEED):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Cast before computing the reference: the NPU receives these exact fp16 values.
    a = rng.uniform(1.0, 10.0, (M, K)).astype('<f2')
    b = rng.uniform(1.0, 10.0, (K, N)).astype('<f2')
    # Exact boundary values are explicitly represented in both inputs.
    a.flat[0], a.flat[-1] = np.float16(1), np.float16(10)
    b.flat[0], b.flat[-1] = np.float16(1), np.float16(10)
    assert a.shape == (M, K) and b.shape == (K, N)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert float(a.min()) >= 1 and float(a.max()) <= 10
    assert float(b.min()) >= 1 and float(b.max()) <= 10
    a.tofile(destination / 'input_a.bin')
    b.tofile(destination / 'input_b.bin')
    print(f'[DATA] seed={seed} A={M}x{K} B={K}x{N} input=float16 ND range=[1,10]', flush=True)
    print('[REFERENCE] CPU NumPy float32 matmul; BLAS thread limit=8', flush=True)
    started = time.perf_counter()
    # This reference does not call Ascend, torch_npu, or the tested kernel.
    reference = (a.astype(np.float32) @ b.astype(np.float32)).astype('<f4', copy=False)
    seconds = time.perf_counter() - started
    assert reference.shape == (M, N) and np.isfinite(reference).all()
    assert (reference > 0).all()
    reference.tofile(destination / 'golden.bin')
    # Validate selected reference coordinates with separate float64 dot products.
    points = {(0, 0), (0, N - 1), (M - 1, 0), (M - 1, N - 1),
              (M // 2, N // 2), (127, 255), (128, 256), (2047, 4095)}
    sample_rng = np.random.default_rng(seed + 1)
    points.update((int(sample_rng.integers(M)), int(sample_rng.integers(N))) for _ in range(24))
    samples = []
    for row, col in sorted(points):
        exact = float(np.dot(a[row, :].astype(np.float64), b[:, col].astype(np.float64)))
        value = float(reference[row, col])
        error = abs(value - exact)
        samples.append({'row': row, 'column': col, 'float64_reference': exact,
                        'float32_golden': value, 'abs_error': error,
                        'rel_error': error / abs(exact)})
    max_sample_rel = max(item['rel_error'] for item in samples)
    # Separate from the device threshold; a bad CPU reference must fail generation.
    if max_sample_rel > 5e-6:
        raise RuntimeError(f'CPU float32 reference failed float64 spot checks: {max_sample_rel}')
    expected_sizes = {'input_a.bin': M * K * 2, 'input_b.bin': K * N * 2,
                      'golden.bin': M * N * 4}
    for name, size in expected_sizes.items():
        if (destination / name).stat().st_size != size:
            raise RuntimeError(f'Unexpected file size for {name}')
    metadata = {
        'seed': seed, 'M': M, 'K': K, 'N': N,
        'input_dtype': 'float16', 'output_dtype': 'float32', 'format': 'ND',
        'byte_order': 'little', 'input_range': [1.0, 10.0],
        'input_a_shape': [M, K], 'input_b_shape': [K, N], 'output_shape': [M, N],
        'input_a_actual_range': [float(a.min()), float(a.max())],
        'input_b_actual_range': [float(b.min()), float(b.max())],
        'golden_range': [float(reference.min()), float(reference.max())],
        'reference_method': 'CPU NumPy: float32(A_fp16) @ float32(B_fp16)',
        'reference_compute_seconds': seconds, 'numpy_version': np.__version__,
        'blas_thread_limit': 8, 'output_elements': M * N,
        'file_bytes': expected_sizes,
        'sha256': {name: sha256_file(destination / name) for name in expected_sizes},
        'float64_spot_check_count': len(samples),
        'float64_spot_check_max_rel_error': max_sample_rel,
        'float64_spot_checks': samples,
    }
    (destination / 'metadata.json').write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'[REFERENCE] complete elements={M*N} range=[{reference.min():.9g},{reference.max():.9g}] '
          f'cpu_seconds={seconds:.3f}', flush=True)
    print(f'[REFERENCE] float64_spot_checks={len(samples)} max_rel_error={max_sample_rel:.9g}', flush=True)
    for name in expected_sizes:
        print(f'[SHA256] {metadata["sha256"][name]}  {name}', flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='data')
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args()
    generate(args.output_dir, args.seed)


if __name__ == '__main__':
    main()
