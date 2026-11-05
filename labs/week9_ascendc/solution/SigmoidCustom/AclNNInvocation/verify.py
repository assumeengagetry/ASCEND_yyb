#!/usr/bin/env python3
"""Generate deterministic data, invoke ACLNN on NPU, and verify against float64."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def stable_sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', default='./build/execute_sigmoid')
    parser.add_argument('--dtype', choices=['all', 'float16', 'float32'], default='all')
    parser.add_argument('--shape', help='Single test shape, e.g. 8,2048 or 257')
    parser.add_argument('--numel', type=int, help='Single one-dimensional tensor size')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--output-dir', default='results')
    args = parser.parse_args()
    if args.shape and args.numel:
        parser.error('Use --shape or --numel, not both')
    if args.shape:
        shapes = [tuple(int(x) for x in args.shape.split(','))]
    elif args.numel:
        shapes = [(args.numel,)]
    else:
        shapes = [(8, 2048), (1,), (15,), (17,), (255,), (257,), (4096,), (65536,)]
    if any(not shape or any(d <= 0 for d in shape) for shape in shapes):
        parser.error('Shape dimensions must be positive')
    dtypes = ['float16', 'float32'] if args.dtype == 'all' else [args.dtype]
    executable = str(Path(args.executable).resolve())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261029)
    # Finite extreme inputs exercise both sigmoid saturation directions without
    # overflowing the stable float64 reference.
    extreme = np.array([-1000, -100, -80, -20, -16, -12, -8, -1, -0.0,
                        0, 1, 8, 12, 16, 20, 80, 100, 1000], dtype=np.float64)
    records = []
    for dtype_name in dtypes:
        dtype = np.dtype(dtype_name)
        rtol, atol = (1e-3, 1e-3) if dtype_name == 'float16' else (1e-4, 1e-6)
        for shape in shapes:
            count = int(np.prod(shape))
            values = rng.uniform(-12.0, 12.0, count)
            values[:min(count, extreme.size)] = extreme[:min(count, extreme.size)]
            # Also test a one-element tensor at an ordinary, non-saturated input.
            if count == 1:
                values[0] = 0.5
            x = values.astype(dtype)
            golden = stable_sigmoid(x)
            shape_text = ','.join(map(str, shape))
            label = f'{dtype_name}_' + 'x'.join(map(str, shape))
            case_dir = output_dir / label
            case_dir.mkdir(exist_ok=True)
            input_path, output_path = case_dir / 'input.bin', case_dir / 'output.bin'
            x.tofile(input_path)
            golden.tofile(case_dir / 'golden_float64.bin')
            if output_path.exists():
                output_path.unlink()
            command = [executable, dtype_name, shape_text, str(input_path), str(output_path), str(args.device)]
            result = subprocess.run(command, capture_output=True, text=True)
            (case_dir / 'npu.log').write_text(result.stdout + result.stderr)
            record = dict(dtype=dtype_name, shape=list(shape), numel=count, rtol=rtol, atol=atol,
                          returncode=result.returncode, input_min=float(x.min()), input_max=float(x.max()))
            if result.returncode == 0 and output_path.exists():
                actual = np.fromfile(output_path, dtype=dtype).astype(np.float64)
                if actual.size == count:
                    error = np.abs(actual - golden)
                    tolerance = atol + rtol * np.abs(golden)
                    good = np.isfinite(actual) & (error <= tolerance)
                    record.update(max_abs_error=float(error.max()),
                                  max_rel_error=float((error / np.maximum(np.abs(golden), 1e-300)).max()),
                                  failed_elements=int((~good).sum()),
                                  passed=bool(good.all()))
                    samples = [0, min(8, count - 1), min(10, count - 1), count - 1]
                    record['samples'] = [dict(index=i, x=float(x[i]), expected=float(golden[i]), actual=float(actual[i]))
                                         for i in sorted(set(samples))]
                else:
                    record.update(passed=False, error=f'Output length {actual.size}, expected {count}')
            else:
                record.update(passed=False, error=(result.stdout + result.stderr)[-8000:])
            records.append(record)
            print(f"{'PASS' if record['passed'] else 'FAIL'} dtype={dtype_name} shape={shape_text} "
                  f"numel={count} max_abs={record.get('max_abs_error', float('nan')):.9g} "
                  f"failed={record.get('failed_elements', 'N/A')} rtol={rtol:g} atol={atol:g}", flush=True)
    summary = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), seed=20261029,
                   reference='stable sigmoid computed in numpy float64 from quantized input',
                   execution='custom ACLNN NPU invocation', device=args.device,
                   passed=all(r['passed'] for r in records),
                   passed_cases=sum(r['passed'] for r in records), total_cases=len(records), cases=records)
    json_path = output_dir / 'verification.json'
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"SUMMARY {summary['passed_cases']}/{summary['total_cases']} PASS; JSON={json_path}")
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
