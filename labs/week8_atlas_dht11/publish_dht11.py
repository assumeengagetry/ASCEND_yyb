#!/usr/bin/env python3
"""Read a real DHT11 frame with the C helper, then optionally publish it.

Python 3.6+ standard library only. Run on the Atlas, not on the PC.
Hardware compatibility is conditional; see 实验操作指南.md.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reader', default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'dht11_read'))
    parser.add_argument('--gpio', type=int, default=504)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=1883)
    parser.add_argument('--topic', default='atlas/dht11')
    parser.add_argument('--interval', type=float, default=3.0)
    parser.add_argument('--count', type=int, default=0,
                        help='number of sampling attempts; 0 means continuous')
    parser.add_argument('--local-only', action='store_true',
                        help='read and print real samples without MQTT')
    args = parser.parse_args()
    if args.interval < 3 or args.count < 0 or args.gpio < 0:
        parser.error('interval must be >=3, count and gpio must be >=0')
    if not os.path.isfile(args.reader) or not os.access(args.reader, os.X_OK):
        parser.error('compile dht11_read.c first; reader is not executable')

    run_id = uuid.uuid4().hex[:12]
    attempted = valid = published = 0
    try:
        while args.count == 0 or attempted < args.count:
            started = time.monotonic()
            attempted += 1
            try:
                result = subprocess.run(
                    [args.reader, '--gpio', str(args.gpio), '--attempts', '1'],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    universal_newlines=True, timeout=10)
                if result.returncode:
                    print('sample {} failed: {}'.format(attempted, result.stderr.strip()),
                          file=sys.stderr, flush=True)
                    # Configuration failures require action, not endless retries.
                    if result.returncode == 2:
                        return 2
                else:
                    frame = json.loads(result.stdout)
                    temperature = float(frame['temperature_c'])
                    humidity = float(frame['humidity_percent'])
                    if not (-20 <= temperature <= 60 and 0 <= humidity <= 100):
                        raise ValueError('measurement outside DHT11 limits')
                    valid += 1
                    frame.update({
                        'device': 'atlas200dk-a1', 'run_id': run_id,
                        'seq': attempted,
                        'ts_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()
                    })
                    payload = json.dumps(frame, ensure_ascii=True, separators=(',', ':'))
                    print(payload, flush=True)
                    if not args.local_only:
                        result = subprocess.run(
                            ['mosquitto_pub', '-h', args.host, '-p', str(args.port),
                             '-t', args.topic, '-q', '1', '-m', payload],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, timeout=10)
                        if result.returncode:
                            print('MQTT seq={} failed: {}'.format(
                                attempted, result.stderr.strip()), file=sys.stderr, flush=True)
                        else:
                            published += 1
                            print('MQTT broker accepted seq={}'.format(attempted),
                                  file=sys.stderr, flush=True)
            except FileNotFoundError as exc:
                print('missing executable: {}'.format(exc), file=sys.stderr)
                return 2
            except (subprocess.TimeoutExpired, ValueError, KeyError) as exc:
                print('sample {} error: {}'.format(attempted, exc), file=sys.stderr, flush=True)
            if args.count == 0 or attempted < args.count:
                time.sleep(max(0, args.interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        pass
    finally:
        print('SUMMARY run_id={} attempts={} valid={} broker_accepted={}'.format(
            run_id, attempted, valid, published), file=sys.stderr, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
