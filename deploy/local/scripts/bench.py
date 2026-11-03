"""Small sequential API benchmark; emits JSON for comparison across settings."""
import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:9931")
    parser.add_argument('--label', default='baseline')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-tokens', type=int, default=128)
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1:
        parser.error('repeats and max-tokens must be positive')
    key = os.environ.get('STRATA_API_KEY')
    if not key:
        try:
            key = subprocess.check_output(
                [str(Path(__file__).with_name('credentials.sh')), 'lookup', 'strata_api_key'],
                text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            raise SystemExit('Cannot decrypt the API key credential; run on the host or set STRATA_API_KEY')
    if not key:
        raise SystemExit('No API key found')
    cfg = json.loads((Path(__file__).parents[1] / 'config/server.json').read_text())
    rows = []
    with requests.Session() as session:
        session.headers['Authorization'] = f'Bearer {key}'
        session.get(args.url + '/health', timeout=10).raise_for_status()
        # One warmup followed by fresh prefixes to avoid prompt-cache reuse.
        for kind, words in [('warmup', 16), ('short', 64), ('long', 2048)]:
            for repeat in range(1 if kind == 'warmup' else args.repeats):
                prompt = f'Request {uuid.uuid4().hex}.\n' + ' '.join(
                    f'item{i % 97}' for i in range(words))
                prompt += '\nWrite a numbered list of observations about this input.'
                start = time.perf_counter()
                first = None
                usage = None
                with session.post(args.url + '/v1/chat/completions', json={
                    'model': cfg['model_name'],
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0, 'max_tokens': args.max_tokens,
                    'stream': True, 'stream_options': {'include_usage': True},
                }, stream=True, timeout=(10, 300)) as response:
                    response.raise_for_status()
                    for line in response.iter_lines(chunk_size=1):
                        if not line.startswith(b'data:'):
                            continue
                        payload = line[5:].strip()
                        if payload == b'[DONE]':
                            break
                        event = json.loads(payload)
                        if event.get('error'):
                            raise RuntimeError(event['error'])
                        for choice in event.get('choices', []):
                            delta = choice.get('delta', {})
                            if first is None and any(delta.get(k) for k in
                                                     ('content', 'reasoning_content', 'reasoning')):
                                first = time.perf_counter() - start
                        if event.get('usage'):
                            usage = event['usage']
                elapsed = time.perf_counter() - start
                rows.append({'workload': kind, 'repeat': repeat + 1,
                             'wall_seconds': elapsed, 'first_token_seconds': first,
                             'usage': usage})
    print(json.dumps({'label': args.label, 'server_config': cfg,
                      'max_tokens': args.max_tokens, 'runs': rows}, indent=2))


if __name__ == '__main__':
    main()
