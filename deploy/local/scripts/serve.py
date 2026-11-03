"""Load the Podman secret without persisting it or passing it in argv."""
import os
from pathlib import Path
import re
import sys


def main():
    key = Path('/run/secrets/strata_api_key').read_text().strip()
    if not key:
        raise SystemExit('The Strata API key is empty; run make set-api-key.')
    # The default model (qwen) uses /config/server.json (its variants: STRATA_VARIANT, read by the server); another
    # uses /config/server-<model>.json.
    model = os.environ.get('STRATA_MODEL', 'qwen')
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', model):
        raise SystemExit(f'Invalid STRATA_MODEL: {model!r}')
    config = '/config/server.json' if model == 'qwen' else f'/config/server-{model}.json'
    if not Path(config).is_file():
        raise SystemExit(f'Missing native Strata config at {config}.')
    # The server reads /config itself ("engine" and "engine_env" fields included: the fork's serve/engine_fields.py),
    # so the web page's Config view edits and applies these very files.
    os.environ['STRATA_API_KEY'] = key
    os.chdir('/opt/strata')
    os.execv(sys.executable, [sys.executable, '-m', 'serve.server',
                            '--engine', 'strata', '--config', str(config),
                            '--host', '0.0.0.0', '--port', '8080'])


if __name__ == '__main__':
    main()
