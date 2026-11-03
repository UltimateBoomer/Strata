"""Run upstream tools/calibrate.py on the served model's run config; with model variants, on one variant's."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, '/opt/strata')
from serve import engine_fields, models  # noqa: E402


def main():
    # The same config serve.py picks; a config with "models" is the common part, so the variant's own (STRATA_VARIANT,
    # else the first) is put together the way the server does.
    model = os.environ.get('STRATA_MODEL', 'qwen')
    path = Path('/config/server.json' if model == 'qwen' else f'/config/server-{model}.json')
    cfg = engine_fields.expand(json.loads(path.read_text()), path.name)
    if cfg.get('models'):
        _, cfg = models.load(cfg, str(path), os.environ.get('STRATA_VARIANT') or None)
    out = Path('/tmp/calibrate.json')
    out.write_text(json.dumps(cfg))
    os.chdir('/opt/strata')
    os.execv(sys.executable, [sys.executable, 'tools/calibrate.py', str(out)])


if __name__ == '__main__':
    main()
