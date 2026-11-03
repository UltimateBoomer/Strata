"""Prepare pinned GGUFs and Strata artifacts without running upstream setup.py."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import quote

import requests

ROOT = Path('/opt/strata')
DATA = Path('/data')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as src:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(session, url, target, size, sha):
    """Reuse existing files by size; always verify downloads before publication."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verify = os.environ.get('STRATA_VERIFY_EXISTING', '0') == '1'
        if target.stat().st_size == size and (not verify or digest(target) == sha):
            print(f'{"Verified" if verify else "Reusing"} {target.name}', flush=True)
            return
        raise RuntimeError(f'{target} failed verification; remove it and retry')
    partial = target.with_name(target.name + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise RuntimeError(f'{partial} is too large; remove it and retry')
    if offset < size:
        with session.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                         stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            if offset and (response.status_code != 206 or not
                           response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                raise RuntimeError('Server did not honor the download resume range')
            print(f'Downloading {target.name} from byte {offset}', flush=True)
            with partial.open('ab' if offset else 'wb') as out:
                for block in response.iter_content(8 * 1024 * 1024):
                    out.write(block)
    if partial.stat().st_size != size or digest(partial) != sha:
        raise RuntimeError(f'{partial} failed verification; remove it and retry')
    partial.replace(target)


def tool(name, *args):
    subprocess.run([sys.executable, str(ROOT / 'tools' / name), *map(str, args)],
                   cwd=ROOT, check=True)


def model_files(cfg):
    if not isinstance(cfg.get('files'), list) or not cfg['files'] or not all(isinstance(f, str) for f in cfg['files']):
        raise ValueError('files must be a nonempty list of repository filenames')
    files = list(cfg['files'])
    if cfg.get('vision_file'):
        files.append(cfg['vision_file'])
    for source in cfg.get('sources', []):
        if (not isinstance(source.get('repository'), str) or not isinstance(source.get('files'), list) or not source['files']
                or not all(isinstance(f, str) for f in source['files'])):
            raise ValueError('Each source needs a repository and a nonempty list of filenames')
        files.extend(source['files'])
    if not files:
        raise ValueError('At least one model file is required')
    names = set()
    for filename in files:
        path = Path(filename)
        if path.is_absolute() or '..' in path.parts or not path.name:
            raise ValueError('Invalid model filename')
        if path.name in names or path.name == 'manifest.json' or path.name.endswith('.partial'):
            raise ValueError('Model filenames must have unique, non-reserved basenames')
        names.add(path.name)
    return files


def prepare(cfg, folder, pack):
    """Run explicitly configured tools once per model/preparation configuration."""
    pack.mkdir(parents=True, exist_ok=True)
    complete = pack / '.complete'
    if complete.exists():
        return
    values = {'model': str(folder), 'pack': str(pack), 'root': str(ROOT),
              'first_file': str(folder / Path(cfg['files'][0]).name)}
    for command in cfg.get('prepare', []):
        if not isinstance(command, list) or not command or not all(isinstance(arg, str) for arg in command):
            raise ValueError('Each prepare command must be a nonempty list of strings')
        name, *args = [arg.format_map(values) for arg in command]
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not name.endswith('.py'):
            raise ValueError('Preparation tools must be Python files under Strata tools/')
        tool(name, *args)
    for copy in cfg.get('copies', []):
        source = Path(copy['source'].format_map(values))
        target = Path(copy['target'].format_map(values))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    complete.write_text('complete\n')


def reuse_cached_file(target, filename, info):
    """Link a file another model folder already holds (an older cache layout, or the same blob under another name:
    the Coder's shard 2 is Q2_0's) without copying large files."""
    if target.exists():
        return
    for candidate in sorted((DATA / 'models').glob('*/manifest.json')):
        if candidate.parent == target.parent:
            continue
        try:
            metadata = json.loads(candidate.read_text())
        except (OSError, ValueError):
            continue
        for entry in metadata.get('siblings', []):
            if entry.get('lfs', {}).get('sha256') != info['sha256'] or not entry.get('rfilename'):
                continue
            source = candidate.parent / Path(entry['rfilename']).name
            if source.is_file() and source.stat().st_size == info['size']:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.hardlink_to(source)
                print(f'Reusing cached {target.name} from {candidate.parent.name}/{source.name}', flush=True)
                return


def resolve(session, repository, revision):
    response = session.get(
        f"https://huggingface.co/api/models/{quote(repository, safe='/')}/revision/{quote(revision, safe='')}",
        params={'blobs': 'true'}, timeout=60)
    response.raise_for_status()
    metadata = response.json()
    resolved = metadata['sha']
    if not re.fullmatch(r'[0-9a-f]{40}', resolved):
        raise RuntimeError('Hugging Face returned an invalid revision')
    if re.fullmatch(r'[0-9a-f]{40}', revision) and resolved != revision:
        raise RuntimeError('Hugging Face returned a different revision')
    return metadata


def fetch(session, repository, metadata, files, folder):
    base = f"https://huggingface.co/{quote(repository, safe='/')}"
    entries = {item['rfilename']: item for item in metadata['siblings']}
    for filename in files:
        if filename not in entries:
            raise ValueError(f'Model file not found in repository: {filename}')
        info = entries[filename].get('lfs')
        if not info or not info.get('sha256') or 'size' not in info:
            raise ValueError(f'Model file requires LFS size and SHA-256 metadata: {filename}')
        target = folder / Path(filename).name
        reuse_cached_file(target, filename, info)
        download(session, f"{base}/resolve/{metadata['sha']}/{quote(filename, safe='/')}",
                 target, info['size'], info['sha256'])


def initialize(cfg, session, suffix=''):
    model_files(cfg)
    metadata = resolve(session, cfg['repository'], cfg.get('revision', 'main'))
    resolved = metadata['sha']
    # Files from other repositories (a drafter, a metadata shard) land in the same folder.
    sources = [(s, resolve(session, s['repository'], s.get('revision', 'main'))) for s in cfg.get('sources', [])]
    # Preserve the original download-cache key. Preparation and projector changes
    # must not invalidate already downloaded model shards.
    key = {'repository': cfg['repository'], 'revision': resolved, 'files': cfg['files']}
    if sources:
        key['sources'] = [{'repository': s['repository'], 'revision': m['sha'], 'files': s['files']} for s, m in sources]
    identity = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    preparation_identity = hashlib.sha256(json.dumps(
        {**cfg, 'revision': resolved}, sort_keys=True).encode()).hexdigest()[:16]
    folder = DATA / 'models' / identity
    pack = DATA / 'packs' / preparation_identity
    fetch(session, cfg['repository'], metadata, cfg['files'] + ([cfg['vision_file']] if cfg.get('vision_file') else []), folder)
    for source, source_metadata in sources:
        fetch(session, source['repository'], source_metadata, source['files'], folder)
    (folder / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    prepare(cfg, folder, pack)
    # Each model has its own links (the default model /data/model and /data/pack, another /data/model-<name> and
    # /data/pack-<name>): an engine still running another model never writes through them (DeepSeek saves its
    # expert profile into its pack when it stops).
    for name, target in (('model' + suffix, folder), ('pack' + suffix, pack)):
        link = DATA / name
        temporary = DATA / (name + '.tmp')
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(target, target_is_directory=True)
        temporary.replace(link)
    print('Strata model artifacts ready.', flush=True)


def config_path(kind, model):
    """The default model (qwen) uses /config/<kind>.json; another uses /config/<kind>-<model>.json."""
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', model):
        raise SystemExit(f'Invalid STRATA_MODEL: {model!r}')
    return Path('/config') / (f'{kind}.json' if model == 'qwen' else f'{kind}-{model}.json')


def variant_model(server_dir, name, file):
    """The model a variant runs: the <name> of its pack, /data/pack-<name> ("engine": {"pack": ...} or --pack in its
    "args"), so variants of one model with other settings (q3s-fast, q3s-slow) share one download; else the variant's
    own name."""
    try:
        cfg = json.loads((Path(server_dir) / file).read_text()) if isinstance(file, str) else file
    except (OSError, ValueError):
        cfg = None
    pack = None
    if isinstance(cfg, dict):
        pack = (cfg.get('engine') or {}).get('pack')
        args = cfg.get('args') or []
        if pack is None and '--pack' in args and args.index('--pack') + 1 < len(args):
            pack = args[args.index('--pack') + 1]
    m = re.fullmatch(r'/data/pack-([a-z0-9][a-z0-9_-]*)', pack) if isinstance(pack, str) else None
    return m.group(1) if m else name


def models_to_prepare(model):
    """(model config, link suffix) pairs for STRATA_MODEL: a server config with "models" (Strata's model variants)
    prepares the model of each variant (variant_model: its pack's name, once for the variants that share it) from
    /config/model-<name>.json, linked as /data/{model,pack}-<name>; else the model's own config."""
    server = config_path('server', model)
    variants = json.loads(server.read_text()).get('models') if server.is_file() else None
    names = list(dict.fromkeys(variant_model(server.parent, n, f) for n, f in variants.items())) if variants else [model]
    return [(config_path('model', n), '' if n == 'qwen' else '-' + n) for n in names]


def main():
    model = os.environ.get('STRATA_MODEL', 'qwen')
    todo = models_to_prepare(model)
    for path, _ in todo:
        if not path.is_file():
            raise SystemExit(f'Missing model config {path} for STRATA_MODEL={model}.')
    with requests.Session() as session:
        token_file = Path('/run/secrets/hf_token')
        token = token_file.read_text().strip() if token_file.exists() else ''
        if token:
            session.headers['Authorization'] = f'Bearer {token}'
        for path, suffix in todo:
            print(f'Preparing {path.name}', flush=True)
            initialize(json.loads(path.read_text()), session, suffix)


if __name__ == '__main__':
    main()
