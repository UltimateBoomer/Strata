import os
from pathlib import Path
import subprocess
import tempfile
import unittest


def stub(root, name, body):
    file = root / name
    file.write_text('#!/bin/sh\n' + body + '\n')
    file.chmod(0o755)


def environment(root):
    # Keep credential files and registration IDs out of the real home directory, and
    # select the local mode even when the caller targets a remote Podman.
    env = {k: v for k, v in os.environ.items() if k not in ('CONTAINER_CONNECTION', 'CONTAINER_HOST')}
    return dict(env, PATH=f'{root}:' + os.environ['PATH'], TEST_STATE=str(root),
                XDG_CONFIG_HOME=str(root / 'config'), XDG_DATA_HOME=str(root / 'data'))


class SecretRegistrationTests(unittest.TestCase):
    def test_repeated_registration_reuses_ids_without_delete(self):
        script = Path(__file__).parents[1] / 'scripts/register-podman-secrets.sh'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            credentials = root / 'config/strata-local/credentials'
            credentials.mkdir(parents=True)
            (credentials / 'strata_api_key.cred').write_text('encrypted\n')
            stub(root, 'systemd-creds', 'exit 0')
            # base64 of "test-key", as the system credential service returns it.
            stub(root, 'varlinkctl', 'printf %s \'{"data": "dGVzdC1rZXk="}\'')
            stub(root, 'podman', '''set -eu
[ "$1" = secret ] && [ "$2" = create ] && [ "$3" = --ignore ]
name=$7
case "$name" in strata-local-api-key|strata-local-hf-token) ;; *) exit 2 ;; esac
cat >/dev/null
if [ ! -f "$TEST_STATE/$name" ]; then
  printf '%s\\n' "$name-created" > "$TEST_STATE/$name"
  printf '%s\\n' "$name" >> "$TEST_STATE/created"
fi
cat "$TEST_STATE/$name"''')
            env = environment(root)
            for _ in range(2):
                subprocess.run([str(script)], env=env, check=True, capture_output=True)
            self.assertEqual((root / 'created').read_text().splitlines(),
                             ['strata-local-api-key', 'strata-local-hf-token'])

    def test_remote_stores_values_with_default_driver(self):
        script = Path(__file__).parents[1] / 'scripts/register-podman-secrets.sh'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            credentials = root / 'config/strata-local/credentials'
            credentials.mkdir(parents=True)
            (credentials / 'strata_api_key.cred').write_text('encrypted\n')
            stub(root, 'systemd-creds', 'exit 0')
            stub(root, 'varlinkctl', 'printf %s \'{"data": "dGVzdC1rZXk="}\'')
            stub(root, 'podman', '''set -eu
printf '%s\\n' "$*" >> "$TEST_STATE/args"
cat > "$TEST_STATE/$4"''')
            env = dict(environment(root), CONTAINER_CONNECTION='remote')
            subprocess.run([str(script)], env=env, check=True, capture_output=True)
            self.assertEqual((root / 'args').read_text().splitlines(),
                             ['secret create --replace strata-local-api-key -',
                              'secret create --replace strata-local-hf-token -'])
            self.assertEqual((root / 'strata-local-api-key').read_text(), 'test-key\n')
            # The absent optional token is stored as a newline, which Podman accepts.
            self.assertEqual((root / 'strata-local-hf-token').read_text(), '\n')

    def test_missing_key_fails_before_podman_changes(self):
        script = Path(__file__).parents[1] / 'scripts/register-podman-secrets.sh'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stub(root, 'systemd-creds', 'exit 0')
            stub(root, 'varlinkctl', 'touch "$TEST_STATE/decrypted"')
            stub(root, 'podman', 'touch "$TEST_STATE/called"')
            result = subprocess.run([str(script)], env=environment(root), capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'make set-api-key', result.stderr)
            self.assertFalse((root / 'called').exists())
            self.assertFalse((root / 'decrypted').exists())
