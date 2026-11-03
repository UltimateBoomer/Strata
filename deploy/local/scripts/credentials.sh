#!/bin/sh
# The project's secrets as user-scoped systemd credentials: encrypted with the host's
# credential key (and the TPM when present), decryptable only by this user on this machine.
#   credentials.sh store NAME            prompt for a value and encrypt it (empty removes an optional one)
#   credentials.sh lookup NAME           print the decrypted value; an absent optional value prints nothing
#   credentials.sh migrate-keyring NAME  copy a value from the GNOME keyring (Secret Service), if present
set -eu

action=${1:?Expected store, lookup or migrate-keyring}
name=${2:?Expected a credential name}
dir=${XDG_CONFIG_HOME:-$HOME/.config}/strata-local/credentials

case "$name" in
  strata_api_key) required=1 ;;
  hf_token) required=0 ;;
  *) echo "Unknown credential: $name" >&2; exit 2 ;;
esac
file=$dir/$name.cred

encrypt() {
  install -d -m 700 "$dir"
  umask 077
  # Write beside the target and rename, so a reader never sees a partial file.
  systemd-creds encrypt --user --name="$name" - "$file.tmp"
  mv -f "$file.tmp" "$file"
}

case "$action" in
  store)
    printf '%s: ' "$name" >&2
    stty -echo 2>/dev/null || true
    IFS= read -r value || value=
    stty echo 2>/dev/null || true
    printf '\n' >&2
    if [ -z "$value" ]; then
      if [ "$required" = 1 ]; then
        echo "$name must not be empty" >&2
        exit 1
      fi
      rm -f "$file"
      echo "Removed $name" >&2
      exit 0
    fi
    printf %s "$value" | encrypt
    echo "Stored $name in $file" >&2
    ;;
  lookup)
    if [ ! -f "$file" ]; then
      if [ "$required" = 1 ]; then
        echo "$name is not stored; run make set-api-key" >&2
        exit 1
      fi
      exit 0
    fi
    # Ask the system credential service over Varlink rather than running systemd-creds decrypt:
    # rootless Podman runs this driver in its user namespace, where the user appears as root, so
    # systemd-creds would open the TPM directly and be denied. The service identifies the caller by
    # its real UID, which a user namespace does not change.
    blob=$(tr -d ' \n\r' < "$file")
    varlinkctl call /run/systemd/io.systemd.Credentials io.systemd.Credentials.Decrypt \
      "{\"name\":\"$name\",\"blob\":\"$blob\",\"scope\":\"user\"}" |
      python3 -I -c 'import base64, json, sys; sys.stdout.buffer.write(base64.b64decode(json.load(sys.stdin)["data"]))'
    ;;
  migrate-keyring)
    if ! value=$(secret-tool lookup application strata-local name "$name" 2>/dev/null) || [ -z "$value" ]; then
      echo "$name: not found in an unlocked keyring; skipped" >&2
      exit 0
    fi
    printf %s "$value" | encrypt
    echo "Migrated $name to $file" >&2
    ;;
  *)
    echo "Unknown action: $action" >&2
    exit 2
    ;;
esac
