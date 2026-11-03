#!/bin/sh
set -eu

action=${1:?Expected a secret driver action}
name=${2:?Expected a credential name}
state_dir=${XDG_DATA_HOME:-$HOME/.local/share}/strata-local/podman-secret-ids
credentials=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)/credentials.sh

case "$name" in
  strata_api_key|hf_token) ;;
  *) echo "Unknown credential: $name" >&2; exit 2 ;;
esac

case "$action" in
  lookup|store)
    case "${SECRET_ID:-}" in
      ''|*[!0-9a-fA-F]*) echo 'Invalid Podman secret ID' >&2; exit 2 ;;
    esac
    ;;
esac

case "$action" in
  lookup|check)
    if [ "$action" = lookup ] && [ ! -f "$state_dir/$SECRET_ID" ]; then
      echo 'Podman secret is not registered' >&2
      exit 1
    fi
    # An absent HF token is valid for public models; credentials.sh fails for a missing API key.
    value=$("$credentials" lookup "$name") || {
      echo "Failed to decrypt the $name credential" >&2
      exit 1
    }
    if [ "$name" = strata_api_key ] && [ -z "$value" ]; then
      echo 'The Strata API key credential is empty' >&2
      exit 1
    fi
    if [ "$action" = lookup ]; then
      printf %s "$value"
    fi
    ;;
  store)
    # Podman requires store during registration. The credential file remains the
    # source of truth; discard the nonsecret registration marker from stdin.
    cat >/dev/null
    install -d -m 700 "$state_dir"
    umask 077
    : > "$state_dir/$SECRET_ID"
    ;;
  list)
    if [ -d "$state_dir" ]; then
      find "$state_dir" -maxdepth 1 -type f -printf '%f\n'
    fi
    ;;
  delete)
    # Deleting Podman registration must not erase the credential.
    case "${SECRET_ID:-}" in
      ''|*[!0-9a-fA-F]*) : ;;
      *) rm -f "$state_dir/$SECRET_ID" ;;
    esac
    ;;
  *)
    echo "Unknown secret driver action: $action" >&2
    exit 2
    ;;
esac
