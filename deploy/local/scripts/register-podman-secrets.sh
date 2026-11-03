#!/bin/sh
set -eu

if ! command -v systemd-creds >/dev/null 2>&1; then
  echo 'systemd-creds is required (systemd 256 or newer)' >&2
  exit 1
fi
if ! command -v podman >/dev/null 2>&1; then
  echo 'podman is required' >&2
  exit 1
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
driver=$script_dir/podman-secret-driver.sh

# A remote Podman cannot run this host's driver scripts, so decrypt here and store the
# values with the remote's default driver, replacing them on every run.
if [ -n "${CONTAINER_CONNECTION:-}${CONTAINER_HOST:-}" ]; then
  api_key=$("$script_dir/credentials.sh" lookup strata_api_key)
  if [ -z "$api_key" ]; then
    echo 'The Strata API key credential is empty' >&2
    exit 1
  fi
  hf_token=$("$script_dir/credentials.sh" lookup hf_token)
  # Podman rejects empty secrets; the trailing newline keeps an absent HF token nonempty,
  # and both readers strip it.
  printf '%s\n' "$api_key" | podman secret create --replace strata-local-api-key - >/dev/null
  printf '%s\n' "$hf_token" | podman secret create --replace strata-local-hf-token - >/dev/null
  exit 0
fi

# Fail before modifying Podman registration if the mandatory key is unavailable.
"$driver" check strata_api_key

register() {
  secret_name=$1
  credential_name=$2
  options="lookup=$driver lookup $credential_name,store=$driver store $credential_name,list=$driver list $credential_name,delete=$driver delete $credential_name"
  # Values are decrypted from the credential files at container creation, so registrations
  # do not need replacing when a credential changes. Avoid --replace's delete
  # path and preserve the existing opaque ID on repeated startup.
  printf %s registered | podman secret create --ignore --driver=shell \
    --driver-opts "$options" "$secret_name" - >/dev/null
}

register strata-local-api-key strata_api_key
register strata-local-hf-token hf_token
