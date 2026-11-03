#!/bin/sh
# Keep config/ and the strata-local-config volume (mounted at /config) in step.
#
#   config-volume.sh push   copy config/ into the volume (make config, run by make up)
#   config-volume.sh pull   copy the volume's run configs back into config/ (make pull-config)
#
# The web page's Config view edits the volume's files. Each push also keeps what it copied in the volume's .pushed/,
# so a later push can tell a file the web page changed (it differs from .pushed/) from one changed here; it refuses to
# overwrite such a file unless config/ already has the same content (pulled), or FORCE=1. Files removed from config/
# stay in the volume.
set -eu
VOLUME=strata-local-config
cd "$(dirname "$0")/.."
tmp=$(mktemp -d "${TMPDIR:-/tmp}/strata-config.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

podman volume create --ignore "$VOLUME" >/dev/null
mkdir "$tmp/volume"
podman volume export "$VOLUME" | tar -x -C "$tmp/volume" 2>/dev/null || true

# the volume's run configs the web page changed since the last push (no .pushed/ yet: a volume from before the web
# page could write it, so none)
web_changed() {
  [ -d "$tmp/volume/.pushed" ] || return 0
  for f in "$tmp"/volume/*.json; do
    [ -f "$f" ] || continue
    name=$(basename "$f")
    cmp -s "$f" "$tmp/volume/.pushed/$name" || echo "$name"
  done
}

case "${1:-}" in
push)
  conflicts=""
  for name in $(web_changed); do
    cmp -s "$tmp/volume/$name" "config/$name" || conflicts="$conflicts $name"
  done
  if [ -n "$conflicts" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "The web page's Config view changed these files in the volume since the last make config:$conflicts" >&2
    echo "Run make pull-config to keep those changes (then review them with git diff), or make config FORCE=1 to" >&2
    echo "replace them with config/." >&2
    exit 1
  fi
  mkdir -p "$tmp/stage/.pushed"
  cp -R config/. "$tmp/stage/"
  cp config/*.json "$tmp/stage/.pushed/"
  tar -C "$tmp/stage" -c --owner=0 --group=0 . | podman volume import "$VOLUME" -
  ;;
pull)
  # only what the web page changed; a file also changed here since the last push is left alone unless FORCE=1
  n=0 both=""
  for name in $(web_changed); do
    f="$tmp/volume/$name"
    cmp -s "$f" "config/$name" && continue
    if [ -f "config/$name" ] && ! cmp -s "config/$name" "$tmp/volume/.pushed/$name" && [ "${FORCE:-0}" != 1 ]; then
      both="$both $name"
      continue
    fi
    cp "$f" "config/$name"
    echo "config/$name"
    n=$((n + 1))
  done
  echo "$n file(s) copied from the volume; review them with git diff" >&2
  if [ -n "$both" ]; then
    echo "Changed both here and in the web page since the last make config, left as they are:$both" >&2
    echo "make pull-config FORCE=1 takes the web page's versions." >&2
    exit 1
  fi
  ;;
*)
  echo "usage: $0 push|pull" >&2
  exit 2
  ;;
esac
