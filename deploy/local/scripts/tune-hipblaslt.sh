#!/bin/sh
# Calibrate a hipBLASLt table for this card and the image's hipBLASLt; the table goes to stdout, progress to stderr.
# The shapes are the 43 rows of upstream's gfx1100 hipBLASLt 1.4.1 table; every solution id is measured fresh.
set -eu
cd /opt/strata
cases=$(awk 'NR > 2 {printf " --case %s,%s,%s,%s,%s", $1, $5, $2, $3, $4}' tools/hip/gfx1100-hipblaslt-100401.txt)
# shellcheck disable=SC2086
engine/tune_hipblaslt $cases --tuning-out /tmp/table.txt >&2
cat /tmp/table.txt
