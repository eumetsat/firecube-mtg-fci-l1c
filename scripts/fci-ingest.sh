#!/usr/bin/env bash
# Copyright 2025-2026 EUMETSAT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# fci-ingest.sh — parallel slot-range ingestion for MTG FCI L1C (FDHSI/HRFI).
#
# Fans out disjoint slot ranges across N pods into one preallocated Zarr store,
# exactly like the OPERA-SEVIRI flow. FCI specifics baked in:
#   * one 10-min repeat cycle = one slot:
#     slot = (timestamp_utc - time_epoch_utc_midnight) / cadence
#   * a pod writes ALL resolution groups for its range -> NO --slot-group
#   * every pod gets the SAME --input-data; the plugin's filter_items_to_slot_range
#     keeps only the ZIPs whose slot falls in [slot_start, slot_end)
#   * mode="floor": split on whole-slot boundaries only (never split a cycle)
#
# Two-phase: (0) generate shared geo grids once + preallocate the axis, then
# (1) fan out ingest pods with force_reingest (idempotent; written slots no-op).
#
# The AXIS (time_epoch + time_slots|time_end) can be as large as you want — it
# fixes the preallocated store shape. The WINDOW (FROM/TO or SLOT_START/SLOT_END)
# is only what THIS run ingests. Grow the window later without re-preallocating.
#
# LOGS: every run writes to /root/logs/fci-ingest-<timestamp>-<pid>/ —
# run.log is the full terminal transcript (review with `less -R` instead of
# scrolling screen/tmux), pod_<start>_<end>.log is each pod's firecube output,
# results.txt the ok/FAIL lines. Override the location with LOGDIR or LOG_ROOT.
set -euo pipefail

# ---- logging ---------------------------------------------------------------
LOG_ROOT="${LOG_ROOT:-/root/logs}"
LOGDIR="${LOGDIR:-$LOG_ROOT/fci-ingest-$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/run.log") 2>&1
RUN_T0=$(date +%s)
fmt_dur() { local s=$1; printf '%dh%02dm%02ds' $((s/3600)) $((s%3600/60)) $((s%60)); }

# ---- what to ingest --------------------------------------------------------
PLUGIN="${PLUGIN:-mtg_fci_l1c}"
PRODUCT_TYPE="${PRODUCT_TYPE:-FDHSI}"                 # FDHSI | HRFI
INPUT="${INPUT:-/data/fci-zips}"                      # dir of FCI L1C .zip files (same for all pods)
TARGET="${TARGET:-s3://mtg-fci-l1c.zarr/}"            # product store URI (s3:// or file://)
PRODUCT_NAME="${PRODUCT_NAME:-}"                      # logical name; default = target basename

# ---- the slot axis (fixes the preallocated store shape) --------------------
TIME_EPOCH="${TIME_EPOCH:?set TIME_EPOCH=YYYY-MM-DD (slot-index anchor, UTC midnight)}"
TIME_SLOTS="${TIME_SLOTS:-}"                          # axis length in slots  (takes precedence)
TIME_END="${TIME_END:-}"                              # OR axis end YYYY-MM-DD ((end-epoch).days*144)

# ---- the window to ingest now (pick ONE way) -------------------------------
FROM="${FROM:-}"                                      # e.g. 2024-08-01T06:00  (inclusive)
TO="${TO:-}"                                          # e.g. 2024-08-01T16:00  (exclusive)
SLOT_START="${SLOT_START:-}"                          # OR explicit slot indices
SLOT_END="${SLOT_END:-}"

# ---- fan-out shape ---------------------------------------------------------
SLOTS_PER_POD="${SLOTS_PER_POD:-6}"                   # 6 = 1 hour/pod
PARALLELISM="${PARALLELISM:-8}"                       # pods running at once

# ---- storage / behavior ----------------------------------------------------
WRITE_MODE="${WRITE_MODE:-direct}"
STORAGE_DRIVER="${STORAGE_DRIVER:-fsspec}"
STORAGE_TYPE="${STORAGE_TYPE:-}"                      # inferred from TARGET scheme if empty
FORCE_REINGEST="${FORCE_REINGEST:-1}"                # 1 = overwrite + idempotent re-run
RESOLUTIONS="${RESOLUTIONS:-}"                        # optional subset, e.g. "1km" or "500m,1km"
FIRECUBE="${FIRECUBE:-firecube}"
ASSUME_YES="${ASSUME_YES:-0}"

# ---- shared geolocation grids (avoid every pod recomputing 200MB-4GB) ------
GRIDS_FILE="${GRIDS_FILE:-}"                          # path to a shared .npz; empty = on-the-fly
GEN_GRIDS="${GEN_GRIDS:-0}"                           # 1 = generate GRIDS_FILE once if missing
DO_PREALLOCATE="${DO_PREALLOCATE:-1}"                 # 1 = run preallocate before fan-out

# ===========================================================================
[[ -n "$STORAGE_TYPE" ]] || case "$TARGET" in
  s3://*)   STORAGE_TYPE=s3 ;;
  file://*) STORAGE_TYPE=local ;;
  *)        STORAGE_TYPE=local ;;
esac
if [[ -z "$PRODUCT_NAME" ]]; then
  PRODUCT_NAME="$(basename "${TARGET%/}")"
fi

# axis option shared by preallocate AND every pod (schema must match the store)
if [[ -n "$TIME_SLOTS" ]]; then
  AXIS_OPT=(--option "time_slots=$TIME_SLOTS")
  AXIS_LEN="$TIME_SLOTS"
elif [[ -n "$TIME_END" ]]; then
  AXIS_OPT=(--option "time_end=$TIME_END")
  AXIS_LEN="$(python3 -c "import datetime as d,sys; e=d.date.fromisoformat('$TIME_EPOCH'); print((d.date.fromisoformat('$TIME_END')-e).days*144)")"
else
  echo "ERROR: set TIME_SLOTS=N or TIME_END=YYYY-MM-DD (the axis length)." >&2; exit 2
fi

# resolve the window -> integer [SLOT_START, SLOT_END)
slot_of() {  # $1 = ISO datetime -> slot index
  python3 - "$TIME_EPOCH" "$1" <<'PY'
import sys, datetime as d
epoch = d.date.fromisoformat(sys.argv[1])
t = d.datetime.fromisoformat(sys.argv[2])
print((t.date()-epoch).days*144 + t.hour*6 + t.minute//10)
PY
}
if [[ -z "$SLOT_START" || -z "$SLOT_END" ]]; then
  if [[ -n "$FROM" && -n "$TO" ]]; then
    SLOT_START="$(slot_of "$FROM")"
    SLOT_END="$(slot_of "$TO")"
  else
    SLOT_START=0
    SLOT_END="$AXIS_LEN"       # default: the whole axis
  fi
fi
if (( SLOT_START < 0 || SLOT_END > AXIS_LEN || SLOT_START >= SLOT_END )); then
  echo "ERROR: window [$SLOT_START,$SLOT_END) invalid or outside axis [0,$AXIS_LEN)." >&2; exit 2
fi

# grids: default resolutions from product type
if [[ -z "$RESOLUTIONS" ]]; then
  case "$PRODUCT_TYPE" in FDHSI) GRIDS_RES="1km,2km" ;; HRFI) GRIDS_RES="500m,1km" ;; *) GRIDS_RES="1km,2km" ;; esac
else
  GRIDS_RES="$RESOLUTIONS"
fi

# assemble the shared --option list
COMMON_OPTS=(--option "product_type=$PRODUCT_TYPE" --option "time_epoch=$TIME_EPOCH" "${AXIS_OPT[@]}")
[[ -n "$RESOLUTIONS" ]] && COMMON_OPTS+=(--option "resolutions=$RESOLUTIONS")
[[ -n "$GRIDS_FILE"  ]] && COMMON_OPTS+=(--option "fci_grids_file=$GRIDS_FILE")
FORCE_OPT=(); [[ "$FORCE_REINGEST" == "1" ]] && FORCE_OPT=(--option "force_reingest=true")

npods=$(( (SLOT_END - SLOT_START + SLOTS_PER_POD - 1) / SLOTS_PER_POD ))
cat <<EOF
============================================================
 plugin        : $PLUGIN   ($PRODUCT_TYPE)
 target        : $TARGET   (product=$PRODUCT_NAME, $STORAGE_TYPE)
 input (shared): $INPUT
 axis          : epoch=$TIME_EPOCH  length=$AXIS_LEN slots  (${AXIS_OPT[*]})
 window        : slots [$SLOT_START,$SLOT_END)  = $((SLOT_END-SLOT_START)) slots
 fan-out       : $SLOTS_PER_POD slots/pod -> $npods pods, $PARALLELISM in parallel
 grids         : ${GRIDS_FILE:-on-the-fly}   force_reingest=$FORCE_REINGEST
 logs          : $LOGDIR
============================================================
EOF
if [[ "$ASSUME_YES" != "1" ]]; then
  read -rp ">> proceed? [y/N] " a; case "${a:-N}" in y|Y|yes|YES) ;; *) echo "Aborted."; exit 0 ;; esac
fi

# ---- phase 0: shared grids + preallocate (once) ---------------------------
if [[ "$GEN_GRIDS" == "1" && -n "$GRIDS_FILE" && ! -f "$GRIDS_FILE" ]]; then
  echo ">> generating shared geo grids ($GRIDS_RES) -> $GRIDS_FILE"
  t0=$(date +%s)
  "$FIRECUBE" plugins "$PLUGIN" geo generate --resolutions "$GRIDS_RES" --output "$GRIDS_FILE"
  echo ">> grids generated in $(( $(date +%s) - t0 ))s"
fi
if [[ "$DO_PREALLOCATE" == "1" ]]; then
  echo ">> preallocating $TARGET to $AXIS_LEN slots (idempotent) ..."
  t0=$(date +%s)
  "$FIRECUBE" zarr preallocate "$PLUGIN" --product-name "$PRODUCT_NAME" --target "$TARGET" \
    --storage-type "$STORAGE_TYPE" --storage-driver "$STORAGE_DRIVER" --write-mode "$WRITE_MODE" \
    --input-data "$INPUT" "${COMMON_OPTS[@]}"
  echo ">> preallocate done in $(( $(date +%s) - t0 ))s"
fi

# ---- phase 1: fan out ingest pods over disjoint slot ranges ----------------
echo ">> ingesting $npods pod(s), $PARALLELISM parallel  (logs: $LOGDIR)"
export FIRECUBE PLUGIN INPUT TARGET STORAGE_TYPE STORAGE_DRIVER WRITE_MODE LOGDIR
export COMMON_STR="${COMMON_OPTS[*]}" FORCE_STR="${FORCE_OPT[*]:-}"

# emit "start end" lines for each disjoint chunk of the window
gen_ranges() {
  python3 - "$SLOT_START" "$SLOT_END" "$SLOTS_PER_POD" <<'PY'
import sys
s, e, step = map(int, sys.argv[1:4])
for a in range(s, e, step):
    print(a, min(a + step, e))
PY
}

FAN_T0=$(date +%s)
gen_ranges | xargs -P "$PARALLELISM" -n2 bash -c '
  s=$1; e=$2
  log="$LOGDIR/pod_${s}_${e}.log"
  t0=$(date +%s)
  # word-split COMMON_STR / FORCE_STR intentionally (they are pre-tokenized --option pairs)
  if "$FIRECUBE" ingest "$PLUGIN" \
       --input-data "$INPUT" --target "$TARGET" \
       --storage-type "$STORAGE_TYPE" --storage-driver "$STORAGE_DRIVER" --write-mode "$WRITE_MODE" \
       $COMMON_STR $FORCE_STR \
       --slot-start "$s" --slot-end "$e" > "$log" 2>&1; then
    echo "ok   [$s,$e)  $(( $(date +%s) - t0 ))s"
  else
    echo "FAIL [$s,$e)  $(( $(date +%s) - t0 ))s  -> $log"
  fi
' _ | tee "$LOGDIR/results.txt"

nfail=$(grep -c "^FAIL" "$LOGDIR/results.txt" || true)
FAN_ELAPSED=$(( $(date +%s) - FAN_T0 ))
RUN_ELAPSED=$(( $(date +%s) - RUN_T0 ))
echo "============================================================"
echo ">> fan-out: $((npods - nfail))/$npods pod(s) OK in $(fmt_dur "$FAN_ELAPSED") (${FAN_ELAPSED}s)"
echo ">> total run time: $(fmt_dur "$RUN_ELAPSED") (${RUN_ELAPSED}s)"
echo ">> logs: $LOGDIR  (full transcript: run.log)"
if [[ "$nfail" -eq 0 ]]; then
  echo ">> all $npods pod(s) OK. Window [$SLOT_START,$SLOT_END) ingested."
else
  echo ">> $nfail pod(s) FAILED. Re-run is safe (written slots no-op):"
  grep '^FAIL' "$LOGDIR/results.txt" | sed 's/^/   /'
  echo "   If a FAIL is a stale claim/run from a crash, clear it like the OPERA flow:"
  echo "     firecube chunks claims list --product-name $TARGET"
  echo "     firecube chunks runs list  --product-name $TARGET --status started"
fi
exit $(( nfail > 0 ? 1 : 0 ))
