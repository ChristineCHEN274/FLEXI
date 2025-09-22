#!/usr/bin/env bash
set -euo pipefail

BASE="your/data/root/path/"
IN_GLOB="$BASE/model_backchannel/*/input.wav"   # Different categories of data paths use wildcards.
SERVER="https://0.0.0.0:7000"     # The server address and port after you deployment

# Tip: refresh the frontend once before running to ensure exactly one active connection
# curl -k -s -X POST "$SERVER/admin/reload_all" >/dev/null || true
# sleep 5

simulate_once () {
  local wav="$1"
  local dst="$2"
  curl -k -s -o /dev/null -w "%{http_code}" \
    -m 600 \
    -X POST "$SERVER/simulate_stream" \
    -H "Content-Type: application/json" \
    -d "{\"wav_path\":\"${wav}\", \"out_wav\":\"${dst}\", \"wait\": true}"
}

shopt -s nullglob
for wav in $IN_GLOB; do
  [[ -f "$wav" ]] || continue
  subdir="$(dirname "$wav")"
  dst="$subdir/output.wav"   

  echo "[INFO] simulate_stream -> $wav"
  code="$(simulate_once "$wav" "$dst")"

  if [[ "$code" == "800" ]]; then
    echo "[WARN] 800: Not exactly one active connection; trying reload and retry once..."
    curl -k -s -X POST "$SERVER/admin/reload_all" >/dev/null || true
    sleep 2
    code="$(simulate_once "$wav" "$dst")"
  fi

  if [[ "$code" != "200" ]]; then
    echo "[ERR] /simulate_stream returned $code, skipping: $wav"
    continue
  fi

  # Validate output: file exists and size > 0
  if [[ ! -s "$dst" ]]; then
    echo "[WARN] Output is empty or missing: $dst; waiting 2s and checking again..."
    sleep 2
    if [[ ! -s "$dst" ]]; then
      echo "[WARN] Still empty: $dst; marking with .empty for review"
      : > "${dst}.empty"
      continue
    fi
  fi

  bytes=$(stat -c%s "$dst" 2>/dev/null || echo 0)
  echo "[OK] Saved output: $dst (${bytes} bytes)"

  sleep 1
done

echo "[DONE] All finished. Results written to each subdirectory."
