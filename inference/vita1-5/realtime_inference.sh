#!/usr/bin/env bash
set -euo pipefail

# === Configurable options ===
BASE="/your/path/store/data"
IN_GLOB="$BASE/user_backchannel/*/input.wav"   # Different categories of data paths use wildcards.
SERVER="https://0.0.0.0:8082"         # Your service address/port


# Tip: Before running, you can refresh once to ensure only one frontend connection remains (optional)
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
    echo "[WARN] 800: Not exactly one active connection, trying reload and retry once..."
    curl -k -s -X POST "$SERVER/admin/reload_all" >/dev/null || true
    sleep 2
    code="$(simulate_once "$wav" "$dst")"
  fi

  if [[ "$code" != "200" ]]; then
    echo "[ERR] /simulate_stream returned $code, skipping: $wav"
    continue
  fi

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
  curl -k -s -X POST "$SERVER/admin/reset_all" >/dev/null
  sleep 1
done

echo "[DONE] All finished. Results have been written back to each subdirectory."
