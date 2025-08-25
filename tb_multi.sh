#!/usr/bin/env bash
set -euo pipefail

# Usage: tb_multi.sh /path/to/root_dir [--port 6006]
# It supports two layouts:
#  a) root_dir/training_*/logs/tensorboard_logs
#  b) root_dir/logs/tensorboard_logs

ROOT_DIR="${1:-}"
PORT=6006

if [[ -z "${ROOT_DIR}" ]]; then
  echo "Usage: $0 /path/to/root_dir [--port PORT]"
  exit 1
fi

shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="${2:-}"
      if [[ -z "${PORT}" ]]; then
        echo "Error: --port needs a value"
        exit 1
      fi
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if ! command -v tensorboard >/dev/null 2>&1; then
  echo "Error: tensorboard not found in PATH"
  exit 1
fi

# Collect candidate log dirs
shopt -s nullglob
declare -a CANDIDATES=()

# Case (a): multiple training_* folders
for d in "$ROOT_DIR"/training_*/logs/tensorboard_logs; do
  [[ -d "$d" ]] && CANDIDATES+=("$d")
done

# Case (b): single logs folder at root
if [[ ${#CANDIDATES[@]} -eq 0 && -d "$ROOT_DIR/logs/tensorboard_logs" ]]; then
  CANDIDATES+=("$ROOT_DIR/logs/tensorboard_logs")
fi

# Fallback: find any directories that actually contain TF event files
if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  while IFS= read -r -d '' p; do
    CANDIDATES+=("$p")
  done < <(find "$ROOT_DIR" -type f -name 'events.out.tfevents.*' -printf '%h\0' | sort -zu)
fi

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "No TensorBoard logs found under: $ROOT_DIR"
  exit 1
fi

# Build --logdir_spec with readable aliases
# Prefer the training_* folder name as alias if available; otherwise use the leaf directory name.
LOGDIR_SPEC=""
for p in "${CANDIDATES[@]}"; do
  # alias = training_* folder if it exists in the path
  alias="$(echo "$p" | awk -F/ '{for(i=1;i<=NF;i++){if($i ~ /^training_[0-9]+$/){print $i; exit}}}')" || true
  if [[ -z "$alias" ]]; then
    alias="$(basename "$(dirname "$p")")_$(basename "$p")"
  fi
  # Append to spec (comma-separated)
  if [[ -z "$LOGDIR_SPEC" ]]; then
    LOGDIR_SPEC="${alias}:${p}"
  else
    LOGDIR_SPEC="${LOGDIR_SPEC},${alias}:${p}"
  fi
done

echo "Found ${#CANDIDATES[@]} log dir(s)."
echo "Launching TensorBoard on port ${PORT} with:"
echo "  --logdir_spec ${LOGDIR_SPEC}"
echo

# Use --logdir_spec so aliases show up in the UI
exec tensorboard --port "${PORT}" --logdir_spec "${LOGDIR_SPEC}"
