#!/usr/bin/env bash
set -euo pipefail

# tb_multi.sh — Launch TensorBoard over multiple experiment folders with aliases.
#
# Usage:
#   tb_multi.sh [--mode auto|training|trial|roots] [--port PORT] ROOT_DIR [ROOT_DIR...]

PORT=6006
MODE="auto"
ROOT_DIRS=()

usage () {
  cat <<'EOF'
Usage: tb_multi.sh [--mode auto|training|trial|roots] [--port PORT] ROOT_DIR [ROOT_DIR...]

Modes:
  training : Use ROOT_DIR/training_*/logs/tensorboard_logs
  trial    : Use ROOT_DIR/trial_*/logs/tensorboard_logs
  roots    : Use each ROOT_DIR/logs/tensorboard_logs
  auto     : Try training, then trial, then roots (default)
EOF
}

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    *) ROOT_DIRS+=("$1"); shift ;;
  esac
done

if ! command -v tensorboard >/dev/null 2>&1; then
  echo "Error: tensorboard not found in PATH" >&2
  exit 1
fi

if [[ ${#ROOT_DIRS[@]} -eq 0 ]]; then
  echo "Error: at least one ROOT_DIR is required." >&2
  usage
  exit 1
fi

shopt -s nullglob

declare -a CANDIDATES=()
declare -A ALIAS_COUNT=()

add_candidates_training () {
  local root="$1" d
  for d in "$root"/training_*/logs/tensorboard_logs; do
    [[ -d "$d" ]] && CANDIDATES+=("$d")
  done
}
add_candidates_trial () {
  local root="$1" d
  for d in "$root"/trial_*/logs/tensorboard_logs; do
    [[ -d "$d" ]] && CANDIDATES+=("$d")
  done
}
add_candidates_roots () {
  local root d
  for root in "${ROOT_DIRS[@]}"; do
    d="$root/logs/tensorboard_logs"
    [[ -d "$d" ]] && CANDIDATES+=("$d")
  done
}
add_candidates_find_events () {
  local root="$1"
  while IFS= read -r -d '' p; do
    CANDIDATES+=("$p")
  done < <(find "$root" -type f -name 'events.out.tfevents.*' -printf '%h\0' | sort -zu)
}

# Collect candidates
case "$MODE" in
  training) add_candidates_training "${ROOT_DIRS[0]}" ;;
  trial)    add_candidates_trial    "${ROOT_DIRS[0]}" ;;
  roots)    add_candidates_roots ;;
  auto)
    add_candidates_training "${ROOT_DIRS[0]}"
    [[ ${#CANDIDATES[@]} -eq 0 ]] && add_candidates_trial "${ROOT_DIRS[0]}"
    [[ ${#CANDIDATES[@]} -eq 0 ]] && add_candidates_roots
    ;;
  *) echo "Error: unknown --mode '$MODE' (use auto|training|trial|roots)" >&2; exit 1 ;;
esac

# Fallback
[[ ${#CANDIDATES[@]} -eq 0 ]] && add_candidates_find_events "${ROOT_DIRS[0]}"

# De-duplicate exact paths
if [[ ${#CANDIDATES[@]} -gt 0 ]]; then
  declare -A seen=()
  declare -a unique=()
  for p in "${CANDIDATES[@]}"; do
    if [[ -z "${seen[$p]+x}" ]]; then
      seen["$p"]=1
      unique+=("$p")
    fi
  done
  CANDIDATES=("${unique[@]}")
fi

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "No TensorBoard logs found for: ${ROOT_DIRS[*]}" >&2
  exit 1
fi

# Short stable hash for suffixing aliases on collision
short_hash () {
  local s="$1"
  if command -v md5sum >/dev/null 2>&1; then
    printf "%s" "$s" | md5sum | awk '{print substr($1,1,6)}'
  elif command -v shasum >/dev/null 2>&1; then
    printf "%s" "$s" | shasum -a 256 | awk '{print substr($1,1,6)}'
  else
    # Fallback: length + basename
    echo "${#s}$(basename "$s" | tr -cd '[:alnum:]' | cut -c1-4)"
  fi
}

# Build readable, unique aliases
make_alias () {
  local path="$1"
  local alias=""

  # Normalize path without trailing slash
  path="${path%/}"

  # 1) Prefer training_* or trial_* folder in the path
  alias="$(echo "$path" | awk -F/ '{for(i=1;i<=NF;i++){if($i ~ /^(training|trial)_[0-9]+$/){print $i; exit}}}')"

  # 2) If it's .../logs/tensorboard_logs, use the grandparent (experiment dir)
  if [[ -z "$alias" && "$path" =~ /logs/tensorboard_logs$ ]]; then
    # grandparent of tensorboard_logs
    alias="$(basename "$(dirname "$(dirname "$path")")")"
  fi

  # 3) Fallback: last 2 meaningful components (skip "logs" and "tensorboard_logs")
  if [[ -z "$alias" ]]; then
    local leaf="$(basename "$path")"            # tensorboard_logs
    local parent="$(basename "$(dirname "$path")")" # logs
    local grandparent="$(basename "$(dirname "$(dirname "$path")")")"
    if [[ "$parent" == "logs" && "$leaf" == "tensorboard_logs" && -n "$grandparent" ]]; then
      alias="$grandparent"
    else
      alias="${parent}_${leaf}"
    fi
  fi

  # Ensure uniqueness: append short hash if already used
  if [[ -n "${ALIAS_COUNT[$alias]+x}" ]]; then
    local h
    h="$(short_hash "$path")"
    alias="${alias}-${h}"
  fi
  ALIAS_COUNT[$alias]=1

  printf "%s" "$alias"
}

# Build --logdir_spec
LOGDIR_SPEC=""
for p in "${CANDIDATES[@]}"; do
  alias="$(make_alias "$p")"
  if [[ -z "$LOGDIR_SPEC" ]]; then
    LOGDIR_SPEC="${alias}:${p}"
  else
    LOGDIR_SPEC+=",${alias}:${p}"
  fi
done

echo "Found ${#CANDIDATES[@]} log dir(s)."
echo "Launching TensorBoard on port ${PORT} with:"
echo "  --logdir_spec ${LOGDIR_SPEC}"
echo

exec tensorboard --port "${PORT}" --logdir_spec "${LOGDIR_SPEC}"
