#!/usr/bin/env bash
# Zero-shot retrieval comparison on one region (default Alps): EarthLoc as the
# baseline, then RemoteCLIP and Qwen3-VL-Embedding. Retrieval recall only, no
# SIFT+LightGlue matching. Results land in output/results/*.json.
#
#   scripts/run_zero_shot_alps.sh                       # all three
#   scripts/run_zero_shot_alps.sh remoteclip            # just one
#   SMOKE=1 scripts/run_zero_shot_alps.sh remoteclip    # minutes-long plumbing check first
#   REGION="Toshka Lakes" PYTHON=~/venv/bin/python USER_CONFIG=my_paths.yaml scripts/run_zero_shot_alps.sh
#
# Follows the shared GPU workstation rules (README "Running on the shared GPU
# workstation"): nothing is installed or downloaded (HF offline mode), a
# retriever whose dependencies or weights are missing is skipped with a note,
# and every GPU run goes through scripts/gpu_queue.py, which waits for an idle
# GPU and never touches other people's processes.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
REGION="${REGION:-Alps}"
USER_CONFIG="${USER_CONFIG:-user_config.yaml}"
EXTRA_ARGS=()
[[ "${SMOKE:-0}" == "1" ]] && EXTRA_ARGS+=(--smoke)
if [[ $# -gt 0 ]]; then RETRIEVERS=("$@"); else RETRIEVERS=(earthloc remoteclip qwen3vl_embedding); fi

# Never fetch models or code from the Hub; missing weights must fail, not download.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

mkdir -p output/logs
for retriever in "${RETRIEVERS[@]}"; do
  log="output/logs/$(date +%Y%m%d-%H%M%S)_${REGION// /_}_${retriever}.log"
  echo "=== ${retriever} on ${REGION} (log: ${log}) ==="
  if ! "$PYTHON" scripts/check_env.py --user-config "$USER_CONFIG" --retriever "$retriever" > "${log}.preflight" 2>&1; then
    echo "skipping ${retriever}: preflight failed, see ${log}.preflight"
    grep '\[!!\]' "${log}.preflight" || true
    continue
  fi
  # ${arr[@]+...} keeps an empty EXTRA_ARGS from tripping set -u on older bash.
  if ! "$PYTHON" scripts/gpu_queue.py -- \
      "$PYTHON" scripts/evaluate.py --user-config "$USER_CONFIG" --region "$REGION" --retriever "$retriever" --skip-matching \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$log"; then
    echo "${retriever} failed, see ${log}; continuing with the next retriever"
  fi
done
