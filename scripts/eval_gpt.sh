#!/usr/bin/env bash

set -x
set -e

DIR="$(cd "$(dirname "$0")" && cd .. && pwd)"
echo "working directory: ${DIR}"

NAME='Qwen7B-FT'
MODEL_NAME_OR_PATH="/mnt/lustre/tangyang2/hjq/model/Qwen7B-gte/"
MAX_LENGTH=512
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${NAME}"

export PYTHONPATH=/mnt/lustre/tangyang2/hjq/mteb_benchmark_zh/:$PYTHONPATH

if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
	MODEL_NAME_OR_PATH=$1
	shift
fi

if [ -z "$OUTPUT_DIR" ]; then
	OUTPUT_DIR="tmp-outputs/"
fi

mkdir -p "${OUTPUT_DIR}"

python -u eval_Qwen.py \
	--model-name-or-path "${MODEL_NAME_OR_PATH}" \
	--pool-type last \
	--prefix-type query_or_passage \
	--max-length $MAX_LENGTH \
	--output-dir "${OUTPUT_DIR}" "$@"
echo "done"
