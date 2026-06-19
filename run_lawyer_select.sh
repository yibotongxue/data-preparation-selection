#!/usr/bin/env bash
# run_lawyer_select.sh
# 只针对 lawyer-llama 汇总数据集 (lawyer_llama_all.jsonl) 跑所有 selector。
# 支持自定义可用 GPU 列表。
#
# 数据格式: {"conversations": [...], "source": "<原文件名>"}，共 23476 条。
# - conversations / source 均为 select 框架默认字段，无需额外 key 参数。
# - source 字段已按来源 json 文件设置，source_balanced_random 可正常发挥作用。
#
# 用法:
#   bash run_lawyer_select.sh                      # 默认 k=1000 10000
#   bash run_lawyer_select.sh 500 1000 5000        # 指定多个 k 值
#   GPUS="0,2,5" bash run_lawyer_select.sh         # 自定义可用 GPU 列表
#   GPUS="0,2,5" bash run_lawyer_select.sh 1000 10000
#
# 注意: lawyer_llama_all.jsonl 共 23476 条，k 超过该值时按全量截断。

set -euo pipefail

PROJECT_DIR="/jizhicfs/linyibo/data-preparation-selection"
OUTPUT_DIR="$PROJECT_DIR/data"
INPUT_FILE="/jizhicfs/linyibo/lawyer-llama/data/lawyer_llama_all.jsonl"
SHORT_NAME="lawyer_llama"

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR"

# Ctrl-C 时杀掉所有子进程
cleanup() {
    echo ""
    echo "[INTERRUPTED] Killing all child processes..."
    kill -- -$$ 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

# ============================================================
# 可用 GPU 列表
#   - 默认使用 0-7
#   - 可通过环境变量 GPUS 覆盖，例如: GPUS="0,2,5" bash run_lawyer_select.sh
#   - 也可直接修改下面的默认值
# ============================================================
if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$GPUS"
else
    GPU_LIST=(1 3 4 6 7)
fi
NUM_GPUS=${#GPU_LIST[@]}
if [[ $NUM_GPUS -eq 0 ]]; then
    echo "[ERROR] No GPU specified. Set GPUS or edit GPU_LIST."
    exit 1
fi
echo "Available GPUs: ${GPU_LIST[*]} (count=$NUM_GPUS)"

# 解析命令行参数：支持传入多个 k 值
if [[ $# -gt 0 ]]; then
    K_VALUES=("$@")
else
    K_VALUES=(1000 10000)
fi
echo "K values: ${K_VALUES[*]}"
echo "Input: $INPUT_FILE"
echo "========================================="

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "[ERROR] Input file not found: $INPUT_FILE"
    exit 1
fi

# 日志目录和缓存目录
LOG_DIR="$OUTPUT_DIR/logs"
CACHE_DIR="$OUTPUT_DIR/score_cache"
mkdir -p "$LOG_DIR" "$CACHE_DIR"

# Random-based 选择器：每个 k 独立跑
# Score-based CPU 选择器：CUDA_VISIBLE_DEVICES="" ，一次传入所有 k
# Score-based GPU 选择器：round-robin 分配到 GPU_LIST，同 GPU 任务顺序执行
RANDOM_SELECTORS=(random source_balanced_random)
SCORE_CPU_SELECTORS=(length_based perplexity_based)
SCORE_GPU_SELECTORS=(deita_quality quality_scorer embedding_similarity diversity_kcenter)

PIDS=()
SKIPPED=0

# --- 辅助函数：检查 score-based 选择器对该文件是否所有 k 的输出都已存在 ---
all_outputs_exist() {
    local sel="$1"
    for k in "${K_VALUES[@]}"; do
        local f="$OUTPUT_DIR/output_${SHORT_NAME}_${sel}_${k}.jsonl"
        if [[ ! -f "$f" ]]; then
            return 1
        fi
    done
    return 0
}

INPUT_ARGS="--input $INPUT_FILE"

# --- Random 类选择器：每个 k 独立并行，已存在则跳过 ---
for sel in "${RANDOM_SELECTORS[@]}"; do
    for k in "${K_VALUES[@]}"; do
        OUTPUT_FILE="$OUTPUT_DIR/output_${SHORT_NAME}_${sel}_${k}.jsonl"
        if [[ -f "$OUTPUT_FILE" ]]; then
            echo "[SKIP] $OUTPUT_FILE exists"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        CUDA_VISIBLE_DEVICES="" uv run select \
            --config "configs.${sel}" \
            --k "$k" \
            $INPUT_ARGS \
            --output "$OUTPUT_FILE" \
            2>&1 | tee "$LOG_DIR/${SHORT_NAME}_${sel}_${k}.log" &
        PIDS+=($!)
    done
done

# --- Score-based CPU 选择器：所有输出都存在则跳过，否则一次传入所有 k ---
for sel in "${SCORE_CPU_SELECTORS[@]}"; do
    if all_outputs_exist "$sel"; then
        echo "[SKIP] All outputs for ${SHORT_NAME}/${sel} exist"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    CUDA_VISIBLE_DEVICES="" uv run select \
        --config "configs.${sel}" \
        --k ${K_VALUES[*]} \
        $INPUT_ARGS \
        --cache-dir "$CACHE_DIR" \
        --output "$OUTPUT_DIR/output_${SHORT_NAME}_${sel}_{k}.jsonl" \
        2>&1 | tee "$LOG_DIR/${SHORT_NAME}_${sel}.log" &
    PIDS+=($!)
done

# --- Score-based GPU 选择器：round-robin 分配到 GPU_LIST ---
declare -a GPU_CMDS
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_CMDS[$i]=""
done

slot=0
for sel in "${SCORE_GPU_SELECTORS[@]}"; do
    if all_outputs_exist "$sel"; then
        echo "[SKIP] All outputs for ${SHORT_NAME}/${sel} exist"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    gpu_id=${GPU_LIST[$slot]}
    # embedding_similarity 使用法律领域 proxy 作为 NEAR query 目标集
    QUERY_ARGS=""
    if [[ "$sel" == "embedding_similarity" ]]; then
        PROXY_FILE="$PROJECT_DIR/proxy/law_proxy.jsonl"
        if [[ -f "$PROXY_FILE" ]]; then
            QUERY_ARGS="--query-path $PROXY_FILE --query-key query"
        fi
    fi
    CMD="CUDA_VISIBLE_DEVICES=$gpu_id uv run select --config configs.${sel} --k ${K_VALUES[*]} $INPUT_ARGS $QUERY_ARGS --cache-dir $CACHE_DIR --output $OUTPUT_DIR/output_${SHORT_NAME}_${sel}_{k}.jsonl 2>&1 | tee $LOG_DIR/${SHORT_NAME}_${sel}.log"
    if [[ -z "${GPU_CMDS[$slot]}" ]]; then
        GPU_CMDS[$slot]="$CMD"
    else
        GPU_CMDS[$slot]="${GPU_CMDS[$slot]} ; $CMD"
    fi
    slot=$(( (slot + 1) % NUM_GPUS ))
done

# 每个 GPU 的命令队列作为一个后台任务
for ((i=0; i<NUM_GPUS; i++)); do
    if [[ -n "${GPU_CMDS[$i]}" ]]; then
        echo "[GPU ${GPU_LIST[$i]}] Queue: ${GPU_CMDS[$i]}"
        eval "${GPU_CMDS[$i]}" &
        PIDS+=($!)
    fi
done

# ============================================================
# 等待所有任务完成
# ============================================================
echo ""
echo "========================================="
echo "Launched ${#PIDS[@]} tasks, skipped $SKIPPED"
echo "GPUs: ${GPU_LIST[*]}"
echo ""

FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================="
if [[ $FAILED -eq 0 ]]; then
    echo "ALL DONE! Results saved to: $OUTPUT_DIR/"
else
    echo "COMPLETED with $FAILED failure(s). Check logs in $LOG_DIR/"
fi
echo "========================================="
ls -lh "$OUTPUT_DIR"/output_${SHORT_NAME}_*.jsonl 2>/dev/null || true
