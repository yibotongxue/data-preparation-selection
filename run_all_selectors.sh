#!/usr/bin/env bash
# run_all_selectors.sh
# 并行运行所有 selector（除 LLMAsSelector 和 Composite）。
# 支持多个输入文件，所有文件 × 所有方法全部并发跑（GPU 的排队执行）。
# Score-based 选择器传入所有 k 值一次完成，random 类每个 k 独立跑。
# 如果输出文件已存在则跳过该任务。
#
# 用法:
#   bash run_all_selectors.sh              # 默认跑 1000, 10000, 100000
#   bash run_all_selectors.sh 1            # debug 模式，k=1
#   bash run_all_selectors.sh 1000 5000 10000   # 指定多个 k 值

set -euo pipefail

PROJECT_DIR="/jizhicfs/linyibo/data-preparation-selection"
OUTPUT_DIR="$PROJECT_DIR/data"

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

# 解析命令行参数：支持传入多个 k 值
if [[ $# -gt 0 ]]; then
    K_VALUES=("$@")
else
    K_VALUES=(1000 10000 100000)
fi

echo "K values: ${K_VALUES[*]}"
echo "========================================="

# --- 输入文件列表 ---
INPUT_FILES=(
    "/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl"
    "/jizhicfs/linyibo/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k/WizardLM_evol_instruct_V2_143k.jsonl"
    "/jizhicfs/linyibo/datasets/TsinghuaC3I/UltraMedical/UltraMedical.jsonl"
    "/jizhicfs/linyibo/datasets/Josephgflowers/Finance-Instruct-500k/Finance-Instruct-500k.jsonl"
    "/jizhicfs/linyibo/datasets/MegaScience/MegaScience/MegaScience.jsonl"
)

# 从文件路径提取短名称作为输出前缀（去掉 .jsonl 后缀的文件名）
get_short_name() {
    local fpath="$1"
    basename "$fpath" .jsonl
}

# 根据数据集短名称设置对应的键名参数
get_key_args() {
    local short_name="$1"
    case "$short_name" in
        Finance-Instruct-500k)
            echo "--instruction-key user --output-key assistant"
            ;;
        MegaScience)
            echo "--instruction-key question --output-key answer"
            ;;
        *)
            echo ""
            ;;
    esac
}

# embedding_similarity 的领域 proxy（NEAR query 目标集），按数据集映射
PROXY_DIR="$PROJECT_DIR/proxy"
get_proxy() {
    local short_name="$1"
    case "$short_name" in
        scalequest_math)                 echo "$PROXY_DIR/math_proxy.jsonl" ;;
        MegaScience)                     echo "$PROXY_DIR/science_proxy.jsonl" ;;
        UltraMedical)                    echo "$PROXY_DIR/medical_proxy.jsonl" ;;
        Finance-Instruct-500k)           echo "$PROXY_DIR/finance_proxy.jsonl" ;;
        WizardLM_evol_instruct_V2_143k)  echo "$PROXY_DIR/general_proxy.jsonl" ;;
        *)                               echo "" ;;
    esac
}

# 日志目录和缓存目录
LOG_DIR="$OUTPUT_DIR/logs"
CACHE_DIR="$OUTPUT_DIR/score_cache"
mkdir -p "$LOG_DIR" "$CACHE_DIR"

# Score-based 选择器：传入多个 k，一次计算，输出用 {k} 模板
# Random-based 选择器：每个 k 独立跑
RANDOM_SELECTORS=(random source_balanced_random)
SCORE_CPU_SELECTORS=(length_based perplexity_based)
SCORE_GPU_SELECTORS=(deita_quality quality_scorer embedding_similarity diversity_kcenter)

PIDS=()
SKIPPED=0

# --- 辅助函数：检查 score-based 选择器对某文件是否所有 k 的输出都已存在 ---
all_outputs_exist() {
    local prefix="$1"
    local sel="$2"
    for k in "${K_VALUES[@]}"; do
        local f="$OUTPUT_DIR/output_${prefix}_${sel}_${k}.jsonl"
        if [[ ! -f "$f" ]]; then
            return 1
        fi
    done
    return 0
}

# debug 模式判断
MAX_K=0
for k in "${K_VALUES[@]}"; do
    if [[ $k -gt $MAX_K ]]; then MAX_K=$k; fi
done

# --- 遍历所有输入文件，启动 Random 和 CPU 任务（全部并发） ---
for INPUT_FILE in "${INPUT_FILES[@]}"; do
    SHORT_NAME=$(get_short_name "$INPUT_FILE")
    KEY_ARGS=$(get_key_args "$SHORT_NAME")
    echo ""
    echo ">>> Processing: $SHORT_NAME ($INPUT_FILE)"
    [[ -n "$KEY_ARGS" ]] && echo "    Key args: $KEY_ARGS"

    ACTUAL_INPUT="$INPUT_FILE"
    if [[ $MAX_K -le 100 ]]; then
        DEBUG_INPUT="$OUTPUT_DIR/debug_input_${SHORT_NAME}_100.jsonl"
        head -100 "$INPUT_FILE" > "$DEBUG_INPUT"
        ACTUAL_INPUT="$DEBUG_INPUT"
        echo "[DEBUG] Using 100-line input: $ACTUAL_INPUT"
    fi

    INPUT_ARGS="--input $ACTUAL_INPUT"

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
                $KEY_ARGS \
                --output "$OUTPUT_FILE" \
                2>&1 | tee "$LOG_DIR/${SHORT_NAME}_${sel}_${k}.log" &
            PIDS+=($!)
        done
    done

    # --- Score-based CPU 选择器：所有输出都存在则跳过，否则一次传入所有 k ---
    for sel in "${SCORE_CPU_SELECTORS[@]}"; do
        if all_outputs_exist "$SHORT_NAME" "$sel"; then
            echo "[SKIP] All outputs for ${SHORT_NAME}/${sel} exist"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        CUDA_VISIBLE_DEVICES="" uv run select \
            --config "configs.${sel}" \
            --k ${K_VALUES[*]} \
            $INPUT_ARGS \
            $KEY_ARGS \
            --cache-dir "$CACHE_DIR" \
            --output "$OUTPUT_DIR/output_${SHORT_NAME}_${sel}_{k}.jsonl" \
            2>&1 | tee "$LOG_DIR/${SHORT_NAME}_${sel}.log" &
        PIDS+=($!)
    done
done

# --- Score-based GPU 选择器：round-robin 分配 GPU，同一 GPU 的任务顺序执行 ---
NUM_GPUS=8
# 为每个 GPU 构建命令队列
declare -a GPU_CMDS
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_CMDS[$i]=""
done

gpu_id=0
for INPUT_FILE in "${INPUT_FILES[@]}"; do
    SHORT_NAME=$(get_short_name "$INPUT_FILE")
    KEY_ARGS=$(get_key_args "$SHORT_NAME")

    ACTUAL_INPUT="$INPUT_FILE"
    if [[ $MAX_K -le 100 ]]; then
        ACTUAL_INPUT="$OUTPUT_DIR/debug_input_${SHORT_NAME}_100.jsonl"
    fi

    for sel in "${SCORE_GPU_SELECTORS[@]}"; do
        if all_outputs_exist "$SHORT_NAME" "$sel"; then
            echo "[SKIP] All outputs for ${SHORT_NAME}/${sel} exist"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        # embedding_similarity 使用领域 proxy 作为 NEAR query 目标集
        QUERY_ARGS=""
        if [[ "$sel" == "embedding_similarity" ]]; then
            PROXY_FILE=$(get_proxy "$SHORT_NAME")
            if [[ -n "$PROXY_FILE" ]]; then
                QUERY_ARGS="--query-path $PROXY_FILE --query-key query"
            fi
        fi
        CMD="CUDA_VISIBLE_DEVICES=$gpu_id uv run select --config configs.${sel} --k ${K_VALUES[*]} --input $ACTUAL_INPUT $KEY_ARGS $QUERY_ARGS --cache-dir $CACHE_DIR --output $OUTPUT_DIR/output_${SHORT_NAME}_${sel}_{k}.jsonl 2>&1 | tee $LOG_DIR/${SHORT_NAME}_${sel}.log"
        if [[ -z "${GPU_CMDS[$gpu_id]}" ]]; then
            GPU_CMDS[$gpu_id]="$CMD"
        else
            GPU_CMDS[$gpu_id]="${GPU_CMDS[$gpu_id]} ; $CMD"
        fi
        gpu_id=$(( (gpu_id + 1) % NUM_GPUS ))
    done
done

# 每个 GPU 的命令队列作为一个后台任务
for ((i=0; i<NUM_GPUS; i++)); do
    if [[ -n "${GPU_CMDS[$i]}" ]]; then
        echo "[GPU $i] Queue: ${GPU_CMDS[$i]}"
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
echo "GPUs: $NUM_GPUS (round-robin assignment)"
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
ls -lh "$OUTPUT_DIR"/output_*.jsonl 2>/dev/null || true
