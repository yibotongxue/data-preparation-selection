#!/usr/bin/env bash
# run_all_and_rerun.sh
# 整合 run_all_selectors.sh + run_rerun_emb_and_lawyer.sh：
#   一次性对 6 个数据集（5 个英文 + 1 个 lawyer）跑全部 selector。
#   - random / CPU 打分类（length / perplexity）：CPU 并行后台跑，不占 GPU
#   - GPU 打分类（deita / quality / embedding / diversity）：按数据集「快->慢」展开后 round-robin 到可用 GPU
#   - embedding_similarity 使用对应领域 proxy 作为 NEAR query 目标集
#
# 不删除任何文件：所有 selector 都保留「输出已存在则跳过」逻辑，
# 已经生成的 output 文件会被复用，不会被覆盖或删除。
#
# 用法:
#   bash run_all_and_rerun.sh                       # 默认 GPU 0-7
#   GPUS="0,2,5" bash run_all_and_rerun.sh          # 自定义可用 GPU 列表
#   ENG_K="1000 10000" bash run_all_and_rerun.sh    # 覆盖英文数据集 k 值
#   LAW_K="1000" bash run_all_and_rerun.sh          # 覆盖 lawyer k 值

set -euo pipefail

PROJECT_DIR="/jizhicfs/linyibo/data-preparation-selection"
OUTPUT_DIR="$PROJECT_DIR/data"
PROXY_DIR="$PROJECT_DIR/proxy"
LOG_DIR="$OUTPUT_DIR/logs"
CACHE_DIR="$OUTPUT_DIR/score_cache"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$CACHE_DIR"
cd "$PROJECT_DIR"

# Ctrl-C 时杀掉所有子进程
cleanup() {
    echo ""
    echo "[INTERRUPTED] Killing all child processes..."
    kill -- -$$ 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

# ---------- 可用 GPU 列表 ----------
if [[ -n "${GPUS:-}" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$GPUS"
else
    GPU_LIST=(0 1 2 3 4 5 6 7)
fi
NUM_GPUS=${#GPU_LIST[@]}
[[ $NUM_GPUS -eq 0 ]] && { echo "[ERROR] No GPU specified."; exit 1; }
echo "Available GPUs: ${GPU_LIST[*]} (count=$NUM_GPUS)"

# ---------- k 值 ----------
ENG_K="${ENG_K:-1000 10000 100000}"   # 英文大数据集
LAW_K="${LAW_K:-1000 10000}"          # lawyer 仅 23476 条，100k 无意义
echo "ENG_K: $ENG_K"
echo "LAW_K: $LAW_K"

# ---------- 数据集表（已按候选量「快->慢」排序）----------
# 字段: SHORT | INPUT | KVALS | KEYARGS | PROXY
LAWYER_FILE="/jizhicfs/linyibo/lawyer-llama/data/lawyer_llama_all.jsonl"
DATASETS=(
  "lawyer_llama|$LAWYER_FILE|$LAW_K||law_proxy.jsonl"
  "WizardLM_evol_instruct_V2_143k|/jizhicfs/linyibo/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k/WizardLM_evol_instruct_V2_143k.jsonl|$ENG_K||general_proxy.jsonl"
  "UltraMedical|/jizhicfs/linyibo/datasets/TsinghuaC3I/UltraMedical/UltraMedical.jsonl|$ENG_K||medical_proxy.jsonl"
  "Finance-Instruct-500k|/jizhicfs/linyibo/datasets/Josephgflowers/Finance-Instruct-500k/Finance-Instruct-500k.jsonl|$ENG_K|--instruction-key user --output-key assistant|finance_proxy.jsonl"
  "scalequest_math|/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl|$ENG_K||math_proxy.jsonl"
  "MegaScience|/jizhicfs/linyibo/datasets/MegaScience/MegaScience/MegaScience.jsonl|$ENG_K|--instruction-key question --output-key answer|science_proxy.jsonl"
)

RANDOM_SELECTORS=(random source_balanced_random)
SCORE_CPU_SELECTORS=(length_based perplexity_based)
# GPU selector 顺序：embedding/diversity/quality 较快，deita(2x7B) 最慢放最后
SCORE_GPU_SELECTORS=(embedding_similarity diversity_kcenter quality_scorer deita_quality)

PIDS=()
SKIPPED=0

# 检查某 selector 对该数据集是否所有 k 的输出都已存在
all_outputs_exist() {
    local short="$1"; local sel="$2"; shift 2
    local k
    for k in $1; do
        [[ -f "$OUTPUT_DIR/output_${short}_${sel}_${k}.jsonl" ]] || return 1
    done
    return 0
}

# ============================================================
# (1) CPU 类 selector：random + CPU 打分，全部并行后台跑，不占 GPU
# ============================================================
echo ""
echo ">>> 启动 CPU 类 selector (random / source_balanced_random / length / perplexity)"
for ds in "${DATASETS[@]}"; do
    IFS='|' read -r SHORT INPUT KVALS KEYARGS PROXY <<< "$ds"

    # random 类：每个 k 独立
    for sel in "${RANDOM_SELECTORS[@]}"; do
        for k in $KVALS; do
            OUT="$OUTPUT_DIR/output_${SHORT}_${sel}_${k}.jsonl"
            if [[ -f "$OUT" ]]; then
                echo "[SKIP] $OUT"
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
            CUDA_VISIBLE_DEVICES="" uv run select \
                --config "configs.${sel}" --k "$k" \
                --input "$INPUT" $KEYARGS \
                --output "$OUT" \
                2>&1 | tee "$LOG_DIR/${SHORT}_${sel}_${k}.log" &
            PIDS+=($!)
        done
    done

    # CPU 打分类：一次传入所有 k
    for sel in "${SCORE_CPU_SELECTORS[@]}"; do
        if all_outputs_exist "$SHORT" "$sel" "$KVALS"; then
            echo "[SKIP] All outputs for ${SHORT}/${sel} exist"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        CUDA_VISIBLE_DEVICES="" uv run select \
            --config "configs.${sel}" --k $KVALS \
            --input "$INPUT" $KEYARGS \
            --cache-dir "$CACHE_DIR" \
            --output "$OUTPUT_DIR/output_${SHORT}_${sel}_{k}.jsonl" \
            2>&1 | tee "$LOG_DIR/${SHORT}_${sel}.log" &
        PIDS+=($!)
    done
done

# ============================================================
# (2) GPU 类 selector：展开为 (dataset × selector) 任务后 round-robin
# ============================================================
declare -a GPU_CMDS
for ((i=0; i<NUM_GPUS; i++)); do GPU_CMDS[$i]=""; done

slot=0
for ds in "${DATASETS[@]}"; do
    IFS='|' read -r SHORT INPUT KVALS KEYARGS PROXY <<< "$ds"
    for sel in "${SCORE_GPU_SELECTORS[@]}"; do
        if all_outputs_exist "$SHORT" "$sel" "$KVALS"; then
            echo "[SKIP] All outputs for ${SHORT}/${sel} exist"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        QUERY_ARGS=""
        if [[ "$sel" == "embedding_similarity" && -n "$PROXY" && -f "$PROXY_DIR/$PROXY" ]]; then
            QUERY_ARGS="--query-path $PROXY_DIR/$PROXY --query-key query"
        fi
        gpu_id=${GPU_LIST[$slot]}
        CMD="CUDA_VISIBLE_DEVICES=$gpu_id uv run select --config configs.${sel} --k ${KVALS} --input $INPUT $KEYARGS $QUERY_ARGS --cache-dir $CACHE_DIR --output $OUTPUT_DIR/output_${SHORT}_${sel}_{k}.jsonl 2>&1 | tee $LOG_DIR/${SHORT}_${sel}.log"
        if [[ -z "${GPU_CMDS[$slot]}" ]]; then
            GPU_CMDS[$slot]="$CMD"
        else
            GPU_CMDS[$slot]="${GPU_CMDS[$slot]} ; $CMD"
        fi
        slot=$(( (slot + 1) % NUM_GPUS ))
    done
done

echo ""
echo ">>> 启动 GPU 任务队列 (按快->慢 round-robin)"
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
echo "Launched ${#PIDS[@]} task groups, skipped $SKIPPED"
echo "GPUs: ${GPU_LIST[*]} (count=$NUM_GPUS)"
echo ""

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
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
