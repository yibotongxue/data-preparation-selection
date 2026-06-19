#!/usr/bin/env bash
# run_rerun_emb_and_lawyer.sh
# 统一跑两批任务：
#   (A) 5 个英文数据集的 embedding_similarity —— 用新的领域 proxy 重跑（覆盖旧的错配数学 proxy 结果）
#   (B) lawyer 数据集的所有 selector
#
# GPU 任务按预估耗时「快→慢」排序后 round-robin 到可用 GPU；
# lawyer 的 CPU 类 selector(random/source_balanced/length/perplexity) 并行后台跑，不占 GPU。
#
# 预估 GPU 耗时（候选全量打分，与 k 无关）：
#   lawyer/embedding(23k)~1m, lawyer/diversity~2m, lawyer/quality~2m,
#   WizardLM/embedding(143k)~6m, lawyer/deita(2x7B)~10m,
#   UltraMedical/embedding(410k)~17m, Finance/embedding(518k)~22m,
#   scalequest/embedding(1.0M)~42m, MegaScience/embedding(1.25M)~52m
#
# 用法:
#   bash run_rerun_emb_and_lawyer.sh                  # 默认 GPU 0-7
#   GPUS="0,2,5" bash run_rerun_emb_and_lawyer.sh     # 自定义可用 GPU 列表

set -euo pipefail

PROJECT_DIR="/jizhicfs/linyibo/data-preparation-selection"
OUTPUT_DIR="$PROJECT_DIR/data"
PROXY_DIR="$PROJECT_DIR/proxy"
LOG_DIR="$OUTPUT_DIR/logs"
CACHE_DIR="$OUTPUT_DIR/score_cache"
LAWYER_FILE="/jizhicfs/linyibo/lawyer-llama/data/lawyer_llama_all.jsonl"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$CACHE_DIR"
cd "$PROJECT_DIR"

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
    GPU_LIST=(1 3 4 6 7)
fi
NUM_GPUS=${#GPU_LIST[@]}
[[ $NUM_GPUS -eq 0 ]] && { echo "[ERROR] No GPU specified."; exit 1; }
echo "Available GPUs: ${GPU_LIST[*]} (count=$NUM_GPUS)"

ENG_K="1000 10000 100000"     # 英文大数据集
LAW_K="1000 10000"            # lawyer 仅 23476 条，100k 无意义

PIDS=()

# ============================================================
# (B-CPU) lawyer 的 CPU 类 selector：并行后台，不占 GPU
# ============================================================
echo ">>> 启动 lawyer CPU 类 selector (random/source_balanced/length/perplexity)"
for k in $LAW_K; do
    for sel in random source_balanced_random; do
        OUT="$OUTPUT_DIR/output_lawyer_llama_${sel}_${k}.jsonl"
        [[ -f "$OUT" ]] && { echo "[SKIP] $OUT"; continue; }
        CUDA_VISIBLE_DEVICES="" uv run select --config "configs.${sel}" --k "$k" \
            --input "$LAWYER_FILE" --output "$OUT" \
            2>&1 | tee "$LOG_DIR/lawyer_llama_${sel}_${k}.log" &
        PIDS+=($!)
    done
done
for sel in length_based perplexity_based; do
    CUDA_VISIBLE_DEVICES="" uv run select --config "configs.${sel}" --k $LAW_K \
        --input "$LAWYER_FILE" --cache-dir "$CACHE_DIR" \
        --output "$OUTPUT_DIR/output_lawyer_llama_${sel}_{k}.jsonl" \
        2>&1 | tee "$LOG_DIR/lawyer_llama_${sel}.log" &
    PIDS+=($!)
done

# ============================================================
# GPU 任务表：已按预估耗时「快→慢」排序
# 字段: CONFIG | INPUT | SHORT | KVALS | KEYARGS | QUERY_PROXY
#   QUERY_PROXY 为空表示该 selector 不需要 proxy
# ============================================================
GPU_TASKS=(
  # 1. lawyer embedding (最快)
  "embedding_similarity|$LAWYER_FILE|lawyer_llama|$LAW_K||law_proxy.jsonl"
  # 2. lawyer diversity
  "diversity_kcenter|$LAWYER_FILE|lawyer_llama|$LAW_K||"
  # 3. lawyer quality
  "quality_scorer|$LAWYER_FILE|lawyer_llama|$LAW_K||"
  # 4. WizardLM embedding (143k)
  "embedding_similarity|/jizhicfs/linyibo/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k/WizardLM_evol_instruct_V2_143k.jsonl|WizardLM_evol_instruct_V2_143k|$ENG_K||general_proxy.jsonl"
  # 5. lawyer deita (2x7B)
  "deita_quality|$LAWYER_FILE|lawyer_llama|$LAW_K||"
  # 6. UltraMedical embedding (410k)
  "embedding_similarity|/jizhicfs/linyibo/datasets/TsinghuaC3I/UltraMedical/UltraMedical.jsonl|UltraMedical|$ENG_K||medical_proxy.jsonl"
  # 7. Finance embedding (518k)
  "embedding_similarity|/jizhicfs/linyibo/datasets/Josephgflowers/Finance-Instruct-500k/Finance-Instruct-500k.jsonl|Finance-Instruct-500k|$ENG_K|--instruction-key user --output-key assistant|finance_proxy.jsonl"
  # 8. scalequest embedding (1.0M)
  "embedding_similarity|/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl|scalequest_math|$ENG_K||math_proxy.jsonl"
  # 9. MegaScience embedding (1.25M, 最慢)
  "embedding_similarity|/jizhicfs/linyibo/datasets/MegaScience/MegaScience/MegaScience.jsonl|MegaScience|$ENG_K|--instruction-key question --output-key answer|science_proxy.jsonl"
)

# 为每个 GPU 构建命令队列
declare -a GPU_CMDS
for ((i=0; i<NUM_GPUS; i++)); do GPU_CMDS[$i]=""; done

slot=0
for task in "${GPU_TASKS[@]}"; do
    IFS='|' read -r CONFIG INPUT SHORT KVALS KEYARGS PROXY <<< "$task"

    QUERY_ARGS=""
    if [[ -n "$PROXY" ]]; then
        QUERY_ARGS="--query-path $PROXY_DIR/$PROXY --query-key query"
    fi

    gpu_id=${GPU_LIST[$slot]}
    CMD="CUDA_VISIBLE_DEVICES=$gpu_id uv run select --config configs.${CONFIG} --k ${KVALS} --input $INPUT $KEYARGS $QUERY_ARGS --cache-dir $CACHE_DIR --output $OUTPUT_DIR/output_${SHORT}_${CONFIG}_{k}.jsonl 2>&1 | tee $LOG_DIR/${SHORT}_${CONFIG}.log"

    if [[ -z "${GPU_CMDS[$slot]}" ]]; then
        GPU_CMDS[$slot]="$CMD"
    else
        GPU_CMDS[$slot]="${GPU_CMDS[$slot]} ; $CMD"
    fi
    slot=$(( (slot + 1) % NUM_GPUS ))
done

echo ""
echo ">>> 启动 GPU 任务队列 (按快->慢顺序 round-robin)"
for ((i=0; i<NUM_GPUS; i++)); do
    if [[ -n "${GPU_CMDS[$i]}" ]]; then
        echo "[GPU ${GPU_LIST[$i]}] Queue: ${GPU_CMDS[$i]}"
        eval "${GPU_CMDS[$i]}" &
        PIDS+=($!)
    fi
done

# ============================================================
# 等待
# ============================================================
echo ""
echo "========================================="
echo "Launched ${#PIDS[@]} task groups. GPUs: ${GPU_LIST[*]}"
FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
done
echo "========================================="
if [[ $FAILED -eq 0 ]]; then
    echo "ALL DONE! Results in: $OUTPUT_DIR/"
else
    echo "COMPLETED with $FAILED failure(s). Check $LOG_DIR/"
fi
echo "========================================="