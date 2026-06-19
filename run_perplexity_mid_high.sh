#!/usr/bin/env bash
# run_perplexity_mid_high.sh
# 对全部 6 个数据集运行新增的两个 perplexity 选择策略：
#   - perplexity_based_mid  (strategy=mid,  取 PPL 最接近均值的样本)
#   - perplexity_based_high (strategy=high, 取 PPL 最高的样本)
#
# 说明:
#   - PPL 分数与策略无关，但 ScoreCache 仅按【类名】命名 (PerplexityBasedSelector_scores.jsonl)，
#     跨数据集共享同一缓存会串味。因此本脚本为【每个数据集】使用独立 cache 目录，
#     且不复用旧的共享缓存(那里可能已被其它数据集污染)。
#   - 同一数据集内先跑 mid(计算并缓存 PPL)，再跑 high(复用该数据集缓存，免重复打分)。
#   - 纯 CPU (Kenlm)，各数据集并行；输出已存在则跳过，不删除任何文件。
#
# 用法:
#   bash run_perplexity_mid_high.sh                     # 默认 k
#   ENG_K="1000 10000" LAW_K="1000" bash run_perplexity_mid_high.sh

set -euo pipefail

PROJECT_DIR="/jizhicfs/linyibo/data-preparation-selection"
OUTPUT_DIR="$PROJECT_DIR/data"
LOG_DIR="$OUTPUT_DIR/logs"
CACHE_ROOT="$OUTPUT_DIR/score_cache_ppl_strategy"   # 独立于旧共享缓存
mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$CACHE_ROOT"
cd "$PROJECT_DIR"

cleanup() {
    echo ""
    echo "[INTERRUPTED] Killing all child processes..."
    kill -- -$$ 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

ENG_K="${ENG_K:-1000 10000 100000}"   # 英文大数据集
LAW_K="${LAW_K:-1000 10000}"          # lawyer 仅 23476 条

# 数据集表: SHORT | INPUT | KVALS | KEYARGS
LAWYER_FILE="/jizhicfs/linyibo/lawyer-llama/data/lawyer_llama_all.jsonl"
DATASETS=(
  "lawyer_llama|$LAWYER_FILE|$LAW_K|"
  "WizardLM_evol_instruct_V2_143k|/jizhicfs/linyibo/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k/WizardLM_evol_instruct_V2_143k.jsonl|$ENG_K|"
  "UltraMedical|/jizhicfs/linyibo/datasets/TsinghuaC3I/UltraMedical/UltraMedical.jsonl|$ENG_K|"
  "Finance-Instruct-500k|/jizhicfs/linyibo/datasets/Josephgflowers/Finance-Instruct-500k/Finance-Instruct-500k.jsonl|$ENG_K|--instruction-key user --output-key assistant"
  "scalequest_math|/jizhicfs/linyibo/datasets/dyyyyyyyy/ScaleQuest-Math/scalequest_math.jsonl|$ENG_K|"
  "MegaScience|/jizhicfs/linyibo/datasets/MegaScience/MegaScience/MegaScience.jsonl|$ENG_K|--instruction-key question --output-key answer"
)

# 检查某策略对该数据集是否所有 k 的输出都已存在
all_outputs_exist() {
    local short="$1"; local label="$2"; local kvals="$3"
    local k
    for k in $kvals; do
        [[ -f "$OUTPUT_DIR/output_${short}_${label}_${k}.jsonl" ]] || return 1
    done
    return 0
}

PIDS=()

# 每个数据集一个后台任务：先 mid 后 high，共用该数据集独立 cache
for ds in "${DATASETS[@]}"; do
    IFS='|' read -r SHORT INPUT KVALS KEYARGS <<< "$ds"
    (
        DS_CACHE="$CACHE_ROOT/$SHORT"
        mkdir -p "$DS_CACHE"
        for variant in mid high; do
            LABEL="perplexity_based_${variant}"
            if all_outputs_exist "$SHORT" "$LABEL" "$KVALS"; then
                echo "[SKIP] all outputs exist: ${SHORT}/${LABEL}"
                continue
            fi
            echo ">>> [$SHORT] running $LABEL (k=$KVALS)"
            CUDA_VISIBLE_DEVICES="" uv run select \
                --config "configs.${LABEL}" \
                --k $KVALS \
                --input "$INPUT" $KEYARGS \
                --cache-dir "$DS_CACHE" \
                --output "$OUTPUT_DIR/output_${SHORT}_${LABEL}_{k}.jsonl" \
                2>&1 | tee "$LOG_DIR/${SHORT}_${LABEL}.log"
        done
    ) &
    PIDS+=($!)
done

echo ""
echo "========================================="
echo "Launched ${#PIDS[@]} dataset tasks (mid+high each)"

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
done

echo "========================================="
if [[ $FAILED -eq 0 ]]; then
    echo "ALL DONE! Results in: $OUTPUT_DIR/"
else
    echo "COMPLETED with $FAILED failure(s). Check logs in $LOG_DIR/"
fi
echo "========================================="
ls -lh "$OUTPUT_DIR"/output_*perplexity_based_mid_*.jsonl "$OUTPUT_DIR"/output_*perplexity_based_high_*.jsonl 2>/dev/null || true
