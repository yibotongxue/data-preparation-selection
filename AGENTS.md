# AGENTS.md

本文件面向 AI Coding Agent，说明该仓库的结构、命令、约定与注意事项。假设读者对项目一无所知。

## 1. 项目概述

- **项目名称**: `data-selection`
- **定位**: 面向 LLM 训练数据的数据选择框架，基于 DataFlow / DataFlex 生态，统一封装多种选择策略。
- **包名**: `data_selection`，源码位于 `src/data_selection/`。
- **Python 版本**: `>=3.12,<3.13`（见 `.python-version` 与 `pyproject.toml`）。
- **包管理器**: `uv`；构建后端为 `uv_build`。
- **入口**:
  - `main.py`：演示性入口，使用 `RandomSelector`。
  - `select` CLI：`pyproject.toml` 中注册的 console script，实际逻辑在 `src/data_selection/cli.py`。

## 2. 技术栈

| 类别 | 关键依赖 | 说明 |
|------|----------|------|
| 包管理 | `uv` | 必须用于安装、运行、添加依赖 |
| 配置解析 | `omegaconf` | `CustomOmegaConfig` 封装 `_target_` 延迟实例化 |
| DataFlow 生态 | `open-dataflow` | 提供各类 scorer（Deita、FineWeb-Edu、PairQual、Perplexity、MetaScorer） |
| DataFlex 生态 | `dataflex` (Git 依赖) | 提供 `offline_near_Selector` / `offline_tsds_Selector` |
| 数值/ML | `numpy`, `scikit-learn`, `sentence-transformers`, `torch`, `faiss-cpu` | 用于 Embedding / Diversity / 模型推理 |
| 数据处理 | `pandas` | 批量构造 scorer 输入 |
| 测试 | `pytest`, `unittest.mock` | 通过 MagicMock 注入 scorer 进行单元测试 |
| 类型检查 | `pyright` | pre-commit 与手动检查 |
| 代码风格 | `black`, `isort`, `autoflake`, `pyupgrade` | 由 pre-commit 统一执行 |
| 安全 | `bandit` | 配置见 `.bandit.yml` |
| 拼写 | `codespell` | pre-commit 最后一步 |

## 3. 仓库结构

```
.
├── pyproject.toml              # 包元数据、依赖、工具配置、console script
├── .python-version             # 3.12
├── .pre-commit-config.yaml     # pre-commit hook 流水线
├── .bandit.yml                 # bandit 扫描配置
├── main.py                     # 演示入口
├── run_*.sh                    # 批量执行脚本（见下文）
├── src/data_selection/         # 核心包
│   ├── __init__.py             # 顶层 re-export
│   ├── protocol.py             # Selector Protocol（结构子类型）
│   ├── config.py               # CustomOmegaConfig：带 _target_ 的延迟实例化
│   ├── dataset.py              # DatasetConfig：输入格式归一化
│   ├── utils.py                # read_jsonl / write_jsonl / extract_text
│   ├── types.py                # Meta TypedDict（meta 字段 schema）
│   ├── score_cache.py          # ScoreCache：可断点续跑的分数缓存
│   ├── cli.py                  # select 命令行入口
│   ├── scorers_truncated.py    # 显存友好的 Deita 子类 scorer
│   └── selectors/              # 选择器实现
│       ├── random_selection.py
│       ├── source_balanced_random.py
│       ├── length_based.py
│       ├── perplexity_based.py       # DataFlow Kenlm PPL
│       ├── embedding_similarity.py   # DataFlex NEAR
│       ├── diversity_kcenter.py      # DataFlex TSDS
│       ├── deita_quality.py          # DataFlow Deita quality × complexity
│       ├── deita_quality_ray.py      # Ray 并行版 Deita
│       ├── quality_scorer.py         # FineWeb-Edu / PairQual
│       ├── llm_selector.py           # DataFlow MetaScorer (LLM 打分)
│       ├── composite.py              # 多选择器链式组合
│       └── __init__.py
├── configs/                    # 每个选择器一个 Python 配置模块
│   ├── random.py
│   ├── source_balanced_random.py
│   ├── length_based.py
│   ├── perplexity_based.py
│   ├── perplexity_based_mid.py
│   ├── perplexity_based_high.py
│   ├── embedding_similarity.py
│   ├── diversity_kcenter.py
│   ├── deita_quality.py
│   ├── deita_quality_finance.py      # 使用 TruncatedDeitaScorer
│   ├── deita_quality_ray.py
│   ├── quality_scorer.py
│   ├── llm_selector.py
│   └── composite.py
├── tests/                      # pytest 测试
├── proxy/                      # 领域 proxy JSONL（NEAR query 目标集）
├── data/                       # 默认输出目录与缓存
│   ├── score_cache/            # 共享 score cache
│   ├── score_cache_ppl_strategy/ # perplexity mid/high 专用 cache
│   └── output_*.jsonl          # 选择结果
└── dataflow_cache/             # DataFlow 模型缓存
```

## 4. 核心架构约定

### 4.1 Selector Protocol

所有选择器遵循结构子类型（`typing.Protocol`），无需显式继承：

```python
class Selector(Protocol):
    def select(self, samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]: ...
```

- **策略参数（包括 `k`）一律在 `__init__` 中配置**，`select()` 只接收 `samples`。
- 返回的样本会附加 `"meta"` 字段，记录选择器名称、分数、种子等元信息（schema 见 `types.py`）。

### 4.2 输入格式

`DatasetConfig.format()` 将原始 JSONL 归一化为统一的 `messages` 格式：

```python
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

支持两种原始格式：

1. Alpaca 格式：`{"instruction": "...", "output": "..."}`（键名可通过 `instruction_key` / `output_key` 覆盖）。
2. Conversations 格式：`{"conversations": [...]}`，内部支持 `{"role", "content"}`、`{"from", "value"}` 或嵌套 `{"messages": [...]}`。

使用 `extract_text()` 可从 `messages` 中提取文本。

### 4.3 配置系统

`CustomOmegaConfig` 继承 `omegaconf.DictConfig`，支持通过 `_target_` 延迟实例化 Python 对象：

```python
CustomOmegaConfig.of(RandomSelector, k=100, seed=42)
```

每个 `configs/*.py` 模块暴露两个函数：

- `dataset()`：返回 `DatasetConfig` 配置（输入路径、输出路径、键名）。
- `selector()`：返回包含 `"selector"` 键的字典，值是 `CustomOmegaConfig`。

### 4.4 可断点续跑的分数缓存

对于计算昂贵的选择器（Deita、Quality、Perplexity、LLM、Ray 版 Deita），通过 `ScoreCache` 将每批分数以 JSONL 形式追加写入磁盘：

```jsonl
{"idx": 0, "scores": {"quality": 3.2, "complexity": 2.1}}
{"idx": 1, "scores": {"quality": 4.5, "complexity": 3.8}}
```

- 中断后重新运行会自动跳过已缓存样本。
- **缓存文件按 selector 类名命名**（如 `PerplexityBasedSelector_scores.jsonl`），跨数据集或跨策略复用同一缓存会串味；实际脚本会为不同数据集/策略使用独立 cache 目录。

## 5. 选择器速查

| 选择器 | 类型 | 依赖 | 关键参数 |
|--------|------|------|----------|
| `RandomSelector` | 随机 | 无 | `k`, `seed` |
| `SourceBalancedRandomSelector` | 随机+源均衡 | 无 | `k`, `source_key`, `seed` |
| `LengthBasedSelector` | 分数（文本长度） | 无 | `k` |
| `PerplexityBasedSelector` | 分数（PPL） | DataFlow Kenlm | `k`, `strategy` (`low`/`high`/`mid`), `scores_cache_path` |
| `EmbeddingSimilaritySelector` | 分数（相似度） | DataFlex NEAR | `k`, `query_path`/`domain_proxy_text`, `embed_model`, `max_chars` |
| `DiversityKCenterSelector` | 分数（多样性） | DataFlex TSDS | `k`, `seed`, `embed_model`, `sigma`, `alpha`, `max_chars` |
| `DeitaQualitySelector` | 分数（quality × complexity） | DataFlow Deita | `k`, `device`, `scores_cache_path` |
| `DeitaQualityRaySelector` | 分数（Ray 并行 Deita） | DataFlow + Ray | `k`, `replicas`, `num_gpus_per_replica`, `scores_cache_path` |
| `QualityScorerSelector` | 分数（FineWeb-Edu / PairQual） | DataFlow | `k`, `strategy` (`fineweb_edu`/`pairqual`/`composite`), `scores_cache_path` |
| `LLMAsSelector` | 分数（LLM 多维打分） | DataFlow MetaScorer | `k`, `llm_serving`, `dimensions` |
| `CompositeSelector` | 组合 | 其它 Selector | `selectors` 列表，`k` |

注意：`DeitaQualityRaySelector` 为实验性实现，依赖 `dataflow.rayorch` 与 `ray`，默认环境可能未安装。

## 6. 常用命令

```bash
# 安装依赖
uv sync

# 运行单个示例
uv run python main.py

# 通过 CLI 运行选择器
uv run select --config configs.random --k 100 --input data/input.jsonl --output data/output.jsonl

# 运行全部测试
uv run pytest

# 运行单个测试文件
uv run pytest tests/test_random_selection.py

# 类型检查
uv run --with pyright pyright

# 运行 pre-commit
uv run pre-commit run --all-files
```

### 6.1 `select` CLI 参数

```bash
uv run select \
  --config configs.deita_quality \
  --k 1000 10000 100000 \
  --input /path/to/input.jsonl \
  --output data/output_{k}.jsonl \
  --cache-dir data/score_cache \
  --instruction-key instruction \
  --output-key output \
  --conversations-key conversations \
  --query-path proxy/math_proxy.jsonl \
  --query-key query
```

- `--k` 支持多个值。对于 score-based 选择器，CLI 会取 `max(k)` 计算一次，再对每个 `k` 截断输出；对于 random-based 选择器，每个 `k` 独立运行。
- `--output` 中可使用 `{k}` 占位符；否则自动在扩展名前插入 `_k`。
- `--cache-dir` 会自动创建 `<SelectorName>_scores.jsonl` 缓存文件。

## 7. 批量执行脚本

项目根目录提供多个 Bash 脚本，用于在多 GPU 环境下批量跑实验：

| 脚本 | 用途 |
|------|------|
| `run_all_selectors.sh` | 对 5 个英文数据集跑全部 selector（除 LLM / Composite），CPU/GPU 任务并发调度 |
| `run_all_and_rerun.sh` | 整合 5 个英文数据集 + lawyer 数据集，支持 `GPUS`、`ENG_K`、`LAW_K` 环境变量 |
| `run_rerun_emb_and_lawyer.sh` | 重跑 embedding_similarity（用新 proxy）+ lawyer 全部 selector |
| `run_lawyer_select.sh` | 仅对 lawyer-llama 数据集跑全部 selector |
| `run_perplexity_mid_high.sh` | 对全部数据集跑 `perplexity_based_mid` 与 `perplexity_based_high` |

这些脚本默认写死数据集路径（`/jizhicfs/linyibo/datasets/...`）、输出到 `data/`、使用 8 卡或 `GPUS` 环境变量指定 GPU。它们都实现了「输出文件已存在则跳过」的幂等逻辑，并会捕获 `SIGINT/SIGTERM` 清理子进程。

## 8. 代码风格与提交前检查

- **格式化**: `black`（行宽 88，`target-version py312`）。
- **导入排序**: `isort`，profile 为 `black`，`src_paths = ["src", "tests"]`。
- **升级语法**: `pyupgrade --py312-plus`。
- **清理未使用导入/变量**: `autoflake`。
- **安全扫描**: `bandit`，配置 `.bandit.yml`：排除 `.venv`、`venv`、`build`、`dist`；中等级别；跳过 `B101`、`B311`、`B615`。
- **类型检查**: `pyright`。
- **拼写检查**: `codespell`。

pre-commit 流水线：

```
check-symlinks → destroyed-symlinks → trailing-whitespace → end-of-file-fixer → check-yaml → check-toml → check-ast → check-added-large-files → check-merge-conflict → check-executables-have-shebangs → check-shebang-scripts-are-executable → detect-private-key → debug-statements → isort → black-jupyter → pyupgrade → autoflake → bandit → pyright → codespell
```

> 部分 hook（如 `black`、`isort`、`autoflake`、`pyupgrade`）会原地修改文件。如果 commit 失败，需要重新 `git add` 后再 commit。

### 8.1 Python 代码约定

- 文件开头使用 `from __future__ import annotations`。
- 使用 Python 3.12 语法（如泛型 `class CustomOmegaConfig[T]`、`type MaybeConfig[T] = ...`）。
- 类型注解尽量完整。
- 选择器构造参数中，允许注入 scorer 对象（`MaybeConfig[T]`），便于测试时通过 `MagicMock` 替换。

## 9. 测试策略

- 框架：`pytest`。
- 测试文件：`tests/test_*.py`，每个选择器对应一个测试文件。
- Mock 策略：对依赖 DataFlow / DataFlex 的选择器，使用 `unittest.mock.MagicMock` 注入 scorer，避免加载真实模型。
- 基础边界用例覆盖：`k=0`、空输入、`k` 大于池大小、确定性种子等。

示例：

```python
mock_q = MagicMock()
mock_q.eval.return_value = [4.0, 2.0, 3.0]
mock_c = MagicMock()
mock_c.eval.return_value = [3.0, 1.0, 5.0]

result = DeitaQualitySelector(k=2, quality_scorer=mock_q, complexity_scorer=mock_c).select(samples)
```

## 10. 安全与运维注意事项

- **模型与数据路径**: 大量数据集路径硬编码为 `/jizhicfs/linyibo/...`，运行批量脚本前请确认这些路径在当前环境存在。
- **GPU 使用**: GPU 选择器默认 `device="cuda"`，批量脚本会设置 `CUDA_VISIBLE_DEVICES`。
- **缓存污染**: 不要在不同数据集或不同策略间共享同一个 `scores_cache_path`；脚本已通过独立 cache 目录规避。
- **API 密钥**: `LLMAsSelector` 通过 `APILLMServing_request` 调用外部 API，依赖环境变量 `DF_API_KEY`（或其它 `key_name_of_api_key` 指定变量）。
- **Ray 实验性**: `DeitaQualityRaySelector` 会拉起 Ray actor，调用后需 `shutdown()` 释放资源。
- **无正式部署流程**: 本项目没有 Dockerfile、CI/CD workflow 或服务化部署；使用方式是本地/集群通过 `uv run select` 或 `run_*.sh` 批量执行。

## 11. 给 Agent 的开发指南

1. **添加新选择器**:
   - 在 `src/data_selection/selectors/` 新建模块，实现 `select(samples)` 接口。
   - 在 `src/data_selection/selectors/__init__.py` 与 `src/data_selection/__init__.py` 中注册。
   - 在 `configs/` 增加对应配置模块。
   - 在 `tests/` 添加测试，复杂 scorer 用 MagicMock 注入。
   - 如果是 score-based 且计算昂贵，建议接入 `ScoreCache` 支持 `--cache-dir` 断点续跑。

2. **修改配置**:
   - 不要手动编辑 `pyproject.toml` 添加依赖，使用 `uv add <package>`。
   - 配置优先放在 `configs/*.py`，通过 `CustomOmegaConfig.of()` 构造。

3. **修改测试**:
   - 不要修改已有测试的断言逻辑；如需重构接口，保持原有测试行为不变。

4. **运行前检查**:
   - 修改后执行 `uv run pytest`、`uv run --with pyright pyright`、`uv run pre-commit run --all-files`。

5. **批量脚本**:
   - 除非明确需要生产数据结果，否则不要随意运行 `run_*.sh`，它们会占用多 GPU 并写入 `data/`。
