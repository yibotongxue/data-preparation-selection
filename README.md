# 数据选择框架

基于 DataFlow/DataFlex 的 LLM 训练数据选择框架，提供多种数据选择策略的统一接口。

## 使用

```bash
# 安装依赖
uv sync

# 运行示例
uv run python main.py

# 运行所有测试
uv run pytest

# 运行单个测试
uv run pytest tests/test_random_selection.py

# 类型检查
uv run --with pyright pyright

# 运行 pre-commit
uv run pre-commit run --all-files
```

## 环境管理

使用 `uv add` 添加依赖，不要手动编辑 `pyproject.toml` 或使用 `pip`。

```bash
uv add <package-name>
```

## 架构

### 目录结构

```
src/data_selection/
├── __init__.py          # 顶层 re-export
├── protocol.py          # SelectionMethod Protocol
├── utils.py             # extract_text 等工具函数
└── selectors/
    ├── __init__.py
    ├── random_selection.py              # 随机选择
    ├── source_balanced_random.py        # 按源均衡随机
    ├── length_based.py                  # 按长度选择
    ├── perplexity_based.py              # 困惑度 (DataFlow Kenlm)
    ├── embedding_similarity.py          # 嵌入相似度 (Near 算法)
    ├── deita_quality.py                 # Deita 质量×复杂度 (DataFlow)
    ├── quality_scorer.py                # FineWeb-Edu / PairQual (DataFlow)
    ├── diversity_kcenter.py             # 多样性 K-Center (TSDS 算法)
    └── llm_selector.py                  # LLM 打分 (DataFlow MetaScorer)
```

### SelectionMethod Protocol

所有选择器遵循统一的 `select(samples, k) -> list[dict]` 接口。策略参数通过 `__init__` 配置，不通过 `select` 传参。

### 输入格式

支持两种格式：
- `{"instruction": "...", "output": "..."}`
- `{"conversations": [{"messages": [{"content": "..."}]}]}`

### 依赖

| 库 | 用途 | 对应方法 |
|---|---|---|
| `numpy` | 矢量运算 | embedding_similarity, diversity_kcenter |
| `open-dataflow` | 质量打分器 | deita_quality, quality_scorer, perplexity_based, llm_selector |

## Pre-commit 检查

`isort` → `black-jupyter` → `autoflake` → `pyupgrade` → `bandit` → `pyright` → `codespell`

部分检查会直接修改文件，commit 失败后需要重新 `git add` 再 commit。
