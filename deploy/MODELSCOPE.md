# EvoResearch @ ModelScope 免费 GPU

在 ModelScope 的免费 GPU Notebook 里跑 Qwen2.5-7B-Instruct（FP16，不量化）。

## 三步

1. **传代码**：把整个项目上传到 Notebook 工作目录，或
   ```
   !git clone https://github.com/Elio896/AutoResearch.git
   ```

2. **装环境**（一个 cell 里）：
   ```
   !bash deploy/modelscope_setup.sh
   ```

3. **跑研究**（另一个 cell）：
   ```
   python deploy/modelscope_run.py --csv data/demo/clean_balanced.csv --target species
   ```

## 三个场景

| 场景 | 命令 |
|---|---|
| 干净三分类 | `--csv data/demo/clean_balanced.csv --target species` |
| 缺失（正例少数） | `--csv data/demo/missing_informative.csv --target churned --metric f1_minority --pos-label 1` |
| 不平衡 | `--csv data/demo/imbalanced_fraud.csv --target is_fraud --metric f1_minority --pos-label 1` |

## 可选参数

- `--max-experiments 5`：实验预算
- `--use-mde`：多种子复核（慢 5 倍，但能过滤噪声提升）

## 模型选择

`modelscope_run.py` 会按顺序找可用的千问：

1. `DASHSCOPE_API_KEY` 设了 → 走 DashScope 官方 API
2. 有 CUDA → 本地 transformers 加载 `Qwen/Qwen2.5-7B-Instruct`（FP16）
3. 都没有 → 回退到本地 ollama

## 注意

- 免费 GPU 一般是 T4 16G 或 V100 16G，7B FP16 约 15G，刚好够。
- 不装 `imbalanced-learn`（SMOTE 依赖未确认）。
- 密钥只从环境变量读，不写进任何文件。
