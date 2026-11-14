#!/usr/bin/env bash
# EvoResearch 在 ModelScope 免费 GPU Notebook 里的环境搭建
# 在 Notebook 里执行：  !bash deploy/modelscope_setup.sh
set -euo pipefail

echo "=== EvoResearch @ ModelScope ==="
echo "python: $(python --version)"
echo "cuda:   $(python -c 'import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")' 2>/dev/null || echo 'torch 未装')"
echo

echo "=== 安装依赖 ==="
pip install -q \
  pandas scikit-learn fastapi uvicorn \
  transformers torch accelerate \
  modelscope dashscope 2>&1 | tail -n 3
echo

echo "=== 依赖就绪 ==="
python -c "import pandas, sklearn, fastapi, transformers; print('import OK')"
echo
echo "=== 下一步 ==="
echo "  python deploy/modelscope_run.py --csv data/demo/clean_balanced.csv --target species"
echo "  # 不平衡任务加: --metric f1_minority --pos-label 1"
echo "  # 多种子复核加: --use-mde"
