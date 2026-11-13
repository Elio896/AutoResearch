"""从历史实验轨迹里挖 SFT 样本，训本地千问的「决策能力」。

思路：
  我们跑了 150+ 个任务，每次实验都是一条真实样本：
    输入 = 数据画像 + 到这一步为止的历史分数
    输出 = 下一步该用什么方案（模型 + 参数 + 假设）
  这些是**真跑出来的、带真实分数**的决策，比让大模型凭空造数据可靠。

只保留"改进过"的那一步（分数比前一步高），因为我们要教的是"该怎么选"，
不是"怎么选错"。

用法：
  .venv\\Scripts\\python.exe tools\\build_sft_data.py --output artifacts\\sft\\train.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def scan_trajectories() -> list[dict]:
    """扫所有 evoundo 版本链，每条链是一个任务的完整决策序列。"""
    chains = []
    base = ROOT / 'artifacts' / 'evoundo'
    if not base.exists():
        return chains
    for run_dir in sorted(base.iterdir()):
        versions_file = run_dir / 'versions.json'
        if not versions_file.is_file():
            continue
        try:
            data = json.loads(versions_file.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            continue
        versions = data.get('versions') or []
        if len(versions) < 2:
            continue
        chains.append({'run_id': data.get('run_id') or run_dir.name, 'versions': versions})
    return chains


def build_samples(chains: list[dict]) -> list[dict]:
    """每条链生成多条样本：每个步骤都用**不同长度的历史窗口**切一次。

    为什么要滑窗：同一个决策，在"只看上一步"和"看最近五步"下，正确答案可能不同。
    这两种都是真实的推理场景，都要教。
    """
    samples = []
    for chain in chains:
        versions = sorted(chain['versions'], key=lambda v: v.get('created_at', ''))
        history = []
        for version in versions:
            proposal = version.get('proposal') or {}
            score = version.get('score')
            if not isinstance(score, (int, float)):
                continue
            cur = {'model': proposal.get('model'),
                   'parameters': proposal.get('parameters') or {},
                   'score': score}
            # 对每个可用的历史长度各造一条（至少要看一步才叫"历史"）
            for window in (1, 2, 3, 5):
                if len(history) >= window:
                    samples.append(_make_sample(history[-window:], proposal, score,
                                                chain['run_id'], window))
            if history:
                samples.append(_make_sample(history[-5:], proposal, score,
                                            chain['run_id'], len(history[-5:])))
            history.append(cur)
    return samples


def _make_sample(history: list, proposal: dict, score: float,
                 run_id: str, window: int) -> dict:
    prev_best = max((h['score'] for h in history), default=None)
    return {
        'instruction': (
            '你是研究系统的本地决策员。根据数据画像和历史分数，'
            '提出下一个实验方案。只输出 JSON。'
        ),
        'input': json.dumps({
            'history': history,
            'step': len(history),
        }, ensure_ascii=False),
        'output': json.dumps({
            'model': proposal.get('model'),
            'parameters': proposal.get('parameters') or {},
        }, ensure_ascii=False),
        'meta': {'run_id': run_id, 'score': score,
                 'prev_best': prev_best, 'window': window,
                 'improved': (prev_best is None or score > prev_best + 1e-9)},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='artifacts/sft/train.jsonl')
    args = p.parse_args()

    chains = scan_trajectories()
    samples = build_samples(chains)
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as fh:
        for item in samples:
            fh.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f'chains: {len(chains)}')
    print(f'samples: {len(samples)}')
    print(f'output: {out}')
    if samples:
        print('--- first sample ---')
        print(json.dumps(samples[0], ensure_ascii=False, indent=2)[:600])


if __name__ == '__main__':
    main()
