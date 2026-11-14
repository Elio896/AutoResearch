"""评估本地模型的「置信度」表现，为路由阈值找依据。

思路（妈妈提的）：
    本地模型给每个提案一个 confidence_before（0~1）
    高置信 → 直接采用；低置信 → 交给云端

要定阈值，先得看：模型自己说"我有信心"的时候，实际真的更好吗？

这个脚本跑 N 轮，收集 (confidence, 实际分数) 对，算：
    - 置信度与实际改进的相关性
    - 不同阈值下的「放行率」和「放行后的成功率」
    - 高低置信两组的表现差异

用法：
  .venv\\Scripts\\python.exe tools\\measure_confidence.py --rounds 6
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents import Budget                     # noqa: E402
from executor import ResearchSession          # noqa: E402
from orchestrator import LocalQwenClient      # noqa: E402
from research_loop import ResearchLoop        # noqa: E402


def run_once(csv: str, target: str, metric: str, pos_label: str,
             model: str, max_experiments: int, base_url: str = 'http://127.0.0.1:11434') -> dict:
    session = ResearchSession(csv, target, metric=metric, pos_label=pos_label)
    # base_url 必须显式传：LocalQwenClient 的默认是 11435 那个旧服务
    proposer = LocalQwenClient(base_url=base_url, model=model)
    budget = Budget(max_experiments=max_experiments, max_seconds=150.0,
                    max_local_calls=max_experiments + 1)
    loop = ResearchLoop(session, proposer=proposer, budget=budget, patience=2)
    result = loop.run()

    # 从事件里把「每个提案的置信度」和「它跑出来的分数」配对。
    # 注意：experiment 事件比 proposal 多一个——baseline 没有对应提案，
    # 所以按模型名+参数对齐，而不是按顺序 zip。
    proposals = [e for e in result['events'] if e.get('event') == 'scientist_proposal']
    experiments = [e for e in result['events'] if e.get('event') == 'experiment']
    pairs = []
    used = set()
    for prop in proposals:
        proposal = prop.get('proposal') or {}
        confidence = proposal.get('confidence_before')
        if confidence is None:
            continue
        key = json.dumps({'model': proposal.get('model'),
                          'parameters': proposal.get('parameters')}, sort_keys=True)
        for i, exp in enumerate(experiments):
            if i in used:
                continue
            exp_key = json.dumps({'model': exp.get('model'),
                                  'parameters': exp.get('parameters')}, sort_keys=True)
            if key == exp_key and exp.get('score') is not None:
                pairs.append({'confidence': float(confidence),
                              'score': float(exp['score'])})
                used.add(i)
                break
    return {'pairs': pairs, 'summary': result['summary']}


def evaluate(pairs: list[dict]) -> dict:
    if len(pairs) < 3:
        return {'error': '样本太少（<3），无法评估', 'count': len(pairs)}
    confidences = [p['confidence'] for p in pairs]
    scores = [p['score'] for p in pairs]
    mean_conf = statistics.mean(confidences)
    mean_score = statistics.mean(scores)

    # 相关系数（皮尔逊，手算避免依赖）
    n = len(pairs)
    cov = sum((p['confidence'] - mean_conf) * (p['score'] - mean_score) for p in pairs) / n
    std_conf = (sum((c - mean_conf) ** 2 for c in confidences) / n) ** 0.5
    std_score = (sum((s - mean_score) ** 2 for s in scores) / n) ** 0.5
    corr = (cov / (std_conf * std_score)) if (std_conf and std_score) else None

    # 各阈值的放行率与放行后的平均分
    thresholds = {}
    for th in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        passed = [p for p in pairs if p['confidence'] >= th]
        rejected = [p for p in pairs if p['confidence'] < th]
        thresholds[f'{th:.1f}'] = {
            'pass_rate': round(len(passed) / n, 3),
            'passed_mean_score': round(statistics.mean([p['score'] for p in passed]), 4) if passed else None,
            'rejected_mean_score': round(statistics.mean([p['score'] for p in rejected]), 4) if rejected else None,
            'n_passed': len(passed),
            'n_rejected': len(rejected),
        }
    return {'count': n, 'mean_confidence': round(mean_conf, 4),
            'mean_score': round(mean_score, 4),
            'correlation': round(corr, 4) if corr is not None else None,
            'thresholds': thresholds}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', default='bench/tasks/t2_imbalanced/data.csv')
    p.add_argument('--target', default='target')
    p.add_argument('--metric', default='f1_minority')
    p.add_argument('--pos-label', default='1')
    p.add_argument('--model', default='qwen2.5:7b-instruct-q4_K_M')
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--max-experiments', type=int, default=4)
    p.add_argument('--output', default='artifacts/confidence_eval.json')
    args = p.parse_args()

    all_pairs = []
    for i in range(args.rounds):
        result = run_once(args.csv, args.target, args.metric, args.pos_label,
                          args.model, args.max_experiments)
        all_pairs.extend(result['pairs'])
        print(f"round {i+1}: {len(result['pairs'])} pairs, "
              f"best={result['summary'].get('best_score')}", flush=True)

    report = evaluate(all_pairs)
    report['pairs'] = all_pairs
    print(json.dumps({k: v for k, v in report.items() if k != 'pairs'},
                     ensure_ascii=False, indent=2))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('saved:', out)


if __name__ == '__main__':
    main()
