"""批量跑研究任务，把轨迹堆起来当训练数据。

为什么值得花时间：训练数据的量决定模型的"见识"。
我们手上只有 113 条链（487 条样本），对"专家"来说太少。
多跑一批任务，链数翻几倍，训练出来的千问才见过足够多的局面。

跑法：同一个任务用不同的**预算 / 耐心 / 模型**组合各跑一遍——
这不是重复劳动，是在制造"同一局面下的不同决策路径"，
这些路径里既有成功也有失败，正好教模型"什么情况下该做什么"。

用法：
  .venv\\Scripts\\python.exe tools\\mine_trajectories.py --rounds 30
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = [
    # (csv, target, metric, pos_label)
    # t1 有 record_id 泄漏风险，必须先人工给 feature_decisions 才能训练
    # （这是完整性门在正常工作，不是 bug）。挖数据先跳过它。
    ('bench/tasks/t2_imbalanced/data.csv', 'target', 'f1_minority', '1'),
    ('bench/tasks/t3_clean/data.csv', 'target', 'f1_minority', '1'),
]
CONFIGS = [
    # (max_experiments, patience, model)
    (3, 2, 'qwen2.5:7b-instruct-q4_K_M'),
    (4, 1, 'qwen2.5:7b-instruct-q4_K_M'),
    (5, 2, 'qwen2.5:7b-instruct-q4_K_M'),
    (3, 2, 'qwen2.5:3b-instruct-q4_K_M'),
    (4, 1, 'qwen2.5:3b-instruct-q4_K_M'),
    (5, 2, 'qwen2.5:0.5b'),
]


def run_one(csv: str, target: str, metric: str, pos_label: str | None,
            cfg: tuple, out_dir: Path, idx: int) -> dict:
    max_exp, patience, model = cfg
    out = out_dir / f'run_{idx:03d}.json'
    cmd = [
        sys.executable, 'tools/run_five_roles.py', csv,
        '--target', target, '--metric', metric,
    ]
    if pos_label:
        cmd += ['--pos-label', pos_label]
    cmd += [
        '--model', model, '--max-experiments', str(max_exp),
        '--max-local-calls', str(max_exp + 1), '--max-seconds', '120',
        '--patience', str(patience), '--output', str(out),
    ]
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                              text=True, timeout=300)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = -1
    elapsed = round(time.time() - started, 1)

    record = {'idx': idx, 'task': Path(csv).parent.name, 'model': model,
              'max_exp': max_exp, 'patience': patience, 'rc': rc, 'seconds': elapsed}
    if out.is_file():
        try:
            data = json.loads(out.read_text(encoding='utf-8'))
            record['best_score'] = data.get('summary', {}).get('best_score')
            record['stop_reason'] = data.get('summary', {}).get('stop_reason')
            record['experiments'] = data.get('summary', {}).get('experiments')
        except (json.JSONDecodeError, OSError):
            pass
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rounds', type=int, default=30)
    p.add_argument('--out', default='artifacts/mine')
    args = p.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    for i in range(args.rounds):
        task = TASKS[i % len(TASKS)]
        cfg = CONFIGS[(i // len(TASKS)) % len(CONFIGS)]
        plan.append((task, cfg, i))

    log_path = out_dir / 'runs.jsonl'
    done = 0
    with log_path.open('w', encoding='utf-8') as log:
        for (csv, target, metric, pos_label), cfg, idx in plan:
            record = run_one(csv, target, metric, pos_label, cfg, out_dir, idx)
            log.write(json.dumps(record, ensure_ascii=False) + '\n')
            log.flush()
            done += 1
            print(f"[{done}/{len(plan)}] {record['task']} {record['model']} "
                  f"exp={record['max_exp']} -> {record.get('best_score')} "
                  f"({record['seconds']}s)", flush=True)

    print('done. logs ->', log_path)
    print('next: .venv\\Scripts\\python.exe tools\\build_sft_data.py')


if __name__ == '__main__':
    main()
