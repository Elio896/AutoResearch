"""评估微调后的千问：和未微调版本在同一批任务上对比。

判据只有三条，和 M3 一样（不许改宽）：
  1. 达同样分数所需实验数
  2. 达同样分数所需 token
  3. 达同样分数所需时间

如果三条都不成立，就照实写"没观察到改进"。

用法：
  # 训完之后
  .venv\\Scripts\\python.exe tools\\eval_lora.py --adapter artifacts/sft/lora_qwen2.5-3b-instruct/adapter
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_base(base: str, adapter: str | None = None):
    """加载基础模型；给了 adapter 就用 peft 挂上。"""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None, trust_remote_code=True)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def predict(model, tokenizer, instruction: str, user_input: str, max_new: int = 120) -> str:
    import torch
    messages = [{'role': 'system', 'content': instruction},
                {'role': 'user', 'content': user_input}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def score_prediction(prediction: str, target: dict) -> dict:
    """只比"模型选对了吗"和"JSON 合法吗"，不做模糊匹配。"""
    try:
        parsed = json.loads(prediction)
    except json.JSONDecodeError:
        return {'valid_json': False, 'model_match': False, 'params_match': False}
    model_ok = parsed.get('model') == target.get('model')
    params_ok = (parsed.get('parameters') or {}) == (target.get('parameters') or {})
    return {'valid_json': True, 'model_match': model_ok, 'params_match': params_ok}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--adapter', default=None)
    p.add_argument('--data', default='artifacts/sft/train.jsonl')
    p.add_argument('--limit', type=int, default=30)
    p.add_argument('--mirror', default='https://hf-mirror.com')
    p.add_argument('--output', default='')
    args = p.parse_args()

    import os
    if args.mirror:
        os.environ.setdefault('HF_ENDPOINT', args.mirror)

    data_path = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    rows = [json.loads(line) for line in data_path.open(encoding='utf-8') if line.strip()]
    rows = rows[: args.limit]

    print(f'base={args.base} adapter={args.adapter or "(none)"} samples={len(rows)}')
    model, tokenizer = load_base(args.base, args.adapter)

    stats = {'valid_json': 0, 'model_match': 0, 'params_match': 0, 'total': len(rows)}
    details = []
    for row in rows:
        prediction = predict(model, tokenizer, row['instruction'], row['input'])
        target = json.loads(row['output'])
        result = score_prediction(prediction, target)
        details.append({'input': row['input'][:120], 'prediction': prediction[:160],
                        'target': row['output'][:160], **result})
        for key in ('valid_json', 'model_match', 'params_match'):
            stats[key] += int(result[key])

    stats['valid_json_rate'] = round(stats['valid_json'] / max(1, stats['total']), 4)
    stats['model_match_rate'] = round(stats['model_match'] / max(1, stats['total']), 4)
    stats['params_match_rate'] = round(stats['params_match'] / max(1, stats['total']), 4)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if args.output:
        out = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'stats': stats, 'details': details},
                                  ensure_ascii=False, indent=2), encoding='utf-8')
        print('saved:', out)


if __name__ == '__main__':
    main()
