"""用我们自己的实验轨迹微调本地千问（LoRA）。

为什么用轨迹而不是网上找的数据：
    这 487 条样本全部来自真跑的实验——输入是真实的数据画像和历史分数，
    输出是真实被试过的方案，分数也是真跑出来的。教的是"我们这套系统的决策方式"。

用法（先小后大）：
  # 先拿 0.5B 验证流程能跑通（几分钟）
  .venv\\Scripts\\python.exe tools\\train_qwen_lora.py --base Qwen/Qwen2.5-0.5B-Instruct --epochs 2

  # 跑通后再上 3B（显存够，但下载慢）
  .venv\\Scripts\\python.exe tools\\train_qwen_lora.py --base Qwen/Qwen2.5-3B-Instruct --epochs 3

产物：artifacts/sft/lora_<模型简称>/
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_samples(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows.append(item)
    return rows


def format_prompt(item: dict) -> dict:
    """把样本拼成 chat 格式。**直接用模型自己的 chat template**，别手写。"""
    messages = [
        {'role': 'system', 'content': item['instruction']},
        {'role': 'user', 'content': item['input']},
        {'role': 'assistant', 'content': item['output']},
    ]
    return {'messages': messages}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='Qwen/Qwen2.5-0.5B-Instruct')
    p.add_argument('--data', default='artifacts/sft/train.jsonl')
    p.add_argument('--output', default=None)
    p.add_argument('--epochs', type=float, default=2.0)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--max-seq-len', type=int, default=768)
    p.add_argument('--lora-r', type=int, default=16)
    p.add_argument('--limit', type=int, default=0, help='只训前 N 条（调试用）')
    p.add_argument('--mirror', default='https://hf-mirror.com', help='下载镜像')
    args = p.parse_args()

    import os
    if args.mirror:
        os.environ.setdefault('HF_ENDPOINT', args.mirror)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments)

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    samples = load_samples(data_path)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print('no samples; run tools/build_sft_data.py first')
        sys.exit(1)

    short = args.base.rstrip('/').split('/')[-1].lower()
    out_dir = Path(args.output) if args.output else ROOT / 'artifacts' / 'sft' / f'lora_{short}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'base={args.base} samples={len(samples)} out={out_dir}')
    print('device:', 'cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print('vram total GB:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    def to_text(item):
        text = tokenizer.apply_chat_template(
            format_prompt(item)['messages'], tokenize=False, add_generation_prompt=False)
        return {'text': text}

    dataset = Dataset.from_list([to_text(s) for s in samples])
    dataset = dataset.map(
        lambda batch: tokenizer(batch['text'], truncation=True,
                                max_length=args.max_seq_len, padding=False),
        batched=True, remove_columns=['text'],
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=str(out_dir / 'checkpoints'),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy='no',
        report_to=[],
        fp16=torch.cuda.is_available(),
        optim='adamw_torch',
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset,
                      data_collator=collator)
    trainer.train()
    model.save_pretrained(out_dir / 'adapter')
    tokenizer.save_pretrained(out_dir / 'adapter')
    print('saved adapter to', out_dir / 'adapter')


if __name__ == '__main__':
    main()
