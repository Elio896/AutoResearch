"""跨任务研究经验记忆 —— 赛题里"可持续提升"的落点。

设计上的三条自我约束：

1. **经验只是假设，不是规则。** 它进 prompt 是"建议先试"，不是"必须这么做"；
   最终是否采信由实验分数决定。
2. **每条经验必须带证据。** 没有 `evidence`（哪个任务、什么分数）的经验不入库，
   否则过两周自己都不知道它是怎么来的。
3. **append-only，不做沉默的删除。** 经验失效了就追加一条 `contradicted`，
   而不是把旧的抹掉——否证也是知识。

存储格式：一行一个 JSON（jsonl），可以直接用文本编辑器看。
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


DEFAULT_PATH = Path('artifacts') / 'experience.jsonl'


class ExperienceStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 写 ----------
    def add(self, condition: dict, advice: str, evidence: dict,
            source_task: str = '', kind: str = 'observed') -> str:
        """加一条经验。没有证据的一律拒收。"""
        if not advice or not str(advice).strip():
            raise ValueError('advice 不能为空')
        if not evidence:
            raise ValueError('经验必须带证据，不能凭空写')
        record = {
            'id': uuid.uuid4().hex[:12],
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'kind': kind,                     # observed / contradicted
            'condition': condition,           # 适用条件（数据画像的子集）
            'advice': str(advice).strip()[:400],
            'evidence': evidence,             # {'task':..., 'score':..., 'baseline':...}
            'source_task': source_task,
        }
        with self.path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')
        return record['id']

    def contradict(self, record_id: str, evidence: dict, reason: str = '') -> str:
        """标记某条经验被否证。不删除原记录。"""
        return self.add(
            condition={'contradicts': record_id},
            advice=reason or f'经验 {record_id} 在新任务上没有复现',
            evidence=evidence,
            kind='contradicted',
        )

    # ---------- 读 ----------
    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue                      # 坏行跳过，不炸整库
        return out

    def match(self, profile: dict, limit: int = 5) -> list[dict]:
        """按数据画像挑相关经验。

        匹配规则刻意保守：**只认条件完全满足的经验**，条件越具体越优先。
        宁可少给建议，也不要给一条不适用的大道理。
        """
        if not profile:
            return []
        contradicted = {r['condition'].get('contradicts') for r in self.all()
                        if r.get('kind') == 'contradicted'}
        scored = []
        for record in self.all():
            if record.get('kind') == 'contradicted':
                continue
            if record.get('id') in contradicted:
                continue                      # 已被否证的，不再给建议
            cond = record.get('condition') or {}
            if not cond:
                continue
            if all(self._satisfied(profile, key, value) for key, value in cond.items()):
                scored.append((len(cond), record))   # 条件越多越具体，排前面
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [record for _, record in scored[:limit]]

    @staticmethod
    def _satisfied(profile: dict, key: str, value) -> bool:
        if key not in profile:
            return False
        actual = profile[key]
        if isinstance(value, bool):
            return bool(actual) == value
        if isinstance(value, (int, float)) and isinstance(actual, (int, float)):
            return abs(float(actual) - float(value)) < 1e-9
        return actual == value

    def stats(self) -> dict:
        records = self.all()
        return {
            'total': len(records),
            'observed': sum(1 for r in records if r.get('kind') == 'observed'),
            'contradicted': sum(1 for r in records if r.get('kind') == 'contradicted'),
            'path': str(self.path),
        }


def advice_lines(records: list[dict]) -> list[str]:
    """把经验变成给模型看的一行行建议。"""
    return [f'- {r["advice"]}（证据: {json.dumps(r["evidence"], ensure_ascii=False)[:120]}）'
            for r in records]
