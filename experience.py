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
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


DEFAULT_PATH = Path('artifacts') / 'experience.jsonl'
_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


class ExperienceStore:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread_lock = _lock_for(self.path)
        self._lock_path = self.path.with_name(self.path.name + '.lock')

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            with self._lock_path.open('a+b') as lock_file:
                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write(b'\0')
                    lock_file.flush()
                lock_file.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                    unlock = lambda: msvcrt.locking(
                        lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    unlock = lambda: fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                try:
                    yield
                finally:
                    unlock()

    # ---------- 写 ----------
    def add(self, condition: dict, advice: str, evidence: dict,
            source_task: str = '', kind: str = 'observed') -> str:
        """加一条经验。没有证据的一律拒收。"""
        if not advice or not str(advice).strip():
            raise ValueError('advice 不能为空')
        if not isinstance(condition, dict) or not isinstance(evidence, dict) or not evidence:
            raise ValueError('经验必须带证据，不能凭空写')
        if kind not in ('observed', 'contradicted'):
            raise ValueError(f'不支持的经验类型: {kind}')
        record = {
            'id': uuid.uuid4().hex[:12],
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'kind': kind,                     # observed / contradicted
            'condition': condition,           # 适用条件（数据画像的子集）
            'advice': str(advice).strip()[:400],
            'evidence': evidence,             # {'task':..., 'score':..., 'baseline':...}
            'source_task': source_task,
            # 使用统计：经验被用过几次、成功几次、被证伪几次
            'used_count': 0,
            'success_count': 0,
            'contradicted_count': 0,
        }
        self._append(record)
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
        with self._locked():
            return self._read_records()

    def _read_records(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        by_id = {}
        for line_number, line in enumerate(
                self.path.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'经验库第 {line_number} 行 JSON 损坏') from exc
            if not isinstance(item, dict):
                raise ValueError(f'经验库第 {line_number} 行不是对象')
            if item.get('event_type') == 'usage':
                target = by_id.get(item.get('record_id'))
                if target is None:
                    raise ValueError(f'经验库第 {line_number} 行引用了未知经验')
                target['used_count'] = int(target.get('used_count', 0)) + 1
                counter = 'success_count' if item.get('improved') is True else 'contradicted_count'
                target[counter] = int(target.get(counter, 0)) + 1
                continue
            record_id = item.get('id')
            if not isinstance(record_id, str) or record_id in by_id:
                raise ValueError(f'经验库第 {line_number} 行 id 缺失或重复')
            item.setdefault('used_count', 0)
            item.setdefault('success_count', 0)
            item.setdefault('contradicted_count', 0)
            by_id[record_id] = item
            records.append(item)
        return records

    def _append(self, item: dict) -> None:
        line = json.dumps(item, ensure_ascii=False) + '\n'
        with self._locked():
            with self.path.open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(line)
                stream.flush()

    def match(self, profile: dict, limit: int = 5) -> list[dict]:
        """按数据画像挑相关经验。

        匹配规则刻意保守：**只认条件完全满足的经验**，条件越具体越优先。
        宁可少给建议，也不要给一条不适用的大道理。
        """
        if not isinstance(profile, dict) or not profile:
            return []
        if type(limit) is not int or limit < 0:
            raise ValueError('limit must be a nonnegative integer')
        records = self.all()
        contradicted = {r['condition'].get('contradicts') for r in records
                        if r.get('kind') == 'contradicted'}
        active_ids = {r['id'] for r in self.active_pool()}
        scored = []
        for record in records:
            if record.get('kind') == 'contradicted':
                continue
            if record.get('id') in contradicted:
                continue                      # 已被否证的，不再给建议
            if record.get('id') not in active_ids:
                continue
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

    def record_usage(self, record_id: str, improved: bool) -> dict | None:
        """追加一次使用结果，读取时再汇总计数。

        这是"自进化"能不能成立的关键：
        经验被用过之后，必须知道自己管不管用。否则经验库只会在膨胀，不会在进化。

        规则：
          - 用过一次就 used_count +1
          - 真的让分数变好才 success_count +1
          - 没用（甚至变差）就 contradicted_count +1
          - **不删记录**，只在 active_pool 里降权
        """
        if type(improved) is not bool:
            raise ValueError('improved must be a bool')
        with self._locked():
            records = self._read_records()
            target = next((r for r in records if r.get('id') == record_id), None)
            if target is None:
                return None
            event = {
                'event_type': 'usage',
                'record_id': record_id,
                'improved': improved,
                'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            }
            with self.path.open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')
                stream.flush()
            target['used_count'] = int(target.get('used_count', 0)) + 1
            counter = 'success_count' if improved else 'contradicted_count'
            target[counter] = int(target.get(counter, 0)) + 1
            return target

    def active_pool(self, max_contradictions: int = 2) -> list[dict]:
        """还能被推荐的经验。被证伪到阈值以上就沉底（但不删）。"""
        if type(max_contradictions) is not int or max_contradictions < 1:
            raise ValueError('max_contradictions must be a positive integer')
        pool = []
        for record in self.all():
            if record.get('kind') == 'contradicted':
                continue
            net_failures = (int(record.get('contradicted_count', 0))
                            - int(record.get('success_count', 0)))
            if net_failures >= max_contradictions:
                continue
            pool.append(record)
        return pool

    def lineage(self) -> dict:
        """进化账：这个库到底学到了多少、有多少真管用。"""
        records = self.all()
        observed = [r for r in records if r.get('kind') != 'contradicted']
        used = [r for r in observed if int(r.get('used_count', 0)) > 0]
        successes = sum(int(r.get('success_count', 0)) for r in observed)
        contradictions = sum(int(r.get('contradicted_count', 0)) for r in observed)
        total_uses = successes + contradictions
        return {
            'total_records': len(records),
            'observed': len(observed),
            'contradicted_records': len(records) - len(observed),
            'ever_used': len(used),
            'total_uses': total_uses,
            'successes': successes,
            'contradictions': contradictions,
            'success_rate': round(successes / total_uses, 4) if total_uses else None,
            'active': len(self.active_pool()),
            'retired': len(observed) - len(self.active_pool()),
            'path': str(self.path),
        }

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
