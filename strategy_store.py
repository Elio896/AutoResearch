"""策略版本存储：策略是**配置**，不是代码。

这一层只做三件事，别的都不做：

1. 把策略当版本存起来（candidate / active / retired / rejected），可 diff、可回滚；
2. 保证「任何一次研究运行用的都是同一个版本」——运行开始 pin 住，运行中途不许换；
3. 升级必须同时拿到：代码侧的验收检查全部通过 + Jev 的明确选择。缺一不切。

设计上刻意保守：
- 只允许改白名单里的字段（prompt 与选路规则），不允许模型碰代码、标签、评分函数、测试集；
- 没有 active 时不会自动挑一个用，调用方必须显式 pin；
- 文件写入走临时文件 + 原子替换，进程重启后读到的还是同一份。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

SCHEMA = 1

# 策略只能改这三类东西：分类提示词、规划提示词、候选选择规则
WHITELIST_FIELDS: dict[str, type] = {
    'classifier_prompt': str,
    'planning_prompt': str,
    'candidate_selection_rules': list,
}

STATUS_CANDIDATE = 'candidate'
STATUS_ACTIVE = 'active'
STATUS_RETIRED = 'retired'
STATUS_REJECTED = 'rejected'
_STATUSES = {STATUS_CANDIDATE, STATUS_ACTIVE, STATUS_RETIRED, STATUS_REJECTED}


def _default_path() -> Path:
    return Path(__file__).resolve().parent / 'runtime' / 'strategy_store.json'


_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        if key not in _LOCKS:
            _LOCKS[key] = threading.RLock()
        return _LOCKS[key]


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def changes_digest(changes: dict) -> str:
    return hashlib.sha256(canonical_json(changes).encode('utf-8')).hexdigest()[:16]


def _validate_changes(changes: dict) -> dict:
    if not isinstance(changes, dict) or not changes:
        raise ValueError('changes 必须是非空字典')
    unknown = sorted(set(changes) - set(WHITELIST_FIELDS))
    if unknown:
        raise ValueError(f'不允许修改的字段：{", ".join(unknown)}（只允许 {", ".join(sorted(WHITELIST_FIELDS))}）')
    cleaned: dict = {}
    for key, value in changes.items():
        expected = WHITELIST_FIELDS[key]
        if not isinstance(value, expected):
            raise ValueError(f'{key} 必须是 {expected.__name__}，收到 {type(value).__name__}')
        if key == 'candidate_selection_rules':
            for index, rule in enumerate(value):
                if not isinstance(rule, dict):
                    raise ValueError(f'candidate_selection_rules[{index}] 必须是对象')
        cleaned[key] = value
    return cleaned


class StrategyStore:
    """一个进程内共享、落盘持久的策略版本库。"""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else _default_path()
        self._lock = _lock_for(self.path)

    # ---------- 读 ----------

    def _read(self) -> dict:
        if not self.path.is_file():
            return {'schema': SCHEMA, 'active_id': None, 'versions': [], 'events': []}
        raw = self.path.read_text(encoding='utf-8')
        data = json.loads(raw)
        if data.get('schema') != SCHEMA:
            raise ValueError(f'策略库 schema={data.get("schema")} 不是 {SCHEMA}')
        data.setdefault('versions', [])
        data.setdefault('events', [])
        return data

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + '.tmp')
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(tmp, self.path)

    def versions(self) -> list[dict]:
        with self._lock:
            return list(self._read()['versions'])

    def get(self, strategy_id: str) -> dict | None:
        with self._lock:
            for version in self._read()['versions']:
                if version['strategy_id'] == strategy_id:
                    return version
        return None

    def active(self) -> dict | None:
        with self._lock:
            data = self._read()
            active_id = data.get('active_id')
            if not active_id:
                return None
            for version in data['versions']:
                if version['strategy_id'] == active_id:
                    return version
        return None

    def events(self) -> list[dict]:
        with self._lock:
            return list(self._read()['events'])

    # ---------- 写 ----------

    def _next_id(self, data: dict) -> str:
        numbers = []
        for version in data['versions']:
            tail = version['strategy_id'].rsplit('_', 1)[-1]
            if tail.isdigit():
                numbers.append(int(tail))
        return f'strategy_{(max(numbers) + 1) if numbers else 1:03d}'

    def _append_version(self, data: dict, version: dict) -> dict:
        version['config_digest'] = changes_digest(version['changes'])
        data['versions'].append(version)
        return version

    def bootstrap(self, changes: dict, reason: str = '初始策略') -> dict:
        """没有版本时建立 strategy_001 并设为 active；已有 active 就原样返回。"""
        with self._lock:
            data = self._read()
            active = None
            if data.get('active_id'):
                for version in data['versions']:
                    if version['strategy_id'] == data['active_id']:
                        active = version
                        break
            if active is not None:
                return active

            cleaned = _validate_changes(changes)
            version = {
                'strategy_id': self._next_id(data),
                'parent_id': None,
                'status': STATUS_ACTIVE,
                'changes': cleaned,
                'reason': reason,
                'evidence_run_ids': [],
                'created_at': time.time(),
                'created_by': 'system',
            }
            self._append_version(data, version)
            data['active_id'] = version['strategy_id']
            data['events'].append({'type': 'bootstrap', 'strategy_id': version['strategy_id'],
                                   'at': time.time(), 'by': 'system', 'reason': reason})
            self._write(data)
            return version

    def propose(self, changes: dict, reason: str, evidence_run_ids: list[str] | None = None,
                parent_id: str | None = None, proposed_by: str = 'deepseek',
                hypothesis: dict | None = None,
                target_failure_types: list[str] | None = None) -> dict:
        """新建一个候选版本。当前正式策略一个字都不动。

        传进来的是**补丁**：没写的字段从父版本继承，存下去的是完整配置。
        这样任何一次运行拿到的策略都是自洽的，不会出现"少了个 prompt"。
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('候选策略必须写明理由（reason）')
        patch = _validate_changes(changes)

        with self._lock:
            data = self._read()
            if parent_id is None:
                parent_id = data.get('active_id')
            parent = next((v for v in data['versions'] if v['strategy_id'] == parent_id), None)
            if parent_id is not None and parent is None:
                raise ValueError(f'父版本不存在：{parent_id}')

            merged = dict(parent['changes']) if parent is not None else {}
            merged.update(patch)
            if not merged:
                raise ValueError('候选策略为空')

            if parent is not None and canonical_json(parent['changes']) == canonical_json(merged):
                raise ValueError('候选策略与父版本完全相同，不算一次修改')

            version = {
                'strategy_id': self._next_id(data),
                'parent_id': parent_id,
                'status': STATUS_CANDIDATE,
                'changes': merged,
                'patch': patch,
                'reason': reason.strip(),
                'evidence_run_ids': [str(x) for x in (evidence_run_ids or [])],
                'hypothesis': dict(hypothesis) if hypothesis else None,
                'target_failure_types': [str(x) for x in (target_failure_types or [])],
                'created_at': time.time(),
                'created_by': proposed_by,
            }
            self._append_version(data, version)
            data['events'].append({'type': 'propose', 'strategy_id': version['strategy_id'],
                                   'parent_id': parent_id, 'at': time.time(), 'by': proposed_by,
                                   'reason': version['reason'],
                                   'evidence_run_ids': version['evidence_run_ids'],
                                   'target_failure_types': version['target_failure_types']})
            self._write(data)
            return version

    def reject(self, strategy_id: str, reason: str, by: str = 'system') -> dict:
        with self._lock:
            data = self._read()
            version = next((v for v in data['versions'] if v['strategy_id'] == strategy_id), None)
            if version is None:
                raise KeyError(strategy_id)
            if version['status'] != STATUS_CANDIDATE:
                raise ValueError(f'只有候选能拒绝，当前状态 {version["status"]}')
            version['status'] = STATUS_REJECTED
            version['rejected_reason'] = reason
            data['events'].append({'type': 'reject', 'strategy_id': strategy_id, 'at': time.time(),
                                   'by': by, 'reason': reason})
            self._write(data)
            return version

    def record_decision(self, strategy_id: str, decision: str, *, source: str,
                        evidence: dict | None = None, reason: str = '') -> dict:
        """把一次「不切换」的决定也留痕：候选状态不动，事件照记。

        defer（证据不足暂缓）和 keep（Jev 说保留）都不能算失败，
        但以后要能回答"当时为什么没升"，所以必须落盘。
        """
        if decision not in ('keep', 'defer'):
            raise ValueError(f'这里只记录不切换的决定，收到 {decision!r}')
        with self._lock:
            data = self._read()
            if not any(v['strategy_id'] == strategy_id for v in data['versions']):
                raise KeyError(strategy_id)
            event = {'type': f'decision_{decision}', 'strategy_id': strategy_id,
                     'at': time.time(), 'source': source, 'reason': reason,
                     'evidence': evidence or {}}
            data['events'].append(event)
            self._write(data)
            return event

    def activate(self, strategy_id: str, *, checks: dict, jev: dict,
                 by: str = 'system', reason: str = '', expected_active_id: str | None = None) -> dict:
        """切换正式策略。两项硬条件缺一不可：

        - checks：代码侧验收检查，必须全部为 True（空字典算没检查，拒绝）；
        - jev：决策记录，decision 必须是 'upgrade'。
        """
        if not checks or not all(v is True for v in checks.values()):
            raise PermissionError(f'代码侧验收未全部通过：{checks}')
        if not isinstance(jev, dict) or jev.get('decision') != 'upgrade':
            raise PermissionError(f'Jev 未给出升级选择：{jev}')

        with self._lock:
            data = self._read()
            if expected_active_id is not None and data.get('active_id') != expected_active_id:
                raise RuntimeError('Active strategy changed after evaluation; re-evaluate before activation.')
            target = next((v for v in data['versions'] if v['strategy_id'] == strategy_id), None)
            if target is None:
                raise KeyError(strategy_id)
            if target['status'] not in (STATUS_CANDIDATE, STATUS_RETIRED):
                raise ValueError(f'{strategy_id} 当前状态 {target["status"]}，不能切换')

            previous = data.get('active_id')
            for version in data['versions']:
                if version['status'] == STATUS_ACTIVE:
                    version['status'] = STATUS_RETIRED

            target['status'] = STATUS_ACTIVE
            data['active_id'] = strategy_id
            data['events'].append({'type': 'activate', 'strategy_id': strategy_id,
                                   'previous_id': previous, 'at': time.time(), 'by': by,
                                   'reason': reason, 'checks': dict(checks),
                                   'jev': dict(jev)})
            self._write(data)
            return target

    def restore(self, strategy_id: str, *, checks: dict, jev: dict,
                by: str = 'system', reason: str = '回滚') -> dict:
        """人在评审之后把某个历史版本重新设为正式策略（需要 Jev 的选择）。"""
        return self.activate(strategy_id, checks=checks, jev=jev, by=by, reason=reason)

    def rollback(self, strategy_id: str, *, triggers: list, evidence: dict,
                 by: str = 'system', reason: str = '') -> dict:
        """自动回滚：只按**事先定义**并已触发的条件切回旧版本。

        和 `restore` 的区别：回滚是代码按规则执行的止损动作，不需要 Jev 再选一次；
        但触发条件和证据必须留痕，否则就成了"谁都能偷偷改回去"。
        """
        if not triggers:
            raise ValueError('回滚必须给出触发条件（traces），不能无理由切回')
        with self._lock:
            data = self._read()
            target = next((v for v in data['versions'] if v['strategy_id'] == strategy_id), None)
            if target is None:
                raise KeyError(strategy_id)
            if target['status'] not in (STATUS_CANDIDATE, STATUS_RETIRED):
                raise ValueError(f'{strategy_id} 当前状态 {target["status"]}，不能回滚到它')

            previous = data.get('active_id')
            if previous == strategy_id:
                raise ValueError('本来就是这个版本，回滚没有意义')

            for version in data['versions']:
                if version['status'] == STATUS_ACTIVE:
                    version['status'] = STATUS_RETIRED
            target['status'] = STATUS_ACTIVE
            data['active_id'] = strategy_id
            data['events'].append({
                'type': 'rollback', 'strategy_id': strategy_id, 'from': previous,
                'at': time.time(), 'by': by, 'reason': reason,
                'triggers': list(triggers), 'evidence': dict(evidence or {}),
            })
            self._write(data)
            return target

    # ---------- 比较与绑定 ----------

    def diff(self, left_id: str, right_id: str) -> dict:
        left, right = self.get(left_id), self.get(right_id)
        if left is None or right is None:
            raise KeyError(f'{left_id} 或 {right_id} 不存在')
        fields = []
        for key in sorted(WHITELIST_FIELDS):
            before = left['changes'].get(key)
            after = right['changes'].get(key)
            if canonical_json(before) != canonical_json(after):
                fields.append({'field': key, 'from': before, 'to': after})
        return {
            'left': left_id, 'right': right_id,
            'changed_fields': [f['field'] for f in fields],
            'changes': fields,
            'identical': not fields,
        }

    def pin(self, strategy_id: str | None = None) -> dict:
        """给一次研究运行固定版本：返回快照，之后库里怎么变都不影响这次运行。"""
        with self._lock:
            version = self.active() if strategy_id is None else self.get(strategy_id)
        if version is None:
            raise LookupError('策略库没有 active 版本，拒绝在无策略状态下运行')
        snapshot = json.loads(json.dumps(version))
        snapshot['pinned_at'] = time.time()
        snapshot['config_digest'] = changes_digest(version['changes'])
        return snapshot

    def verify_pin(self, snapshot: dict) -> bool:
        """运行结束时自查：这次运行用的版本有没有被替换。"""
        version = self.get(snapshot['strategy_id'])
        if version is None:
            return False
        return changes_digest(version['changes']) == snapshot['config_digest']
