"""EvoUndo：版本链 + 可回滚的状态机。

三件事，一句一句说清：

1. **一个版本 = 一次实验**。参数、分数、父版本，全部留档。
2. **回滚不改历史**。回滚只是把 `current` 指回某一代，并把被抛弃的那一代标成
   `rolled_back`——**不重训、不删记录**。重训会让分数变，那就是伪造历史。
3. **版本树必须能给人看**。所以除了 JSON，还输出一个零依赖 HTML。

用法：
    chain = VersionChain(run_id='abc')
    v1 = chain.add(proposal={'model': 'lr'}, score=0.30)
    v2 = chain.add(proposal={'model': 'gbdt'}, score=0.66)
    v3 = chain.add(proposal={'model': 'rf'}, score=0.43)
    chain.rollback(chain.best().version_id)      # 回到 v2
    chain.dump('versions.json'); chain.render_html('versions.html')
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
import tempfile
import threading
import time
import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path

ACTIVE = 'active'
ROLLED_BACK = 'rolled_back'


def _new_id(seed: str) -> str:
    return 'v_' + hashlib.sha256(seed.encode('utf-8')).hexdigest()[:10]


@dataclass
class Version:
    version_id: str
    parent_id: str | None
    proposal: dict
    score: float
    status: str = ACTIVE
    created_at: str = field(default_factory=lambda: time.strftime('%Y-%m-%dT%H:%M:%S'))
    note: str = ''

    def to_dict(self) -> dict:
        return asdict(self)


class VersionChain:
    """一条研究任务的版本链。append-only，只改当前指针和被回滚版本的 status。"""

    def __init__(self, run_id: str = ''):
        self.run_id = run_id or _new_id(str(time.time()))
        self.versions: list[Version] = []
        self.current_id: str | None = None
        self._seq = 0
        self._lock = threading.RLock()

    # ---------- 写 ----------
    def add(self, proposal: dict, score: float, parent_id: str | None = None,
            note: str = '') -> Version:
        if not isinstance(proposal, dict):
            raise ValueError('proposal 必须是 dict')
        if isinstance(score, bool):
            raise ValueError(f'score 不合理: {score}')
        try:
            score = float(score)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f'score 不合理: {score}') from exc
        if not math.isfinite(score) or not (0.0 <= score <= 1.5):
            raise ValueError(f'score 不合理: {score}')
        stored_proposal = copy.deepcopy(proposal)
        try:
            proposal_json = json.dumps(stored_proposal, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError('proposal 必须可序列化为 JSON。') from exc

        with self._lock:
            parent = parent_id if parent_id is not None else self.current_id
            if parent is not None and self.get(parent) is None:
                raise ValueError(f'未知父版本: {parent}')
            self._seq += 1
            version_id = _new_id(f'{self.run_id}:{self._seq}:{proposal_json}')
            version = Version(
                version_id=version_id,
                parent_id=parent,
                proposal=stored_proposal,
                score=score,
                note=note,
            )
            self.versions.append(version)
            self.current_id = version.version_id
            return version

    def rollback(self, to_version_id: str, reason: str = '') -> Version:
        """把 current 指回指定版本。

        - 目标必须存在
        - 被离开的那一代标 rolled_back（**不删除**）
        - 目标版本如果之前被回滚过，重新激活
        """
        with self._lock:
            target = self.get(to_version_id)
            if target is None:
                raise ValueError(f'未知版本: {to_version_id}')
            leaving = self.get(self.current_id) if self.current_id else None
            if leaving is not None and leaving.version_id != target.version_id:
                if leaving.status == ACTIVE:
                    leaving.status = ROLLED_BACK
                    leaving.note = (leaving.note + ' | ' if leaving.note else '') + (
                        reason or 'rolled back')
            target.status = ACTIVE
            self.current_id = target.version_id
            return target

    # ---------- 读 ----------
    def get(self, version_id: str | None) -> Version | None:
        with self._lock:
            if not version_id:
                return None
            for v in self.versions:
                if v.version_id == version_id:
                    return v
            return None

    @property
    def current(self) -> Version | None:
        return self.get(self.current_id)

    def best(self) -> Version | None:
        """历史最优（不论状态）。回滚的目标永远是它。"""
        with self._lock:
            if not self.versions:
                return None
            return max(self.versions, key=lambda v: (v.score, v.created_at))

    def active_path(self) -> list[Version]:
        """从根到 current 的链路——这条链就是"当前有效的研究史"。"""
        path, node = [], self.current
        seen = set()
        while node is not None and node.version_id not in seen:
            seen.add(node.version_id)
            path.append(node)
            node = self.get(node.parent_id)
        return list(reversed(path))

    # ---------- 落盘 ----------
    def to_dict(self) -> dict:
        with self._lock:
            best = self.best()
            return {
                'run_id': self.run_id,
                'current_id': self.current_id,
                'best_id': best.version_id if best else None,
                'best_score': best.score if best else None,
                'rollbacks': sum(1 for v in self.versions if v.status == ROLLED_BACK),
                'versions': [v.to_dict() for v in self.versions],
            }

    @staticmethod
    def _atomic_write(path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', encoding='utf-8', newline='\n', dir=path.parent,
                    prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
                temporary_path = Path(stream.name)
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            return path
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def dump(self, path: str | Path) -> Path:
        p = Path(path)
        return self._atomic_write(
            p, json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    def render_html(self, path: str | Path) -> Path:
        """零依赖版本树。给评委看"哪一代被抛弃了、为什么"。"""
        p = Path(path)
        best = self.best()
        rows = []
        for v in self.versions:
            marker = ''
            if best and v.version_id == best.version_id:
                marker = '★'
            cls = 'current' if v.version_id == self.current_id else (
                'dead' if v.status == ROLLED_BACK else '')
            model = html.escape(str(v.proposal.get('model', '?')))
            params = html.escape(json.dumps(v.proposal.get('parameters', {}), ensure_ascii=False))
            rows.append(
                f'<tr class="{cls}"><td>{marker}</td>'
                f'<td><code>{v.version_id}</code></td>'
                f'<td>{v.parent_id or "—"}</td>'
                f'<td>{model}</td>'
                f'<td><code>{params}</code></td>'
                f'<td class="score">{v.score:.4f}</td>'
                f'<td>{v.status}</td>'
                f'<td>{html.escape(v.note)}</td></tr>')

        best_score = f'{best.score:.4f}' if best else '—'
        doc = f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>EvoUndo · {html.escape(self.run_id)}</title>
<style>
body{{font:14px/1.6 -apple-system,Segoe UI,sans-serif;margin:24px;color:#222}}
h1{{font-size:18px;margin:0 0 4px}}
.meta{{color:#666;margin-bottom:16px}}
table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid #e5e5e5;padding:8px;text-align:left;vertical-align:top}}
th{{background:#fafafa;font-weight:600}}
tr.current{{background:#eef7ee}}
tr.dead{{color:#999}}
tr.dead td{{text-decoration:line-through}}
.score{{font-variant-numeric:tabular-nums;font-weight:600}}
code{{background:#f3f3f3;padding:1px 4px;border-radius:3px}}
.legend{{margin-top:16px;color:#666;font-size:12px}}
</style></head><body>
<h1>EvoUndo · 版本树</h1>
<div class="meta">run: <code>{html.escape(self.run_id)}</code> ·
当前: <code>{self.current_id or '—'}</code> ·
最优: <code>{best.version_id if best else '—'}</code> ({best_score}) ·
回滚次数: {sum(1 for v in self.versions if v.status == ROLLED_BACK)}</div>
<table><thead><tr>
<th></th><th>版本</th><th>父版本</th><th>模型</th><th>参数</th><th>分数</th><th>状态</th><th>备注</th>
</tr></thead><tbody>
{''.join(rows)}
</tbody></table>
<div class="legend">★ = 历史最优 · 绿底 = 当前有效 · 删除线 = 已被回滚（记录保留，它是"假设被否证"的证据）</div>
</body></html>"""
        return self._atomic_write(p, doc)
