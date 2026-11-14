"""Local-only API for the competition campaign, with persistent evidence exports."""
import io
import json
import re
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Literal

from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from research_system import ResearchCampaign, save
from corrections import CorrectionStore, CorrectionGuard


class CampaignInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    mode: Literal['rules', 'single_model'] = 'rules'
    max_calls: int = Field(default=36, ge=0, le=50, strict=True)
    max_tokens: int = Field(default=150000, ge=0, le=500000, strict=True)
    max_seconds: int = Field(default=180, ge=10, le=300, strict=True)
    max_fits: int = Field(default=80, ge=1, le=120, strict=True)
    csv_text: str | None = Field(default=None, max_length=2_000_000)
    target: str = Field(default='target', min_length=1, max_length=200)


class FeedbackInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(min_length=3, max_length=2000)
    kind: Literal['reject_duplicates', 'forbid_model', 'require_missing_indicator', 'no_paid_calls', 'protect_holdout'] | None = None
    value: Literal['logistic_regression', 'random_forest', 'hist_gradient_boosting'] | None = None
    scope: Literal['global', 'dataset'] = 'global'
    dataset_sha256: str | None = Field(default=None, pattern='^[0-9a-f]{64}$')
    source_run_id: str | None = Field(default=None, pattern='^[0-9a-f]{32}$')


def register_research_routes(app, root, static_root):
    corrections = CorrectionStore(Path(root) / 'user-corrections.sqlite3')
    root = Path(root) / 'campaigns'
    root.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def directory(run_id):
        if not re.fullmatch(r'[0-9a-f]{32}', run_id):
            raise HTTPException(404, 'Unknown campaign.')
        path = root / run_id
        if not (path / 'state.json').is_file():
            raise HTTPException(404, 'Unknown campaign.')
        return path

    @app.get('/research-agent', include_in_schema=False)
    def research_agent_page():
        return FileResponse(static_root / 'research-agent.html')

    @app.get('/system/config')
    def research_config():
        import os
        return {'single_model_configured': bool(os.getenv('LIGHT_MODEL_API_KEY') or os.getenv('DEEPSEEK_API_KEY')),
                'default_mode': 'rules', 'local_model_required': False,
                'note': 'Rules mode executes real experiments without LLMs. Single-model mode uses server environment credentials.'}

    @app.get('/system/feedback')
    def feedback_state():
        return {'records': corrections.all(), 'events': corrections.events(),
                'guarantee': 'Recognized active constraints are enforced before execution; unknown feedback stays pending.'}

    @app.post('/system/feedback', status_code=201)
    def add_feedback(body: FeedbackInput):
        if body.source_run_id:
            directory(body.source_run_id)
        try:
            return corrections.add(**body.model_dump())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.post('/system/feedback/{feedback_id}/revoke')
    def revoke_feedback(feedback_id: str):
        try:
            return corrections.revoke(feedback_id)
        except KeyError:
            raise HTTPException(404, 'Feedback record not found.') from None

    @app.get('/system/innovations')
    def innovation_evidence():
        evidence = Path(__file__).parent / 'artifacts' / 'feedback-learning-final' / 'report.json'
        return {'designs': [
            {'title': '把用户纠错变成执行约束', 'detail': '先检查再训练；拦截重复实验和禁用模型，自动补齐缺失指示。'},
            {'title': '带作用域的跨会话记忆', 'detail': '记忆可跨重启；数据专属反馈不影响其他数据，冲突和未知反馈不会直接生效。'},
            {'title': '研究与策略双层可回退', 'detail': '单次实验按证据回退，策略需独立验收才能升级；反馈记录也可撤销。'},
            {'title': '预算与证据一起传递', 'detail': '多个角色共享调用和 token 配额；未知用量保留预算，结果带源码与数据快照。'}],
                'evaluation': json.loads(evidence.read_text(encoding='utf-8')) if evidence.is_file() else None,
                'claim_boundary': 'Project design contributions, not a claim of global novelty. Scripted replay verifies constraints, not live LLM learning.'}

    def execute(path, settings, client):
        try:
            save(path / 'state.json', {'state': 'running', 'run_id': path.name})
            development_csv = None
            if settings.csv_text:
                development_csv = path / 'development-upload.csv'
                development_csv.write_text(settings.csv_text, encoding='utf-8')
            campaign = ResearchCampaign(path, mode=settings.mode, client=client,
                                        max_calls=settings.max_calls, max_tokens=settings.max_tokens,
                                        max_seconds=settings.max_seconds, max_fits=settings.max_fits,
                                        correction_store=corrections)
            report = campaign.run(development_csv, settings.target)
            save(path / 'state.json', {'state': 'completed', 'run_id': path.name,
                                      'rounds': len(report['rounds'])})
        except Exception as exc:
            # Arbitrary provider errors can contain secrets; persist only an error type.
            save(path / 'state.json', {'state': 'failed', 'run_id': path.name,
                                      'error_type': type(exc).__name__,
                                      'reason': 'Campaign could not finish. Partial evidence is retained; check data, model and budgets.'})
        finally:
            lock.release()

    @app.post('/system/campaigns', status_code=202)
    def start_campaign(body: CampaignInput, background_tasks: BackgroundTasks):
        client = None
        if body.mode == 'single_model':
            if CorrectionGuard(corrections.snapshot()).no_paid_calls:
                raise HTTPException(409, 'An active user correction forbids paid model calls. Use rules mode or revoke that correction.')
            from light_model import from_env
            try:
                client = from_env()
            except (ValueError, RuntimeError):
                raise HTTPException(409, 'Invalid server model configuration.') from None
            if client is None:
                raise HTTPException(409, 'Configure a model API key on the server before choosing single-model mode.')
            client.max_calls = body.max_calls
        if not lock.acquire(blocking=False):
            raise HTTPException(409, 'A research campaign is already running.')
        path = root / uuid.uuid4().hex
        try:
            path.mkdir()
            save(path / 'state.json', {'state': 'queued', 'run_id': path.name})
        except Exception:
            lock.release()
            raise
        background_tasks.add_task(execute, path, body, client)
        return {'run_id': path.name, 'state': 'queued'}

    @app.get('/system/campaigns/{run_id}')
    def show_campaign(run_id: str):
        path = directory(run_id)
        state = json.loads((path / 'state.json').read_text(encoding='utf-8'))
        for name in ('report', 'progress'):
            if (path / f'{name}.json').is_file():
                state[name] = json.loads((path / f'{name}.json').read_text(encoding='utf-8'))
        if (path / 'events.jsonl').is_file():
            state['events'] = [json.loads(line) for line in
                               (path / 'events.jsonl').read_text(encoding='utf-8').splitlines()[-80:]]
        if (path / 'strategy.json').is_file():
            from strategy_store import StrategyStore
            store = StrategyStore(path / 'strategy.json')
            state['current_strategy'] = store.active()
            state['strategy_events'] = store.events()
        return state

    @app.post('/system/campaigns/{run_id}/rollback')
    def operator_rollback(run_id: str):
        path = directory(run_id)
        if not lock.acquire(blocking=False):
            raise HTTPException(409, 'Wait for the active research campaign to finish.')
        try:
            state = json.loads((path / 'state.json').read_text(encoding='utf-8'))
            if state['state'] != 'completed':
                raise HTTPException(409, 'Only a completed campaign can be restored.')
            from strategy_store import StrategyStore
            store = StrategyStore(path / 'strategy.json')
            active = store.active()
            parent = store.get(active['parent_id']) if active and active.get('parent_id') else None
            if parent is None:
                raise HTTPException(409, 'No previous accepted strategy to restore.')
            if parent['status'] != 'retired':
                raise HTTPException(409, 'Previous strategy is not a stable retired version.')
            restored = store.rollback(parent['strategy_id'], triggers=['operator_request'],
                                      evidence={'run_id': run_id, 'source': 'explicit_restore_button'},
                                      by='operator', reason='User requested previous stable strategy')
            return {'strategy_id': restored['strategy_id'], 'restored': True,
                    'note': 'Research history and the original final test remain immutable.'}
        finally:
            lock.release()

    @app.get('/system/campaigns/{run_id}/export')
    def export_campaign(run_id: str):
        path = directory(run_id)
        state = json.loads((path / 'state.json').read_text(encoding='utf-8'))
        if state['state'] in ('queued', 'running'):
            raise HTTPException(409, 'Wait for completion before exporting a consistent evidence snapshot.')
        output = io.BytesIO()
        source = Path(__file__).parent
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in path.rglob('*'):
                if file.is_file() and file.suffix in ('.json', '.jsonl', '.csv'):
                    archive.write(file, 'evidence/' + file.relative_to(path).as_posix())
            for file in (path / 'source').rglob('*'):
                if file.is_file():
                    archive.write(file, 'source/' + file.relative_to(path / 'source').as_posix())
            archive.writestr('REPRODUCE.txt', 'Evidence and source snapshot. From source/:\n'
                             'python tools/run_research_system.py --output replay\n'
                             'Rules replay retrains with frozen seeds. Model replay requires server credentials; '
                             'nondeterministic providers may differ. For custom development data add '
                             '--development-csv ../evidence/development-upload.csv --target YOUR_TARGET. '
                             'If protocol.json includes user_corrections, also add '
                             '--corrections-json ../evidence/protocol.json (same constraints, new local record IDs). '
                             'See evidence/protocol.json. Source bytes are captured when the run starts.\n')
        return Response(output.getvalue(), media_type='application/zip', headers={
            'Content-Disposition': f'attachment; filename="research-{run_id}.zip"', 'Cache-Control': 'no-store'})
