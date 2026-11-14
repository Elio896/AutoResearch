"""Loopback-only development API for persistent classification tasks."""
from typing import Literal
from pathlib import Path
import sqlite3
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from task_store import TaskStore
from research_jobs import ResearchJobs


class TaskInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    csv_text: str = Field(min_length=1, max_length=2_000_000)
    target: str = Field(min_length=1, max_length=200)
    metric: str = 'f1_macro'
    pos_label: str = Field(default='1', max_length=200)
    max_experiments: int = Field(default=3, ge=1, le=20, strict=True)


class ExperimentInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: str = 'logistic_regression'
    parameters: dict = Field(default_factory=dict)


class FiveRolesInput(BaseModel):
    """现场演示用：跑一遍五角色研究闭环。"""
    model_config = ConfigDict(extra='forbid')
    csv_text: str = Field(min_length=1, max_length=2_000_000)
    target: str = Field(min_length=1, max_length=200)
    metric: str = 'f1_minority'
    pos_label: str = Field(default='1', max_length=200)
    max_experiments: int = Field(default=3, ge=1, le=6, strict=True)
    model: str = 'qwen2.5:7b-instruct-q4_K_M'
    offline: bool = True    # True: 不依赖 Ollama，用确定性 proposer（演示兜底）


class ResearchInput(BaseModel):
    baseline_policy: Literal["logreg", "fixed_search"] = "logreg"
    model_config = ConfigDict(extra='forbid')
    max_calls: int = Field(default=4, ge=0, le=10, strict=True)
    max_seconds: int = Field(default=120, ge=5, le=600, strict=True)


class FeatureDecision(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: str
    reason: str = Field(min_length=3, max_length=500)


class ReviewInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    decisions: dict[str, FeatureDecision]


def create_app(data_directory=None):
    root = Path(data_directory or Path(__file__).parent / 'artifacts' / 'api').resolve()
    uploads = root / 'uploads'
    uploads.mkdir(parents=True, exist_ok=True)
    store = TaskStore(root / 'research.sqlite3')
    jobs = ResearchJobs(store)
    app = FastAPI(title='EvoResearch', version='0.1.0', description='Local development API; no authentication or paid model calls.')
    static_root = Path(__file__).parent / 'static'
    app.mount('/static', StaticFiles(directory=static_root), name='static')

    @app.get('/', include_in_schema=False)
    def workbench():
        return FileResponse(static_root / 'index.html')

    @app.get('/demo', include_in_schema=False)
    def demo_page():
        """现场演示页：一屏讲完整条链，自带示例数据，不依赖模型服务。"""
        return FileResponse(static_root / 'demo.html')

    def invoke(action):
        try:
            return action()
        except KeyError:
            raise HTTPException(status_code=404, detail='Task not found.') from None
        except (ValueError, UnicodeError):
            raise HTTPException(status_code=400, detail='Invalid dataset or experiment configuration.') from None
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        except sqlite3.OperationalError:
            raise HTTPException(status_code=503, detail='Task database busy or unavailable.') from None
        except OSError:
            raise HTTPException(status_code=409, detail='Task dataset is unavailable.') from None

    @app.get('/health')
    def health():
        return {'status': 'ok', 'cloud_enabled': False}

    @app.get('/tasks')
    def list_tasks():
        return invoke(store.list_tasks)

    @app.post('/tasks', status_code=201)
    def create_task(body: TaskInput):
        raw = body.csv_text.encode('utf-8')
        if len(raw) > 2_000_000:
            raise HTTPException(status_code=413, detail='CSV exceeds 2 MB.')
        # Clients upload text, never server-side paths or executable objects.
        path = uploads / (uuid.uuid4().hex + '.csv')
        with path.open('xb') as file:
            file.write(raw)
        try:
            task_id = invoke(lambda: store.create(path, body.target, max_experiments=body.max_experiments, metric=body.metric, pos_label=body.pos_label))
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return {'task_id': task_id, 'state': 'open', 'risk_report': store.show(task_id)['risk_report']}

    @app.post('/research/five-roles', status_code=201)
    def run_five_roles(body: FiveRolesInput):
        """现场演示：跑一遍五角色研究闭环，返回版本树和事件。

        offline=True（默认）时用确定性 proposer，不依赖 Ollama——
        现场万一模型服务挂了也能演示整条链（角色/回滚/轨迹都在跑）。
        """
        import tempfile
        from pathlib import Path as P
        from agents import Budget
        from executor import ResearchSession
        from research_loop import ResearchLoop

        tmp = tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False, encoding='utf-8')
        try:
            tmp.write(body.csv_text)
            tmp.close()
            session = ResearchSession(tmp.name, body.target, metric=body.metric,
                                      pos_label=body.pos_label)

            if body.offline:
                # 确定性 proposer：按"没试过的模型优先"轮流给方案，不调模型
                tried = {'n': 0}
                catalogue = [
                    {'model': 'hist_gradient_boosting', 'parameters': {'max_depth': 4}},
                    {'model': 'random_forest', 'parameters': {'n_estimators': 50}},
                    {'model': 'logistic_regression', 'parameters': {'C': 0.5}},
                ]

                def proposer(messages, timeout=None):
                    idx = tried['n'] % len(catalogue)
                    tried['n'] += 1
                    proposal = dict(catalogue[idx])
                    proposal.update({'hypothesis': f'offline proposal {idx}',
                                     'expected_direction': 'up',
                                     'falsifier': 'validation score does not improve'})
                    import json as _json
                    return {'message': {'content': _json.dumps(proposal)},
                            'prompt_eval_count': 0, 'eval_count': 0}
            else:
                from orchestrator import LocalQwenClient
                proposer = LocalQwenClient(model=body.model)

            budget = Budget(max_experiments=body.max_experiments, max_seconds=120.0,
                            max_local_calls=body.max_experiments + 1)
            loop = ResearchLoop(session, proposer=proposer, budget=budget, patience=2)
            result = loop.run()
            return {'summary': result['summary'], 'events': result['events'],
                    'experiments': result['experiments'], 'offline': body.offline}
        finally:
            try:
                P(tmp.name).unlink()
            except OSError:
                pass

    @app.get('/tasks/{task_id}')
    def show_task(task_id: str):
        return invoke(lambda: store.show(task_id))

    @app.get('/evoundo/runs')
    def list_evoundo_runs():
        """列出所有研究任务的版本树（只读）。"""
        import json
        from pathlib import Path as P
        base = P(data_directory or '.').resolve().parent / 'artifacts' / 'evoundo'
        if not base.exists():
            base = P('artifacts') / 'evoundo'
        runs = []
        if base.exists():
            for run_dir in sorted(base.iterdir()):
                vf = run_dir / 'versions.json'
                if not vf.is_file():
                    continue
                try:
                    data = json.loads(vf.read_text(encoding='utf-8'))
                except (json.JSONDecodeError, OSError):
                    continue
                runs.append({'run_id': data.get('run_id') or run_dir.name,
                             'versions': len(data.get('versions') or []),
                             'rollbacks': data.get('rollbacks', 0),
                             'best_score': data.get('best_score')})
        return {'runs': runs, 'count': len(runs)}

    @app.get('/evoundo/{run_id}')
    def show_evoundo(run_id: str):
        """一个任务的完整版本树（只读）。"""
        import json
        from pathlib import Path as P
        for base in (P('artifacts') / 'evoundo', P(data_directory or '.').resolve().parent / 'artifacts' / 'evoundo'):
            vf = base / run_id / 'versions.json'
            if vf.is_file():
                return json.loads(vf.read_text(encoding='utf-8'))
        raise HTTPException(status_code=404, detail='No such run.')

    @app.get('/experience')
    def show_experience():
        """经验池的进化账（只读）：学到多少、用过多少、有效率。"""
        from experience import ExperienceStore
        from pathlib import Path as P
        store = ExperienceStore(P('artifacts') / 'experience.jsonl')
        return {'lineage': store.lineage(),
                'active': [{'id': r.get('id'), 'advice': r.get('advice'),
                            'used_count': r.get('used_count', 0),
                            'success_count': r.get('success_count', 0)}
                           for r in store.active_pool()[:20]]}

    @app.get('/tasks/{task_id}/export')
    def export_task(task_id: str, format: Literal['json', 'markdown', 'zip'] = 'markdown'):
        import json
        from research_export import build_manifest, markdown, bundle
        manifest = invoke(lambda: build_manifest(store.show(task_id)))
        if format == 'json':
            content, media, extension = json.dumps(manifest, ensure_ascii=False, indent=2), 'application/json', 'json'
        elif format == 'zip':
            content, media, extension = bundle(manifest), 'application/zip', 'zip'
        else:
            content, media, extension = markdown(manifest), 'text/markdown', 'md'
        return Response(content, media_type=media, headers={
            'Content-Disposition': f'attachment; filename="evoresearch-{manifest["task"]["id"]}.{extension}"',
            'Cache-Control': 'no-store'})

    @app.post('/tasks/{task_id}/review')
    def review_features(task_id: str, body: ReviewInput):
        return invoke(lambda: store.review(task_id, {key: value.model_dump() for key, value in body.decisions.items()}))

    @app.post('/tasks/{task_id}/experiments')
    def run_experiment(task_id: str, body: ExperimentInput):
        return invoke(lambda: store.run(task_id, body.model, body.parameters))

    @app.post('/tasks/{task_id}/finalize')
    def finalize_task(task_id: str):
        return invoke(lambda: store.finalize(task_id))

    @app.post('/tasks/{task_id}/research', status_code=202)
    def start_research(task_id: str, body: ResearchInput, background_tasks: BackgroundTasks):
        result = invoke(lambda: jobs.start(task_id, body.max_calls, body.max_seconds, body.baseline_policy))
        background_tasks.add_task(jobs.execute, task_id)
        return result

    @app.get('/demo')
    def demo_dataset():
        from sklearn.datasets import load_iris
        return {'csv_text': load_iris(as_frame=True).frame.to_csv(index=False), 'target': 'target', 'name': 'iris.csv'}

    return app
