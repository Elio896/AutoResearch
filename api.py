"""Loopback-only development API for persistent classification tasks."""
from pathlib import Path
import sqlite3
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
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


class ResearchInput(BaseModel):
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

    @app.get('/tasks/{task_id}')
    def show_task(task_id: str):
        return invoke(lambda: store.show(task_id))

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
        result = invoke(lambda: jobs.start(task_id, body.max_calls, body.max_seconds))
        background_tasks.add_task(jobs.execute, task_id)
        return result

    @app.get('/demo')
    def demo_dataset():
        from sklearn.datasets import load_iris
        return {'csv_text': load_iris(as_frame=True).frame.to_csv(index=False), 'target': 'target', 'name': 'iris.csv'}

    return app
