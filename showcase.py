"""Isolated, actual-training demonstration. No model keys or external calls."""
import hashlib
import json
import threading
import uuid
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
from corrections import CorrectionStore, fingerprint
from research_system import save
from tools.evaluate_user_corrections import run_case


VERIFIED_STRATEGY=Path(__file__).parent/'artifacts/topic-delivery-verified/strategy.json'

class DemoFeedback(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(min_length=3, max_length=2000)


def register_showcase(app, root):
    root = Path(root) / 'showcase'
    root.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    source = Path(__file__).parent

    def locate(run_id):
        if len(run_id) != 32 or any(c not in '0123456789abcdef' for c in run_id):
            raise HTTPException(404, 'Unknown demo.')
        path = root / run_id
        if not (path / 'state.json').is_file():
            raise HTTPException(404, 'Unknown demo.')
        return path

    def execute(path, label, memory=None):
        proposal = {'model': 'random_forest', 'parameters': {'n_estimators':30,'max_depth':4}}
        result = run_case(path / label, path / 'data.csv', [proposal, proposal], memory)
        keys = [fingerprint(e) for e in result['experiments']]
        return {'label':label, 'fits':len(keys), 'duplicate_fits':len(keys)-len(set(keys)),
                'score':result['summary']['best_score'], 'seconds':result['measured_seconds'],
                'blocked':sum(e['event']=='correction_blocked' for e in result['events']),
                'test_evaluated':False, 'model_calls':0, 'source':'scripted_proposals_real_training',
                'events':[e for e in result['events'] if e['event'] in
                          ('analyst_profile','manager_plan','scientist_proposal','experiment','reviewer_verdict',
                           'correction_blocked','correction_applied','rollback')],
                'evidence': label + '/run.json'}

    @app.get('/showcase', include_in_schema=False)
    def page():
        return FileResponse(source / 'static/showcase.html')

    @app.post('/showcase/runs', status_code=201)
    def start():
        if not lock.acquire(False):
            raise HTTPException(409, 'Another demonstration is running.')
        try:
            from sklearn.datasets import make_moons
            import pandas as pd
            path = root / uuid.uuid4().hex
            path.mkdir()
            x,y = make_moons(n_samples=400,noise=.28,random_state=71)
            pd.DataFrame({'signal_a':x[:,0],'signal_b':x[:,1],'target':y}).to_csv(path/'data.csv',index=False)
            state = {'run_id':path.name, 'phase':'before', 'feedback':None,
                     'dataset_sha256':hashlib.sha256((path/'data.csv').read_bytes()).hexdigest(),
                     'source':'Actual sklearn training with scripted proposals. No live LLM learning claim.'}
            state['before'] = execute(path,'before')
            save(path/'state.json',state)
            return state
        finally:
            lock.release()

    @app.get('/showcase/runs/{run_id}')
    def state(run_id:str):
        path=locate(run_id)
        return json.loads((path/'state.json').read_text(encoding='utf-8'))

    @app.post('/showcase/runs/{run_id}/feedback')
    def feedback(run_id:str,body:DemoFeedback):
        path=locate(run_id)
        if not lock.acquire(False): raise HTTPException(409,'Demonstration busy.')
        try:
            current=state(run_id)
            if current['phase']!='before': raise HTTPException(409,'Feedback is already frozen for this demo.')
            try:
                row=CorrectionStore(path/'memory.sqlite3').add(body.text)
            except ValueError as exc:
                raise HTTPException(400,str(exc)) from None
            current['feedback']=row
            if row['status']=='active': current['phase']='feedback'
            save(path/'state.json',current)
            return current
        finally: lock.release()

    @app.post('/showcase/runs/{run_id}/next')
    def next_session(run_id:str):
        path=locate(run_id)
        if not lock.acquire(False): raise HTTPException(409,'Demonstration busy.')
        try:
            current=state(run_id)
            if current['phase']!='feedback': raise HTTPException(409,'Submit an active correction first.')
            current['after']=execute(path,'after',CorrectionStore(path/'memory.sqlite3'))
            current['memory_events']=CorrectionStore(path/'memory.sqlite3').events()
            current['phase']='after'
            save(path/'state.json',current)
            return current
        finally: lock.release()

    @app.post('/showcase/runs/{run_id}/revoke')
    def revoke(run_id:str):
        path=locate(run_id)
        if not lock.acquire(False): raise HTTPException(409,'Demonstration busy.')
        try:
            current=state(run_id)
            if current['phase']!='after': raise HTTPException(409,'Run the corrected session first.')
            memory=CorrectionStore(path/'memory.sqlite3')
            memory.revoke(current['feedback']['id'])
            current['revoked']=execute(path,'revoked',CorrectionStore(path/'memory.sqlite3'))
            current['memory_events']=memory.events()
            current['phase']='revoked'
            save(path/'state.json',current)
            return current
        finally: lock.release()

    @app.get('/showcase/runs/{run_id}/evidence')
    def evidence(run_id:str):
        path=locate(run_id)
        return FileResponse(path/'state.json',media_type='application/json',filename='showcase-evidence.json')

    @app.post('/showcase/runs/{run_id}/strategy-rollback')
    def rollback(run_id:str):
        path=locate(run_id)
        if not lock.acquire(False): raise HTTPException(409,'Demonstration busy.')
        try:
            current=state(run_id)
            if current.get('strategy_rollback'): raise HTTPException(409,'Already restored; history is retained.')
            original=VERIFIED_STRATEGY
            if not original.is_file(): raise HTTPException(409,'No verified strategy evidence available.')
            import shutil
            from strategy_store import StrategyStore
            target=path/'strategy-copy.json'
            shutil.copyfile(original,target)
            versions=StrategyStore(target)
            active=versions.active()
            if not active or not active.get('parent_id'): raise HTTPException(409,'No previous strategy available.')
            restored=versions.rollback(active['parent_id'],triggers=['explicit_demo_restore'],
                                      evidence={'run_id':run_id,'source':'isolated_copy_of_verified_strategy'},
                                      by='showcase',reason='Explicit restoration in isolated demonstration')
            current['strategy_rollback']={'from':active['strategy_id'],'to':restored['strategy_id'],
                                           'source':'isolated_copy_of_verified_strategy','events':versions.events()}
            save(path/'state.json',current)
            return current
        finally: lock.release()
