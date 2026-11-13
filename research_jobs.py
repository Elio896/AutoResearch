"""One bounded local research job at a time; persistent claims, no paid API."""
import json
import uuid

from orchestrator import LocalQwenClient, ResearchOrchestrator
from task_store import StoredResearchSession


class ResearchJobs:
    def __init__(self, store, proposer_factory=None):
        self.store = store
        self.proposer_factory = proposer_factory or (lambda: LocalQwenClient())

    def start(self, task_id, max_calls=4, max_seconds=120):
        if type(max_calls) is not int or not 0 <= max_calls <= 10:
            raise ValueError('Local call budget must be 0..10.')
        if type(max_seconds) not in (int, float) or not 5 <= max_seconds <= 600:
            raise ValueError('Time budget must be 5..600 seconds.')
        task = self.store.show(task_id)
        if task['state'] != 'open' or task['experiments'] or task['research_report']:
            raise RuntimeError('Autonomous research requires a fresh open task.')
        self.store._session(task)  # Verify data/runtime before reserving a job.
        settings = {'max_calls': max_calls, 'max_seconds': max_seconds}
        with self.store.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            current = db.execute('SELECT state FROM tasks WHERE id=?', (task_id,)).fetchone()
            if current['state'] != 'open' or db.execute('SELECT 1 FROM experiments WHERE task_id=?', (task_id,)).fetchone():
                raise RuntimeError('Task is no longer fresh.')
            if db.execute('SELECT 1 FROM research_jobs WHERE state IN (?,?)', ('queued', 'running')).fetchone():
                raise RuntimeError('Another local research job is active. Wait for it to finish.')
            if db.execute('SELECT 1 FROM research_jobs WHERE task_id=?', (task_id,)).fetchone():
                raise RuntimeError('This task already has a research run; inspect its history.')
            db.execute('INSERT INTO research_jobs (task_id,token,state,settings) VALUES (?,?,?,?)',
                       (task_id, uuid.uuid4().hex, 'queued', json.dumps(settings)))
        self.store.add_event(task_id, {'event': 'research_queued', 'settings': settings})
        return {'task_id': task_id, 'job_status': 'queued'}

    def execute(self, task_id):
        with self.store.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            job = db.execute('SELECT * FROM research_jobs WHERE task_id=?', (task_id,)).fetchone()
            if not job or job['state'] != 'queued':
                return
            db.execute('UPDATE research_jobs SET state=? WHERE task_id=?', ('running', task_id))
            token, settings = job['token'], json.loads(job['settings'])
        try:
            task = self.store.show(task_id)
            self.store.add_event(task_id, {'event': 'research_started'})
            report = ResearchOrchestrator(
                StoredResearchSession(self.store, task_id, research_token=token), self.proposer_factory(),
                max_experiments=task['max_experiments'], max_calls=settings['max_calls'],
                max_seconds=settings['max_seconds'],
                emit=lambda event: self.store.add_event(task_id, event)).run()
            report['task_id'] = task_id
            self.store.save_report(task_id, report)
            state = 'completed'
        except Exception:
            # Persist a safe failure without provider responses, paths or secrets.
            self.store.add_event(task_id, {'event': 'research_failed',
                'reason': 'Research interrupted by an execution error; inspect task history.'})
            state = 'failed'
        with self.store.connection() as db:
            db.execute('UPDATE research_jobs SET state=? WHERE task_id=?', (state, task_id))
