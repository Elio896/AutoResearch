"""Durable local task/experiment history and at-most-once final evaluation."""
import argparse
from contextlib import contextmanager
from importlib.metadata import version
import json
from pathlib import Path
import sqlite3
import uuid

from executor import ResearchSession


def runtime_versions():
    return {package: version(package) for package in ('numpy', 'pandas', 'scikit-learn')}


class TaskStore:
    def __init__(self, database='artifacts/research.sqlite3'):
        self.database = Path(database).resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY, csv TEXT NOT NULL, target TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL, seed INTEGER NOT NULL,
                    max_experiments INTEGER NOT NULL, versions TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'open', final_result TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS experiments (
                    task_id TEXT NOT NULL REFERENCES tasks(id), number INTEGER NOT NULL,
                    proposal TEXT NOT NULL, status TEXT NOT NULL,
                    result TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (task_id, number)
                );
                CREATE TABLE IF NOT EXISTS events (
                    number INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(id), payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reports (
                    task_id TEXT PRIMARY KEY REFERENCES tasks(id), payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_jobs (
                    task_id TEXT PRIMARY KEY REFERENCES tasks(id), token TEXT NOT NULL,
                    state TEXT NOT NULL, settings TEXT NOT NULL
                );
            ''')
            columns = {row['name'] for row in db.execute('PRAGMA table_info(tasks)')}
            for name, definition in [('feature_decisions', "TEXT NOT NULL DEFAULT '{}'"),
                                     ('risk_report', 'TEXT'), ('metric', "TEXT NOT NULL DEFAULT 'f1_macro'"), ('pos_label', "TEXT NOT NULL DEFAULT '1'")]:
                if name not in columns:
                    db.execute(f'ALTER TABLE tasks ADD COLUMN {name} {definition}')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, csv, target, seed=42, max_experiments=3, feature_decisions=None, metric="f1_macro", pos_label="1"):
        if type(max_experiments) is not int or not 1 <= max_experiments <= 20:
            raise ValueError('Experiment budget must be an integer between 1 and 20.')
        path = Path(csv).resolve()
        session = ResearchSession(path, target, seed=seed, feature_decisions=feature_decisions, metric=metric, pos_label=pos_label)
        task_id = uuid.uuid4().hex
        with self.connection() as db:
            db.execute('INSERT INTO tasks (id,csv,target,dataset_hash,seed,max_experiments,versions,feature_decisions,risk_report,metric,pos_label) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                       (task_id, str(path), target, session.dataset_sha256, seed,
                        max_experiments, json.dumps(runtime_versions(), sort_keys=True),
                        json.dumps(feature_decisions or {}), json.dumps(session.risk_report), metric, str(pos_label)))
        return task_id

    def show(self, task_id):
        with self.connection() as db:
            row = db.execute('SELECT * FROM tasks WHERE id=?', (task_id,)).fetchone()
            if row is None:
                raise KeyError('Unknown task ID.')
            task = dict(row)
            task['feature_decisions'] = json.loads(task['feature_decisions'])
            task['risk_report'] = json.loads(task['risk_report']) if task['risk_report'] else None
            task['versions'] = json.loads(task['versions'])
            task['final_result'] = json.loads(task['final_result']) if task['final_result'] else None
            rows = db.execute('SELECT * FROM experiments WHERE task_id=? ORDER BY number', (task_id,)).fetchall()
            task['experiments'] = [{**dict(item), 'proposal': json.loads(item['proposal']),
                'result': json.loads(item['result']) if item['result'] else None} for item in rows]
            task['events'] = [json.loads(item['payload']) for item in db.execute(
                'SELECT payload FROM events WHERE task_id=? ORDER BY number', (task_id,))]
            report = db.execute('SELECT payload FROM reports WHERE task_id=?', (task_id,)).fetchone()
            task['research_report'] = json.loads(report['payload']) if report else None
            job = db.execute('SELECT state,settings FROM research_jobs WHERE task_id=?', (task_id,)).fetchone()
            task['research_job'] = {'state': job['state'], 'settings': json.loads(job['settings'])} if job else None
        # Legacy histories stay untouched. Attach a current review warning instead
        # of silently replacing previously recorded scores or feature decisions.
        if task['risk_report'] is None:
            try:
                session = ResearchSession(task['csv'], task['target'], seed=task['seed'],
                                          feature_decisions=task['feature_decisions'], metric=task['metric'], pos_label=task['pos_label'])
                if session.dataset_sha256 != task['dataset_hash']:
                    raise ValueError('Legacy data changed.')
                task['risk_report'] = session.risk_report
            except (ValueError, OSError):
                task['risk_report'] = {'blocked': True, 'findings': [], 'unresolved_columns': [],
                    'excluded_columns': [], 'decisions': {}, 'scope': 'unavailable',
                    'limitation': 'Original data unavailable or changed; integrity cannot be rechecked.'}
        task['legacy_results_at_risk'] = bool(task['experiments'] and task['risk_report']['blocked'])
        return task

    def review(self, task_id, decisions):
        task = self.show(task_id)
        session = ResearchSession(task['csv'], task['target'], seed=task['seed'], feature_decisions=decisions, metric=task['metric'], pos_label=task['pos_label'])
        if session.dataset_sha256 != task['dataset_hash']:
            raise ValueError('Dataset changed since task creation.')
        if task['versions'] != runtime_versions():
            raise ValueError('Runtime changed since task creation.')
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self._check_job(db, task_id)
            if db.execute('SELECT state FROM tasks WHERE id=?', (task_id,)).fetchone()['state'] != 'open':
                raise RuntimeError('Closed tasks cannot change feature decisions; create a new task.')
            if db.execute('SELECT 1 FROM experiments WHERE task_id=?', (task_id,)).fetchone():
                raise RuntimeError('Feature decisions are immutable after experiments; create a new task.')
            db.execute('UPDATE tasks SET feature_decisions=?,risk_report=? WHERE id=?',
                       (json.dumps(decisions), json.dumps(session.risk_report), task_id))
            db.execute('INSERT INTO events (task_id,payload) VALUES (?,?)', (task_id,
                json.dumps({'event': 'feature_reviewed', 'decisions': decisions,
                            'unresolved_columns': session.risk_report['unresolved_columns']})))
        return self.show(task_id)

    def add_event(self, task_id, event):
        with self.connection() as db:
            db.execute('INSERT INTO events (task_id,payload) VALUES (?,?)', (task_id, json.dumps(event)))

    def save_report(self, task_id, report):
        with self.connection() as db:
            db.execute('INSERT INTO reports (task_id,payload) VALUES (?,?)', (task_id, json.dumps(report)))

    def list_tasks(self):
        with self.connection() as db:
            return [dict(row) for row in db.execute('SELECT id,state,target,created_at FROM tasks ORDER BY created_at,id')]

    def _session(self, task):
        if task['versions'] != runtime_versions():
            raise ValueError('Runtime versions changed; cannot reproduce this task.')
        session = ResearchSession(task['csv'], task['target'], seed=task['seed'],
                                  feature_decisions=task['feature_decisions'], metric=task['metric'], pos_label=task['pos_label'])
        if session.dataset_sha256 != task['dataset_hash']:
            raise ValueError('Dataset changed since task creation; restore the original file.')
        session.ensure_ready()
        return session

    def _check_job(self, db, task_id, research_token=None):
        job = db.execute('SELECT state,token FROM research_jobs WHERE task_id=?', (task_id,)).fetchone()
        if job and job['state'] in ('queued', 'running') and research_token != job['token']:
            raise RuntimeError('Autonomous research is active; manual actions are locked.')

    def run(self, task_id, model='logistic_regression', parameters=None, research_token=None):
        from orchestrator import validate_proposal
        proposal = validate_proposal(json.dumps({'model': model, 'parameters': parameters or {}}))
        task = self.show(task_id)
        if task['state'] != 'open':
            raise RuntimeError('Task is not open; further tuning is forbidden.')
        session = self._session(task)
        # Reserve and COMMIT the attempt before fitting. A failed or interrupted fit
        # consumes a slot and cannot silently reset the task's experiment budget.
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self._check_job(db, task_id, research_token)
            current = db.execute('SELECT state,max_experiments FROM tasks WHERE id=?', (task_id,)).fetchone()
            if current['state'] != 'open':
                raise RuntimeError('Task is not open.')
            rows = db.execute('SELECT number,status FROM experiments WHERE task_id=?', (task_id,)).fetchall()
            if any(row['status'] == 'running' for row in rows):
                raise RuntimeError('Task has a running or interrupted experiment; manual recovery required.')
            if len(rows) >= current['max_experiments']:
                raise RuntimeError('Persistent experiment budget exhausted.')
            number = len(rows) + 1
            db.execute('INSERT INTO experiments (task_id,number,proposal,status) VALUES (?,?,?,?)',
                       (task_id, number, json.dumps(proposal, sort_keys=True), 'running'))
        try:
            result = session.run(**proposal)
            result['experiment_id'] = f'exp_{number:03d}'
        except Exception:
            with self.connection() as db:
                db.execute('UPDATE experiments SET status=? WHERE task_id=? AND number=?',
                           ('failed', task_id, number))
            raise RuntimeError('Experiment failed; attempt recorded and budget consumed.') from None
        with self.connection() as db:
            db.execute('UPDATE experiments SET status=?, result=? WHERE task_id=? AND number=?',
                       ('success', json.dumps(result), task_id, number))
        return result

    def finalize(self, task_id):
        task = self.show(task_id)
        if task['state'] != 'open':
            raise RuntimeError('Task is not open; final test cannot be repeated.')
        session = self._session(task)
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            self._check_job(db, task_id)
            state = db.execute('SELECT state FROM tasks WHERE id=?', (task_id,)).fetchone()['state']
            if state != 'open':
                raise RuntimeError('Task is not open; final test cannot be repeated.')
            rows = db.execute('SELECT * FROM experiments WHERE task_id=? ORDER BY number', (task_id,)).fetchall()
            if any(row['status'] == 'running' for row in rows):
                raise RuntimeError('Cannot finalize with an unfinished experiment.')
            successes = [row for row in rows if row['status'] == 'success']
            if not successes:
                raise RuntimeError('No successful experiment to evaluate.')
            selected = max(successes, key=lambda row: json.loads(row['result']).get('validation_score', json.loads(row['result'])['validation_f1']))
            proposal = json.loads(selected['proposal'])
            selected_id = f"exp_{selected['number']:03d}"
            # Durable claim BEFORE test access. A crash after this point fails
            # closed: the task must never automatically re-evaluate its holdout.
            db.execute('UPDATE tasks SET state=? WHERE id=?', ('finalizing', task_id))
        try:
            # Refit the fixed chosen configuration on the original TRAIN split.
            # No pickle loading and no new tuning; runtime/data are fingerprinted.
            replay = session.run(**proposal)
            final = session.finalize(replay['experiment_id'])
            final['experiment_id'] = selected_id
        except Exception:
            with self.connection() as db:
                db.execute('UPDATE tasks SET state=? WHERE id=?', ('final_failed', task_id))
            raise RuntimeError('Final evaluation failed; task remains sealed to avoid a repeated test.') from None
        with self.connection() as db:
            db.execute('UPDATE tasks SET state=?,final_result=? WHERE id=?',
                       ('finalized', json.dumps(final), task_id))
        return final


class StoredResearchSession:
    """Executor facade for NEW orchestrated tasks; manual resume uses TaskStore.run."""
    def __init__(self, store, task_id, research_token=None):
        self.store, self.task_id = store, task_id
        self.research_token = research_token

    @property
    def experiments(self):
        return {row['result']['experiment_id']: (None, row['result'])
                for row in self.store.show(self.task_id)['experiments'] if row['status'] == 'success'}

    @property
    def closed(self):
        return self.store.show(self.task_id)['state'] != 'open'

    def run(self, model='logistic_regression', parameters=None):
        return self.store.run(self.task_id, model, parameters, research_token=self.research_token)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', default='artifacts/research.sqlite3')
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create')
    create.add_argument('csv')
    create.add_argument('--target', required=True)
    create.add_argument('--metric', choices=['f1_macro', 'f1_minority'], default='f1_macro')
    create.add_argument('--pos-label', default='1')
    create.add_argument('--max-experiments', type=int, default=3)
    commands.add_parser('list')
    for command in ('show', 'run', 'finalize', 'review'):
        child = commands.add_parser(command)
        child.add_argument('task_id')
        if command == 'run':
            child.add_argument('--model', default='logistic_regression')
            child.add_argument('--parameters', default='{}')
        if command == 'review':
            child.add_argument('--decisions', required=True)
    args = parser.parse_args()
    store = TaskStore(args.database)
    if args.command == 'create':
        result = {'task_id': store.create(args.csv, args.target, max_experiments=args.max_experiments, metric=args.metric, pos_label=args.pos_label)}
    elif args.command == 'list':
        result = store.list_tasks()
    elif args.command == 'show':
        result = store.show(args.task_id)
    elif args.command == 'run':
        result = store.run(args.task_id, args.model, json.loads(args.parameters))
    elif args.command == 'review':
        result = store.review(args.task_id, json.loads(args.decisions))
    else:
        result = store.finalize(args.task_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
