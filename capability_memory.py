"""Capability memory keyed by (model, task type, data bucket).

The point is the "placement test" idea: a cheap model is tried on many kinds of data,
the outcomes are recorded per data bucket, and routing then only hands it the kinds of
work where it has actually succeeded. Buckets are coarse on purpose - a handful of
samples per bucket is all a team of this size can honestly collect.
"""
import sqlite3
import time
from contextlib import closing
from pathlib import Path

MIN_ATTEMPTS = 2
MAX_FAILURE_RATE = 0.5
_SCHEMA = '''create table if not exists attempts(
    id integer primary key autoincrement, model text, category text, bucket text,
    ok integer, error_type text, created_at real)'''


def bucket(profile):
    """Coarse description of the data, used as the routing key."""
    profile = profile or {}
    rows = profile.get('rows') or 0
    size = 'small' if rows < 2000 else 'medium' if rows < 8000 else 'large'
    classes = profile.get('class_count') or 2
    cls = '2' if classes <= 2 else '3-4' if classes <= 4 else '5+'
    features = profile.get('features') or 0
    feat = 'few' if features <= 10 else 'mid' if features <= 25 else 'many'
    missing = profile.get('missing_fraction') or 0
    miss = 'clean' if missing < 0.05 else 'holes'
    return '%s|cls%s|feat%s|%s' % (size, cls, feat, miss)


class CapabilityMemory:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(str(self.path), timeout=10)) as conn:
            with conn:
                conn.execute(_SCHEMA)

    def _connect(self):
        conn = sqlite3.connect(str(self.path), timeout=10)
        conn.execute(_SCHEMA)
        return conn

    def record(self, model, category, ok, error_type=None, profile=None):
        if not model or not category:
            return
        with closing(self._connect()) as conn, conn:
            conn.execute('insert into attempts(model,category,bucket,ok,error_type,created_at) values(?,?,?,?,?,?)',
                         (model, category, bucket(profile), 1 if ok else 0, error_type, time.time()))

    def _stats(self, sql, args):
        with closing(self._connect()) as conn:
            total, wins = conn.execute(sql, args).fetchone()
        total, wins = total or 0, wins or 0
        return {'attempts': total, 'successes': wins, 'failures': total - wins,
                'failure_rate': (total - wins) / total if total else 0.0}

    def stats(self, model, category=None):
        sql = 'select count(*), sum(ok) from attempts where model=?'
        args = [model]
        if category:
            sql += ' and category=?'
            args.append(category)
        return self._stats(sql, args)

    def bucket_stats(self, model, category, profile):
        return self._stats('select count(*), sum(ok) from attempts where model=? and category=? and bucket=?',
                           [model, category, bucket(profile)])

    def last_error(self, model, category):
        with closing(self._connect()) as conn:
            row = conn.execute(
                'select error_type from attempts where model=? and category=? and ok=0 order by id desc limit 1',
                (model, category)).fetchone()
        return row[0] if row else None

    def should_skip(self, model, category, profile=None, min_attempts=MIN_ATTEMPTS,
                    max_failure_rate=MAX_FAILURE_RATE):
        """Prefer evidence from the same data bucket; fall back to the overall record."""
        candidates = []
        if profile:
            candidates.append(('bucket', self.bucket_stats(model, category, profile)))
        candidates.append(('all', self.stats(model, category)))
        for scope, stat in candidates:
            if stat['attempts'] < min_attempts:
                continue
            if stat['failure_rate'] < max_failure_rate:
                return None
            return {'model': model, 'category': category, 'scope': scope,
                    'bucket': bucket(profile) if profile else None, **stat,
                    'last_error': self.last_error(model, category),
                    'rule': 'failed %d/%d attempts (%s)' % (stat['failures'], stat['attempts'], scope)}
        return None

    def summary(self, limit=8):
        with closing(self._connect()) as conn:
            rows = conn.execute(
                'select model, category, bucket, count(*), sum(ok) from attempts '
                'group by model, category, bucket order by count(*) desc limit ?', (limit,)).fetchall()
        return [{'model': m, 'category': c, 'bucket': b, 'attempts': n, 'successes': s or 0,
                 'failure_rate': round((n - (s or 0)) / n, 2)} for m, c, b, n, s in rows]

    def capable(self, model, min_attempts=2, min_success_rate=0.5):
        """Buckets where this model has actually succeeded - the placement result."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                'select category, bucket, count(*), sum(ok) from attempts where model=? '
                'group by category, bucket', (model,)).fetchall()
        return [{'category': c, 'bucket': b, 'attempts': n, 'successes': s or 0}
                for c, b, n, s in rows if n >= min_attempts and (s or 0) / n >= min_success_rate]



    def route(self, model, category, profile=None, min_attempts=MIN_ATTEMPTS, good_rate=0.5):
        """Positive assignment: send the work where this model has actually succeeded."""
        if profile:
            stat = self.bucket_stats(model, category, profile)
            scope = 'bucket'
        else:
            stat = self.stats(model, category)
            scope = 'all'
        if stat['attempts'] >= min_attempts:
            if stat['failure_rate'] <= (1 - good_rate):
                return {'tier': 'low', 'reason': 'proven_capable', 'scope': scope,
                        'bucket': bucket(profile) if profile else None, **stat}
            return {'tier': 'high', 'reason': 'proven_incapable', 'scope': scope,
                    'bucket': bucket(profile) if profile else None, **stat}
        return {'tier': 'low', 'reason': 'unproven_explore', 'scope': scope,
                'bucket': bucket(profile) if profile else None, **stat}

    def close(self):
        return None
