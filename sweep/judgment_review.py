"""Local case review UI with durable, versioned human corrections.

Run: python -m sweep.judgment_review --data <queue.json> --db <reviews.sqlite3>
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import sqlite3
from urllib.parse import urlparse

from sweep.codegraph_judgment import OPTIONS, identity

ASSETS = Path(__file__).with_name('judgment_review_ui')
STATUSES = {'approved', 'corrected', 'excluded', 'deferred', 'pending'}


class Conflict(ValueError):
    pass


class ReviewStore:
    def __init__(self, data, db):
        self.data = data
        self.cases = {c['key']: c for c in data['cases']}
        if len(self.cases) != len(data['cases']):
            raise ValueError('duplicate case key')
        self.dataset_id = identity(data)
        for case in self.cases.values():
            if (case['draft_label'] not in OPTIONS or not 0 <= case['confidence'] <= 1
                    or not isinstance(case['state']['observations'], list)):
                raise ValueError('invalid case')
        self.db = Path(db)
        self.db.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript('''
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS reviews (
                    key TEXT PRIMARY KEY, revision INTEGER NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, key TEXT NOT NULL, revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL, payload TEXT NOT NULL,
                    UNIQUE(key, revision));
            ''')
            previous = con.execute("SELECT value FROM metadata WHERE key='dataset_id'").fetchone()
            if previous and previous[0] != self.dataset_id:
                raise ValueError('database belongs to a different case snapshot; use a fresh database')
            con.execute("INSERT OR IGNORE INTO metadata VALUES ('dataset_id', ?)", (self.dataset_id,))

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    def snapshot(self):
        with self.connect() as con:
            reviews = {r['key']: json.loads(r['payload']) for r in con.execute('SELECT * FROM reviews')}
        cases = [case | {'review': reviews.get(key, {'status': 'pending', 'revision': 0})}
                 for key, case in self.cases.items()]
        return {'dataset_id': self.dataset_id, 'options': OPTIONS, 'cases': cases,
                'metadata': self.data.get('metadata', {}), 'threshold': 0.8}

    def save(self, key, payload):
        if key not in self.cases:
            raise KeyError('unknown case')
        if not isinstance(payload, dict):
            raise ValueError('invalid review')
        status = payload.get('status')
        if status not in STATUSES:
            raise ValueError('invalid review status')
        expected = payload.get('revision')
        if type(expected) is not int or expected < 0:
            raise ValueError('expected revision is required')
        reviewer = payload.get('reviewer', '').strip()
        notes = payload.get('notes', '').strip()
        if not reviewer or len(reviewer) > 120 or len(notes) > 10000:
            raise ValueError('reviewer is required; reviewer/notes exceed limits')
        original = self.cases[key]
        state = payload.get('state', original['state'])
        if (not isinstance(state, dict) or set(state) != {'report', 'observations'}
                or not isinstance(state['report'], str) or not state['report'].strip()
                or len(state['report']) > 10000 or not isinstance(state['observations'], list)
                or not 1 <= len(state['observations']) <= 100
                or any(not isinstance(x, str) or not x.strip() or len(x) > 10000 for x in state['observations'])):
            raise ValueError('a report and nonempty observation lines are required')
        label = payload.get('label')
        if status in {'approved', 'corrected'} and label not in OPTIONS:
            raise ValueError('choose a valid label before approving')
        if label is not None and label not in OPTIONS:
            raise ValueError('unknown label')
        changed = label != original['draft_label'] or state != original['state']
        if status == 'approved' and changed:
            status = 'corrected'
        if status == 'corrected' and not changed:
            status = 'approved'
        if status in {'corrected', 'excluded'} and not notes:
            raise ValueError('explain the correction or exclusion')
        if status in {'approved', 'corrected'} and payload.get('facts_checked') is not True:
            raise ValueError('check that the facts and decision point are supported')
        family_concern = payload.get('family_concern', '').strip()
        if len(family_concern) > 4000:
            raise ValueError('family concern exceeds limit')
        review = {'status': status, 'revision': expected+1, 'reviewer': reviewer,
                  'label': label, 'state': state, 'notes': notes,
                  'facts_checked': payload.get('facts_checked') is True,
                  'family_concern': family_concern,
                  'created_at': datetime.now(timezone.utc).isoformat(),
                  'reviewer_kind': 'human_ui'}
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE')
            previous = con.execute('SELECT revision FROM reviews WHERE key=?', (key,)).fetchone()
            actual = previous['revision'] if previous else 0
            if actual != expected:
                raise Conflict('This case changed in another tab. Reload it before saving.')
            con.execute('INSERT OR REPLACE INTO reviews VALUES (?, ?, ?)',
                        (key, review['revision'], json.dumps(review)))
            con.execute('INSERT INTO events(key,revision,created_at,payload) VALUES (?, ?, ?, ?)',
                        (key, review['revision'], review['created_at'], json.dumps(review)))
        return review

    def history(self, key):
        if key not in self.cases:
            raise KeyError('unknown case')
        with self.connect() as con:
            return [json.loads(r['payload']) for r in con.execute(
                'SELECT payload FROM events WHERE key=? ORDER BY revision', (key,))]

    def export(self):
        snapshot = self.snapshot()
        eligible = []
        for case in snapshot['cases']:
            review = case['review']
            if review['status'] not in {'approved', 'corrected'} or review['family_concern']:
                continue
            label = review['label']; options = list(case['question']['criteria'])
            eligible.append({'key': identity([review['state'], case['question']])[:24],
                             'original_key': case['key'], 'state': review['state'],
                             'question': case['question'], 'label': label,
                             'target': [float(k == label) for k in options],
                             'hard_index': options.index(label), 'keys': options,
                             'source': 'codegraph/judgment', 'family_id': case['family_id'],
                             'role': case['role'], 'gold_status': 'reviewed',
                             'reviewer': review['reviewer'], 'review_revision': review['revision']})
        # Human-edited states can create new duplicates across frozen roles. Keep
        # them out of the training export rather than silently merging/resplitting.
        state_groups = {}
        for row in eligible:
            norm = json.dumps(row['state'], sort_keys=True).casefold().split()
            state_groups.setdefault(identity(norm), []).append(row)
        duplicate_keys = {r['original_key'] for group in state_groups.values() if len(group) > 1 for r in group}
        return snapshot | {'exported_at': datetime.now(timezone.utc).isoformat(),
                           'reviewed_cases': [r for r in eligible if r['original_key'] not in duplicate_keys],
                           'duplicate_cases_held_back': sorted(duplicate_keys),
                           'calibration_ready': False,
                           'readiness_note': 'Human approval exports labels only. Recheck family/state separation, class coverage, sample support and serving parity before fitting or certification.'}


def make_server(store, host='127.0.0.1', port=8765, benchmark=None):
    if host not in {'127.0.0.1', 'localhost', '::1'}:
        raise ValueError('review UI must bind to loopback')
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # Avoid writing case contents or query strings into terminal logs.
            pass

        def reply(self, status, body, content_type='application/json; charset=utf-8'):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.end_headers(); self.wfile.write(body)

        def allowed_host(self):
            return self.headers.get('Host') in {
                f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

        def do_GET(self):
            if not self.allowed_host():
                return self.reply(403, {'error': 'loopback Host required'})
            path = urlparse(self.path).path
            if path == '/api/benchmark' and benchmark is not None:
                return self.reply(200, benchmark)
            if path == '/api/cases':
                snapshot = store.snapshot()
                if benchmark is not None:
                    snapshot['metadata'] = snapshot['metadata'] | {'archived':True,
                        'archive_note':'This judgment-label study is retired. Existing corrections are preserved; no further review is needed.'}
                return self.reply(200, snapshot | {'csrf_token': token})
            if path == '/api/export':
                return self.reply(200, store.export())
            if path.startswith('/api/history/'):
                try:
                    return self.reply(200, {'events': store.history(path.split('/')[-1])})
                except KeyError:
                    return self.reply(404, {'error': 'unknown case'})
            assets = {'/': ('benchmark.html' if benchmark is not None else 'index.html', 'text/html; charset=utf-8'),
                      '/review': ('index.html', 'text/html; charset=utf-8'),
                      '/benchmark.js': ('benchmark.js', 'text/javascript; charset=utf-8'),
                      '/benchmark.css': ('benchmark.css', 'text/css; charset=utf-8'),
                      '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                      '/style.css': ('style.css', 'text/css; charset=utf-8')}
            if path not in assets:
                return self.reply(404, {'error': 'not found'})
            name, content_type = assets[path]
            return self.reply(200, (ASSETS/name).read_bytes(), content_type)

        def do_POST(self):
            path = urlparse(self.path).path
            origin = self.headers.get('Origin')
            if (not self.allowed_host() or self.headers.get('X-Review-Token') != token
                    or (origin is not None and origin not in {
                        f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'})):
                return self.reply(403, {'error': 'local review token and matching origin required'})
            if path != '/api/review':
                return self.reply(404, {'error': 'not found'})
            if benchmark is not None:
                return self.reply(410, {'error':'This study is retired. Existing corrections are preserved in a read-only archive; no further review is needed.'})
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.reply(415, {'error': 'JSON required'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 131072:
                    return self.reply(413, {'error': 'invalid review size'})
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('invalid review object')
                review = store.save(payload.get('key'), payload)
                return self.reply(200, {'review': review})
            except Conflict as exc:
                return self.reply(409, {'error': str(exc)})
            except KeyError:
                return self.reply(404, {'error': 'unknown case'})
            except (ValueError, TypeError, AttributeError) as exc:
                return self.reply(400, {'error': str(exc)})

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--export', type=Path, help='Export a review snapshot and exit without starting a server')
    parser.add_argument('--benchmark', type=Path, help='Serve the corrected replay audit; keep prior reviews read-only')
    args = parser.parse_args()
    store = ReviewStore(json.loads(args.data.read_text()), args.db)
    if args.export:
        with args.export.open('x') as handle:
            json.dump(store.export(),handle,indent=2)
            handle.write('\n')
        print(f'Review snapshot saved: {args.export.resolve()}',flush=True)
        return
    benchmark = json.loads(args.benchmark.read_text()) if args.benchmark else None
    server = make_server(store, port=args.port, benchmark=benchmark)
    print(f'LARS review: http://127.0.0.1:{server.server_port}', flush=True)
    print(f'{len(store.cases)} cases; corrections saved to {args.db.resolve()}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
