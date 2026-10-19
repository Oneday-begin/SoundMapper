import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from .config import STATE

@contextmanager
def connect():
    db = sqlite3.connect(STATE / 'soundmap.sqlite', timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA foreign_keys=ON')
    try:
        with db: yield db
    finally: db.close()

def init():
    with connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS tracks(
          id TEXT PRIMARY KEY, title TEXT NOT NULL, path TEXT NOT NULL, hash TEXT UNIQUE NOT NULL,
          source TEXT NOT NULL, metadata TEXT NOT NULL DEFAULT '{}', created TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS analyses(
          track_id TEXT REFERENCES tracks(id) ON DELETE CASCADE, config_key TEXT, result TEXT,
          PRIMARY KEY(track_id,config_key));
        CREATE TABLE IF NOT EXISTS jobs(
          id TEXT PRIMARY KEY, track_id TEXT REFERENCES tracks(id) ON DELETE CASCADE, status TEXT,
          progress REAL DEFAULT 0, message TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS ratings(
          id TEXT PRIMARY KEY, a TEXT REFERENCES tracks(id) ON DELETE CASCADE,
          b TEXT REFERENCES tracks(id) ON DELETE CASCADE, rater TEXT, session TEXT,
          rhythm INTEGER, tune INTEGER, instruments INTEGER, overall INTEGER,
          created TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE TABLE IF NOT EXISTS import_events(path TEXT PRIMARY KEY, status TEXT, reason TEXT);
        CREATE TABLE IF NOT EXISTS annotations(
          track_id TEXT REFERENCES tracks(id) ON DELETE CASCADE, kind TEXT, config_key TEXT, result TEXT,
          PRIMARY KEY(track_id,kind,config_key));
        CREATE INDEX IF NOT EXISTS idx_tracks_source_title ON tracks(source,title);
        CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status,created);
        ''')
        if 'kind' not in {r[1] for r in db.execute('PRAGMA table_info(jobs)')}:
            db.execute("ALTER TABLE jobs ADD COLUMN kind TEXT NOT NULL DEFAULT 'analysis'")

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def register(path, source='sample', metadata=None, title=None):
    path = Path(path).resolve()
    hash_ = digest(path)
    with connect() as db:
        prior = db.execute('SELECT id FROM tracks WHERE hash=?', (hash_,)).fetchone()
        if prior: return prior['id'], False
        id_ = uuid.uuid4().hex
        db.execute('INSERT INTO tracks(id,title,path,hash,source,metadata) VALUES(?,?,?,?,?,?)',
                   (id_, title or path.stem, str(path), hash_, source, json.dumps(metadata or {})))
    return id_, True

def track(id_):
    with connect() as db: row = db.execute('SELECT * FROM tracks WHERE id=?', (id_,)).fetchone()
    if not row: raise KeyError('Track not found')
    result = dict(row)
    result['metadata'] = json.loads(result['metadata'])
    return result

def tracks(config_key=None, source=None):
    """Fetch the library in one query; optionally join its active analysis cache."""
    with connect() as db:
        if config_key:
            rows=db.execute('''SELECT t.*, a.result AS _analysis_json FROM tracks t
                LEFT JOIN analyses a ON a.track_id=t.id AND a.config_key=?
                WHERE (? IS NULL OR (?='personal' AND t.source!='fma') OR t.source=?)
                ORDER BY t.created,t.title''',(config_key,source,source,source)).fetchall()
        else:
            rows=db.execute('SELECT * FROM tracks ORDER BY created,title').fetchall()
    result=[]
    for row in rows:
        item=dict(row);item['metadata']=json.loads(item['metadata'])
        if config_key:item['_analysis']=json.loads(item.pop('_analysis_json')) if item['_analysis_json'] else None
        result.append(item)
    return result

init()
