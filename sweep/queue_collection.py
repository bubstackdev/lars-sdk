"""Local prospective queue evidence journal and blinded target-utility review."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import sqlite3
from urllib.parse import urlparse

ASSET=Path(__file__).with_name('judgment_review_ui')/'collection.html'
RATINGS={'reprocess_only','discard_only','either','insufficient'}


def now():return datetime.now(timezone.utc).isoformat()


def instant(value):
    try: result=datetime.fromisoformat(value.replace('Z','+00:00'))
    except (AttributeError,ValueError):raise ValueError('Use a timestamp with timezone.')
    if result.tzinfo is None:raise ValueError('Timestamp needs a timezone.')
    return result


def text(value,name,limit=10000):
    if not isinstance(value,str) or not value.strip() or len(value)>limit:raise ValueError(name+' is required or too long.')
    if re.search(r'(?:password|pwd|client_secret|access_token|api_key)\s*[=:]|<(?:\w+:)?pw>',value,re.I):
        raise ValueError('Remove credential fields before saving evidence.')
    return value.strip()


def url(value):
    value=text(value,'Source URL',2000);p=urlparse(value)
    if p.scheme not in {'http','https'} or not p.netloc or p.username or p.password:raise ValueError('Use an HTTP(S) source URL without credentials.')
    return value


class Collection:
    def __init__(self,path):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as con:
            con.executescript('''
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, incident TEXT NOT NULL, kind TEXT NOT NULL,
                    reviewer TEXT NOT NULL, recorded_at TEXT NOT NULL, payload TEXT NOT NULL,
                    previous_hash TEXT NOT NULL, hash TEXT NOT NULL,
                    UNIQUE(incident,kind,reviewer));
                CREATE TRIGGER IF NOT EXISTS no_event_update BEFORE UPDATE ON events
                    BEGIN SELECT RAISE(ABORT,'Journal entries are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS no_event_delete BEFORE DELETE ON events
                    BEGIN SELECT RAISE(ABORT,'Journal entries are immutable'); END;
            ''')

    @contextmanager
    def connect(self):
        con=sqlite3.connect(self.path,timeout=10);con.row_factory=sqlite3.Row
        try:
            with con:yield con
        finally:con.close()

    @staticmethod
    def checksum(incident,kind,reviewer,recorded,payload,previous):
        return hashlib.sha256(json.dumps([incident,kind,reviewer,recorded,payload,previous],sort_keys=True).encode()).hexdigest()

    def entries(self,con=None):
        if con is None:
            with self.connect() as c:return self.entries(c)
        rows=[dict(r) for r in con.execute('SELECT * FROM events ORDER BY id')];previous=''
        for r in rows:
            if r['previous_hash']!=previous or r['hash']!=self.checksum(r['incident'],r['kind'],r['reviewer'],r['recorded_at'],r['payload'],previous):
                raise ValueError('Journal integrity check failed.')
            previous=r['hash'];r['payload']=json.loads(r['payload'])
        return rows

    def save(self,kind,payload):
        if not isinstance(payload,dict):raise ValueError('Expected a JSON object.')
        incident=text(payload.get('incident_id'),'Incident ID',160)
        with self.connect() as con:
            con.execute('BEGIN IMMEDIATE');rows=self.entries(con)
            incident_rows=[r for r in rows if r['incident']==incident]
            checkpoint=next((r for r in incident_rows if r['kind']=='checkpoint'),None)
            recorded=now();reviewer=''
            if kind=='checkpoint':
                if checkpoint:raise ValueError('Checkpoint already frozen. Start a separately identified decision checkpoint if facts change.')
                obs=payload.get('observations')
                if not isinstance(obs,list) or not 1<=len(obs)<=100:raise ValueError('Provide 1–100 observation lines.')
                refs=payload.get('related_refs',[])
                if not isinstance(refs,list) or len(refs)>100:raise ValueError('Invalid related incident/fix references.')
                data={'source_url':url(payload.get('source_url')),'observations':[text(q,'Observation') for q in obs],
                      'related_refs':[text(q,'Related reference',2000) for q in refs],
                      'captured_by':text(payload.get('captured_by'),'Captured by',120)}
            elif kind=='assessment':
                if not checkpoint:raise ValueError('Freeze a checkpoint first.')
                reviewer=text(payload.get('reviewer'),'Reviewer',120).casefold()
                rating=payload.get('rating');confidence=payload.get('confidence')
                if rating not in RATINGS or type(confidence) not in (int,float) or not 0<=confidence<=1:raise ValueError('Choose an assessment and confidence from 0 to 1.')
                if payload.get('checkpoint_only') is not True:raise ValueError('Confirm a checkpoint-only assessment without consulting the outcome.')
                data={'rating':rating,'confidence':confidence,'rationale':text(payload.get('rationale'),'Evidence and rationale'),
                      'checkpoint_only':True,'reviewer_identity':'self_reported','checkpoint_hash':checkpoint['hash']}
            elif kind=='outcome':
                if not checkpoint:raise ValueError('Freeze a checkpoint before recording an outcome.')
                action=payload.get('mechanism')
                if action not in {'reprocess','discard'}:raise ValueError('Choose the actual completed mechanism.')
                started=instant(payload.get('action_started_at'))
                if not instant(checkpoint['recorded_at'])<started<=instant(recorded):raise ValueError('Action must start after the saved checkpoint and no later than now; retrospective cases cannot enter prospective collection.')
                data={'mechanism':action,'action_started_at':started.isoformat(),'source_url':url(payload.get('source_url')),
                      'quote':text(payload.get('quote'),'Completed-action quote'),
                      'recorded_by':text(payload.get('recorded_by'),'Recorded by',120),
                      'timing_verification':'operator_reported_action_time; local_checkpoint_clock',
                      'success':payload.get('success') if payload.get('success') in {'confirmed','failed','unknown'} else 'unknown'}
            else:raise ValueError('Unknown journal entry kind.')
            previous=rows[-1]['hash'] if rows else '';body=json.dumps(data,sort_keys=True)
            h=self.checksum(incident,kind,reviewer,recorded,body,previous)
            try:con.execute('INSERT INTO events(incident,kind,reviewer,recorded_at,payload,previous_hash,hash) VALUES (?,?,?,?,?,?,?)',
                            (incident,kind,reviewer,recorded,body,previous,h))
            except sqlite3.IntegrityError:raise ValueError('That entry is already frozen; it cannot be replaced.')
        return {'saved':True,'incident_id':incident,'kind':kind,'recorded_at':recorded,'hash':h}

    def summary(self):
        rows=self.entries();cases=[]
        for r in rows:
            if r['kind']!='checkpoint':continue
            linked=[x for x in rows if x['incident']==r['incident']]
            cases.append({'incident_id':r['incident'],'recorded_at':r['recorded_at'],'checkpoint_hash':r['hash'],
                **r['payload'],'assessment_count':sum(x['kind']=='assessment' for x in linked),
                'outcome_recorded':any(x['kind']=='outcome' for x in linked)})
        # Outcome values and earlier reviewer ratings are intentionally absent.
        return {'cases':cases,'gpu_work':False,'automatic_ingestion':False}

    def export(self):
        rows=self.entries();summary=self.summary();completed=[];review=[];counts={}
        for c in summary['cases']:
            linked=[r for r in rows if r['incident']==c['incident_id']]
            outcome=next((r for r in linked if r['kind']=='outcome'),None)
            assessments=[r for r in linked if r['kind']=='assessment']
            if any(r['payload']['confidence']<.8 or r['payload']['rating'] in {'either','insufficient'} for r in assessments):review.append(c['incident_id'])
            if not outcome:continue
            label=outcome['payload']['mechanism'];counts[label]=counts.get(label,0)+1
            ratings=[r['payload']['rating'] for r in assessments]
            paired=len(ratings)>=2
            decisive=paired and len(set(ratings))==1 and ratings[0] in {'reprocess_only','discard_only'} and all(r['payload']['confidence']>=.8 for r in assessments)
            completed.append({'incident_id':c['incident_id'],'checkpoint_hash':c['checkpoint_hash'],
                'state':{'observations':c['observations']},'source_url':c['source_url'],'related_refs':c['related_refs'],
                'outcome':outcome,'assessments':assessments,'two_reviewer_agreement_on_single_action':decisive,
                'agrees_with_recorded_action':bool(decisive and ratings[0]==label+'_only'),
                'independence_status':'needs_family_audit','gold_status':'prospective_source_documented_action'})
        paired=[c for c in completed if len(c['assessments'])>=2]
        n=len(paired);decisive=sum(c['two_reviewer_agreement_on_single_action'] for c in paired)
        return {'cases':completed,'review_needed':review,'counts':{'checkpoints':len(summary['cases']),'completed':len(completed),'per_mechanism':counts,'two_reviewers':n,'decisive_agreement':decisive},
                'target_review':{'minimum_paired_cases':20,'decisive_agreement_fraction':decisive/n if n else None,
                    'candidate_threshold':.8,'ready_for_discussion':n>=20,
                    'interpretation':'Frequent either/insufficient or reviewer disagreement challenges a binary target. Agreement is expert opinion, not proof of counterfactual safety.'},
                'validation_ready':False,'reason':'Family separation, source timing, class coverage and target utility require audit before scoring. No fitting or deployment export.',
                'journal_head':rows[-1]['hash'] if rows else None}


def server(store,port):
    token=secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,data,mime='application/json'):
            body=(json.dumps(data) if mime=='application/json' else data).encode()
            self.send_response(status);self.send_header('Content-Type',mime+'; charset=utf-8');self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.send_header('X-Frame-Options','DENY');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.headers.get('Host') not in {f'127.0.0.1:{port}',f'localhost:{port}'}:return self.reply(403,{'error':'Loopback host required.'})
            if self.path=='/':return self.reply(200,ASSET.read_text(),'text/html')
            if self.path=='/api/cases':return self.reply(200,store.summary()|{'token':token})
            if self.path=='/api/export':return self.reply(200,store.export())
            return self.reply(404,{'error':'Not found.'})
        def do_POST(self):
            origin=self.headers.get('Origin')
            if self.headers.get('Host') not in {f'127.0.0.1:{port}',f'localhost:{port}'} or self.headers.get('X-Collection-Token')!=token or (origin and origin not in {f'http://127.0.0.1:{port}',f'http://localhost:{port}'}):return self.reply(403,{'error':'Reload this local page before saving.'})
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=150000:raise ValueError('Invalid request size.')
                if self.path not in {'/api/checkpoint','/api/assessment','/api/outcome'}:return self.reply(404,{'error':'Not found.'})
                result=store.save(self.path.rsplit('/',1)[1],json.loads(self.rfile.read(length)))
                self.reply(201,result)
            except (ValueError,TypeError) as error:self.reply(400,{'error':str(error)})
    httpd=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    port=httpd.server_port
    return httpd


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['serve','export'])
    parser.add_argument('--db',type=Path,required=True);parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args();store=Collection(args.db)
    if args.command=='export':print(json.dumps(store.export(),indent=2))
    else:
        httpd=server(store,args.port)
        print(f'Collection ready: http://127.0.0.1:{httpd.server_port}/',flush=True)
        httpd.serve_forever()
