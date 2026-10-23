"""Build reviewable synthetic contract cases from pinned Codegraph source sections.

No model labels, production records, operational commands or external writes are used.
These draft rule-derived labels are NOT independent company calibration gold.
"""
from __future__ import annotations

import argparse
from collections import Counter
import itertools
import json
from pathlib import Path

from lars.train.accuracy import record_key
from lars.train.data import Record, write_records
from sweep.accuracy_ladder import digest
from sweep.experiment import atomic_json
from sweep.local_calibration import identity, write_jsonl


def billing_action(error, delivery, attempt):
    if error == 'DuplicateLineItemException':
        return 'skip_duplicate'
    if error != 'BillingMethodNotFoundException':
        return 'fault'
    return 'retry_in_process' if attempt < 4 else ('delayed_redelivery' if delivery < 4 else 'fault')


def http_action(status, retries_used):
    if (500 <= status < 600 or status in (408,429)) and retries_used < 3:
        return f'retry_after_{[2,4,8][retries_used]}s'
    return 'fail'


def payout_action(status, error):
    if error:
        return 'needs_attention'
    if status in ('Paid','Cleared'):
        return 'completed'
    if status in ('Rejected','Deferred','DeferredInternal','Canceled'):
        return 'unsuccessful'
    return 'in_progress'


def build(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'cases-manifest.json').exists():
        raise ValueError('preserve existing cases; choose a fresh output directory')
    sources=[]
    for i in range(7):
        path=args.sources/f'source-{i}.json';envelope=json.loads(path.read_text());r=envelope['result']
        ref=r['documentReference']
        if not envelope['snapshot']['cacheable'] or not ref.get('revision') or r.get('truncated'):
            raise ValueError('sources must be pinned, cacheable, complete sections')
        sources.append({'reference':ref,'content':r['content'],'file':str(path.resolve()),'sha256':digest(path)})
    # Refuse to apply these rules to unrelated retrievals or changed source sections.
    for i,phrases in {0:['Completed','Withdrawal'],1:['balance reaches zero','InvoiceClosed'],
        2:['Payment plan returns','no invoice'],3:['16 attempts','DuplicateLineItemException'],
        4:['Error','Paid','PendingPayerFunds'],5:['429','408','2s, 4s, 8s'],6:['5 min','DB-Core','Infra-Messaging']}.items():
        if any(phrase not in sources[i]['content'] for phrase in phrases):
            raise ValueError('retrieved contract does not match the implemented rule')
    records=[];evidence=[];wire=[]

    def add(name,source,facts,question,options,gold,family):
        src=sources[source];hard=options.index(gold)
        instruction='Apply only this supplied policy to the stated facts.\n\n'+src['content']+'\n\n'+question
        r=Record(identity([name,facts])[:20],'codegraph/'+name,'choice',facts,instruction,options,
                 [float(i==hard) for i in range(len(options))])
        r.validate();key=record_key(r);records.append(r)
        meta={'key':key,'family_id':name+'/'+family,'gold_status':'rule_derived_unreviewed',
            'label_origin':'deterministic rule over constructed facts, based on retrieved documentation',
            'review_required':True,'source_reference':src['reference'],'source_file_sha256':src['sha256']}
        evidence.append(meta|{'hard_index':hard,'answer':gold})
        wire.append({'record_key':key,'state':facts,'question':r.to_question(),'label':gold,'soft_label':r.target,
                     'source':r.source,'hard_index':hard,**meta})

    errors=['BillingMethodNotFoundException','DuplicateLineItemException','invalid_message',
            'debit_below_zero','deadlock','lock_wait_timeout','dropped_connection']
    for error,delivery,attempt in itertools.product(errors,range(1,5),range(1,5)):
        add('billing_retry',3,{'failure':error,'delivery_number':delivery,'attempt_number_this_delivery':attempt},
            'Which documented action applies now? Delivery and attempt numbers are one-based. Each delivery allows the first attempt plus three in-process retries; there are four deliveries total.',
            ['retry_in_process','delayed_redelivery','fault','skip_duplicate'],billing_action(error,delivery,attempt),error)
    for status,used in itertools.product([400,401,403,404,408,429,500,502,503,504],range(4)):
        add('story_http_retry',5,{'http_status':status,'retries_already_used':used},
            'What should the documented retry policy do following this response? retries_already_used excludes the initial request.',
            ['fail','retry_after_2s','retry_after_4s','retry_after_8s'],http_action(status,used),str(status))
    statuses=['Paid','Cleared','Rejected','Deferred','DeferredInternal','Canceled','Scheduled','Submitted','FraudReview','PendingPayerFunds']
    for status,error in itertools.product(statuses,[False,True]):
        add('payout_result',4,{'batch_request_succeeded':True,'item':{'Status':status,'Error':'simulated item error' if error else None}},
            'How should this individual payout item be classified? Batch transport success does not replace the per-item checks.',
            ['needs_attention','completed','unsuccessful','in_progress'],payout_action(status,error),status)
    for kind,completed,partial in itertools.product(['Account Funding','Sale Proceeds','User Purchase','Withdrawal'],[False,True],[False,True]):
        yes=completed and kind!='Withdrawal'
        add('internal_refund',0,{'transaction_type':kind,'transaction_completed':completed,'partial_refund_requested':partial},
            'Does this transaction meet the documented type/status prerequisites for the internal refund operation? Do not assume an unfinished transaction has settled.',
            ['eligible','ineligible'],'eligible' if yes else 'ineligible',kind+'/'+str(completed))
    for owed,fraction in itertools.product([100,1000,100000],[0,.25,.99,1.]):
        paid=int(owed*fraction)
        add('invoice_closure',1,{'previous_status':'Open','debits_cents':owed,'credits_cents':paid,'remaining_balance_cents':owed-paid},
            'Has the documented zero-balance condition for automatic invoice closure and InvoiceClosed publication been met?',
            ['close_and_publish','remain_open'],'close_and_publish' if owed==paid else 'remain_open','zero' if owed==paid else 'positive')
    for seconds in [0,60,299,300,301,600]:
        add('escalation_contact',6,{'single_symptom':'MySQL unavailable','unavailable_seconds':seconds},
            'Which contact does this matrix explicitly prescribe for this single symptom? Use not_specified when the matrix condition is not met; do not infer a broader no-escalation policy.',
            ['DB-Core','Infra-Messaging','Domain Intelligence','Architecture lead','not_specified'],
            'DB-Core' if seconds>300 else 'not_specified','mysql_outage')
    for symptom,gold in [('Message bus outage','Infra-Messaging'),('Analyzer algorithm bug','Domain Intelligence'),('Persistent data corruption','Architecture lead')]:
        add('escalation_contact',6,{'single_symptom':symptom},
            'Which contact does this matrix explicitly prescribe for this single symptom? Use not_specified when the matrix condition is not met; do not infer a broader no-escalation policy.',
            ['DB-Core','Infra-Messaging','Domain Intelligence','Architecture lead','not_specified'],gold,symptom)
    for facts,gold in [({'return_type':'payment plan','return_initiated':True},'publish_PaymentPlanReturned'),
        ({'return_type':'full price','event':'DomainHeldForRefund'},'create_refund_invoice'),
        ({'return_type':'full price','event':'InvoiceClosed','admin_confirmed_seller_funds_returned':True},'return_domain_to_seller')]:
        add('domain_return',2,facts,'Which documented return action applies at this stage?',
            ['publish_PaymentPlanReturned','create_refund_invoice','return_domain_to_seller'],gold,facts['return_type'])
    if len({record_key(r) for r in records})!=len(records):
        raise ValueError('duplicate generated decision')
    write_records(records,out/'cases.records.jsonl');write_jsonl(out/'cases.wire.jsonl',wire);write_jsonl(out/'evidence.jsonl',evidence)
    manifest={'format':1,'cases':len(records),'predicates':dict(Counter(r.source for r in records)),
        'families':len({e['family_id'] for e in evidence}),'sources':[s['reference']|{'local_sha256':s['sha256']} for s in sources],
        'code_sha256':digest(Path(__file__)),'outputs':{name:digest(out/name) for name in ['cases.records.jsonl','cases.wire.jsonl','evidence.jsonl']},
        'gold_status':'rule_derived_unreviewed','production_traffic':False,'production_representative':False,
        'interpretation':'Synthetic contract probes from seven pinned Codegraph sections, not company incident history or independently reviewed gold. Preserve case families across splits. Small family count limits valid learning curves; do not pad with paraphrases to inflate sample size. Review source interpretation and labels before calibration/certification. No model inference or training performed.'}
    atomic_json(out/'cases-manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['cases','predicates','families','gold_status']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--sources',type=Path,default=Path('sweep/out/codegraph-calibration/sources'))
    p.add_argument('--out',type=Path,default=Path('sweep/out/codegraph-calibration'))
    build(p.parse_args())
