const $=s=>document.querySelector(s);
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!=null)n.textContent=text;if(cls)n.className=cls;return n;};
let data,view='cases',selected,term='';
const date=s=>new Date(s).toLocaleString(undefined,{dateStyle:'medium',timeStyle:'short'});
function link(url,text){try{const u=new URL(url);if(u.protocol!=='https:'||!['app.shortcut.com','gitlab.tcdevops.com'].includes(u.hostname))return el('span',text);const a=el('a',text);a.href=u.href;a.target='_blank';a.rel='noopener noreferrer';return a;}catch{return el('span',text);}}
function detail(row){const p=$('#audit-detail');p.replaceChildren();if(!row){p.append(el('p','No entries in this view.'));return;}
 p.append(el('span',row.status==='excluded'?'Excluded · no review requested':row.scope==='pre_action'?'Pre-action observations':row.scope==='pre_resolution'?'Pre-resolution observations':'Pre-finding observations','badge '+(row.status==='excluded'?'gray':'green')));
 p.append(el('h2',row.title||`SC-${row.story_id}`));
 if(view==='exclusions'){p.append(el('div',row.reason,'audit-reason'),el('p','This case is excluded from fitting and evaluation. It does not create a manual labeling task.','help'));return;}
 const latest=row.input_evidence.map(r=>r.updated_at).sort().at(-1),first=row.outcome_evidence[0].created_at;
 const timing=el('div',null,'chronology');for(const [title,value] of [['Latest input edit',latest],[row.scope==='pre_action'?'Action cutoff':'Finding / correction cutoff',row.cutoff.created_at],['Recorded outcome',first]]){const n=el('div');n.append(el('small',title),el('strong',date(value)));timing.append(n);}p.append(timing);
 p.append(el('p','MODEL OBSERVATIONS · VERBATIM EARLIER EXCERPTS','section-label'));const input=el('div',null,'input-block');row.model_input.state.observations.forEach(x=>input.append(el('p',x)));p.append(input);
 if(row.action_label)p.append(el('p',`RECORDED MECHANISM · ${row.action_label.toUpperCase()}`,'section-label'));
 p.append(el('p','EXPECTED OUTCOME · LATER SOURCE RECORD','section-label'));p.append(el('div',row.expected_outcome,'outcome-block'));
 p.append(el('p',row.notes,'help'));const evidence=el('details',null,'evidence');evidence.append(el('summary','Inspect dated source references'));
 for(const [kind,anchors] of [['Observation',row.input_evidence],['Outcome',row.outcome_evidence]])for(const a of anchors){const n=el('div',null,'source-row');n.append(el('span',`${kind} · ${date(a.updated_at)}`),link(a.url,'Open source ↗'));evidence.append(n);}p.append(evidence);
 p.append(el('p',row.limitations,'help'),el('p',row.scope==='pre_action'?`Frozen group: ${row.family_id} · role: ${row.role} · ${row.evidence_level==='diagnostic'?'diagnostic checkpoint':'early alert or symptom only'}`:`Preserved group: ${row.family_id} · prior role: ${row.role} · not used for fitting`,'audit-group'));
}
function list(){const container=$('#audit-list');container.replaceChildren();let rows=view==='cases'?data.cases:data.audit.filter(r=>r.status==='excluded');
 if(view==='exclusions'){const search=el('input',null,'exclusion-search');search.type='search';search.placeholder='Find ticket or reason…';search.value=term;search.setAttribute('aria-label','Search exclusions');search.oninput=()=>{term=search.value;renderRows();};container.append(search);}
 const items=el('div');container.append(items);
 function renderRows(){items.replaceChildren();const shown=rows.filter(r=>!term||view==='cases'||JSON.stringify(r).toLowerCase().includes(term.toLowerCase()));for(const r of shown){const b=el('button',null,'audit-row'+(selected?.story_id===r.story_id?' active':''));b.append(el('small',`SC-${r.story_id}`),el('strong',r.title||r.reason_code.replaceAll('_',' ')));b.onclick=()=>{selected=r;list();detail(r);};items.append(b);}if(!shown.length)items.append(el('p','No matching cases.','empty'));}renderRows();
}
function setView(next){view=next;term='';$('#show-cases').classList.toggle('active',view==='cases');$('#show-exclusions').classList.toggle('active',view==='exclusions');selected=(view==='cases'?data.cases:data.audit.filter(r=>r.status==='excluded'))[0];list();detail(selected);}
$('#show-cases').onclick=()=>setView('cases');$('#show-exclusions').onclick=()=>setView('exclusions');
$('#download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download='lars-resolution-replay-audit.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
function showPilot(report){
 const r=report.pilot_results;if(!r)return;
 const section=$('#pilot-results');section.hidden=false;section.replaceChildren(el('h2','Calibration matches the majority baseline'));
 section.append(el('p',report.pilot_conclusion,'finding'));
 const wrap=el('div',null,'table-wrap'),table=el('table'),head=el('tr');['Fitting labels','Held-out correct','Brier ↓','Log loss ↓'].forEach(t=>head.append(el('th',t)));table.append(head);
 const rows=[['0 · existing pack',`${r.baseline.correct}/${r.baseline.n}`,r.baseline.brier,r.baseline.log_loss]];
 for(const budget of [10,25]){const runs=r.runs.filter(x=>x.budget===budget),summary=r.summary[String(budget)];const hits=runs.map(x=>x.metrics.correct);rows.push([String(budget),new Set(hits).size===1?`${hits[0]}/${r.baseline.n}`:`${hits.join(', ')} / ${r.baseline.n}`,summary.mean_brier,summary.mean_log_loss]);}
 const prior=r.runs.find(x=>x.budget===25).prior_only;rows.push(['25 · class prior only',`${prior.correct}/${prior.n}`,prior.brier,prior.log_loss]);
 for(const values of rows){const row=el('tr');values.forEach(v=>row.append(el('td',typeof v==='number'?v.toFixed(3):v)));table.append(row);}wrap.append(table);section.append(wrap);
 section.append(el('p','Three fitting draws share the same 16 held-out groups. Diagnostic checkpoints and sparse alerts are tracked separately. The paired accuracy interval includes zero. Source-recorded actions have not been independently human-adjudicated.'));
 $('.replay-notice strong').textContent='A frozen historical action-prediction pilot';
 $('#retained').nextElementSibling.textContent='Source-backed cases';$('#audited').nextElementSibling.textContent='Cases admitted to this study';
 $('.footer-note').textContent='The evidence, frozen split, and results are preserved. This study does not authorize operating any queue.';
}
fetch('/api/benchmark').then(r=>{if(!r.ok)throw Error('The local audit endpoint is unavailable.');return r.json();}).then(r=>{data=r;showPilot(r);$('#retained').textContent=r.retained_cases;$('#excluded').textContent=r.excluded_cases;$('#reviews').textContent=r.human_review_required;$('#audited').textContent=r.previous_cases;$('#limits').textContent=r.pilot_results?`${r.retained_cases} admitted cases in ${r.families} dependence groups. One representative per group supplies 25 fitting labels and 16 held-out outcomes. ${r.source_inventory_count} source tickets were retrieved; this page shows the admitted cohort. Earlier findings are kept separate from later actions.`:`${r.retained_cases} timestamp-supported fixtures remain. This is too small for an accuracy or calibration claim. Cases without a clean earlier observation or a specific documented outcome are excluded.`;setView('cases');}).catch(e=>$('#audit-detail').replaceChildren(el('p',e.message)));
