/* Local UI. Retrieved text is always rendered with textContent, never HTML. */
const $ = selector => document.querySelector(selector);
const labels = {
  application_engineering: 'Application engineering', operational_state_recovery: 'Recover operational state',
  customer_or_integration_configuration: 'Configuration or integration', infrastructure_or_upstream: 'Infrastructure or upstream',
  expected_policy_behavior: 'Expected policy or workflow', insufficient_evidence: 'Collect more evidence',
};
let snapshot, selected, dirty = false, busy = false;
let visible = [];
let lastFilter = 'uncertain';
const el = (tag, text, className) => { const node = document.createElement(tag); if (text != null) node.textContent = text; if (className) node.className = className; return node; };
const append = (parent, ...children) => { parent.append(...children); return parent; };
function toast(message, error = false) { const node = $('#toast'); node.textContent = message; node.className = error ? 'error' : ''; node.style.display = 'block'; clearTimeout(toast.timer); toast.timer = setTimeout(() => { node.style.display = 'none'; }, error ? 9000 : 3500); }
function statusBadge(status,confidence) { const assessed=status==='pending'&&confidence>=.8;return el('span', snapshot?.metadata.archived&&status==='pending'?'Archived label':assessed?'Agent assessed':{approved:'Reviewed',corrected:'Corrected',excluded:'Excluded',deferred:'Deferred',pending:'Needs your judgment'}[status], 'badge ' + ({approved:'green',corrected:'green',excluded:'red',deferred:'gray',pending:'gray'}[status])); }
function isReviewed(c) { return ['approved', 'corrected'].includes(c.review.status); }
function filtered() {
  const term = $('#search').value.trim().toLowerCase(); const filter = $('#filter').value;
  return snapshot.cases.filter(c => {
    const status = c.review.status;
    const match = filter === 'all' || filter === 'uncertain' && c.confidence < .8 && !isReviewed(c) && status !== 'excluded' || filter === 'confident' && c.confidence >= .8 || filter === 'pending' && status === 'pending' || filter === 'reviewed' && isReviewed(c) || filter === status;
    return match && (!term || JSON.stringify([c.state,c.assessment,c.evidence,c.story_id]).toLowerCase().includes(term));
  }).sort((a,b) => a.confidence - b.confidence || String(a.story_id).localeCompare(String(b.story_id)));
}
function stats() {
  const cases = snapshot.cases; const reviewed = cases.filter(isReviewed).length;
  $('#stat-uncertain').textContent = cases.filter(c=>c.confidence < .8 && !isReviewed(c) && c.review.status !== 'excluded').length;
  $('#stat-reviewed').textContent = reviewed;
  $('#stat-corrected').textContent = cases.filter(c=>c.review.status === 'corrected').length;
  $('#stat-total').textContent = cases.length;
  const queued=cases.filter(c=>c.confidence<.8);const resolved=queued.filter(c=>isReviewed(c)||c.review.status==='excluded').length;
  $('#progress-label').textContent = `${resolved} of ${queued.length} queued cases resolved`;
  $('#progress').value = queued.length ? resolved / queued.length * 100 : 100;
  if(snapshot.metadata.archived){$('#stat-uncertain').textContent='0';$('#progress-label').textContent='Archived · no review requested';$('#progress').value=100;}
}
function renderQueue() {
  visible = filtered(); $('#queue-count').textContent = visible.length;
  const list = $('#case-list'); list.replaceChildren();
  if (!visible.length) list.append(el('div', 'No cases in this view. Try another filter.', 'empty'));
  for (const c of visible) {
    const button = el('button', null, 'queue-item' + (selected?.key === c.key ? ' active' : ''));
    button.dataset.key = c.key;
    button.setAttribute('aria-current', selected?.key === c.key ? 'true' : 'false');
    const top = append(el('div',null,'item-top'), el('span',`SC ${c.story_id}`,'item-id'), el('span', c.confidence.toFixed(2), 'badge ' + (c.confidence < .8 ? 'amber':'green')));
    append(button, top, el('p',c.review.state?.report || c.state.report,'item-title'),el('div',`${c.review.status === 'pending' ? (snapshot.metadata.archived?'Archived assessment':c.confidence>=.8?'Agent assessed · optional review':'Needs your judgment') : c.review.status} · ${c.domain || 'Support'}`,'item-meta'));
    button.addEventListener('click', () => select(c)); list.append(button);
  }
  stats();
}
function safeLink(url, text) {
  const parsed = new URL(url);
  if (parsed.protocol !== 'https:' || !['app.shortcut.com','gitlab.tcdevops.com'].includes(parsed.hostname)) return el('span',text);
  const a = el('a',text,'source-link'); a.href = parsed.href; a.target = '_blank'; a.rel = 'noopener noreferrer'; return a;
}
function details(title) { const node = el('details',null,'evidence'); node.append(el('summary',title)); return node; }
function renderCase() {
  const panel = $('#case-panel'); const decision = $('#decision-panel'); panel.replaceChildren(); decision.replaceChildren();
  if (!selected) { panel.append(el('div','Choose a case from the queue.','empty')); return; }
  const c = selected, review = c.review, state = review.state || c.state;
  const top = el('div',null,'case-top'); top.append(el('span',`CASE SC-${c.story_id} · ${c.domain || 'Support'}`,'case-kicker'));
  const nav = el('div',null,'nav-buttons'); for (const [text,delta] of [['←',-1],['→',1]]) { const b = el('button',text,'button'); b.setAttribute('aria-label',delta<0?'Previous case':'Next case'); b.onclick=()=>navigate(delta); nav.append(b); } top.append(nav); panel.append(top);
  append(panel, append(el('div',null,'status-line'), el('span', `${c.confidence.toFixed(2)} agent confidence`, 'badge '+(c.confidence<.8?'amber':'green')), statusBadge(review.status,c.confidence)), el('h2',state.report));
  panel.append(el('p','Available observations','section-label'));
  const observations = el('ol',null,'observations'); for (const text of state.observations) observations.append(el('li',text)); panel.append(observations);
  const caution = el('div',null,'uncertainty'); append(caution,el('h3',snapshot.metadata.archived?'Archived assessment':c.confidence < .8 ? 'Where your judgment is needed' : 'Assessment and remaining limits'),el('p',c.assessment.reason));
  if(c.assessment.unresolved?.length){ const ul=el('ul');c.assessment.unresolved.forEach(x=>ul.append(el('li',x)));caution.append(ul); } panel.append(caution);
  const proposed=details('Reveal proposed track and audit'); append(proposed,el('p',labels[c.draft_label]),el('p',c.assessment.audit_summary || c.assessment.reason));
  const use=el('button','Use proposed track','button'); use.onclick=()=>{ const radio=document.querySelector(`input[name="label"][value="${c.draft_label}"]`);radio.checked=true;dirty=true; }; proposed.append(use); panel.append(proposed);
  const evidence = details(`Evidence & sources · ${c.evidence.length}`);
  for(const source of c.evidence){ append(evidence,el('p',source.title || 'Source excerpt'),el('blockquote',source.quote),safeLink(source.url,'Open source ↗')); if(source.revision) evidence.append(el('p',`Snapshot: ${source.revision}`,'source-meta')); }
  panel.append(evidence);
  const edit = details('Correct the report or observations'); edit.classList.add('edited-facts');
  const reportInput=el('textarea'); reportInput.id='edit-report';reportInput.value=state.report;reportInput.setAttribute('aria-label','Correct report');
  const obsInput=el('textarea'); obsInput.id='edit-observations';obsInput.value=state.observations.join('\n');obsInput.rows=5;obsInput.setAttribute('aria-label','Correct observations, one per line');
  append(edit,el('p','Keep only supported observations. One observation per line.'),reportInput,obsInput); panel.append(edit);
  const provenance = details('Case grouping & study details');append(provenance,el('p',`Family: ${c.family_id}`),el('p',`Frozen role: ${c.role}`),el('p','This is a reconstructed diagnostic checkpoint, not a verified original intake. Agent confidence does not make a label human-reviewed.'));panel.append(provenance);
  const history=details('Correction history'); history.addEventListener('toggle',async()=>{if(!history.open)return;try{const response=await fetch(`/api/history/${c.key}`);const data=await response.json();history.querySelectorAll('.history-event').forEach(n=>n.remove());for(const h of data.events){const n=el('div',null,'history-event');append(n,el('p',`${h.status} · ${h.label?labels[h.label]:''}`),el('small',`${h.reviewer} · ${new Date(h.created_at).toLocaleString()} · revision ${h.revision}`),el('p',h.notes||'No note'));history.append(n);}if(!data.events.length)history.append(el('div','No corrections saved yet.','history-event'));}catch{toast('Could not load correction history.',true);}});panel.append(history);
  panel.append(el('p','Inspect the sources before approving uncertain facts. Exclude a case if the evidence cannot support a fair decision.','audit-note'));
  append(decision,el('h2','Your decision'),el('p','Choose the best supported track for the primary symptom.','help'));
  const options=el('div',null,'label-options');
  for(const [key,description] of Object.entries(snapshot.options)) {const label=el('label',null,'label-option');const radio=el('input');radio.type='radio';radio.name='label';radio.value=key;radio.checked=review.label===key;append(label,radio,append(el('span'),el('strong',labels[key]),el('small',description)));options.append(label);}decision.append(options);
  const check=el('label',null,'check');const checkbox=el('input');checkbox.type='checkbox';checkbox.id='facts-checked';checkbox.checked=!!review.facts_checked;append(check,checkbox,el('span','I checked that the facts and decision point are supported.'));decision.append(check);
  const noteLabel=el('label','Reason / correction note','field-label');noteLabel.htmlFor='notes';const notes=el('textarea');notes.id='notes';notes.placeholder='What should change, or what is still unclear?';notes.value=review.notes||'';append(decision,noteLabel,notes);
  const grouping=el('details');grouping.append(el('summary','Flag related cases or a family concern'));const family=el('textarea');family.id='family-concern';family.placeholder='Related incident, duplicate, or grouping correction…';family.value=review.family_concern||'';family.setAttribute('aria-label','Family concern');grouping.append(family);decision.append(grouping);
  const save=el('button','Save review & next →','button primary full');save.id='save-review';save.onclick=()=>saveReview('approved');decision.append(save);
  const other=el('div',null,'two-actions');for(const [text,status] of [['Defer','deferred'],['Exclude','excluded']]){const b=el('button',text,'button'+(status==='excluded'?' danger':''));b.onclick=()=>saveReview(status);other.append(b);}decision.append(other);
  if(review.revision){const reopen=el('button','Reopen for review','button full');reopen.onclick=()=>saveReview('pending');decision.append(reopen);}
  decision.append(el('p',review.revision?`Saved locally · revision ${review.revision} · ${review.reviewer}`:'No decision saved yet. Your review will be stored locally.','saved-label'));
  if(snapshot.metadata.archived){
    decision.querySelector('h2').textContent='Archived decision';
    decision.querySelector('.help').textContent=snapshot.metadata.archive_note;
    decision.querySelectorAll('button,input,textarea').forEach(n=>n.disabled=true);
  }
  for(const node of [...panel.querySelectorAll('textarea'),...decision.querySelectorAll('input,textarea')])node.addEventListener('input',()=>{dirty=true;});
  dirty=false;
}
async function discardAllowed(){if(!dirty)return true;const dialog=$('#leave-dialog');return new Promise(resolve=>{dialog.showModal();$('#stay').onclick=()=>{dialog.close();resolve(false)};$('#discard').onclick=()=>{dialog.close();dirty=false;resolve(true)};dialog.oncancel=()=>resolve(false);});}
async function select(c){if(!await discardAllowed())return;selected=c;renderQueue();renderCase();}
async function navigate(delta){if(!visible.length)return;let i=visible.findIndex(c=>c.key===selected?.key);i=Math.max(0,Math.min(visible.length-1,i+delta));await select(visible[i]);}
async function saveReview(status){
 if(!selected||busy)return;
 const reviewer=$('#reviewer').value.trim();if(!reviewer){$('#reviewer').focus();return toast('Enter your name to save a review.',true);}
 const key=selected.key;const oldIndex=visible.findIndex(c=>c.key===key);
 const payload={key,revision:selected.review.revision,status,reviewer,label:document.querySelector('input[name="label"]:checked')?.value||null,state:{report:$('#edit-report').value.trim(),observations:$('#edit-observations').value.split('\n').map(s=>s.trim()).filter(Boolean)},notes:$('#notes').value.trim(),facts_checked:$('#facts-checked').checked,family_concern:$('#family-concern').value.trim()};
 busy=true;$('#save-review').disabled=true;
 try{const response=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':snapshot.csrf_token},body:JSON.stringify(payload)});const result=await response.json();if(!response.ok)throw Error(result.error||'Save failed');selected.review=result.review;localStorage.setItem('lars-reviewer',reviewer);dirty=false;renderQueue();selected=visible[Math.min(Math.max(oldIndex,0),visible.length-1)]||null;renderQueue();renderCase();toast('Saved locally.');}
 catch(error){toast(error.message,true);}
 finally{busy=false;if($('#save-review'))$('#save-review').disabled=false;}
}
async function init(){
 try{const response=await fetch('/api/cases');if(!response.ok)throw Error('Could not load cases');snapshot=await response.json();$('#reviewer').value=localStorage.getItem('lars-reviewer')||'';if(snapshot.metadata.archived){$('h1').textContent='Archived judgment study';$('.lede').textContent=snapshot.metadata.archive_note;$('#filter').value='all';lastFilter='all';$('#reviewer').disabled=true;}renderQueue();selected=visible[0]||null;renderQueue();renderCase();}
 catch(error){$('#case-panel').replaceChildren(el('div',error.message,'empty'));toast('The local review server is unavailable. '+error.message,true);}
}
$('#search').addEventListener('input',()=>renderQueue());
$('#filter').addEventListener('change',async()=>{if(!await discardAllowed()){$('#filter').value=lastFilter;return;}lastFilter=$('#filter').value;renderQueue();selected=visible[0]||null;renderQueue();renderCase();});
$('#export').addEventListener('click',async()=>{try{const r=await fetch('/api/export');if(!r.ok)throw Error('Export failed');const data=await r.json();const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=el('a');a.href=url;a.download='lars-case-reviews.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast(`Exported ${data.reviewed_cases.length} reviewed cases with audit state.`);}catch(error){toast(error.message,true);}});
document.addEventListener('keydown',event=>{if(event.ctrlKey||event.metaKey||event.altKey||/INPUT|TEXTAREA|SELECT/.test(event.target.tagName)||$('#leave-dialog').open)return;if(event.key==='j')navigate(1);if(event.key==='k')navigate(-1);});
window.addEventListener('beforeunload',event=>{if(dirty){event.preventDefault();event.returnValue='';}});
init();
