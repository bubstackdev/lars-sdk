// Exercise the full workflow in an isolated temporary database, never the live journal.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile,rm,mkdir} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';
const temp=await mkdtemp(path.join(tmpdir(),'lars-collection-qa-'));
const out=process.argv[2]||'sweep/out/codegraph-judgment/queue-prospective-20261023';
await mkdir(out,{recursive:true});
let worker,chrome,socket;let serial=0;const pending=new Map(),errors=[];
const checks=[];
function send(method,params={},sessionId){return new Promise((resolve,reject)=>{const id=++serial,timer=setTimeout(()=>reject(Error('CDP timeout')),10000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}));});}
async function stop(child){if(child&&child.exitCode===null&&child.signalCode===null){child.kill('SIGTERM');await Promise.race([new Promise(r=>child.once('exit',r)),delay(3000)]);if(child.exitCode===null&&child.signalCode===null){child.kill('SIGKILL');await new Promise(r=>child.once('exit',r));}}}
try{
 worker=spawn('.venv-train/bin/python',['-m','sweep.queue_collection','serve','--db',path.join(temp,'qa.sqlite3'),'--port','0'],{stdio:['ignore','pipe','pipe']});
 const base=await new Promise((resolve,reject)=>{let b='';const timer=setTimeout(()=>reject(Error('QA collector timeout')),10000);worker.stdout.on('data',s=>{b+=s;const m=b.match(/http:\/\/127\.0\.0\.1:\d+/);if(m){clearTimeout(timer);resolve(m[0]);}});worker.once('exit',()=>reject(Error('QA collector exited')));});
 let r=await fetch(base+'/api/checkpoint',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});assert.equal(r.status,403);checks.push('Mutation requires same-origin local token');
 chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking',`--user-data-dir=${temp}/chrome`,'--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','about:blank'],{stdio:['ignore','ignore','pipe']});
 const endpoint=await new Promise((resolve,reject)=>{let b='';const timer=setTimeout(()=>reject(Error('Chrome timeout')),15000);chrome.stderr.on('data',s=>{b+=s;const m=b.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(m){clearTimeout(timer);resolve(m[1]);}});});
 socket=new WebSocket(endpoint);await new Promise((r,j)=>{socket.addEventListener('open',r,{once:true});socket.addEventListener('error',j,{once:true});});
 socket.addEventListener('message',event=>{const m=JSON.parse(event.data),p=pending.get(m.id);if(p){clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);});
 const {targetId}=await send('Target.createTarget',{url:base});const {sessionId}=await send('Target.attachToTarget',{targetId,flatten:true});
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true},sessionId);if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 const until=async expression=>{for(let i=0;i<100;i++){if(await evaluate(expression))return;await delay(50);}throw Error('Condition timed out: '+expression);};
 await send('Runtime.enable',{},sessionId);await send('Page.enable',{},sessionId);
 await send('Emulation.setDeviceMetricsOverride',{width:1200,height:1000,deviceScaleFactor:1,mobile:false},sessionId);
 await until(`document.querySelector('#stats')?.textContent.includes('0 frozen')`);
 await evaluate(`document.querySelector('#incident').value='qa-only';document.querySelector('#source').value='https://example.invalid/incident';document.querySelector('#observations').value='Original fault, no final action known.\\n<svg onload=alert(1)> literal text';document.querySelector('#captured').value='QA';document.querySelector('#checkpoint-form').requestSubmit();`);
 await until(`document.querySelector('#stats').textContent.includes('1 frozen')`);checks.push('Browser freezes checkpoint');
 await evaluate(`document.querySelector('[data-tab="assessment"]').click();document.querySelector('#assess-case').value='qa-only';document.querySelector('#assess-case').dispatchEvent(new Event('change'));document.querySelector('#reviewer').value='QA Alice';document.querySelector('#rating').value='either';document.querySelector('#confidence').value='0.7';document.querySelector('#rationale').value='The original fault does not establish a unique mechanism.';document.querySelector('[name="checkpoint_only"]').checked=true;document.querySelector('#assessment-form').requestSubmit();`);
 await until(`document.querySelector('#message').textContent.startsWith('Saved assessment')`);checks.push('Checkpoint-only assessment saved');
 let snapshot=await(await fetch(base+'/api/cases')).json();const saved=snapshot.cases[0].recorded_at;
 await delay(20);const actionTime=new Date().toISOString();assert.ok(Date.parse(actionTime)>Date.parse(saved));
 await evaluate(`document.querySelector('[data-tab="outcome"]').click();document.querySelector('#outcome-case').value='qa-only';document.querySelector('#outcome-case').dispatchEvent(new Event('change'));document.querySelector('#action-time').value=${JSON.stringify(actionTime)};document.querySelector('#mechanism').value='discard';document.querySelector('#outcome-source').value='https://example.invalid/outcome';document.querySelector('#quote').value='I purged the failed messages.';document.querySelector('#recorded-by').value='QA';document.querySelector('#outcome-form').requestSubmit();`);
 await until(`document.querySelector('#stats').textContent.includes('1 completed')`);checks.push('Browser records later outcome');
 snapshot=await(await fetch(base+'/api/cases')).json();assert.equal(JSON.stringify(snapshot).includes('purged'),false);assert.equal(JSON.stringify(snapshot).includes('either'),false);checks.push('Assessment API hides action and reviewer ratings');
 const report=await(await fetch(base+'/api/export')).json();assert.deepEqual(report.review_needed,['qa-only']);assert.equal(report.validation_ready,false);checks.push('Ambiguous low-confidence case is flagged; no training promotion');
 const token=snapshot.token;
 r=await fetch(base+'/api/checkpoint',{method:'POST',headers:{'Content-Type':'application/json','X-Collection-Token':token},body:JSON.stringify({incident_id:'qa-only'})});assert.equal(r.status,400);checks.push('Frozen checkpoint cannot be overwritten');
 await evaluate(`document.querySelector('[data-tab="assessment"]').click();document.querySelector('#assess-case').value='qa-only';document.querySelector('#assess-case').dispatchEvent(new Event('change'));`);
 assert.equal(await evaluate(`document.querySelector('#assess-evidence').textContent.includes('<svg onload=alert(1)>')`),true);
 assert.equal(await evaluate(`document.querySelector('#assess-evidence svg')!==null`),false);checks.push('Evidence renders as text');
 let shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true},sessionId);await writeFile(out+'/collection-desktop.png',Buffer.from(shot.data,'base64'));
 await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true},sessionId);
 assert.equal(await evaluate(`document.documentElement.scrollWidth<=window.innerWidth`),true);checks.push('Mobile layout fits viewport');
 shot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true},sessionId);await writeFile(out+'/collection-mobile.png',Buffer.from(shot.data,'base64'));
 assert.deepEqual(errors,[]);checks.push('No browser exceptions');
 await writeFile(out+'/browser-validation.json',JSON.stringify({passed:true,isolated_database:true,checks},null,2));
 console.log(`PASS: ${checks.length} browser checks; only temporary QA cases written.`);
}finally{if(socket?.readyState===WebSocket.OPEN)socket.close();await stop(chrome);await stop(worker);await rm(temp,{recursive:true,force:true});}
