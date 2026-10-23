// Real Chrome browser checks, using isolated fixtures and a temporary profile.
// All child processes here are created by this test; existing browsers are untouched.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';

const temp = await mkdtemp(path.join(tmpdir(),'lars-review-browser-'));
const fixture = JSON.parse(await readFile('sweep/out/codegraph-judgment/review-20261022/queue.json','utf8'));
fixture.cases = fixture.cases.slice(0,3).map((row,i)=>({...row,key:`fixture-${i}`,story_id:900+i,
  confidence:[.7,.8,.9][i],state:{report:`Fixture report ${i+1}`,observations:['An observation requiring review.']},
  draft_label:'application_engineering',evidence:[{title:'Untrusted fixture',quote:'<script>window.BAD=true</script>',url:'https://app.shortcut.com/turncommerce/story/900'}]}));
fixture.metadata={test_fixture:true};
await writeFile(path.join(temp,'data.json'),JSON.stringify(fixture));
let server,chrome,socket;const pending=new Map();let serial=0;const browserErrors=[];
function waitText(child,stream,pattern){return new Promise((resolve,reject)=>{let text='';const timer=setTimeout(()=>reject(Error('Process readiness timed out')),20000);child[stream].on('data',chunk=>{text+=chunk;const m=text.match(pattern);if(m){clearTimeout(timer);resolve(m[1]);}});child.once('exit',code=>{clearTimeout(timer);reject(Error(`Process exited before readiness: ${code}`));});});}
function command(method,params={},sessionId){return new Promise((resolve,reject)=>{const id=++serial;const timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout: '+method));},15000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}));});}
async function stop(child){if(!child||child.exitCode!==null||child.signalCode!==null)return;child.kill('SIGTERM');await Promise.race([new Promise(resolve=>child.once('exit',resolve)),delay(5000)]);if(child.exitCode===null&&child.signalCode===null){child.kill('SIGKILL');await new Promise(resolve=>child.once('exit',resolve));}}
try {
 server=spawn('.venv-train/bin/python',['-m','sweep.judgment_review','--data',path.join(temp,'data.json'),'--db',path.join(temp,'reviews.sqlite3'),'--port','0'],{stdio:['ignore','pipe','pipe']});
 const url=await waitText(server,'stdout',/LARS review: (http:\/\/127\.0\.0\.1:\d+)/);
 chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking',`--user-data-dir=${path.join(temp,'chrome')}`,'--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','about:blank'],{stdio:['ignore','ignore','pipe']});
 const endpoint=await waitText(chrome,'stderr',/DevTools listening on (ws:\/\/[^\s]+)/);
 socket=new WebSocket(endpoint);await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
 socket.addEventListener('message',event=>{const message=JSON.parse(event.data);if(message.id){const p=pending.get(message.id);if(p){clearTimeout(p.timer);pending.delete(message.id);message.error?p.reject(Error(JSON.stringify(message.error))):p.resolve(message.result);}}if(message.method==='Runtime.exceptionThrown')browserErrors.push(message.params.exceptionDetails.text);});
 const {targetId}=await command('Target.createTarget',{url});
 const {sessionId}=await command('Target.attachToTarget',{targetId,flatten:true});
 const evaluate=async expression=>{const r=await command('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true},sessionId);if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 const until=async expression=>{for(let i=0;i<100;i++){if(await evaluate(expression))return;await delay(50);}throw Error('Browser condition timed out: '+expression);};
 await command('Runtime.enable',{},sessionId);await command('Page.enable',{},sessionId);
 await command('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false},sessionId);
 await until(`document.querySelector('#stat-total')?.textContent === '3'`);
 assert.equal(await evaluate(`document.querySelector('#queue-count').textContent`),'1');
 assert.equal(await evaluate(`window.BAD === undefined`),true,'source evidence must not execute');
 await evaluate(`document.querySelector('#reviewer').value='Browser fixture reviewer';document.querySelector('input[value="insufficient_evidence"]').click();document.querySelector('#facts-checked').click();document.querySelector('#notes').value='The trace does not support the original proposed track.';document.querySelector('#save-review').click()`);
 await until(`document.querySelector('#stat-corrected').textContent === '1'`);
 await command('Page.reload',{},sessionId);await until(`document.querySelector('#stat-reviewed')?.textContent === '1'`);
 await evaluate(`document.querySelector('#filter').value='confident';document.querySelector('#filter').dispatchEvent(new Event('change'))`);
 await until(`document.querySelector('#queue-count').textContent === '2'`);
 assert.equal(await evaluate(`document.querySelector('.case h2').textContent`),'Fixture report 2');
 // Unsaved-edit navigation must be explicitly discarded, including filter changes.
 await evaluate(`document.querySelector('#notes').value='Unsaved';document.querySelector('#notes').dispatchEvent(new Event('input'));document.querySelector('#filter').value='all';document.querySelector('#filter').dispatchEvent(new Event('change'))`);
 await until(`document.querySelector('#leave-dialog').open`);
 await evaluate(`document.querySelector('#stay').click()`);await delay(30);
 assert.equal(await evaluate(`document.querySelector('#filter').value`),'confident');
 await evaluate(`document.querySelector('[data-key="fixture-2"]').click()`);await until(`document.querySelector('#leave-dialog').open`);await evaluate(`document.querySelector('#discard').click()`);await until(`document.querySelector('.case h2').textContent==='Fixture report 3'`);
 await evaluate(`document.querySelector('#notes').value='Duplicate fixture excluded for the test.';[...document.querySelectorAll('.two-actions button')].find(b=>b.textContent==='Exclude').click()`);
 await until(`document.querySelector('#toast').textContent==='Saved locally.'`);
 const exported=await (await fetch(url+'/api/export')).json();
 assert.equal(exported.reviewed_cases.length,1);assert.equal(exported.reviewed_cases[0].label,'insufficient_evidence');
 assert.equal(exported.cases.find(c=>c.key==='fixture-2').review.status,'excluded');
 await evaluate(`document.querySelector('#filter').value='all';document.querySelector('#filter').dispatchEvent(new Event('change'))`);await delay(70);
 const desktop=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:true},sessionId);
 await writeFile('sweep/out/codegraph-judgment/review-20261022/qa-desktop.png',Buffer.from(desktop.data,'base64'));
 await command('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true},sessionId);await delay(70);
 assert.equal(await evaluate(`document.documentElement.scrollWidth <= window.innerWidth`),true,'mobile layout must fit');
 const mobile=await command('Page.captureScreenshot',{format:'png',captureBeyondViewport:true},sessionId);
 await writeFile('sweep/out/codegraph-judgment/review-20261022/qa-mobile.png',Buffer.from(mobile.data,'base64'));
 assert.deepEqual(browserErrors,[]);
 await writeFile('sweep/out/codegraph-judgment/review-20261022/browser-validation.json',JSON.stringify({passed:true,checks:['uncertain queue','0.8 boundary','XSS text rendering','correct label','durable reload','unsaved edits','filter cancellation','exclusion','reviewed-only export','responsive layout','no browser exceptions'],fixture_only:true},null,2));
 console.log('PASS: 11 real-browser checks on isolated fixture data; no production reviews created.');
}finally{if(socket?.readyState===WebSocket.OPEN)socket.close();await stop(chrome);await stop(server);await rm(temp,{recursive:true,force:true});}
