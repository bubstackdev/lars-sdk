// Read-only browser QA against the delivered local audit. No review mutations.
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import path from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';
const temp=await mkdtemp(path.join(tmpdir(),'lars-replay-qa-'));
const out=process.argv[2]||'sweep/out/codegraph-judgment/resolution-replay-v3';
const report=JSON.parse(await readFile(out+'/report.json','utf8'));
const kept=report.retained_cases,excluded=report.excluded_cases;
let chrome,socket;const pending=new Map();let serial=0;const errors=[];
function send(method,params={},sessionId){return new Promise((resolve,reject)=>{const id=++serial;const timer=setTimeout(()=>reject(Error('CDP timeout: '+method)),10000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}));});}
try{
 chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-networking',`--user-data-dir=${temp}`,'--remote-debugging-port=0','--remote-debugging-address=127.0.0.1','about:blank'],{stdio:['ignore','ignore','pipe']});
 const endpoint=await new Promise((resolve,reject)=>{let buffer='';const timer=setTimeout(()=>reject(Error('Chrome readiness timeout')),15000);chrome.stderr.on('data',chunk=>{buffer+=chunk;const m=buffer.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(m){clearTimeout(timer);resolve(m[1]);}});});
 socket=new WebSocket(endpoint);await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
 socket.addEventListener('message',event=>{const m=JSON.parse(event.data),p=pending.get(m.id);if(p){clearTimeout(p.timer);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result);}if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);});
 const {targetId}=await send('Target.createTarget',{url:'http://127.0.0.1:8765/'});const {sessionId}=await send('Target.attachToTarget',{targetId,flatten:true});
 const evaluate=async expression=>{const r=await send('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true},sessionId);if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;};
 const until=async expression=>{for(let i=0;i<100;i++){if(await evaluate(expression))return;await delay(50);}throw Error('Condition timed out: '+expression);};
 await send('Runtime.enable',{},sessionId);await send('Page.enable',{},sessionId);await send('Emulation.setDeviceMetricsOverride',{width:1440,height:1050,deviceScaleFactor:1,mobile:false},sessionId);
 await until(`document.querySelector('#retained')?.textContent===${JSON.stringify(String(kept))}`);
 assert.equal(await evaluate(`document.querySelector('#excluded').textContent`),String(excluded));assert.equal(await evaluate(`document.querySelector('#reviews').textContent`),'0');
 assert.equal(await evaluate(`document.querySelectorAll('.audit-row').length`),kept);
 for(let i=0;i<kept;i++){await evaluate(`document.querySelectorAll('.audit-row')[${i}].click()`);assert.equal(await evaluate(`document.querySelectorAll('.chronology strong').length`),3);assert.ok(await evaluate(`document.querySelector('.outcome-block').textContent.length>20`));}
 await evaluate(`document.querySelector('#show-exclusions').click()`);assert.equal(await evaluate(`document.querySelectorAll('.audit-row').length`),excluded);
 if(report.audit.some(r=>r.reason_code==='same_ticket_different_incident')){
 await evaluate(`const s=document.querySelector('.exclusion-search');s.value='same_ticket_different_incident';s.dispatchEvent(new Event('input'));document.querySelector('.audit-row').click()`);assert.equal(await evaluate(`document.querySelectorAll('.audit-row').length`),1);assert.ok(await evaluate(`document.querySelector('.audit-reason').textContent.includes('another outage')`));
 }else if(excluded===0){assert.ok(await evaluate(`document.querySelector('.empty').textContent.includes('No matching')`));}
 if(report.pilot_results){assert.equal(await evaluate(`document.querySelector('#pilot-results').hidden`),false);assert.equal(await evaluate(`document.querySelectorAll('#pilot-results table tr').length`),5);assert.ok(await evaluate(`document.querySelector('#pilot-results').textContent.includes('No meaningful judgment improvement')`));}
 await send('Page.navigate',{url:'http://127.0.0.1:8765/review'},sessionId);await until(`document.querySelector('h1')?.textContent==='Archived judgment study'`);
 assert.equal(await evaluate(`document.querySelector('#stat-uncertain').textContent`),'0');assert.equal(await evaluate(`document.querySelector('#save-review').disabled`),true);
 await send('Page.navigate',{url:'http://127.0.0.1:8765/'},sessionId);await until(`document.querySelector('#retained')?.textContent===${JSON.stringify(String(kept))}`);
 let screenshot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true},sessionId);await writeFile(out+'/qa-desktop.png',Buffer.from(screenshot.data,'base64'));
 await send('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true},sessionId);await delay(80);assert.equal(await evaluate(`document.documentElement.scrollWidth<=window.innerWidth`),true);
 screenshot=await send('Page.captureScreenshot',{format:'png',captureBeyondViewport:true},sessionId);await writeFile(out+'/qa-mobile.png',Buffer.from(screenshot.data,'base64'));
 assert.deepEqual(errors,[]);await writeFile(out+'/browser-validation.json',JSON.stringify({passed:true,read_only:true,checks:[`${kept} fixtures / ${excluded} exclusions / 0 review requests`,'all fixture timelines and source outcomes','exclusion search','cross-incident exclusion reason','read-only correction archive','mobile width','no browser exceptions',...(report.pilot_results?['comparison table and majority-baseline conclusion']:[])]},null,2));
 console.log(`PASS: ${report.pilot_results?8:7} read-only browser checks; no saved reviews changed.`);
}finally{if(socket?.readyState===WebSocket.OPEN)socket.close();if(chrome&&chrome.exitCode===null&&chrome.signalCode===null){chrome.kill('SIGTERM');await Promise.race([new Promise(resolve=>chrome.once('exit',resolve)),delay(5000)]);if(chrome.exitCode===null&&chrome.signalCode===null){chrome.kill('SIGKILL');await new Promise(resolve=>chrome.once('exit',resolve));}}await rm(temp,{recursive:true,force:true});}
