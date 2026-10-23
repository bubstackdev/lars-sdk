> 2026-10-23 — user requested a committed and remotely pushed research checkpoint.
> Branch: wip/lars-research-checkpoint-20261023. This preserves the accumulated
> training, Flash bridge, calibration/evidence experiments, tests and research notes.
> Work remains paused; no release, model promotion or GPU experiment is authorized
> by this checkpoint. The prior wind-down state and stopped local UIs remain in effect.
> Checkpoint validation: Python suite110 tests, OK(1 optional real-model test skipped),
> MLX forced to CPU;3 Node pack tests PASS; git diff --check PASS. A stale existing
> Flash mock initially failed3 tests; a fresh isolated mock-only CMake build resolved
> all3, without editing/rebuilding the separate Flash checkout or running GPU models.
> To reproduce: cmake -S sweep/flash_teacher -B <fresh-build-dir>
> -DFLASH_BUILD_RUNTIME=OFF -DCMAKE_BUILD_TYPE=Release; build with-j2;
> set FLASH_TEST_BUILD=<fresh-build-dir>; load training.tests.test_* modules explicitly
> (namespace-package discovery with top_level_dir='.' is unsupported), set
> mlx.core.set_default_device(mlx.core.cpu), then run unittest suite.
> Source scan found no live credentials; credential URL fixture uses example.invalid.
> Ignored sweep/out (45GB), weights, runtime builds, local SQLite and source snapshots
> are NOT in this Git checkpoint. They remain on this machine and need separate backup
> if required. No artifact deletion or external-system mutation occurred.

> 2026-10-23 — USER REQUESTED WIND-DOWN; research PAUSED.
> Queue workflow experiment parked: historical evidence too sparse, target may encode
> operator preference, prospective UI assumes operational work the user does not do.
> No task for user to populate collector. No automatic collection/training scheduled.
> Own UI servers on8765(PID76672) and8766(PID34912) stopped after command verification.
> Earlier RUNNING entries below are historical. 8B remains paused; other sessions untouched.
> Repo snapshot before this handoff: main,8 modified tracked files,69 untracked files
> (~9,980 lines),nothing staged. Experimental training/backend work and numerous studies
> remain mixed in working tree; do not bulk-add, reset, delete or call it release-ready.
> Ignored sweep/out is45GB: accuracy-full-smoke24GB,accuracy-full18GB dominate.
> No artifacts deleted, commits made, or weights promoted. Recent queue modules passed
> 25 targeted tests +10 isolated browser checks; whole dirty repo not revalidated.
> Next session: review existing changes, decide which experiments/code to retain, separate
> coherent commits, then consider archiving redundant checkpoints after provenance review.
> No new GPU work until requested. Latest evidence/protocol/artifact locations below.

> 2026-10-23 — prospective collection and target-utility review SET UP.
> Local UI http://127.0.0.1:8766/ is RUNNING. Own PID34912, saved under
> sweep/out/codegraph-judgment/queue-prospective-20261023/server.pid and server.log.
> New sweep/queue_collection.py + judgment_review_ui/collection.html. Fresh append-only
> collection.sqlite3 starts EMPTY. No synthetic QA or historical cases in live journal.
> User saves literal observations and source links BEFORE queue action. Current local
> UTC checkpoint immutable; new decision checkpoint IDs for changed facts, same family.
> Separate assessment supports reprocess_only/discard_only/either/insufficient, confidence,
> rationale and checkpoint-only attestation. Other reviewers' answers and actual action
> absent from assessment API. Reviewer identity/blinding self-reported, not authenticated.
> Outcome requires explicit source quote and reported action start strictly after saved
> checkpoint and no later than now. Success separate. SQLite immutable triggers + journal
> hash chain check; not independent trusted execution-time proof. Low confidence<.8,
> either/insufficient cases get review_needed in export. No automatic fitting promotion.
> Target review begins at20 completed cases with2 reviewers; candidate threshold80%
> single-action agreement is a discussion trigger, not validation. Inspect both-acceptable,
> insufficient evidence, reviewer disagreement and observed-action disagreement. Family,
> source timing and class support audits still mandatory before any scoring.
> No automatic ingestion is connected. This is a local capture/review tool, not a claim
> that future cases are accumulating unattended. No messages sent or systems changed.
> Available Codegraph tools have no Slack/PagerDuty reader. Limited read-only probes:
> resolved Grafana incidents(limit10) returnednone; DomainInventoryApi Republish logs
> 2026-10-22T00Z..2026-10-23T00Z returned0. Not exhaustive; source-access-probes.json saved.
> Existing four historical candidates remain quarantined; old audit UI8765 unchanged.
> Frozen20-group/5-per-label/10-diagnostic validation gate remains; model unchanged.
> 25 targeted Python tests +10 isolated browser workflow/security/mobile checks PASS.
> Live HTTP checks PASS, empty journal. protocol.json pins code/UI/test hashes; validation
> and desktop/mobile screenshots saved. 8B paused, GPU untouched; other sessions untouched.
> Export: .venv-train/bin/python -m sweep.queue_collection export --db
> sweep/out/codegraph-judgment/queue-prospective-20261023/collection.sqlite3

> 2026-10-23 — fresh validation admission audit COMPLETE; evidence gate FAILED.
> Frozen structured extractor/prompt/rules and affine(a=.9827921721,b=1.2155592281)
> from original25 fit groups BEFORE fresh searches. Pack choice temperature1.3810558844;
> Jeffreys reprocess prior20.5/26. Old361 full stories excluded, old16 test remains spent.
> Artifacts: sweep/out/codegraph-judgment/queue-fresh-validation-20261023/.
> 16 prespecified/extended searches fully paginated (43 pages; no pagination truncation),
> 377 unseen full stories saved sanitized,14 selected histories audited. This is bounded
> search coverage, NOT all Codegraph. Four source-backed independent groups admitted:
> 113637 discard(diagnostic);119161 reprocess(diagnostic);116075 reprocess(queue-count
> symptom);132446 reprocess(customer symptom). All exact input quotes precede outcome
> record; no same-note diagnostic enrichment. Actual execution clocks often unavailable.
> Full formal-link/parent/shared-PagerDuty/shared-MR/commit graph over old+new snapshots
> excludes24 new stories dependent on prior work. Four admitted remain independent.
> No_verified_completed_queue_disposition is a screening exclusion, not an exhaustive
> historical proof of no action. Explicit exclusions distinguish proposals, action before
> first record, domain portfolio purges and unclear final mechanisms.
> Gate requires20 groups,>=5 per mechanism,>=10 diagnostic. Actual4 groups (3reprocess,
> 1discard),2diagnostic. Insufficient: NO scoring, comparison, fitting or GPU work started.
> Fresh cases quarantined; do not tune on them. Never relax timing to fill the cohort.
> prospective-case-template.json captures immutable pre-action checkpoint, source links,
> related incidents/fixes and separate post-action outcome; collect all eligible consecutive
> incidents. Need at least16 additional independent groups, including4 discard and8
> additional diagnostic groups, before this small pilot can run. More needed for precision.
> New sweep/queue_validation.py records admission gate, evidence and dependence audit;
> explicit past return to consumer queue is accepted, generic 'cleared' alone is not.
> 19 targeted tests PASS; all artifact/source hashes and four timelines verified.
> 8B paused; other sessions untouched; existing UI still displays prior25-label pilot.

> 2026-10-23 — structured pre-action exploration COMPLETE (fitting-only).
> User authorized brief current-2B inference, not 8B training. Fresh artifacts:
> sweep/out/codegraph-judgment/queue-structured-20261023/{study-plan,results,manifest}.json.
> Exactly 25 original fit groups; old 16 examined test groups excluded before extraction.
> Original quotes retained; literal structured matches carry exact quote evidence;
> unrecorded fields explicitly unknown. No outcome/current-description fields used.
> Original versus structured pack: 16/25 versus13/25 accuracy; AUC .64 versus .78;
> balanced accuracy .625 versus .70. Fixed-penalty fivefold stratified group OOF
> calibration: both19/25 (76%), discard1/5; structured Brier .310/logloss .484 versus
> original .353/.530. Jeffreys class-prior baseline20/25 (80%), Brier .320/logloss .501.
> Simple exploratory rules decide6/25, all6 match source actions; prior fallback22/25.
> Rule design saw fitting cases/labels: neither six-of-six nor22/25 validates rules.
> Seven cases are sparse alerts/symptoms. Ownership mismatch coverage0/25; no facts
> were inferred to fill gaps. All figures exploratory; no independent accuracy gain.
> Ranking is mildly encouraging, accuracy is not. Next useful experiment is a fresh,
> independent, timestamp-audited holdout with both outcomes; freeze these features,
> prompt, rules, and calibration procedure first. No retuning on old16 or claiming
> this cross-validation validates human feature selection. No further GPU run started.
> 25 native captures+8 controls took4.469s; parity probability delta0, flips0; worker
> exited. 15 targeted Python tests PASS. Frozen input/output hashes verified.
> No model weights, shipped pack, certificate, UI or external systems changed.
> Other sessions untouched; 8B stays paused. Existing audit UI shows prior pilot,
> not these new fitting-only results. See results.json for detailed class metrics.

> Latest, 2026-10-23: expanded mining reached a pilot cohort; comparison COMPLETE.
> User explicitly requires evidence recorded BEFORE the action. Same-note retrospective
> findings are not features. Source metadata and any stated earlier operation time are
> checked; independent execution clocks are unavailable for many historical records.
> Read sweep/out/codegraph-judgment/queue-pilot-20261023/report.json first.
> 361 full Shortcut story records and histories saved in deep-mining-20261023/.
> Admitted 42 cases / 41 dependence groups: 29 diagnostic, 13 early-alert/symptom-only.
> Only 41 hash-selected representatives are scored. Source links, same PD incidents,
> shared fixes, and a conservative same-day outage join families. Earlier case cohorts
> and the old 19-review task remain archived; no inferred triage categories were revived.
> Frozen cohort-v3: 25 fit groups (20 reprocess / 5 discard), 16 held-out (12 / 4),
> including 11 diagnostic and 5 sparse test cases. One fitting label per family.
> Original creation titles may supply sparse inputs; current titles/descriptions and
> later findings do not. Historical pre-action description revisions are permitted.
> Sparse recurring inputs in separate real incidents are tracked explicitly; their
> identical logits were verified. Do not call such repetitions diagnostic information.
> Labels are SOURCE-DOCUMENTED FINAL QUEUE MECHANISMS (reprocess versus remove), not
> invented investigation tracks, optimal policy, or independent human adjudication.
> Two ambiguous 'cleared' action records were excluded rather than inferred as purges.
> 0/10/25-label experiment is frozen and DONE. Existing 2B pack 10/16 (62.5%);
> 10-label draws 12/16,12/16,10/16 (mean70.8%); 25-label all12/16 (75%).
> Class-prior-only baseline also12/16 and BETTER Brier/log loss (.378/.567 vs .400/.587).
> Diagnostic accuracy remains9/11; calibrated model misses all4 discard outcomes.
> The 25-label paired accuracy interval is [-12.5,+37.5] percentage points. NO meaningful
> judgment/accuracy gain established. Do not present the raw10->12 as improved reasoning.
> Fitting/CV used only fitting labels and proper log loss; all outputs were scored once
> on the pre-frozen held-out groups. Draws share one holdout, not independent replications.
> New scripts sweep/queue_evidence.py and queue_calibration.py; source/action guards,
> immutable evidence anchors, group split, source hashes, and explicit limitations saved.
> Standard reviewed-only runner and retired-question guard unchanged. No certificate,
> shipped pack, hidden head, deployment, or backbone training. 8B remains paused.
> 41 native2B captures +8 controls took8.18s; control probability delta0; worker exited.
> 37 relevant Python tests +8 read-only Chrome checks PASS. All frozen hashes verified.
> UI http://127.0.0.1:8765 now serves queue-pilot-20261023/audit-v1/report.json, including
> comparison table, source timelines and sparse/diagnostic distinctions. Own server
> PID76672 in review-20261022/server.pid. /review retains original SQLite read-only.
> Other sessions untouched. New snapshots redact credential values found in old logs;
> no credential values are model features. No production systems were mutated.
> Next research must address weak conditional discrimination/diagnostic inputs; simple
> probability calibration learned the majority preference here. Do not tune prompts or
> features on the now-examined16-group holdout and call that a new validation result.
> Earlier v4/v3/v2 replay notes below are historical; artifacts remain preserved.

> Latest, 2026-10-23: remaining history audit COMPLETE; workflow feasibility checked.
> All 32 pending histories adjudicated from their source evidence: 1 admitted (142001),
> 31 excluded with specific reasons. No pending-history queue or user relabeling task.
> Active resolution-replay-v4: 12 cases / 12 preserved groups, 63 exclusions, zero reviews.
> New 142001 recovery comes from a timestamped description edit; builder now permits
> history edits as outcomes/cutoffs while retaining creation-only history inputs.
> Observations never use current description or later outcome bytes.
> Nameserver-change support selected for deeper mining after cart/refund/nameserver
> first-page searches. 16 new story records audited, 10 histories pinned, one related
> repair story and merged premiumdomains MR86 inspected. Search is NOT exhaustive.
> Workflow corpus has 4 fixtures / 4 groups (3 new + existing 140396). Each has a
> different corrective action: no repeated target to fit and test. Multiple domains
> in 136973 and linked 137059 are ONE group. New workflow roles stay unassigned.
> Do not fabricate a learning curve, action-category gold labels or a held-out split.
> Frozen study-plan.json records gates before capture/scoring; no comparison/GPU started.
> Read workflow-audit-20261023/{report,workflow-report,study-plan}.json and admission
> audits for case-level evidence. Hash manifests cover source snapshots and outputs.
> 26 relevant Python tests and 7 read-only Chrome checks PASS; all pinned hashes verified.
> UI http://127.0.0.1:8765 serves v4; own server PID85494 in server.pid. Original review
> DB/archive unchanged; prior judgment metrics remain retired. 8B remains paused.
> Next data need: independent repeated action/outcome cases with diagnostic checkpoint
> facts. Sparse intake tickets alone cannot establish that the correct repair was
> inferable. Do not resume training to compensate for absent labels or observations.
> Earlier v3/v2 notes below are historical and superseded.

> Latest, 2026-10-23: live Codegraph access RESTORED; further evidence mining done.
> Active audit is resolution-replay-v3: 11 cases / 11 preserved groups, 64 exclusions,
> zero manual reviews. Six added cases use exact original-description excerpts from
> recorded story creation events, with later source repair/outcome excerpts. No
> curator-created categories or new performance claim. Sparse observations remain
> compatible with multiple causes; these are replay fixtures, not calibration proof.
> Pinned 73 ticket histories under mining-20261023/history/; selection.json pins hashes.
> Re-fetched seven candidate full records; comment text matches earlier snapshots.
> Six admitted: 132795, 136164, 141213, 121405, 128839, 140396. Additional 141835 candidate
> remains outside the frozen 75-case cohort; saved separately pending group admission.
> 32 prior description-version exclusions now have recovered histories, but have not
> passed an observation/outcome admission audit. Remain excluded without user tasks.
> Local audit http://127.0.0.1:8765 uses v3; own server PID65115, also in server.pid.
> Original review SQLite untouched/read-only via /review. Prior retired metrics remain
> retired. 8B paused; no model scoring, fitting or GPU work. Other sessions untouched.
> Validation: 24 relevant Python tests + 7 read-only Chrome checks pass; bundle hashes verified.
> Builder supports pinned creation-event evidence and rejects post-cutoff events,
> updates passed as creation, invented quotes and modified snapshot hashes.
> Next: finish source admission on recovered histories; obtain multiple independent
> cases per outcome before any learning curve. Historical v2 notes below are superseded.

> Latest,2026-10-23: user challenged re-labeling fixed bugs; benchmark correction DONE.
> RETIRE the19-case manual queue and previous judgment-category accuracy comparison.
> Old inputs included later diagnostic facts; categories were curator-created. Prior
>4/9 -> mean5/9 is invalid as adaptation evidence. All old artifacts/corrections retained.
> study-retirement.json records retired question signature; probe CLI blocks its rerun.
> Active resolution-replay-v2:5 exact-source replay fixtures/5groups,70 excluded with
> reasons,zero reviews requested.4 pre-finding +1 pre-resolution case. No original-intake
> claim. Earlier comment quotes only; latest edit before explicit cutoff (including
> related-ticket cutoff); later source findings/resolutions supply direct output labels.
> No inferred triage categories, confidence-based pseudo-gold, or diagnoses inserted into
> observations. Source labels documented_outcome; no independent human-review fiction.
> Source history unavailable: live Codegraph tools absent, plugin search no match.36
> cases excluded for unverifiable earlier versions; further unresolved/ambiguous cases
> excluded. One later OOM outcome belonged to another outage in the same ticket.
>5groups too few for calibration/accuracy claims: no scoring, fitting or GPU work done.
> UI http://127.0.0.1:8765 now AUDIT, serving resolution-replay-v2/report.json. Detached
> own server PID44826 (server.pid records it). Prior reviews at /review READ-ONLY; writes
>410. Original SQLite database retained, backed up via SQLite into new bundle;0 recorded
>events at backup time. Never overwrite/reset the original DB; it remains the archive.
>25 targeted Python tests +7 read-only Chrome checks pass. Inputs/expected outcomes,
>source timestamps/offsets, exclusions and frozen builder in resolution-replay-v2.
>sweep/resolution_replay.py and audit UI assets implement the correction. Earlier v1 is
>superseded preparation output. Read guide's2026-10-23 section for restart command.
>Next useful data: original ticket versions or independently dated earlier observations
>with explicit later outcomes, yielding more independent incidents. No user relabeling
>of closed bugs required.8B remains paused; other sessions/processes untouched.

> Latest,2026-10-22: autonomous audit, provisional comparison and LOCAL REVIEW UI done.
> User rule: below0.8 confidence -> human queue.103 ticket snapshots,75 curated cases,
>64 proposed groups; active corpus bundle-v4. Original44 inputs/roles preserved.56 cases
>meet agent threshold,19 uncertain; confidence is subjective and uncalibrated, never
>independent human gold. Five cases have additional pinned source/change corroboration;
>C# tests read, not executed. UI http://127.0.0.1:8765 is RUNNING (our server session92738).
>Retain it for the user. No other session controlled. No GPU worker remains;8B paused.
>App sweep/judgment_review.py; assets sweep/judgment_review_ui/. Queue and uncertainty
>CSV/JSON in sweep/out/codegraph-judgment/review-20261022/. Persistent corrections in
>reviews.sqlite3, bound to queue hash, append-only revisions, optimistic concurrency.
>Do not overwrite/reset that DB or regenerate queue in place; user may already review.
>Supports labels/facts, reasons, exclude/defer/reopen, family concerns, exports, unsaved
>edits. High-confidence agent assessments distinguished from human-reviewed decisions.
>18 Python tests +11 real-Chrome checks pass; all browser mutations in isolated fixtures.
>75 raw2B outputs cached capture-v4;8 controls exact,all IDs/keys checked,~8s GPU run.
>CPU-only exploratory probe-v4-final: confidence>=.8,one case/family,23fit/14cert/9test.
>0-label pack4/9;10-label draws6/9,4/9,5/9 (mean5/9); hardBrier .70984 ->.58208mean.
>Tiny agent-labeled diagnostic, no established accuracy gain;25+ labels unsupported
>after confidence filtering. No certification, deployment, hidden-head or backbone run.
>Standard reviewed-only runner still rejects draft labels. Probe accepts future UI
>snapshots with human corrections, omits excluded/deferred/family-concern cases; fact
>edits need recapture. Strict reviewed-only join also exists in judgment_capture.py.
>Next: user reviews19; export live DB and run a NEW corrected comparison directory.
>Restart/export commands and full limitations in sweep/LOCAL_CALIBRATION.md.

> Latest,2026-10-22: historical judgment-task preparation COMPLETE; no GPU used.
> Codegraph read-only harvest:67 Shortcut story snapshots,4 pinned contextual runbooks;
>44 reconstructed diagnostic checkpoints selected,23 stories not curated. Active bundle:
> sweep/out/codegraph-judgment/bundle-v3/.44 cases/34 conservative dependence groups;
> fit13/10groups,certificate19/14groups,test12/10groups. Formal links through shared
> external stubs join related reports. No paraphrase grid or independent-incident claim.
> Labels source_interpreted_unreviewed; exact evidence excerpts verified, independently
> adjudicated labels0. Several sources include AI-assisted or revised diagnoses. Inputs
> contain diagnostic observations, not verified original intake. Labels/outcomes/source
> titles/URLs excluded from inference input; blind-review CSV has blank answer fields.
> Source snapshots local/gitignored; credential-like values redacted; curated inputs
> omit customer identities/domains. No operational instructions or external writes.
> Driver sweep/codegraph_judgment.py; study-plan.json fixes review and later0/10-label
> comparison. More data needed: fit/test lack infrastructure examples; certificate lacks
> configuration;14cert groups cannot establish5% bound. No model scoring, fitting,
> training or certification started. Seven relevant tests pass; frozen bundle verified.
> Next gate is independent label/chronology/family review plus case expansion, before
> a credible adaptation comparison. Existing runner still rejects draft labels.
> Guide sweep/LOCAL_CALIBRATION.md.8B remains paused; other sessions untouched.

> Latest,2026-10-22: Codegraph source audit and trained 2B baseline COMPLETE.
> sweep/out/codegraph-baseline/:212 synthetic draft policy probes,143 correct overall.
> Frozen tiers: implementation corroborated95/136 (69.9%), documentation only16/29,
> quarantined32/47. Corroboration means inspected pinned implementation/test source,
> not executed remote tests or independently reviewed company gold.47 quarantined:
> HTTP retry40 (wiki/code conflict), withdrawal4 (README/code conflict), domain-return3
> (missing runtime guards). Original draft inputs/gold unchanged; no labels promoted.
> Billing retries contribute38/41 corroborated errors,74/112 correct from only7 families;
> invoice closure12/12, non-withdrawal refunds9/12. Descriptive source-majority reference
> on corroborated cases70.6%; no aggregate advantage established. error-analysis.json
> records confusion counts/failing inputs; all212 row identities checked.
> Eight hidden controls failed probability parity (maxdelta0.00395974 >0.001); logits-only
> fallback exact, used for main capture. No212-row hidden capture. Runtime about24s;
> our workers exited. Three targeted CPU tests pass. No training/fitting/certification.
> Next: review and expand independent case families before any split/learning curve;
> exact retry/time rules belong in code. Target remaining judgment tasks for adaptation.
> Guide sweep/LOCAL_CALIBRATION.md; driver sweep/codegraph_baseline.py.
>8B remains paused; other sessions untouched. Earlier notes below describe prior states.

> Latest,2026-10-22: local calibration setup COMPLETE and CPU rehearsal finished.
> User chose saved workflows first and authorized broad Codegraph mining for cases.
> New runner sweep/local_calibration.py; guide sweep/LOCAL_CALIBRATION.md. Output
> sweep/out/local-calibration/:20 predicates, four trained/previously evaluated
> synthetic business workflows; whole normalized states/families disjoint across fit,
> certificate,test and excluded from base-training state overlap.100 fit states,
>300 certificate states,91 evaluation states (2455 decisions;455 eval decisions).
> Cached data supports0/10/25 fitting labels per predicate;50/100/200 are explicitly
> unsupported.25 fit +75 certificate +21–26 eval labels per predicate. Three nested
> draws share evaluation cases and are not independent tests. CV guard pack->fitted
> ->recal; optional heads only with supplied hidden vectors and>=50 fit labels.
> Current rehearsal has no hidden vectors/heads. Macro accuracy74.76/74.97/75.41%;
> hard-label Brier0.41956/0.39556/0.34067; soft Brier0.06672/0.12198/0.10884.
> No5%-error threshold qualified; coverage0, accepted error null.75 certificate labels
> barely exceed73-clean minimum. No new-company or deployment guarantee established.
> Results.json/summary.json retain per-predicate metrics, fit keys, CV and label costs.
> Five CPU tests pass, covering gold isolation, role/family leakage, frozen identities,
> unsupported budgets, thresholds, optional heads and source-rule boundaries.
> SciPy1.18.1 installed in .venv-train; training[calibration-study] extra added.
>
> Codegraph READ-ONLY harvest: seven pinned README/runbook sections from six systems,
> used to construct212 synthetic contract probes in43 case families across7 predicates.
> Output sweep/out/codegraph-calibration/:sources/,cases.records.jsonl,cases.wire.jsonl,
> evidence.jsonl,cases-manifest.json,review.csv,review-manifest.json. Rules cover billing
> retries, HTTP retries, payout status/error precedence, refunds, invoice closure,
> escalation and domain returns. All labels rule-derived/unreviewed, not production
> history or human/company gold; remain separate and are rejected by calibration runner.
> Review CSV has blank reviewed_label/reviewer columns. No operational instructions from
> source docs executed, no external writes/messages/publishing. No GPU or training used;
>8B remains paused; other sessions untouched. Next: review/extend source-derived cases
> with representative company gold before claiming larger domain learning curves.
> Earlier notes below describe previous states.

> Latest,2026-10-22: authorized reasoning teacher pilot COMPLETE; no GPU worker
> remains, pressure normal.32 state-disjoint selected cases,24 prediction disagreements
> +8 agreement controls. Trained2B19/32; Flash direct logits19/32; reasoning19/32.
> Reasoning got12/24 disagreements vs11/24 each baseline but lost one of8 agreement
> controls. Against Flash logits:1 correction,1 regression,5 answer changes overall.
> All32 reasoning outputs valid; generated-direct control16/32 with26 valid outputs
> (5 token-budget failures,1 format failure). Different budgets128 vs1024 are explicit.
> Runtime277.93s including8 sequential model loads; reasoning mean5.84s/decision.
> No teacher advantage established; no distillation/training/calibration/promotion.
> User asked whether local calibration is needed: yes, recommend representative local
> gold labels, per-predicate calibration/adaptation, then independent accepted-error and
> coverage evaluation. Not a universal model capacity ceiling. Local calibration learning
> curve is the recommended next experiment, not started/authorized as a separate run.
> Artifacts: sweep/out/flash-reasoning-pilot/{plan,status,comparison,analysis}.json,
> cases.jsonl (gold separate from prompts),capture.jsonl,smoke.json,cpu-validation.json,
> frozen-source/. Actual greedy generation now exists in native/Python bridge;
> worker capability greedy-v1; CPU vocabulary mode refuses it.21 CPU tests pass.
> Build target/flash-teacher-reasoning/. Workers rotated every4 cases, memory guard
> stayed enabled; other sessions untouched.8B remains paused.
> Earlier notes below describe previous states.

> Latest, 2026-10-22: Flash prompt/token investigation complete. Original control
> reproduced exactly on250/250 decisions. Removing assistant `Answer:` prefill gives
>75.6% (189/250), vs Flash original74.8% (187/250), tying trained2B75.6%.
> Raw Brier remains worse: native Flash0.32235 vs2B0.29519. Format-effect paired95%CI
>[-1.56,+3.07]pp includes zero. Bare-letter variant74.78% on230 eligible decisions,
> versus2B/control74.35% on the identical230; no useful overall rescue established.
> Three labels CJ/CZ/EJ are multi-token without spaces;20 LEDGAR/CLINC rows excluded
> only from the bare diagnostic, with all alternatives preserved elsewhere.
> Memory guard exited our worker after243 rows; validated saved rows and completed
> only7 in a fresh worker. Both exited. No other process controlled. Future diagnostic
> workers rotate every100 rows, with no overlap; memory guard remains enabled.
> New prefix capability negotiation rejects old workers for bare-token requests.
> Current diagnostic build: target/flash-teacher-current/. Original binaries/results
> preserved. Frozen diagnostic source/binary and recovery script are in
> sweep/out/flash-prompt-investigation/; comparison.json contains results and CIs.
> No training/calibration/promotion.8B remains paused. Next useful teacher test would
> require actual reasoning/generation and gold-checked disagreements, not blanket
> direct-logit relabeling. It has not been implemented or benchmarked.
> Earlier notes below describe previous states.

> Latest, 2026-10-22: user authorized GPU use for the Flash teacher pilot. Complete;
> worker exited and GPU allocation released. Real cached/full/replay/label-order logits
> matched exactly on six decisions plus a shared-prefix batch.250 dev decisions across
> 25 sources: Flash raw accuracy74.8% vs trained2B75.6% (-0.8pp; paired95%CI
> [-6.51,+4.44]pp). Raw Brier worsened0.29519 ->0.32951. Flash corrected29 student
> errors and lost31 student hits. Capture189.56s + load6.54s (~0.758s/decision).
> Artifacts: sweep/out/flash-teacher-pilot/{gpu-parity,comparison,paired-analysis}.json.
> No training, distillation, promotion or certification performed.8B remains paused.
> Next: broader frozen teacher evaluation or diagnostic prompt/label-format checks;
> current evidence does not support blanket teacher-target replacement.
> Earlier notes below describe previous states.

> Latest, user requested Flash teacher setup with CPU tests only. Complete: native
> persistent worker + Python adapter + prepare/capture/report tooling;15 CPU tests
> passed, actual qwen4exp vocabulary/template included.250-row pilot is CPU-prepared
> in sweep/out/flash-teacher-pilot/ (~94MiB vocabulary-worker RSS; no weights/GPU).
> Instructions: sweep/flash_teacher/README.md. GPU numerical parity/teacher quality
> remain unmeasured. Do not launch capture or resume8B without user direction; the
> user's separate Flash workload remains untouched. Existing8B is paused and later
> resume needs provenance revalidation because the new backend changes code hashes.
> Older activity notes below are historical.

> Latest user instruction: CANCEL/PAUSE 8B training to free the GPU; user will ask
> to restart later. Verified capacity-8b runner and its own training child exited;
> status.json is paused_by_user. Do not restart automatically. The base download and
> completed 50-step probe are saved. Full training stopped around step290/5021, before
> its first step500 checkpoint; no full-run checkpoint exists. The current trainer
> cannot resume optimizer state, so full training restarts on a later explicit resume.
> Command then: .venv-train/bin/python -m sweep.capacity_8b (Metal access required).
> Saved stages are content-verified and skipped. Saved PIDs are historical; verify
> identity before process actions. Other sessions and jobs were untouched.

> Latest authorization: trained8B artifact is missing locally and from HF; user
> explicitly authorized rebuilding it. Active output: sweep/out/capacity-8b/. Base
> Qwen3-8B pinned at b968826d9c46dd6066d109eabc6255188de91218 is downloading via the
> current session's download job. Detached runner waits for base-path.txt, then runs
> sweep.capacity_8b:50-step probe, <=12 projected training-hours gate, one gold-only
> epoch, Q3 build/parity/calibration/SDK checks, paired dev/latency comparison. It then
> stops. No other sweep resumes; no publication. See INVESTIGATION.md for scope and
> recovery limitations. runner.pid/status.json identify current work; verify PID
> identities before signals. Older notes below are historical.

> Latest user instruction: STOP data comparison; identify a stronger intervention.
> data-correction-s0 runner and its child exited; state stopped_by_user. No new GPU
> work started afterward. CPU-only expert headroom analysis is saved in
> sweep/out/stronger-intervention/headroom.json and INVESTIGATION.md. Naive confidence
> fallback loses; exploratory source routing gains1.60 points. Existing trained8B
> and per-predicate heads have stronger historical evidence. Next proposed work is a
> bounded inference-only capacity/expert benchmark, not more training. No benchmark
> or training is running. Older active-run notes below are historical.

> Current active experiment: targeted data refresh, explicitly authorized 2026-10-22.
> `sweep/out/data-correction-s0/`, command `.venv-train/bin/python -m sweep.data_correction
> --run`. Two sequential matched 5366-step arms, baseline vs 577 refreshed training
> examples; 39846 records/source counts preserved. Select-only audit, raw select/dev
> evaluations, no test/confirmation or automatic additional work. See INVESTIGATION.md
> for audit, exclusions and interpretation. Expected 7–9 active hours. Eleven relevant
> tests pass. Older notes below are historical; other sweeps remain stopped.

> Latest, 2026-10-22: five-arm source/mixture ablation COMPLETE; runner stopped
> automatically. No new GPU work started. Verbosity-only mix50 is the only arm above
> baseline on development selection score: 67.9243% vs 67.6593% (+0.265 points),
> macro Brier 0.293878 vs 0.298820. Other isolated sources and all25 lose; see
> INVESTIGATION.md / sweep/out/distillation-ablation-s0/results.json. Single-seed
> development evidence only, no promotion. No further training is authorized by the
> completed five-arm sweep; wait for user direction. Earlier running notes are historical.

> Current active run: user authorized and started the five-arm development-only
> source/mixture ablation. `sweep/out/distillation-ablation-s0/` is authoritative for
> current status/log/PID. Command: `.venv-train/bin/python -m sweep.distillation_ablation
> --run`. ARC-only50, MMLU-only50, MNLI-only50, verbosity-only50, all25, sequential,
> seed0/full epoch/final checkpoint. No test evaluation or promotion. Setup/tests passed;
> estimated 16–20 hours. See INVESTIGATION.md. The original full sweep stays stopped.
> Older completion/stop notes below describe the earlier experiments.

> Latest: checkpoint comparison COMPLETE. Both recipes select final step 5366 over
> step 2500 on the frozen 2642-row selection split. Selection score baseline 0.676593,
> distilled 0.664957; distilled loses 1.16 points, driven by unseen-source accuracy
> 0.543333 vs 0.568333. The earlier-checkpoint hypothesis is unsupported for these
> candidates. No GPU work remains from the comparison; the full sweep is still stopped.
> See INVESTIGATION.md and sweep/out/checkpoint-selection-s0/results.json. Older
> running-state notes below are historical. Next candidate is a development-only
> target-source/mixture ablation; do not automatically launch new training.

> Current authorization: user requested the proposed checkpoint comparison. Inference-only
> `sweep.checkpoint_selection` is now running in `sweep/out/checkpoint-selection-s0/`.
> It compares saved steps 2500/final for both seed-0 recipes on the 2642-row selection
> split and stops after writing results. No new training, original sweep restart or
> publication. See INVESTIGATION.md for protocol and monitoring/resume commands.
> The stopped-sweep notes below still apply to `accuracy-full/`.

> Latest state: STOPPED FOR INVESTIGATION at user direction after seed-0 results.
> Runner and its own seed-1 training child exited. No automatic restart. See
> `INVESTIGATION.md` for the completed target/source/checkpoint audit and proposed
> development-only follow-up. Both seed-0 models and results are preserved; the original
> protocol stays unchanged. The running/resume notes below are historical.

> Current update: the SDK parity failure was a checker false positive, now fixed.
> `sweep/check_pack.mjs` recomputed choice argmax from four-decimal public probabilities
> instead of using `answer.choice`, which the core selects at full precision. Banking77
> parity row 15 (record 6b28701213ff4348) had cancel_transfer=0.13425982243657128 and
> transfer_not_received_by_recipient=0.13432584945162154. Both display as 0.1343, so
> recomputing argmax selected the first key; the SDK correctly returned the second key.
> The checker now compares the explicit returned choice, preserving zero-label-flips and
> the 0.001 probability tolerance. Three Node regression tests pass. Actual 80-prompt
> model rerun passes with zero flips, max probability difference 0.00035968667908942435.
> The original failing diagnostic remains in baseline-s0-sdk-rca.json. Misleading error
> reporting now includes both flip count and maximum probability difference. No model,
> training data, calibration, SDK behavior or frozen policy was changed. User's continue
> authorization remains in force; the runner will skip all verified completed stages.

# Handoff, 2026-10-21: SDK checker RCA resolved; resume authorized

**Current state: resumed on explicit user request.** The user revoked the pause and
asked to continue. A new detached runner was launched with the original configuration;
`runner.pid` and `resume-request.json` record the new launch. Other sessions remain
untouched. The paused state described below is historical.

At approximately
12:13 MDT the user requested a handoff and stopped GPU work because they need the machine.
The current session's verified runner and its own capture child were terminated with SIGTERM;
both exited. Other sessions and processes were left untouched. No automatic restart or
scheduler was installed. The subsequent explicit resume request restores authorization for this run.

## Saved state and results

- Run directory: `sweep/out/accuracy-full/`. `status.json` tracks the resumed run;
  the prior state was `paused_by_user`.
  `pause-snapshot.json` preserves the pre-stop status. Saved PIDs are historical and must
  never be signaled without fresh identity verification.
- Baseline seed 0 and distilled seed 0 both completed all 5,366 training steps, adapter
  export, fusion, GGUF conversion, imatrix quantization, conversion checks and calibration.
  Models: `baseline-s0-Q4_K_M.gguf` and `distilled-s0-Q4_K_M.gguf` in the run directory.
  Histories and every checkpoint are under `training/`; adapters under `adapters/`.
- Baseline reproduction passed on 4,400 historical benchmark rows: macro accuracy
  **0.7118181818** vs reference **0.7134090909**; Brier **0.3309798733** vs
  **0.3312996024**. See `baseline-s0-reproduction.json`.
- Both 80-prompt conversion checks passed with 0.9625 label agreement. These checks
  establish conversion fidelity; they do not establish an accuracy gain.
- Teacher calibration, dev gating, candidate captures and distilled targets are saved.
  Baseline certificate capture and canonicalization completed (3,371 decisions).
- Interrupted stage: **`baseline-s0-confirmation`**, the 4,397-row fresh confirmation
  capture. It has no completed-stage marker and no output capture file at stop time.
  It will restart from the beginning on resume. Completed training does not need repeating
  when command/input/output hashes still match.
- No full matched-seed `results.json` or `decision.json` exists yet. Distilled confirmation,
  both models' remaining regression/OOD/Open-Jev suites, certificate analysis and actual
  SDK pack checks remain. Seeds 1 and 2 have not started. No pack was promoted/published.
- Final small 300-row heldout monitoring accuracy was 0.69 baseline and 0.6667 distilled;
  Brier 0.32798 and 0.33000 respectively. These are monitoring figures, not the independent
  confirmation comparison. No confirmed distillation gain can yet be claimed.

## Resume only when requested

From `/Users/ZKeown/Code/lab/lars-sdk`, run the same configuration (Metal access requires
execution outside the restricted sandbox):

```sh
.venv-train/bin/python -m sweep.full_accuracy \
  --model /Users/ZKeown/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc \
  --teacher /Users/ZKeown/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --out sweep/out/accuracy-full
```

Use an append-only `runner.log` if detaching and update `runner.pid` with the new runner PID.
The runner verifies completed-stage commands and input/output hashes before skipping them;
keep code, protocol and dataset inputs unchanged to preserve valid expensive stages.
Review logs if any unexpected training rebuild is requested. The file lock is released
when the old runner exits. No cleanup of saved artifacts is needed.

The next meaningful checkpoint is the complete seed-0 comparison, roughly 25–40 minutes
of active evaluation from the interruption point at the observed throughput. This is
not a programmed pause: the runner then continues seeds 1/2 unless a gate fails. If the
user requests stopping after seed 0, implement that explicitly before launching. Preserve
the frozen promotion policy and do not weaken gates to accommodate results.

All implementation changes remain uncommitted in the working tree, including new sweep
workers and training tests. Seventeen tests and the full tiny smoke workflow passed before
the full run; the pause changed documentation/run status only. See the earlier handoff
below and `training/README.md` for architecture, commands and the completed pilot.

## Earlier handoff, 2026-10-20: full-corpus confirmation launched

The user explicitly authorized the next full-corpus baseline/distillation comparison with
"Make it so" after the pilot. GPU authorization continues. Never interfere with any other
session or process. The new runner controls only its own sequential subprocesses.

- **Active run:** `sweep/out/accuracy-full/`, detached with `runner.log`, `runner.pid`,
  and `status.json`. Verify PID identity before any process action; old saved PIDs can be reused
  by the OS. The command is `.venv-train/bin/python -m sweep.full_accuracy --model <2B snapshot>
  --teacher <9B snapshot> --out sweep/out/accuracy-full`; exact paths are in `config.json`.
  No scheduler, publication, or pack replacement was authorized or performed.
- **Reference established:** restored `models/reference-2b/packs/lars-2b-dev/0.0.1` from
  the existing model backup. Rust SDK evaluation reproduces 0.7134090909 accuracy exactly
  and 0.3312996024 Brier (recorded 0.3312935317). `reference-parity.json` records the check.
- **Prepared:** 39,846 training records, one full epoch, final checkpoint, seeds 0/1/2,
  4,096-token batches. The historical 700-record tasksource budget is fixed. All distillation
  runs retain the same population/order, changing only qualified targets. Source eligibility
  compares the separately calibrated teacher with the calibrated full baseline on dev rows.
- **Independent evaluation:** 1,836 calibration labels, 3,371 certificate labels, 4,397 fresh
  jev-bench confirmation rows, 1,000 fresh Open-Jev OOD and 1,000 fresh Open-Jev test rows.
  Confirmation excludes all previously examined pilot states. The old 2,000-row typed-decisions
  test is a regression check, not fresh evidence. All role intersections were checked; hard
  labels are retained independently of soft targets. Every evaluation prompt fits 8,192 tokens.
- **Predeclared gates:** `protocol.json` freezes reproduction, conversion, serving, and promotion
  criteria. Reproduction must be within 1 accuracy point and +0.015 Brier of the reference.
  A failed gate writes `stopped_at_gate` and stops further GPU spending. Promotion requires
  useful gains across seeds, no material accuracy/Brier regressions, and retained certified
  coverage with each seed's accepted-error upper bound at most 5%. No automatic promotion.
- **Tools restored:** `target/release/lars-dev`, `target/llama-tools/bin/llama-{quantize,imatrix}`,
  and the isolated conversion venv `sweep/.venv-convert`. No vendored source was edited.
- **Verified:** seventeen tests pass. The full smoke workflow `sweep/out/accuracy-full-smoke`
  completed training, fusion, norm repair, conversion, imatrix quantization, independent
  calibration, evaluation, certification, and loading both experimental packs through the
  actual TypeScript SDK. Serving preserved every tested label within the declared 0.001
  probability tolerance (observed differences below 0.0001). Tiny smoke results cannot qualify
  for promotion. Commands and detailed stage hashes are saved for resumption.
- **Results:** pilot breakdown by source, primitive and score level is `pilot-breakdown.json`.
  OOD categorical accuracy regressed as confidence decreased; this is not just overconfidence.
  Full `results.json`/`decision.json` appear after matched seed pairs, with eligibility disabled
  until all three seeds finish. Update RESULTS.md when the run completes or stops at a gate.

## Earlier handoff, 2026-10-20: accuracy pilot complete

The user authorized the accuracy experiments and GPU use. Other sessions and their processes
must remain untouched. Only this session's pilot runner and training child were stopped when
a measured feature-readout confound needed correction. Do not reuse their process IDs.

- **Completed run:** `sweep/out/accuracy-pilot/`, finished 14:37 MDT, three seeds, 4,000 training decisions, 250
  steps per recipe. `status.json` is authoritative for the current stage/state; stage logs
  retain failures. The runner is `sweep/accuracy_ladder.py`; its exact arguments are in
  `sweep/out/accuracy-pilot/config.json`. Re-running the same command resumes verified
  completed stages; incomplete training restarts. No scheduler was installed.
  `runner.log` retains its output. The saved `runner.pid` is historical; do not signal it.
- **Implemented:** ordinal ranked-probability loss, conservative targeted distillation,
  shared pointer-head extraction/training/serving comparison, state-disjoint data partitions,
  paired comparisons with matching seeds, and fresh test/OOD evaluation after recipe selection.
  Details and commands: `training/README.md`, "Accuracy experiments".
- **Environment restored:** `.venv-train`, benchmark and corpus data in `data/`, typed-decisions
  in `sweep/out/typed-decisions`. Base weights are read-only cached Qwen3.5-2B and Qwen3.5-9B
  snapshots under `~/.cache/huggingface/hub/`; the output config records their exact paths.
  Python/package versions, corpus hashes, and dataset revisions are recorded with the run.
- **Verified:** eleven tests pass. The Python feature path previously reconstructed logits in
  float32 and could disagree with ordinary BF16 inference. It now uses the model's own
  projection; `sweep/out/feature-readout-parity.json` records 80/80 real prompts with zero
  probability differences. `training/check_features.py` reruns this check on another model.
- **Results:** all reports are complete and summarized in RESULTS.md. Distillation was selected
  on validation. Across three seeds, jev-bench test macro accuracy improves 0.5910 -> 0.6021,
  exact score accuracy 0.4281 -> 0.4731. typed-decisions is flat (0.5720 -> 0.5708).
  OOD accuracy is 0.6027 -> 0.6070, but Brier worsens 0.5104 -> 0.5432 and gains vary by seed.
  Ordinal weight 1 regresses on every seed; the shared scorer ties validation accuracy while
  improving Brier. No broad win or new shipping recipe is established.
- **Limits:** this pilot reports uncalibrated MLX results. It does not reproduce the old
  full-corpus pack or certify thresholds. A useful recipe needs full-corpus confirmation, independent calibration,
  GGUF conversion, and evaluation through the SDK before promotion.

## Earlier handoff, 2026-10-12 08:00: the 8B moves to a bigger machine

## Where things stand

- **Rebuilt on this Mac** (`models/packs/`, gitignored), backed up on 2026-10-12 to the private
  model repo `aether-models/lars-packs` on Hugging Face, under `packs/<name>/<version>/` (sizes
  checked against the local files; `lars-0.6b-dev-step2500` too):
  - `lars-0.6b-dev` 0.0.2: 0.637 macro accuracy, Brier 0.418 (lost pack: 0.650 / 0.419).
  - `lars-2b-dev` 0.0.1: 0.713, Brier 0.331 (lost: 0.716 / 0.333); typed-decisions 0.757
    (lost: 0.759). Build check MLX 0.8375, GGUF 0.85.
- **Not rebuilt: `lars-8b-dev` 0.0.2** (Qwen3-8B, Q3_K_M, was 0.748). This 64 GB Mac can't train
  it at the lost machine's speed: BF16 8B LoRA peaks at 63.4 GB here and swaps.
  - 2026-10-10 night: 460 of 5,023 steps in 10 h (1.2 min per step; another app held 6.6 GB).
  - 2026-10-11 night: 1,500 steps in 9.3 h (0.37 min per step), about 31 h for the epoch. Stopped
    at 07:48 on 2026-10-12 to free the GPU.
  - Its checkpoints at steps 500, 1,000 and 1,500 are in
    `checkpoints/lora-qwen3-8b-sdk-etd-s0.steps` on this Mac. Step 1,500: selection macro 0.777
    (untrained 0.653), held-out 0.740, Brier 0.299. Not a jev-bench number. Training can't resume
    from it; it would only serve as a fallback pack.
- Nothing is running, and the 22:00 LaunchAgent (`~/Library/LaunchAgents/dev.lars-sdk.rebuild.plist`)
  unloaded itself after its last run. It is still on disk; `launchctl bootstrap gui/$(id -u)` with
  that path would arm it again.

## Running the 8B on the bigger machine

The lost machine trained the 8B in about 9 h (it held a 9B at an 89 GB peak), so plan for 96 GB or
more. Everything below is gitignored, so a fresh clone needs it rebuilt first:

1. Tools: llama.cpp at its pinned commit, `cargo build --release` for `lars-dev`, the llama tools
   in `target/llama-tools/bin`, `sh training/setup.sh` (`.venv-train`), `sweep/.venv-convert`,
   `fixtures/.venv` (with pyarrow and scipy).
2. Data: `tools/fetch_jevbench.py` (`data/evals`), the corpus (`training/README.md`, "The corpus"),
   `python sweep/typed_decisions.py --convert` then `--records` (`sweep/out/typed-decisions`).
3. Base model into the repo-local cache (`training/README.md`, "Base models"): `Qwen/Qwen3-8B`.
4. `SMOKE=1 sh sweep/rebuild.sh`, then `nohup sh sweep/rebuild.sh > sweep/out/rebuild.log 2>&1 &`.
   On a machine without the 2B pack, it trains the Qwen3.5-2B first (about 7 h); comment out that
   `one` line to run the 8B alone, then copy the resulting `models/packs/lars-8b-dev/0.0.2` and
   `sweep/out/heads/lars-8b-dev` back here.
5. `LARS_MLX_CACHE_GB=16` in `rebuild.sh` is the MLX buffer cache; lower it on a tight machine.

Then: `sweep/release_packs.py` stages the 0.1.0 packs again (from `models/packs/<name>/<version>`),
the RC's macOS packages need rebuilding, and the new pack goes to `aether-models/lars-packs`
beside the others:

    .venv-train/bin/hf upload aether-models/lars-packs models/packs/lars-8b-dev packs/lars-8b-dev

Add its row to that repo's README.

## Changed since 2026-10-08

- Base models live in `.hf/hub` inside lars-sdk (gitignored; `HF_HUB_CACHE` in `rebuild.sh` and
  `retrain06.sh`). The shared `~/.cache/huggingface` was pruned on 2026-10-09, which failed that
  night's 8B run. Only Qwen3-8B is in `.hf/hub` here; the 0.6B and 2B bases would need fetching
  to retrain them.
- `rebuild.sh` skips a pack that is already packed with typed-decisions captured.

## What happened on 2026-10-08

lars-sdk had been copied from a computer that died; everything gitignored was gone: the packs,
`dist/`, `models/`, `checkpoints/`, `data/`, `sweep/out/`, the venvs and the llama.cpp checkout.
Nothing was ever published, and no backup holds it (this Mac's Time Machine never had lars-sdk).

**Rebuilt today:**
- **Tools:** llama.cpp at its pinned commit, `lars-dev`, `llama-quantize`, `llama-imatrix`,
  `.venv-train` (now with `datasets`), `sweep/.venv-convert`, `fixtures/.venv` (with pyarrow).
- **Eval data:** `data/evals` (8,520 rows) and typed-decisions (`sweep/out/typed-decisions`).
- **Base models:** Qwen3.5-2B and Qwen3-8B downloaded in full (offline lookups need complete
  snapshots; Qwen3-0.6B's cache entry needed its ref and README).
- **Corpus E, reconstructed** (`training/README.md`, `lars.train.corpus_e`):
  - `build.py` reproduces lars's 2026-09-22 corpus byte for byte (all 11 files).
  - The "soft-target repair": `data.py` only read dict soft labels, so civil_comments (a bare
    P(yes)) and measuring_hate_speech (a list) trained on hard labels. `_soft_target` decodes them
    as the evaluation does.
  - Corpus E = `build.py` defaults, then without jev-bench sst5/yelp5/stsb and tasksource ANLI
    (the release NOTICE's list). Filtering after the caps matches the old runs' step counts:
    4,449 for the 1.7B (2 h 35 min at 2.1 s per step, as logged) and 5,370 for the 2B (logged
    5,279). Not byte-verified.
- **The 0.6B check** (`sweep/retrain06.sh`): 0.625 from the step selection chose (2,500), **0.637**
  (Brier 0.418) from the final step (4,449), against the lost pack's 0.650 (Brier 0.419). Agreed:
  close enough; packs now build from the final step. The lost trainer's selection fixes probably
  explain part of the gap. `models/packs/lars-0.6b-dev/0.0.2` is the step-4,449 pack; the 2,500
  one is in `lars-0.6b-dev-step2500/`.
- **Fixed:** the trainer never checkpointed a last step that fell on `--eval-every`;
  `analyze.py` split helpsteer2's rows on Unicode line separators; `analyze.py` and
  `imatrix_text.py` read lars from the deleted checkout.

**Still pointing at `~/Code/personal/lars`** (not needed by the rebuild): `train.sh`, `train4b.sh`,
`train35.sh`, `train35_4b.sh`, `finish2b.sh`, `moe.sh`, `qwen35.sh`, `night.sh`, `imatrix.sh`,
`moe_profile.py`, `fixtures/generate*.py`.

**Lost for good:** the Qwen3.5-4B checkpoints (`lars-4b35-dev` would need a full retrain), every
retired dev pack, `sweep/out`'s captures and heads data, and the RC's built packages.

## Earlier notes, 2026-10-08 07:45 (before the second loss)


**What happened.** On 2026-10-07 the old `~/Code/personal/lars` checkout was deleted. The user
didn't know it was still in use. I had kept work this project depended on there: lars's
unpushed branch `codex-wip-2026-10-02`, the training venv, the jev-bench eval data, the training
corpus and every LoRA adapter. That was my mistake: all working state now lives in lars-sdk and is
pushed. The SDK, the packs, the release candidate and `sweep/out` were never affected.

**Recovered, inside lars-sdk:**
- `tools/fetch_jevbench.py` rebuilds the eval files into gitignored `data/evals`. All 8,520 rows
  have the row ids of the old captures, so they're exact.
- `training/`: the trainer, lars's `src/` at `ab59a0b` (it survived in `fixtures/.reference/`).
  - **My lost commits** are replayed from the session transcript: the chunked gated delta rule,
    exact again against mlx-lm; the noul-criteria fix; the lora.py hook. `--max-steps` and progress
    lines are added.
  - **Environment:** `sh training/setup.sh` builds `.venv-train` (mlx-lm 0.31.3).
- **Build check:** `lars-dev prompts` plus `training/score_mlx.py` replace lars's MLX evaluator
  (sweep/lora_lib.sh `check`).
- **Qwen3.5-4B:** the trained checkpoints from step 1,500 on are in `checkpoints/`. Steps 500 and
  1,000 were lost, so the run's own selection fell back to the untrained adapter.
  - `models/Qwen3.5-4B-lora-etd-s0-Q4_K_M.gguf` is built from step 4,500 (selection 0.803, unseen
    0.709) and passed the build check: MLX 0.800, GGUF 0.813.
  - **To finish it:** rerun `sh sweep/package35_4b.sh`. It skips what is built, then captures,
    packs `lars-4b35-dev` and runs typed-decisions, about 50 minutes. Then compare it with the 8B
    (0.748, 19% certified).

**Not recovered:**
- **The training corpus** (`data/train-e-soft`) and the scripts that built it. Codex's soft-target
  repair went with them. It can be rebuilt from the public datasets (Open-Jev, tasksource-jev,
  jev-bench train), but not byte-verified.
- **Codex's other uncommitted trainer work:** selection fixes and an MLX prefix cache. Its
  ordinal loss and `--init-adapter` were off in our recipe.
- **Proof the rebuilt training reproduces the old packs.** Retrain the 0.6B and compare: 0.650
  jev-bench through the imatrix pack, 0.646 plain.
- **Still pointing at the deleted folder:** `sweep/analyze.py`, which imports from
  `fixtures/.reference` and works, and the training scripts (`train*.sh`), whose corpus paths
  wait on the corpus rebuild. `sweep/imatrix_text.py` imports lars from the old path; the
  Qwen3.5 imatrix texts already exist in `models/`.

**Release candidate 0.1.0-rc.1** (CHANGELOG.md) is unchanged and on hold:
- **Packs:** `dist/packs`, staged in the Hub's layout.
- **Packages:** `dist/rc`, macOS only.
- **Blocked:** GitHub Actions billing (Friday at the earliest) for Linux and Windows, and the
  user's sign-off for publishing.

## Earlier notes


## Running (unattended), 2026-10-06

Two launchd jobs, one after the other; each unloads itself. Stop one with
`launchctl bootout gui/$(id -u)/dev.lars-sdk.<name>`.

1. `dev.lars-sdk.train4b` (`sweep/train4b.sh`, log `sweep/out/train4b.log`), started training
   07:45. The 4B trained exactly like the 8B (BF16, corpus E soft + typed-decisions train), then
   imatrix Q4_K_M, a jev-bench capture, pack `lars-4b-etd-dev` and typed-decisions
   (`sweep/out/train4b-results.txt`). It decides the top tier: keep the 4B if it lands within
   about 1 point of accuracy and 3 points of Wilson coverage of the 8B; else the 8B at Q3_K_M
   replaces it. Never both.
2. `dev.lars-sdk.train35` (`sweep/train35.sh`, log `sweep/out/train35.log`) waits for it, then
   trains Qwen3.5-2B the same way (from scratch: the overnight run was stopped at step 2,300 to
   let the 4B go first), converts with `--no-mtp` (mlx_lm drops the MTP block when it fuses), and
   captures. With lars `585ffae` (a blocked inverse in the chunked gated delta rule, 2.9x faster,
   23 GB instead of 64 GB peak) training should take about 2.5 hours instead of 7.

Overlapping GPU jobs costs a lot: the old rule ran at 16.2 s per step next to the 4B training,
against 4.7 s alone. Run one GPU job at a time.

The overnight run (`sweep/archive/overnight.sh`) finished everything for the 8B by 04:37:
- `lars-8b-dev`: jev-bench 0.749 (Q4_K_M, 5.0 GB) and 0.748 (Q3_K_M, 4.1 GB), Brier 0.285,
  Wilson coverage 27.6%.
- The untrained 8B scored 0.627.
- On typed-decisions test (it trained on train): 0.773, KL 0.095, Brier 0.050.
- Per-predicate heads on top: 0.806 at 300 labels.

## Decided today

- **Release candidate 0.1.0-rc.1, 2026-10-07 (CHANGELOG.md):** built and verified on macOS, not
  published, not tagged.
  - **Packs:** `sweep/release_packs.py` stages `lars-2b`, `lars-8b` and `lars-0.6b` 0.1.0 in
    `dist/packs/` in the Hub's layout, with model cards and NOTICEs.
  - **Packages:** `dist/rc/` holds the npm tarballs and the NuGet package, for macOS arm64 and x64.
  - **Blocked:** Linux and Windows wait on GitHub Actions billing; publishing waits on sign-off.
- **Tonight, 2026-10-07:** `dev.lars-sdk.train35-4b` (`sweep/train35_4b.sh`, log
  `sweep/out/train35_4b.log`) started at 11:13 (moved up from 18:00) and trains Qwen3.5-4B like the 2B.
  - **Expected:** about 8-9 h of training (48.6 GB peak in the smoke test), then the build check,
    jev-bench, pack `lars-4b35-dev` and typed-decisions, done around 21:00-22:00.
  - **The question:** does it match the 8B (0.748) at about 2.5 GB? The smoke test passed end to
    end on the real base (check: MLX 0.775, GGUF 0.788).
- **2026-10-07 lineup (agreed):**
  - `lars-2b-dev` 0.0.1: Qwen3.5-2B, the default.
  - `lars-8b-dev` 0.0.2: Q3_K_M, the top tier.
  - `lars-0.6b-dev`: the smallest.
  - `lars-4b-dev`, `lars-1.7b-dev` and `lars-8b-dev` 0.0.1 are retired but left on disk.
  - Built by `sweep/promote.sh`.
  - Next: prefix reuse for Qwen3.5 (snapshot the recurrent state instead of re-reading the input).
- **2026-10-07 results (RESULTS.md):**
  - 4B + td 0.740, 8B + td 0.749, Qwen3.5-2B + td 0.716.
  - Top tier: the 8B (Q3_K_M), by the coverage rule.
  - The 2B is the best small model. Its pack, typed-decisions and latency come from
    `sweep/finish2b.sh` (running from 09:00; results in `sweep/out/finish2b-results.txt` and
    `sweep/out/latency-tiers.md`).
  - Builds now gate on `check` (MLX vs GGUF on 80 rows).
- **2026-10-07, two pipeline bugs fixed:**
  - **4B fuse failed offline.** mlx_lm fetches the base's README for a model card; `fuse` now
    reads the local snapshot. The 4B resumed from its finished adapter at 07:26.
  - **The fused Qwen3.5-2B was broken (0.316, against 0.493 untrained).** mlx_lm adds 1.0 to five
    zero-centered norm tensors on load, the fused save keeps the shift, and the converter adds
    1.0 again. `sweep/unshift_norms.py` now runs after every fuse. The 2B rebuilds from its
    adapter after the 4B finishes. Training was fine: selection 0.43 -> 0.778. Broken outputs
    are in `models/broken` and `sweep/out/broken`.
- **Certification fixes (review from Moe, `~/Code/moe/docs/integrations/lars-sdk-findings.md`):**
  - a failed certification accepts nothing
  - calibration files (format 2) carry their questions
  - loading replaces certificates and is atomic
  - thresholds come from a fixed sequence, valid at about 2.5%
  Coverage drops by about a fifth to a third; RESULTS.md 2026-10-06.
- **Rust SDK:** `moedex-lars` (`rust/`, with `#[derive(Options)]` in `rust/derive`), on the same
  runtime as TypeScript and C#. Molar Triage gives the same answers as TypeScript. `lars` is
  taken on crates.io. `USING.md` is the handoff for other projects.
- **Per-predicate heads are in the SDK** (`calibrate`: pack/fitted → recal → head, each step
  significant on 5-fold CV). typed-decisions with 100 labels: 0.6B 0.44 → 0.66, 4B 0.61 → 0.72.
  In-domain (jev-bench), they add 0.5–1 point and do no net harm. RESULTS.md has the tables.
- **Training on typed-decisions train is approved.** The test split stays held out.
- **The Qwen3.5-9B LoRA is infeasible on this machine for now.** mlx_lm's gated-delta kernel
  has no gradient. lars now has a chunked, differentiable version (`f42781a`, exact against the
  reference loop), but the 9B trains at about 23 s per step at 2,048-token batches, 89 GB peak:
  about 73 hours for one epoch. The 2B is about 6.75 s per step, about 10 hours.
- **Default tier:** still open. Suggestion in the conversation: lead with labels, so the 0.6B or
  1.7B plus `calibrate` is the default, and the 4B is the "no labels" tier.

## Open

1. Morning: read `overnight-results.txt` and `summary.md`; decide whether the 8B becomes a tier
   (rule: at least 2 points over the tier it would replace, Wilson coverage no worse).
2. Rebuild the three packs from the imatrix Q4_K_M files (a small, free gain).
3. If the 8B with typed-decisions helps, retrain the 4B the same way to separate data from size.
4. Phase 6: the published packs need your sign-off before anything goes to the Hub.
5. Quiet-machine latency (never measured), `lars pull`, an API reference, TS cancellation,
   ASan on Linux, a C# enum source generator.

## Things to know

- **lars repo:** local branch `codex-wip-2026-10-02`, not pushed. Today's commits there:
  `3485841` (training records keep noul yes/no criteria) and `f42781a` (chunked gated delta rule).
  Adapters in `checkpoints/`: `lora-{1.7b,0.6b}-sdk-e-s0`, tonight `lora-8b-sdk-etd-s0` and
  `lora-qwen3.5-2b-sdk-etd-s0`.
- **typed-decisions:** `sweep/typed_decisions.py --convert` (eval JSONL), `--records` (training
  records: 1,000 train cases + 200 for checkpoint selection), and scoring. Data in
  `sweep/out/typed-decisions/`.
- **Heads offline:** `sweep/heads_capture.sh` (hidden states through a pack) and `sweep/heads.py`.
  Hidden states for the 4B and 0.6B packs are in `sweep/out/heads/`.
- **`models/`:** large and gitignored. The 30B GGUFs (k128 18.6 GB, k64 9.8 GB, k64-IQ3 7.7 GB)
  and the Qwen3.5 BF16 files can be deleted if space is needed.
- Disabled launchd plists are in `~/Library/LaunchAgents/lars-sdk-disabled/`.
