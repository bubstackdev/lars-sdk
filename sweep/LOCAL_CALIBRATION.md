# Local calibration learning curves

> **Latest, 2026-10-23:** the new source-documented queue pilot is complete. It has
> 42 cases / 41 groups with a frozen25-fit /16-test split. Calibration reaches12/16
> from10/16, matching the constant majority baseline; diagnostic accuracy remains9/11.
> No meaningful gain established. Details and current restart path are at the end.


> **2026-10-23 correction:** the Codegraph judgment-category study and its19-case
> relabeling queue are retired. The local UI now shows an evidence audit with12
> timestamp-supported replay fixtures,63 exclusions and zero requested reviews.
> Its previous4/9 → mean5/9 result must not be used as evidence of adaptation.
> The saved-workflow CPU rehearsal is a separate experiment and is unchanged.

This CPU study measures held-out accuracy, hard/soft Brier scores, log loss,
accepted error and coverage against a fitting-label budget per predicate. It does
not train a backbone, publish a pack, or export a deployable SDK calibration.

The completed saved-workflow rehearsal is in `sweep/out/local-calibration/`.
It reuses raw captures of the same trained 2B Q4_K_M model, plus its pooled pack
calibrator. No GPU inference was needed.

## What the rehearsal measures

There are 20 fixed predicates across agent-trace observability, customer service,
invoice processing and security incidents. Those workflow types were included in
backbone training and earlier evaluations. Distinct input states are held out here;
this is a development rehearsal, not evidence for a new company or pristine final test.

The cached dev and typed-regression populations provide 121–126 states per workflow.
After excluding normalized base-training overlaps, whole input states receive one
fixed role across all their questions: 25 fitting, 75 threshold-selection, and the
remaining 21–26 evaluation states per workflow. Globally that is 100 fitting states,
300 threshold-selection states and 91 evaluation states, giving 2,455 decisions.
No production traffic mix or company labels are inferred from these synthetic benchmarks.

The available learning curve is 0, 10 and 25 fitting labels per predicate. Budgets
50, 100 and 200 are explicitly unsupported with these data. Three deterministic
draws use nested subsets of the same fitting pool; they are stability checks, not
independent evaluations. Evaluation labels never choose a method or threshold.

For each nonzero budget, five-fold cross-validation on fitting labels compares the
pack with a per-predicate refit and option-bias recalibration. A richer method is
kept only when its paired log-loss improvement passes the existing 1.96-SE guard.
Optional hidden-state heads are considered from 50 fitting labels when aligned
hidden vectors exist. These cached captures have no hidden states, so no head curve
was measured. The Python study follows the SDK's fitting/selection principles;
byte-for-byte SDK calibration parity has not been established.

The frozen method predicts the separate threshold-selection population. The SDK's
fixed-sequence Wilson rule chooses an acceptance threshold at 5% error. The final
evaluation measures that frozen threshold's accepted error and coverage, without
retuning it. A failed certificate accepts nothing; accepted error is null, not zero.
Per-predicate Wilson bounds describe the finite evaluation sample and its assumptions;
they are not a company-wide guarantee or a multiplicity-adjusted claim for all predicates.
Fixed confidence-threshold diagnostics are reported separately and must not be mined
for a supposedly certified deployment threshold.

At 5%, at least 73 clean accepted labels are needed even to pass the Wilson rule.
The rehearsal's 75 threshold-selection labels barely exceed that minimum. They
should not be mistaken for a well-powered coverage study. Their label cost is
reported separately: fitting budget + 75 threshold labels per predicate, with
evaluation labels additional. Repeated draws never multiply the effective sample size.

## Run or reproduce

```sh
uv pip install --python .venv-train/bin/python -e './training[calibration-study]'
.venv-train/bin/python -m sweep.local_calibration prepare \
  --out sweep/out/local-calibration-new --draws 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv-train/bin/python -m sweep.local_calibration run \
  --out sweep/out/local-calibration-new
.venv-train/bin/python -m unittest training.tests.test_local_calibration -v
```

Preparation freezes capture, calibrator, fitting-code and source-input hashes.
Runs reject changed frozen inputs and completed output reuse. Partial results are
saved per predicate; automatic interrupted-study resume is not implemented. No
other session's jobs are controlled.

## Bring a real workflow

Pass `prepare --input domain-captures.jsonl --calibration pack-calibration.json`.
Each line contains these fields:

```json
{
  "key": "unique-decision-id",
  "model_id": "one-pinned-model-and-runtime",
  "predicate_id": "company/workflow/question-version",
  "workflow": "company/workflow",
  "role": "fit",
  "state": {"the": "actual model input"},
  "family_id": "related-case-or-incident-group",
  "question": {"type": "choice", "instructions": "Fixed question and criteria"},
  "kind": "choice",
  "keys": ["route_a", "route_b"],
  "logits": [1.2, 0.4],
  "hard_index": 0,
  "target": [1.0, 0.0],
  "gold_status": "reviewed"
}
```

Roles are `fit`, `certificate`, and `test`. Entire states and related case families
must stay in one role across all predicates. Each predicate must keep identical
question/option semantics, with unique states, all three roles, and finite aligned
logits. All rows must describe one model identity. Optional `hidden` arrays must be
present, finite and have the same dimension throughout each predicate.

Representative independent company/domain labels should have `gold_status: reviewed`.
`benchmark` is also accepted for existing benchmark studies. Rule-derived, inferred
or unreviewed labels are rejected. Preserve hard gold separately from any soft
target; accuracy and certificate errors always use `hard_index`. Fit targets are
one-hot hard labels, reflecting a person choosing an answer.

The input format is independent of model inference, so larger fitting budgets can
be studied from one pinned capture. Supply enough separate threshold-selection and
evaluation examples in addition to the desired fitting budget. Sample representative
traffic, including rare failures; do not use a disagreement-enriched teacher sample
to estimate production prevalence or certify a threshold.

## Codegraph case seed

The user authorized Codegraph mining for additional cases. Seven revision-pinned
README/runbook sections were retrieved read-only through Codegraph RAG. Sources,
links, revisions and retrieval snapshots are in
`sweep/out/codegraph-calibration/sources/`. No operational commands embedded in the
documents were executed; no stories, messages, databases or infrastructure were changed.

`sweep.codegraph_calibration_cases` builds 212 synthetic contract probes in 43
families: billing retry (112), HTTP retry (40), payout-result classification (20),
internal-refund prerequisites (16), invoice closure (12), escalation contacts (9),
and domain-return transitions (3). They exercise boundaries such as exhausted
retries, error precedence over a successful status, and strictly more than five
minutes. Expected answers are derived mechanically from constructed facts and the
retrieved rule; they were not generated by Flash.

These are **draft rule-derived labels**, not production cases or independently
reviewed company gold. They remain separate from the saved-workflow rehearsal.
Repeated variants in a case family must stay together when partitioning. The 43
families and seven source sections are a much smaller independent evidence base
than the row count; do not pad with paraphrases to claim larger sample support.
Review the source interpretation, cases and labels before using them for a real
calibration/evaluation study. This runner rejects their unreviewed gold status.

The bundle contains `cases.records.jsonl`, `cases.wire.jsonl`, `evidence.jsonl`,
`cases-manifest.json` and an annotation CSV. Wire/evidence records retain their
draft status and source references. Source material and generated cases remain in
gitignored local outputs; nothing was uploaded or published.

To rebuild after reviewing or changing the generator, use a fresh output directory:

```sh
.venv-train/bin/python -m sweep.codegraph_calibration_cases \
  --sources sweep/out/codegraph-calibration/sources \
  --out sweep/out/codegraph-calibration-new
```


## Completed CPU rehearsal

| Fitting labels per predicate | Accuracy | Hard-label Brier | Soft-target Brier | Accepted coverage at5% |
|---|---:|---:|---:|---:|
|0 |74.76% |0.41956 |0.06672 |0% |
|10 |74.97% |0.39556 |0.12198 |0% |
|25 |75.41% |0.34067 |0.10884 |0% |

The accuracy change at25 labels is+0.65 percentage points and is descriptive, not a
statistically established gain. Hard-label probability quality improves; matching
the benchmark's soft targets worsens. No threshold qualified on the independent
75-label threshold population for any predicate/draw. Accepted error is undefined
because no answers were accepted; this is not evidence of zero deployment error.
Larger budgets remain unsupported. No hidden-head results were measured.

Full per-predicate/draw metrics, CV choices, selected fitting keys, fixed-threshold
diagnostics and sample costs are in `sweep/out/local-calibration/results.json`.
`summary.json` adds both Brier definitions and explicit costs. `validation.json`
records five passing tests, including evaluation-label isolation and an optional
hidden-head signal test. `environment.json` and frozen source snapshots support
reproduction. The completed rehearsal used no GPU or backbone training.

Review Codegraph candidates in `sweep/out/codegraph-calibration/review.csv`; expected
labels are deliberately named draft_label, and reviewed_label/reviewer fields are
blank. The review manifest pins the CSV and wire input hashes. All212 cases remain
unreviewed; none was folded into the benchmark curve or used for certification.

## Source audit and completed 2B baseline

`sweep/codegraph_baseline.py` freezes the audit before inference and checks the
existing trained 2B model, runtime and template identities. The completed run is
in `sweep/out/codegraph-baseline/`; `results.json` reports scores and
`error-analysis.json` contains confusion counts and individual failures.

| Audit tier | Correct / cases | Accuracy |
|---|---:|---:|
| Implementation corroborated | 95 / 136 | 69.9% |
| Documentation only | 16 / 29 | 55.2% |
| Quarantined policy probes | 32 / 47 | 68.1% |

These remain synthetic, unreviewed labels. Corroboration used revision-pinned
implementation and test source; the remote C# tests were not executed. All prompts
include their source policy, so following a conflicting policy is a distinct task
from predicting actual service behavior. Do not silently relabel such prompts.

The quarantine contains 40 FinancialReporting HTTP retry cases (wiki and code
disagree about attempts, delays and permanent errors), four FABAL withdrawal cases
(README excludes refunds but the inspected implementation supports them), and
three DomainOwner cases missing runtime guards. The 29 documentation-only cases
cover payout-result classification and escalation contacts.

Billing retries account for 38 of the 41 implementation-corroborated errors:
74/112 correct, with all three delayed-redelivery cases and 35 fault cases
predicted as immediate retry. Those 112 rows come from only seven families.
Invoice closure is 12/12; non-withdrawal refund prerequisites are 9/12. On the
documentation-only probes, all four short MySQL outages incorrectly trigger the
strictly-more-than-five-minutes escalation, and seven of ten error-bearing payout
cases miss error precedence. These findings identify failure modes; they do not
establish that confidence calibration can repair the decisions. The observed
per-source majority reference is 70.6% on the corroborated tier, so 69.9% does not
demonstrate useful aggregate performance over that descriptive reference.

The eight-row hidden-output control changed probabilities by up to 0.00395974,
exceeding the frozen 0.001 parity gate despite zero answer flips. The planned
logits-only fallback matched exactly and was used for all 212 cases. No usable
212-row hidden-feature capture exists. All row identities were checked against
the original inputs. The run took about 24 seconds; its workers exited. Three
targeted tests cover audit tiers, serving parity and control of only the runner's
own worker under memory pressure. No training, fitting or certification ran.

Next, gather independent billing-retry case families and review their source
interpretations before freezing disjoint fitting, certification and evaluation
sets. Keep exact retry limits and time thresholds enforced in application code.
Use any subsequent learning curve to test adaptation on remaining judgment tasks,
with raw accuracy and accepted-error/coverage reported separately. Do not fit to
this inspected evaluation cohort and report the same cohort as held out.

## Historical judgment-task seed (2026-10-22)

The next authorized step harvested 67 historical Shortcut story snapshots through
Codegraph, plus four revision-pinned contextual runbooks. Five discovery searches
covered payment, transfer, cart, WHOIS and DNS bugs. This was a targeted first-page
harvest, not an exhaustive or representative sample of accessible company work.
Retrieved runbook commands and remediation instructions were treated as evidence;
none were executed. Credential-like XML/JSON values were redacted from local story
snapshots, and one body containing an authorization code was omitted. Raw snapshots
can still contain customer details; they remain gitignored and local. Curated model
inputs omit customer identities, actual domains, credentials and source URLs.

`sweep/codegraph_judgment.py` builds the active frozen bundle at
`sweep/out/codegraph-judgment/bundle-v3/`. It contains **44 cases in 34 proposed
dependence groups**, asking which investigation track the available observations
support: application engineering, operational state recovery, configuration,
infrastructure/upstream, intended policy behavior, or further evidence collection.
Unlike the earlier synthetic rule grid, each case reconstructs a diagnostic
checkpoint from an actual historical ticket. It is not a verified prospective
intake: descriptions can be edited after resolution and investigative facts may
already narrow the answer. The intended task is choosing a track after those
observations are available, not predicting an unseen outcome from an original title.

The labels remain **source-interpreted drafts**, with exact source excerpts,
revisions/timestamps, uncertainty notes and detected AI-assisted evidence flags.
A closed ticket, an automated triage report, a workaround, or a successful retry
does not become independently reviewed company gold. Some source explanations
were superseded or contradicted; examples retain those caveats for adjudication.
The source-excerpt checks verify provenance, not semantic correctness of our labels.
Twenty-three harvested stories were not curated into this seed.

Family assignment combines conservative mechanism groups with formal story links,
including shared external ticket stubs. This joins a transfer-authorization report
to the cart investigation chain and reduces the initial 35 groups to 34. Every
connected group gets one deterministic, label-independent role:

| Role | Cases | Proposed dependence groups |
|---|---:|---:|
| Fitting | 13 | 10 |
| Certification | 19 | 14 |
| Evaluation | 12 | 10 |

`inputs.jsonl` contains only opaque keys, states and the fixed question. Draft labels,
provenance and review notes stay in separate artifacts. `review-blind.csv` contains
empty reviewer/label fields and no proposed answers. Copy it outside the frozen
bundle before annotating, then consult `evidence.jsonl` and source snapshots for
adjudication. Reject cases whose chronology or intended track cannot be supported.
Review group membership before any model scoring, and freeze reviewed material as
a new version. Versions 1 and 2 are superseded preparation artifacts; no predictions
were produced for them.

`study-plan.json` fixes the subsequent comparison: current 2B raw/pack-calibrated
predictions versus fit-only guarded adaptation, initially at 0/10 labels if review
leaves sufficient independent support. Preserve held-out groups, use fitting labels
only for method selection, report hard accuracy/Brier and correction/regression
counts, and resample at group level. Larger budgets are unsupported. Ten evaluation
groups make this exploratory. Fourteen certification groups cannot establish the
existing 5% accepted-error gate (at least 73 all-correct independent cases are
required even before considering the practical need for more data).

There are also class-coverage gaps: infrastructure/upstream is absent from fitting
and evaluation; configuration is absent from certification. Expand independent
cases without moving existing groups to improve outcomes. No calibration or
training has started, and no GPU was used for this preparation. The existing
learning-curve validator still rejects these unreviewed labels. Seven relevant
tests pass, covering input/outcome separation, duplicate rejection, source anchors,
linked-family isolation, frozen-file changes, serving parity and own-worker control.

```sh
.venv-train/bin/python -m sweep.codegraph_judgment verify \
  --out sweep/out/codegraph-judgment/bundle-v3
```

## Confidence audit and local review UI (2026-10-22)

The user authorized completing autonomous work and queuing cases below **0.8** for
human correction. The expansion retrieved 36 more ticket snapshots (103 total),
curated 31 additional cases, and audited all **75 cases in 64 dependence groups**.
Active corpus: `sweep/out/codegraph-judgment/bundle-v4/`. All original 44 inputs and
roles were retained. Raw splits are fitting34/31groups, certificate20/15groups,
evaluation21/18groups. All classes occur in fit/test; configuration is still absent
from the certificate population. No class-balanced or representative sampling claim.

Per-case assessments and reasons are in
`expansion-20261022/assessments.json`. **56 cases meet the confidence threshold;
19 fall below it.** Confidence is an uncalibrated agent judgment about whether the
proposed investigation track follows the reconstructed facts. It is not a measured
probability, model output, or independent human adjudication. The audit inspected
complete ticket narratives and later comments. Five cases also have pinned code or
change evidence: wallet rendering, privacy policy conditions, transfer-lock behavior,
and the payment-group refund path (two repositories). C# test assertions were read,
not executed. The privacy case stays uncertain because the applicable account/contact
conditions are missing; a conditional code path alone does not prove intended behavior.

The local review app is `sweep/judgment_review.py` with buildless assets in
`sweep/judgment_review_ui/`. Start it with:

```sh
.venv-train/bin/python -m sweep.judgment_review \
  --data sweep/out/codegraph-judgment/review-20261022/queue.json \
  --db sweep/out/codegraph-judgment/review-20261022/reviews.sqlite3 \
  --port 8765
```

Open `http://127.0.0.1:8765`. The default queue is the 19 cases below0.8, ordered by
confidence; all cases remain inspectable. Each case shows observations, uncertainty,
source excerpts/links, optional proposed label, and family details. Select a track,
check the facts/decision point, and save. Labels and facts can be corrected; exclusion,
deferral, reopening and family concerns are supported. Correction/exclusion reasons
are required. Unsaved changes are protected during navigation. The app marks high-
confidence cases agent-assessed, separately from human decisions.

Corrections persist in SQLite with append-only revision history. Concurrent stale
tabs receive a conflict instead of overwriting another decision. The database is
bound to the exact queue snapshot. Original sources and frozen cases are untouched.
Family concerns and duplicate edited states hold cases out of reviewed exports.
The app binds to loopback, uses no CDN or remote storage, renders retrieved text as
text, and requires same-origin/token JSON writes. Source material never grants
permission to run operational commands. `Export reviews` downloads a snapshot.
An equivalent offline export, using a fresh output path, is:

```sh
.venv-train/bin/python -m sweep.judgment_review \
  --data sweep/out/codegraph-judgment/review-20261022/queue.json \
  --db sweep/out/codegraph-judgment/review-20261022/reviews.sqlite3 \
  --export sweep/out/codegraph-judgment/review-snapshot-new.json
```

`sweep/prepare_judgment_review.py` freezes the queue and the standalone
`review-20261022/needs-review.{csv,json}` list. Eighteen targeted Python tests and
eleven actual Chrome checks pass. Browser mutations used an isolated fixture DB and
temporary browser profile; production reviews were not created by tests. Checks cover
durability, correction history, 0.8 boundary, stale tabs, unsafe source markup,
unsaved edits, export gates, and responsive desktop/mobile layouts. The QA browser
and fixture server exited. The user-facing review server is intentionally left running.

### Cached outputs and completed provisional comparison

`sweep/judgment_capture.py` cached all75 model outputs in `capture-v4/`. Eight SDK
controls matched exactly; native row IDs and option sets were checked for all75.
Runtime was about8seconds, logits-only. No hidden extraction, backbone training or
certification ran. The GPU worker exited. Predictions are not exposed in the review
UI, preserving the reviewer's separation from model performance.

Under the user's confidence rule, a **separate exploratory** CPU runner,
`sweep/judgment_probe.py`, used cases meeting0.8. It selected one example per family
without inspecting labels/logits:23 fitting families,14 certificate families,9 held-
out evaluation families. Five-fold method selection uses only fitting labels. Three
nested10-label draws share the same evaluation population and are not independent.
The standard human-reviewed learning-curve validator remains unchanged.

| Method | Correct on nine evaluation groups | Hard Brier |
|---|---:|---:|
| Raw 2B |4/9 |0.68014 |
| Pack calibration, zero local labels |4/9 |0.70984 |
| Guarded local adaptation,10 labels |6/9,4/9,5/9 across draws |0.58208 mean |

The adaptation mean is5/9 versus4/9: an average of one additional correct answer,
not an established gain. Corrections/regressions are2/0,2/2,3/2 respectively. Larger
budgets are unsupported after the confidence/group filters (23 fitting families).
Current labels are agent-assessed, and the tiny selected evaluation does not establish
company accuracy or a deployment guarantee. No certificate or deployable calibration
was produced. Active artifacts are `probe-v4-final/{plan,results}.json`, selected
captures and frozen driver; `probe-v4/` is a superseded equivalent preparation run.

A future probe can take the UI export as `--queue`: it applies explicit human label
corrections, omits excluded/deferred cases and family concerns, and keeps untouched
high-confidence cases agent-assessed with separate provenance. Edited facts require
recapture; stale logits cannot be reused. For a strictly reviewed-only study,
`sweep.judgment_capture join` aligns exported reviewed labels to cached native option
order and flags changed inputs for recapture. Neither workflow silently promotes an
agent judgment to human-reviewed gold. Freeze each corrected comparison in a new
directory before evaluating it.

## Corrected source-outcome replay audit (2026-10-23)

The user challenged the need to re-label already resolved bugs and authorized
rebuilding the benchmark from the recorded outcomes. The previous inputs sometimes
included later diagnostic statements, while the six investigation tracks were our
own categories. That created review work unrelated to validating the original
resolutions. The judgment-category comparison is now explicitly retired in
`sweep/out/codegraph-judgment/study-retirement.json`; the probe CLI refuses that
question signature. Prior results and captures remain preserved as retired artifacts.

`sweep/resolution_replay.py` rebuilt the evidence using only exact earlier comment
excerpts, with separate exact outcome excerpts. Current descriptions, curator summaries,
case titles and outcome text never enter model observations. Input comment creation
and **last edit** must precede an explicit finding/correction cutoff; the cutoff can
come from an earlier related ticket. Outcomes must be recorded at or after that
cutoff. Exact substring offsets, comment timestamps, source links and hashes are kept.
Outcome changes cannot alter feature construction. A comment created early but edited
after diagnosis is rejected. This establishes the selected chronology from the saved
metadata, not independent completeness of the historical record.

Active bundle: `sweep/out/codegraph-judgment/resolution-replay-v2/`. Five retained
fixtures in five existing groups cover orphaned catalog cart items, NB1/NB2 transfer
security scope, missing payment-group refunds, resilient domain push transport, and
partial-transfer refund reconciliation. Four use pre-finding observations; one uses
pre-resolution observations and does not claim a confirmed cause for the original
refund delay. Earlier comments are not represented as original ticket intake.

Seventy of the75 prior cases were excluded:36 lack a verified pre-finding version,
20 have an unresolved/unconfirmed outcome,5 have an ambiguous resolution/policy,
4 already expose diagnostic interpretation or lack a sufficiently specific fix,
3 have insufficient earlier observations,1 is a linked duplicate, and1 pairs an
earlier outage with a later **different incident** in the same ticket. The last case
shows why comment ordering alone does not establish that an outcome labels an event.
The audit is not an assertion that all excluded bug fixes were wrong or unrecorded.

Live Codegraph/history tools were not exposed this turn, and plugin discovery returned
no matching connector. Original description versions could therefore not be retrieved.
The saved snippets were used without pretending that current descriptions were original
reports. Suitable version history could recover further cases later without user
re-labeling. Sparse earlier observations can also fit multiple causes; these are replay
fixtures, not proof that the later outcome was uniquely inferable at intake.

No fitting, certification or model scoring ran on the corrected set. Five groups cannot
support a meaningful local-adaptation curve or accepted-error bound. The prior roles are
retained for provenance only (2fit,2certificate,1test), with no usable study implied.
Expected outputs carry `documented_outcome`, not a fabricated human-review status. They
are taken directly from source findings/resolutions rather than the retired taxonomy.

The page at `http://127.0.0.1:8765` now displays the retained evidence timelines and all
exclusion reasons. **No manual review is requested.** Prior corrections remain at
`/review` as a read-only archive; the write endpoint returns410 in audit mode. The original
SQLite database is retained, with a consistent SQLite backup in the new bundle. Start
the corrected app with the existing queue/database plus the audit report:

```sh
.venv-train/bin/python -m sweep.judgment_review \
  --data sweep/out/codegraph-judgment/review-20261022/queue.json \
  --db sweep/out/codegraph-judgment/review-20261022/reviews.sqlite3 \
  --benchmark sweep/out/codegraph-judgment/resolution-replay-v2/report.json \
  --port 8765
```

The corrected bundle contains `inputs.jsonl`, `expected.jsonl`, `cases.jsonl`,
`audit.jsonl`, `report.json`, hashes and a frozen builder. Twenty-five targeted Python
checks pass. Seven read-only Chrome checks cover fixture timelines, all70 exclusions,
search, zero review requests, the read-only archive, mobile layout and browser errors.
QA used a separate browser profile and did not modify the review database. Only the
conversation's recorded review-server process was replaced; other sessions were untouched.


## 2026-10-23: original ticket histories recovered

Live Codegraph access is restored. The active evidence bundle is now
`sweep/out/codegraph-judgment/resolution-replay-v3/`: **11 fixtures in 11 groups,
64 exclusions, zero requested reviews**. Six new cases use exact excerpts from
recorded story creation events, rather than today's edited description. Those
creation events precede the selected finding/correction cutoff. Later comment
excerpts supply the documented repair or outcome. The added cases cover partial
account creation, cart recovery, missing sale proceeds, production configuration,
WHOIS recovery, and custom-nameserver deletion.

`mining-20261023/history/` pins 73 ticket histories. `selection.json` records all
history hashes and the six new source selections. Seven candidate current story
records were also fetched; their comment text matches the prior pinned snapshots.
The seventh candidate (141835, transfer-lock race) is outside the frozen cohort
and remains separate in `additional-candidate.json` until source/group admission.
32 prior version-history exclusions have recovered history but still need an
observation/outcome admission audit; no manual review task was created.

The builder never reads the current description for model features. It permits
only a creation event for original-description evidence, verifies exact substrings,
checks its timestamp against the finding cutoff, and pins the history file hash.
Earlier descriptions can already contain a diagnosis: creation history alone does
not qualify a case. The prior accuracy comparison remains retired. These 11
fixtures are still too few and too sparsely replicated for a calibration study.

The existing local audit now serves v3. For a later restart, use the previous
server command with `--benchmark` pointing to `resolution-replay-v3/report.json`.
The original review database remains read-only in the UI. No GPU work was done.


## 2026-10-23: remaining evidence audit and workflow feasibility

All 32 previously pending histories have a recorded decision in
`workflow-audit-20261023/admission-audit.jsonl`. One qualifies (142001); 31 remain
excluded with specific reasons. Fourteen were already diagnosed when created;
others record proposals, vague changes, unverified recovery, a rollback, or an
unresolved upstream incident. There is no remaining history-review queue.

Active `resolution-replay-v4` has **12 fixtures / 12 groups and 63 exclusions**.
The added case uses the original proxy activity alert and a separately timestamped
edit documenting the obsolete DNS settings, correction, and restored service.
The builder accepts description revisions as outcomes or cutoffs; historical
inputs still accept only creation events. Current descriptions never become
features. The nine chronology tests include late-revision isolation and cutoff
checks. In total, 26 relevant Python tests and seven read-only Chrome checks pass.

First-page resolved-bug searches covered cart, refund, and nameserver titles.
Nameserver-change support was selected for deeper inspection: 16 full stories,
ten histories, and a linked repair record were saved. The frozen workflow corpus
has **four fixtures in four groups**: successful domain sync, missing reseller
contact repair, deployment/replay of the `.is` nameserver fix, and the existing
custom-nameserver deletion repair. Each has a different corrective action.
Multiple domains and linked ticket 137059 remain one group with 136973. The three
new cases are not inserted into the old 75-case cohort or assigned fitting roles.

No held-out split or comparison was created: no repeated action currently has
independent fitting and evaluation examples. `workflow-report.json` records the
feasibility result; `study-plan.json` freezes the prospective decision and gates
before scoring. Verbatim source outcomes remain labels; the report's human-readable
action descriptions are not model classes. This search was not exhaustive.
Further sparse symptom tickets alone may not help. A useful next collection needs
the diagnostic facts available when the action was chosen and repeated independent
successful outcomes for that action, with credible alternative-action cases.

The UI serves `resolution-replay-v4/report.json`; the previous restart command works
with that benchmark path. No GPU, capture, fitting, model scoring, or training was
started. The prior correction archive remains read-only and all retired metrics
remain retired.

## 2026-10-23: source-documented queue pilot completed

Continued Codegraph mining retrieved 361 full story records and histories. The
admitted cohort has 42 cases in 41 dependence groups: 29 diagnostic checkpoints
and 13 early-alert/symptom-only observations. Remaining source records were not
admitted; retrieval is not equivalent to complete semantic adjudication.

The user's timing requirement is explicit: observations must be recorded before
the action. `sweep/queue_evidence.py` verifies exact source substrings and event
chronology, refuses same-note retrospective findings and late comment edits, and
handles earlier stated action dates conservatively. Original creation titles and
historical description revisions are eligible evidence; current versions are not.
Actual execution logs are not independently available for every old action, so
recorded chronology must not be described as independently proven execution time.

The new target is the final **documented queue mechanism**: consumer redelivery
(after any required repair, including expected no-ops), versus removal without
further delivery. These are literal action records, not hypothetical best actions
or the retired investigation-track labels. Two ambiguous records saying only
'cleared' were excluded instead of being inferred as purges. Human adjudication is
not claimed. Sparse inputs remain a separate stratum: predicting a recurring
alert's base rate does not measure diagnosis.

The frozen `queue-pilot-20261023/cohort-v3/` supplies one representative per group:
25 fitting cases (20 reprocess, five discard), and 16 held-out cases (12 reprocess,
four discard), with 11 diagnostic and five sparse held-out cases. Formal links
through all saved intermediate stories, shared incidents/fixes, and a conservative
same-day outage grouping remain within one role. Identical sparse observations in
different real incidents are tracked in `input-collisions.json`; they are not
paraphrased synthetic examples. Their native logits were verified identical.

Before scoring, `study-plan.json` froze the split, budgets0/10/25, three nested
fitting draws, five-fold fitting-only CV, proper log loss, affine binary
calibration with ridge toward the pack, class-prior baseline, and paired
family-bootstrap intervals. There is no risk-certification set or threshold.
The standard reviewed-only runner and previous retirement guard remain unchanged;
`sweep/queue_calibration.py` is a separate source-action pilot with explicit
provenance and independent-adjudication limitations.

| Fitting labels | Correct on the same16 held-out groups | Brier | Log loss |
| --- | --- | --- | --- |
| 0: existing pack | 10/16 | .469 | .659 |
| 10: three draws | 12/16,12/16,10/16 | .446 mean | .721 mean |
| 25: all three draws | 12/16 | .400 | .587 |
| 25: class prior only | 12/16 | .378 | .567 |

The fitted model predicts reprocess for all16 cases. Diagnostic accuracy remains
9/11, all four discard outcomes are missed, and the class-prior baseline has
better Brier score and log loss. The paired25-label accuracy improvement interval
is [-12.5,+37.5] percentage points. **No meaningful judgment gain is established.**
The three draws share one held-out set and cannot be counted as independent trials.
This is a selected historical sample, not a production accuracy estimate.

Results are in `queue-pilot-20261023/comparison-v1/`; the concise overall report is
`queue-pilot-20261023/report.json`. Forty-one native2B captures plus eight controls
took8.18 seconds, with zero control probability difference. The worker exited.
Only CPU probability-calibration parameters were fitted; no backbone was trained,
pack shipped, hidden head fitted, or production queue changed. The8B run stays paused.
All37 relevant Python tests and eight read-only Chrome checks pass; source/split
hashes and repeated-input serving parity were verified.

The local UI now uses
`--benchmark sweep/out/codegraph-judgment/queue-pilot-20261023/audit-v1/report.json`
with the existing review-server command. It displays the honest comparison and
all42 source timelines. `/review` remains the original read-only correction archive.
Do not tune new prompts/features against this now-examined16-group holdout and
report the result as another independent validation.


2026-10-23 — structured pre-action exploration COMPLETE (fitting-only).
User authorized brief current-2B inference, not 8B training. Fresh artifacts:
sweep/out/codegraph-judgment/queue-structured-20261023/{study-plan,results,manifest}.json.
Exactly 25 original fit groups; old 16 examined test groups excluded before extraction.
Original quotes retained; literal structured matches carry exact quote evidence;
unrecorded fields explicitly unknown. No outcome/current-description fields used.
Original versus structured pack: 16/25 versus13/25 accuracy; AUC .64 versus .78;
balanced accuracy .625 versus .70. Fixed-penalty fivefold stratified group OOF
calibration: both19/25 (76%), discard1/5; structured Brier .310/logloss .484 versus
original .353/.530. Jeffreys class-prior baseline20/25 (80%), Brier .320/logloss .501.
Simple exploratory rules decide6/25, all6 match source actions; prior fallback22/25.
Rule design saw fitting cases/labels: neither six-of-six nor22/25 validates rules.
Seven cases are sparse alerts/symptoms. Ownership mismatch coverage0/25; no facts
were inferred to fill gaps. All figures exploratory; no independent accuracy gain.
Ranking is mildly encouraging, accuracy is not. Next useful experiment is a fresh,
independent, timestamp-audited holdout with both outcomes; freeze these features,
prompt, rules, and calibration procedure first. No retuning on old16 or claiming
this cross-validation validates human feature selection. No further GPU run started.
25 native captures+8 controls took4.469s; parity probability delta0, flips0; worker
exited. 15 targeted Python tests PASS. Frozen input/output hashes verified.
No model weights, shipped pack, certificate, UI or external systems changed.
Other sessions untouched; 8B stays paused. Existing audit UI shows prior pilot,
not these new fitting-only results. See results.json for detailed class metrics.

2026-10-23 — fresh validation admission audit COMPLETE; evidence gate FAILED.
Frozen structured extractor/prompt/rules and affine(a=.9827921721,b=1.2155592281)
from original25 fit groups BEFORE fresh searches. Pack choice temperature1.3810558844;
Jeffreys reprocess prior20.5/26. Old361 full stories excluded, old16 test remains spent.
Artifacts: sweep/out/codegraph-judgment/queue-fresh-validation-20261023/.
16 prespecified/extended searches fully paginated (43 pages; no pagination truncation),
377 unseen full stories saved sanitized,14 selected histories audited. This is bounded
search coverage, NOT all Codegraph. Four source-backed independent groups admitted:
113637 discard(diagnostic);119161 reprocess(diagnostic);116075 reprocess(queue-count
symptom);132446 reprocess(customer symptom). All exact input quotes precede outcome
record; no same-note diagnostic enrichment. Actual execution clocks often unavailable.
Full formal-link/parent/shared-PagerDuty/shared-MR/commit graph over old+new snapshots
excludes24 new stories dependent on prior work. Four admitted remain independent.
No_verified_completed_queue_disposition is a screening exclusion, not an exhaustive
historical proof of no action. Explicit exclusions distinguish proposals, action before
first record, domain portfolio purges and unclear final mechanisms.
Gate requires20 groups,>=5 per mechanism,>=10 diagnostic. Actual4 groups (3reprocess,
1discard),2diagnostic. Insufficient: NO scoring, comparison, fitting or GPU work started.
Fresh cases quarantined; do not tune on them. Never relax timing to fill the cohort.
prospective-case-template.json captures immutable pre-action checkpoint, source links,
related incidents/fixes and separate post-action outcome; collect all eligible consecutive
incidents. Need at least16 additional independent groups, including4 discard and8
additional diagnostic groups, before this small pilot can run. More needed for precision.
New sweep/queue_validation.py records admission gate, evidence and dependence audit;
explicit past return to consumer queue is accepted, generic 'cleared' alone is not.
19 targeted tests PASS; all artifact/source hashes and four timelines verified.
8B paused; other sessions untouched; existing UI still displays prior25-label pilot.

2026-10-23 — prospective collection and target-utility review SET UP.
Local UI http://127.0.0.1:8766/ is RUNNING. Own PID34912, saved under
sweep/out/codegraph-judgment/queue-prospective-20261023/server.pid and server.log.
New sweep/queue_collection.py + judgment_review_ui/collection.html. Fresh append-only
collection.sqlite3 starts EMPTY. No synthetic QA or historical cases in live journal.
User saves literal observations and source links BEFORE queue action. Current local
UTC checkpoint immutable; new decision checkpoint IDs for changed facts, same family.
Separate assessment supports reprocess_only/discard_only/either/insufficient, confidence,
rationale and checkpoint-only attestation. Other reviewers' answers and actual action
absent from assessment API. Reviewer identity/blinding self-reported, not authenticated.
Outcome requires explicit source quote and reported action start strictly after saved
checkpoint and no later than now. Success separate. SQLite immutable triggers + journal
hash chain check; not independent trusted execution-time proof. Low confidence<.8,
either/insufficient cases get review_needed in export. No automatic fitting promotion.
Target review begins at20 completed cases with2 reviewers; candidate threshold80%
single-action agreement is a discussion trigger, not validation. Inspect both-acceptable,
insufficient evidence, reviewer disagreement and observed-action disagreement. Family,
source timing and class support audits still mandatory before any scoring.
No automatic ingestion is connected. This is a local capture/review tool, not a claim
that future cases are accumulating unattended. No messages sent or systems changed.
Available Codegraph tools have no Slack/PagerDuty reader. Limited read-only probes:
resolved Grafana incidents(limit10) returnednone; DomainInventoryApi Republish logs
2026-10-22T00Z..2026-10-23T00Z returned0. Not exhaustive; source-access-probes.json saved.
Existing four historical candidates remain quarantined; old audit UI8765 unchanged.
Frozen20-group/5-per-label/10-diagnostic validation gate remains; model unchanged.
25 targeted Python tests +10 isolated browser workflow/security/mobile checks PASS.
Live HTTP checks PASS, empty journal. protocol.json pins code/UI/test hashes; validation
and desktop/mobile screenshots saved. 8B paused, GPU untouched; other sessions untouched.
Export: .venv-train/bin/python -m sweep.queue_collection export --db
sweep/out/codegraph-judgment/queue-prospective-20261023/collection.sqlite3
