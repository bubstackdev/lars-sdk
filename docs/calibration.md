# Calibration

A predicate's answer is a probability. This guide covers what makes that probability
trustworthy, when to calibrate on your own labels, and how to turn a probability into an
automated decision with a known error rate.

## What you get without doing anything

Every pack ships a **pooled calibrator**: a temperature per question kind, and a Platt scaling
for yes/no questions. It was fitted on all 4,120 jev-bench validation rows, through this SDK on
that pack's own model, quantization and runtime.

On jev-bench test, an answer the 4B pack gives with confidence 0.8 is right about 80% of the
time (expected calibration error 0.08). That holds for data like jev-bench: routing, triage,
sentiment, NLI, short-text classification. On your data it is a reasonable start, not a promise.

## Calibrating a predicate on your labels

```ts
const fit = await isPissed.calibrate(
  [{ input: "This is the third time...", label: true }, { input: "Thanks!", label: false }, /* ... */],
);
```
```csharp
var fit = await isPissed.CalibrateAsync([("This is the third time...", true), ("Thanks!", false) /* ... */]);
```

This runs the model on each example once and fits a calibrator for this predicate alone:
- **noul:** a Platt scaling, which can move the yes/no boundary
- **choice and score:** a temperature, which changes confidence but never the answer

From then on, calls to this predicate use it. Other predicates keep the pack's.

**A fit only replaces the pack's calibrator when it is clearly better.** The SDK cross-validates
it against the pack's calibrator on your labels (5 folds, log loss per example). It keeps the
fit only when the 95% upper bound on the improvement is still an improvement.
- `fit.kept` (C#: `Kept`) says which one won.
- Without the guard, a fit on 30 Molar Triage labels moved the 1.7B pack's "in pain" boundary
  the wrong way, from 27 to 24 of 30 right on held-out messages. With it, the pack's
  calibrator stays and the score stays at 27.

How many labels:

| labels per predicate | what to expect |
|---|---|
| under 25 | usually changes nothing: the fit can't beat the pack's significantly |
| 50–200 | Brier improves by 0.04–0.05 on zero-shot models, less on packs with a LoRA; Platt can fix a biased yes/no boundary |
| any number | a temperature never changes which option a choice or score picks |

Label examples the way the predicate will see them: the same field selection (`state`), the same
kind of input. A calibrator describes the inputs it was fitted on.

## Certified thresholds: automating at an error budget

Calibration makes probabilities honest. To act on answers automatically and be wrong at most,
say, 5% of the time, ask `calibrate` to certify a threshold:

```ts
const fit = await isPissed.calibrate(labeled, { maxError: 0.05 });
fit.certified;  // e.g. { maxError: 0.05, threshold: 0.93, accepted: 180, errors: 2, coverage: 0.45, errorUpper95: 0.0396 }
```

The threshold is the lowest confidence at which, among your labeled examples it would accept,
the 95% upper bound on the error rate (Wilson score interval) is at most `maxError`. From then
on, for this predicate:
- `decide()` / `AskAsync()` report `accepted: false` for answers below the threshold.
- The plain call is conservative: a noul resolves to `false` unless it is a yes **and**
  accepted, and a choice resolves to its `fallback`, if you gave one.

So `if (await isPissed(message))` only takes the branch when the answer is good enough to
automate. Send the rest to a person.

`coverage` is the share of your labeled examples the threshold accepts: how much of the work
it automates.

**It takes more labels than you might expect.** To certify an error rate at 95% confidence:

| error budget | accepted answers needed, with no mistakes among them |
|---|---|
| 10% | 35 |
| 5% | 73 |
| 2% | 189 |

With fewer, `calibrate` returns `threshold: null` and a `reason`, rather than a threshold that
would not hold. On jev-bench, the empirical alternative ran at 7–20% error on test when it
promised 5%: that alternative is "accept wherever the labels themselves showed at most 5%
errors." The Wilson threshold held 1.6–4.7% (`RESULTS.md`).

Typical coverage at a 5% budget with 200 labels per decision, on jev-bench:

| pack | coverage |
|---|---|
| 0.6B | 7% |
| 1.7B | 12% |
| 4B | 22–27% |

It is much higher on easy, well-defined decisions and lower on subjective ones.

## Saving calibrations

Calibrators belong to the pack they were fitted on, and the SDK refuses to load one into any
other pack. Save them with your application and load them at startup:

```ts
await lars.saveCalibrations("calibrations.json");
await lars.loadCalibrations("calibrations.json");
```
```csharp
lars.SaveCalibrations("calibrations.json");
lars.LoadCalibrations("calibrations.json");
```

The file holds each predicate's calibrator, bound to the pack's name, version and model hash,
and its certified threshold.

## When to recalibrate

- **When your inputs change:** a new product, a new customer base, a new language mix.
- **When you change the predicate's instructions or options.** It is a different question.
- **When you change packs.** Old calibrators won't load.
