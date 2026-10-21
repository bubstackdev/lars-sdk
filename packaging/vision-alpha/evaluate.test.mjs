import { test } from "node:test";
import assert from "node:assert/strict";
import { summarize, validateSuite } from "./evaluate.mjs";

test("controls are paired by case and substituted pixels do not acquire an accuracy claim", () => {
  const base = { group: "one-image", category: "photos", label: true, target_probability: .8, brier: .08, duration_ms: 10 };
  const s = summarize([
    { ...base, id: "a", mode: "intact", prediction: true },
    { ...base, id: "b", mode: "intact", prediction: false, target_probability: .2 },
    { ...base, id: "a", mode: "blank", prediction: false },
    { ...base, id: "b", mode: "blank", prediction: false },
  ]);
  assert.equal(s.accuracy, .5); assert.equal(s.source_groups, 1);
  assert.equal(s.controls.blank.prediction_change_rate, .5);
  assert.equal(Object.hasOwn(s.controls.blank, "accuracy"), false);
});

test("runtime errors and invalid ground truth cannot masquerade as successful evaluation", () => {
  const s = summarize([{ id: "a", group: "a", category: "docs", mode: "intact", error: { message: "bad image" } }]);
  assert.equal(s.errors, 1); assert.equal(s.accuracy, null);
  assert.throws(() => validateSuite({ format: 1, cases: [{ id: "a", group: "a", category: "docs", kind: "choice", question: "Status?", label: "missing", options: { a: "A", b: "B" } }] }));
});

test("numeric JavaScript option keys do not inflate the number of order controls", () => {
  const numeric = { "0": "Zero", "1": "One", unknown: "Unknown" };
  assert.deepEqual(Object.keys(Object.fromEntries(Object.entries(numeric).reverse())), Object.keys(numeric));
  const base = { id: "a", group: "a", category: "counting", label: "0", prediction: "0", target_probability: 1, brier: 0, duration_ms: 1 };
  const result = summarize([
    { ...base, mode: "intact" },
    { ...base, mode: "reversed_options", option_order_changed: false },
    { ...base, id: "b", mode: "reversed_options", option_order_changed: true },
  ]);
  assert.equal(result.controls.reversed_options.actually_reordered, 1);
  assert.equal(result.controls.reversed_options.unchanged_key_order, 1);
});
