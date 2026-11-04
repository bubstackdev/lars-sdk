import test from "node:test";
import assert from "node:assert/strict";
import { answerIndex } from "./check_pack.mjs";

test("choice uses the returned decision when display rounding erases the margin", () => {
  const keys = ["cancel_transfer", "transfer_not_received_by_recipient"];
  const full = [0.13425982243657128, 0.13432584945162154];
  const rounded = full.map((p) => Number(p.toFixed(4)));
  assert.deepEqual(rounded, [0.1343, 0.1343]);
  assert.equal(answerIndex({ type: "choice", choice: keys[1] }, keys, rounded), 1);
  // An actual wrong returned choice must still fail the zero-label-flips check.
  assert.notEqual(answerIndex({ type: "choice", choice: keys[0] }, keys, rounded), 1);
});

test("unknown returned choice is rejected", () => {
  assert.throws(() => answerIndex({ type: "choice", choice: "missing" }, ["a", "b"], [0.2, 0.8]), /Unknown SDK choice/);
});

test("score and binary answers retain their existing probability decisions", () => {
  assert.equal(answerIndex({ type: "score" }, ["0", "1"], [0.2, 0.8]), 1);
  assert.equal(answerIndex({ type: "noul" }, ["yes", "no"], [0.8, 0.2]), 0);
});
