// Molar Triage in TypeScript: sort a dental practice's inbox with three predicates, then calibrate
// them on 30 labeled messages and measure on 30 others. The messages are fictional
// (examples/molar_triage, from the lars repository).
//
//   npx tsx examples/molar-triage.ts <pack>      e.g. ../models/packs/lars-2b-dev/0.0.1

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { Lars } from "../src/index.js";

const DATA = join(import.meta.dirname, "..", "..", "examples", "molar_triage");

const lars = await Lars.open(process.argv[2] ?? "lars-2b");

const inPain = lars.noul("inPain", "The sender says they are in pain right now.");
const department = lars.choice("department", "Which department should take this message?", {
  hygiene: "Cleanings, checkups, whitening, sensitivity advice",
  restorative: "Fillings, crowns, chipped or broken teeth, lost fillings",
  orthodontics: "Braces, aligners, retainers, wires",
  surgery: "Extractions, wisdom teeth, implants, post-operative care",
  billing: "Insurance, invoices, payment plans, estimates",
  front_desk: "Scheduling, forms, hours, directions, records requests",
});
const urgency = lars.score("urgency", "How urgently does this message need attention?", [
  "Routine: can wait for the next regular visit",
  "Soon: book within a few weeks",
  "This week: schedule in the next few days",
  "Today: needs same-day attention",
  "Emergency: go to the practice or an emergency room now",
]);

// The predicates are functions, so the triage reads like the policy it implements.
async function triage(message: string): Promise<string> {
  const r = await lars.ask(message, { inPain, department, urgency });
  if (r.urgency.level >= 4) return "call the patient now";
  if (r.inPain.value && r.urgency.level >= 3) return `same-day slot with ${r.department.value}`;
  return `queue for ${r.department.value}`;
}

console.log(await triage("I knocked out my front tooth playing hockey twenty minutes ago and it's bleeding a lot."));
console.log(await triage("Can I get a copy of my last invoice for my insurance?"));

// Each labeled row is one message and one question; group them per predicate.
type Row = { state: string; question: { instructions: string }; label: string };
const rows = (file: string): Row[] =>
  readFileSync(join(DATA, file), "utf8").trim().split("\n").map((line) => JSON.parse(line) as Row);
const forQuestion = (all: Row[], instructions: string) => all.filter((r) => r.question.instructions === instructions);

const calibration = rows("molar_triage.calibration.jsonl");
const test = rows("molar_triage.test.jsonl");
const pain = "The sender says they are in pain right now.";
const dept = "Which department should take this message?";
const urg = "How urgently does this message need attention?";

async function accuracy(): Promise<string> {
  const [p, d, u] = await Promise.all([
    Promise.all(forQuestion(test, pain).map(async (r) => (await inPain.decide(r.state)).value === (r.label === "yes"))),
    Promise.all(forQuestion(test, dept).map(async (r) => (await department(r.state)) === r.label)),
    Promise.all(forQuestion(test, urg).map(async (r) => (await urgency.decide(r.state)).level === Number(r.label))),
  ]);
  const share = (hits: boolean[]) => `${hits.filter(Boolean).length}/${hits.length}`;
  return `in pain ${share(p)}, department ${share(d)}, urgency ${share(u)}`;
}

console.log(`\ntest accuracy, pack calibrator:       ${await accuracy()}`);
await inPain.calibrate(forQuestion(calibration, pain).map((r) => ({ input: r.state, label: r.label === "yes" })));
await department.calibrate(forQuestion(calibration, dept).map((r) => ({ input: r.state, label: r.label as never })));
await urgency.calibrate(forQuestion(calibration, urg).map((r) => ({ input: r.state, label: Number(r.label) })));
console.log(`test accuracy, calibrated on 30 rows: ${await accuracy()}`);
lars.close();
