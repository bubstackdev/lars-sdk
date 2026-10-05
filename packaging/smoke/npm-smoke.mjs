// Install test for the npm packages: run from a fresh project where only the built tarballs were
// installed, with no LARS_NATIVE and no repository around, on a mock pack.
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { Lars } from "@moedex/lars";

delete process.env.LARS_NATIVE;
const dir = mkdtempSync(join(tmpdir(), "lars-npm-smoke-"));
writeFileSync(join(dir, "pack.json"), JSON.stringify({ format: 1, name: "smoke", version: "0.0.0", backend: "mock", template: "plain" }));
const lars = await Lars.open(dir);
const isRefund = lars.noul("isRefund", "The customer asks for a refund");
const team = lars.choice("team", "Which team?", { billing: "Charges", technical: "Bugs" });
const answers = await Promise.all([isRefund("refund me"), team("I was charged twice")]);
if (typeof answers[0] !== "boolean" || !["billing", "technical"].includes(answers[1])) throw new Error(`unexpected ${answers}`);
console.log(`ok: @moedex/lars ${Lars.version()} on ${process.platform}-${process.arch}, node ${process.version}:`, answers);
lars.close();
