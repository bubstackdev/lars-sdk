// Bun loads the same addon: `bun test/bun-smoke.ts` (run by CI).
import { Lars } from "../src/index.ts";
import { mkdtempSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
const dir = mkdtempSync(join(tmpdir(), "lars-bun-"));
writeFileSync(join(dir, "pack.json"), JSON.stringify({ format: 1, name: "bun", version: "0.0.0", backend: "mock", template: "plain" }));
const lars = await Lars.open(dir);
const isRefund = lars.noul("isRefund", "The customer asks for a refund");
const team = lars.choice("team", "Which team?", { billing: "Charges", technical: "Bugs" });
console.log("bun", Bun.version, await isRefund("refund me"), await team("my charges are wrong"), (await Promise.all(Array.from({ length: 100 }, () => isRefund("x")))).length);
lars.close();
