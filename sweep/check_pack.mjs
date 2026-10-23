// Check the actual Node SDK pack against calibrated lars-core evaluation, in input order.
import { readFile, writeFile } from "node:fs/promises";
import { Lars } from "../ts/dist/index.js";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

// Choice answers select from full precision internally; displayed probabilities are rounded.
// Recomputing their argmax can invent a tie and discard the actual returned decision.
export function answerIndex(answer, keys, probs) {
  if (answer.type === "choice") {
    const index = keys.indexOf(answer.choice);
    if (index < 0) throw new Error(`Unknown SDK choice: ${answer.choice}`);
    return index;
  }
  return probs.reduce((best, value, j) => value > probs[best] ? j : best, 0);
}

async function main() {

  const [pack, wirePath, expectedPath, output] = process.argv.slice(2);
  const lines = (s) => s.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
  const rows = lines(await readFile(wirePath, "utf8"));
  const expected = lines(await readFile(expectedPath, "utf8"));
  if (rows.length !== expected.length) throw new Error("SDK check populations differ");
  const lars = await Lars.open({ pack, verify: true });
  let maxDifference = 0;
  let labelFlips = 0;
  const labelMismatches = [];
  try {
    for (let i = 0; i < rows.length; i++) {
      const response = await lars.evaluate({ state: rows[i].state, questions: { q: rows[i].question } });
      const answer = response.answers.q;
      const probs = answer.type === "noul" ? [answer.noul, 1 - answer.noul]
        : expected[i].keys.map((key) => answer.probabilities[key]);
      const argmax = (v) => v.reduce((best, value, j) => value > v[best] ? j : best, 0);
      const expectedIndex = argmax(expected[i].p);
      const actualIndex = answerIndex(answer, expected[i].keys, probs);
      if (actualIndex !== expectedIndex) {
        labelFlips++;
        labelMismatches.push({ row: i, kind: answer.type,
          expected: expected[i].keys[expectedIndex], actual: expected[i].keys[actualIndex] });
      }
      for (let k = 0; k < probs.length; k++) {
        if (!Number.isFinite(probs[k])) throw new Error(`Missing SDK probability at row ${i}`);
        maxDifference = Math.max(maxDifference, Math.abs(probs[k] - expected[i].p[k]));
      }
    }
  } finally {
    lars.close();
  }
  const result = { rows: rows.length, labelFlips, maxProbabilityDifference: maxDifference,
    probabilityTolerance: 0.001, labelMismatches, passed: maxDifference <= 0.001 && labelFlips === 0 };
  await writeFile(output, `${JSON.stringify(result, null, 2)}\n`);
  if (!result.passed) throw new Error(`SDK pack parity failed: ${labelFlips} label flips; maximum probability difference ${maxDifference}`);
  console.log(JSON.stringify(result));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
