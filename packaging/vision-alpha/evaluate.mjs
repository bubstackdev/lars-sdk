// Run against either an installed alpha or --sdk /absolute/path/to/ts/dist/index.js.
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname, resolve, relative, isAbsolute } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import os from "node:os";
import { performance } from "node:perf_hooks";

const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const mean = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
const percentile = (values, p) => values.length ? [...values].sort((a, b) => a - b)[Math.ceil(p * values.length) - 1] : null;

export function summarize(records) {
  const intact = records.filter(r => r.mode === "intact");
  const valid = intact.filter(r => !r.error);
  const summary = {
    total: intact.length, completed: valid.length, errors: intact.length - valid.length,
    source_groups: new Set(intact.map(r => r.group)).size,
    accuracy: mean(valid.map(r => Number(r.prediction === r.label))),
    mean_log_loss: mean(valid.map(r => -Math.log(Math.max(1e-15, r.target_probability)))),
    mean_brier: mean(valid.map(r => r.brier)),
    latency_ms: { p50: percentile(valid.map(r => r.duration_ms), 0.5), p95: percentile(valid.map(r => r.duration_ms), 0.95) },
    categories: {}, controls: {},
  };
  for (const category of [...new Set(intact.map(r => r.category))].sort()) {
    const all = intact.filter(r => r.category === category), rows = all.filter(r => !r.error);
    summary.categories[category] = { total: all.length, errors: all.length - rows.length,
      accuracy: mean(rows.map(r => Number(r.prediction === r.label))) };
  }
  for (const mode of [...new Set(records.map(r => r.mode))].filter(m => m !== "intact")) {
    const rows = records.filter(r => r.mode === mode && !r.error);
    const paired = rows.map(r => [r, valid.find(v => v.id === r.id)]).filter(([, original]) => original);
    summary.controls[mode] = {
      completed: rows.length, errors: records.filter(r => r.mode === mode && r.error).length,
      prediction_change_rate: mean(paired.map(([r, original]) => Number(r.prediction !== original.prediction))),
      mean_original_target_probability: mean(rows.map(r => r.target_probability)),
      // Only order reversal preserves the original image/question/ground truth.
      ...(mode === "reversed_options" && { accuracy: mean(rows.map(r => Number(r.prediction === r.label))) }),
      ...(mode === "reversed_options" && {
        actually_reordered: rows.filter(r => r.option_order_changed).length,
        unchanged_key_order: rows.filter(r => r.option_order_changed === false).length,
      }),
    };
  }
  return summary;
}

export function validateSuite(suite) {
  if (suite.format !== 1 || !Array.isArray(suite.cases) || !suite.cases.length) throw new Error("unsupported/empty suite");
  const ids = new Set();
  for (const c of suite.cases) {
    if (!c.id || ids.has(c.id) || !c.group || !c.category || typeof c.question !== "string" || !c.question.trim()) throw new Error("invalid/duplicate case");
    ids.add(c.id);
    if (c.kind === "noul" && typeof c.label === "boolean") continue;
    if (c.kind === "choice" && c.options && Object.keys(c.options).length >= 2 && Object.hasOwn(c.options, c.label)) continue;
    throw new Error(`invalid kind/label/options: ${c.id}`);
  }
}

async function main() {
  const args = {};
  for (let i = 2; i < process.argv.length; i += 2) {
    const key = process.argv[i], value = process.argv[i + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--")) throw new Error("usage: node evaluate.mjs --pack PACK [--suite suite.json] [--out result.json] [--sdk index.js] [--controls all|none]");
    if (!["--pack", "--suite", "--out", "--sdk", "--controls", "--gpu-layers"].includes(key)) throw new Error(`unknown option ${key}`);
    args[key.slice(2)] = value;
  }
  if (!args.pack) throw new Error("--pack is required");
  if (args.controls && !["all", "none"].includes(args.controls)) throw new Error("--controls must be all or none");
  const suitePath = resolve(args.suite ?? fileURLToPath(new URL("./fixtures/suite.json", import.meta.url)));
  const suiteBytes = await readFile(suitePath), suite = JSON.parse(suiteBytes);
  validateSuite(suite);
  const suiteDir = dirname(suitePath), sourcesBytes = await readFile(resolve(suiteDir, "SOURCES.json"));
  const sources = JSON.parse(sourcesBytes), images = new Map();
  async function bytes(path) {
    if (images.has(path)) return images.get(path);
    const full = resolve(suiteDir, path), rel = relative(suiteDir, full);
    if (rel.startsWith("..") || isAbsolute(rel)) throw new Error("image path escapes suite directory");
    const data = await readFile(full), item = sources.files[path.split("/").at(-1)];
    if (!item || hash(data) !== item.sha256) throw new Error(`fixture checksum mismatch: ${path}`);
    images.set(path, data); return data;
  }
  for (const path of new Set([...suite.cases.map(c => c.image), suite.blank_image])) await bytes(path);
  const { Lars } = await import(args.sdk ? pathToFileURL(resolve(args.sdk)).href : "@moedex/lars");
  const gpuLayers = args["gpu-layers"] === undefined ? -1 : Number(args["gpu-layers"]);
  if (![-1, 0].includes(gpuLayers)) throw new Error("--gpu-layers must be -1 or 0");
  const start = performance.now(), lars = await Lars.open({ pack: args.pack, threads: 4, gpuLayers });
  const report = {
    format: 1, suite: suite.name, suite_sha256: hash(suiteBytes), fixture_sources_sha256: hash(sourcesBytes),
    started_at: new Date().toISOString(), sdk_runtime_version: Lars.version(), pack: lars.pack,
    host: { platform: process.platform, arch: process.arch, node: process.version, os_release: os.release(), cpu: os.cpus()[0]?.model, memory_bytes: os.totalmem() },
    settings: { threads: 4, gpu_layers: gpuLayers, controls: args.controls ?? "all", protocol: "direct-options-v1", calibration: "none" },
    load_ms: performance.now() - start, records: [],
    interpretation: "Development diagnostics, not held-out accuracy or calibration. Blank/shuffled controls retain the original target only to measure visual dependence; their labels are not ground truth for substituted images. Source groups contain correlated questions/variants.",
  };
  if (lars.pack.calibration) { lars.close(); throw new Error("this alpha evaluation requires an uncalibrated pack"); }
  const groups = [...new Set(suite.cases.map(c => c.group))];
  try {
    for (const [index, c] of suite.cases.entries()) {
      const modes = args.controls === "none" ? ["intact"] : ["intact", "blank", "shuffled", ...(c.kind === "choice" ? ["reversed_options"] : [])];
      for (const mode of modes) {
        let path = c.image;
        if (mode === "blank") path = suite.blank_image;
        if (mode === "shuffled") {
          const next = groups[(groups.indexOf(c.group) + 1) % groups.length];
          path = suite.cases.find(other => other.group === next).image;
        }
        let image = Lars.imageBytes(await bytes(path));
        if (c.context !== undefined) image = image.withContext(c.context);
        const options = mode === "reversed_options" ? Object.fromEntries(Object.entries(c.options).reverse()) : c.options;
        const name = `${c.id}-${mode}`;
        const p = c.kind === "noul" ? lars.noul(name, c.question) : lars.choice(name, c.question, options);
        const begin = performance.now();
        const record = { id: c.id, group: c.group, category: c.category, kind: c.kind, mode, image: path, image_sha256: hash(await bytes(path)), question: c.question, label: c.label };
        if (c.kind === "choice") record.option_order = Object.keys(options);
        if (mode === "reversed_options") {
          record.option_order_changed = JSON.stringify(Object.keys(options)) !== JSON.stringify(Object.keys(c.options));
        }
        try {
          const d = await p.decide(image);
          const probabilities = c.kind === "noul" ? { true: d.p, false: 1 - d.p } : d.probabilities;
          if (!Object.hasOwn(probabilities, String(c.label)) || Object.values(probabilities).some(v => !Number.isFinite(v) || v < 0 || v > 1)
              || Math.abs(Object.values(probabilities).reduce((a, b) => a + b, 0) - 1) > 0.001 || !Object.hasOwn(probabilities, String(d.value))) throw new Error("invalid structured distribution");
          if (d.calibrated !== false) throw new Error("uncalibrated image status is missing");
          Object.assign(record, { prediction: d.value, probabilities, confidence: d.confidence, calibrated: d.calibrated,
            target_probability: probabilities[String(c.label)],
            brier: Object.entries(probabilities).reduce((sum, [key, value]) => sum + (value - Number(key === String(c.label))) ** 2, 0) });
        } catch (error) { record.error = { type: error.type ?? error.name, message: error.message }; }
        record.duration_ms = performance.now() - begin;
        report.records.push(record);
      }
      console.log(`${index + 1}/${suite.cases.length}: ${c.id}`);
    }
  } finally { lars.close(); }
  report.summary = summarize(report.records);
  const out = resolve(args.out ?? "vision-alpha-result.json");
  await mkdir(dirname(out), { recursive: true });
  await writeFile(out, JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify(report.summary, null, 2));
  console.log(`Saved ${out}`);
  if (report.records.some(r => r.error)) process.exitCode = 1;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(error); process.exitCode = 1; });
}
