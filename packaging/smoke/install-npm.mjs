// Install both distributions under the Node version selected by the workflow, in isolated
// projects with empty npm caches. No registry credentials or source-tree native lookup.
import { mkdtempSync, readFileSync, writeFileSync, copyFileSync, rmSync } from "node:fs";
import { join, resolve, dirname } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const dist = resolve(process.argv[2] ?? join(root, "dist"));
const { version } = JSON.parse(readFileSync(join(root, "ts/package.json"), "utf8"));
const triple = `${process.platform}-${process.arch}`;
const filename = `moedex-lars-${version}.tgz`;
const npm = process.platform === "win32" ? "npm.cmd" : "npm";
for (const flavor of ["npm", "github"]) {
  const consumer = mkdtempSync(join(tmpdir(), `lars-${flavor}-node${process.versions.node.split(".")[0]}-`));
  try {
    writeFileSync(join(consumer, "package.json"), '{"private":true,"type":"module"}\n');
    const packages = [join(dist, flavor, filename)];
    if (flavor === "npm") packages.push(join(dist, flavor, `moedex-lars-${triple}-${version}.tgz`));
    execFileSync(npm, ["install", "--offline", "--ignore-scripts", "--no-audit", "--no-fund",
      "--cache", join(consumer, ".cache"), ...(flavor === "npm" ? ["--omit=optional"] : []), ...packages],
    { cwd: consumer, stdio: "inherit", shell: process.platform === "win32" });
    copyFileSync(join(root, "packaging/smoke/npm-smoke.mjs"), join(consumer, "smoke.mjs"));
    execFileSync(process.execPath, ["smoke.mjs"], { cwd: consumer, stdio: "inherit" });
  } finally {
    rmSync(consumer, { recursive: true, force: true });
  }
}
