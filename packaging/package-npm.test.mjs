import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
const platforms = ["darwin-arm64", "darwin-x64", "linux-arm64", "linux-x64", "win32-x64"];

test("GitHub tarball installs offline by itself and carries every required native addon", () => {
  const dir = mkdtempSync(join(tmpdir(), "lars-package-test-"));
  try {
    for (const triple of platforms) {
      mkdirSync(join(dir, "native", triple), { recursive: true });
      writeFileSync(join(dir, "native", triple, "lars.node"), `fixture-${triple}`);
    }
    execFileSync(process.execPath, ["ts/scripts/package-npm.mjs", "--native", join(dir, "native"),
      "--out", join(dir, "npm"), "--github-out", join(dir, "github"), "--require-platforms", platforms.join(",")],
    { cwd: root, stdio: "pipe" });
    const { version } = JSON.parse(readFileSync(join(root, "ts/package.json")));
    const filename = `moedex-lars-${version}.tgz`;
    const consumer = join(dir, "consumer");
    mkdirSync(consumer);
    writeFileSync(join(consumer, "package.json"), '{"private":true,"type":"module"}');
    const npm = process.platform === "win32" ? "npm.cmd" : "npm";
    execFileSync(npm, ["install", "--offline", "--ignore-scripts", "--no-audit", "--no-fund",
      "--cache", join(dir, "empty-cache"), join(dir, "github", filename)],
    { cwd: consumer, stdio: "pipe", shell: process.platform === "win32" });
    const installed = join(consumer, "node_modules/@moedex/lars");
    const pkg = JSON.parse(readFileSync(join(installed, "package.json")));
    assert.equal(pkg.optionalDependencies, undefined);
    assert.equal(pkg.dependencies, undefined);
    assert.equal(pkg.scripts, undefined);
    assert.equal(pkg.version, version);
    assert.ok(readFileSync(join(installed, "dist/index.js"), "utf8").includes("class Lars"));
    for (const triple of platforms) {
      assert.equal(readFileSync(join(installed, "native", `lars.${triple}.node`), "utf8"), `fixture-${triple}`);
    }
    const split = JSON.parse(execFileSync("tar", ["-xOf", join(dir, "npm", filename), "package/package.json"], { encoding: "utf8" }));
    assert.equal(split.optionalDependencies["@moedex/lars-darwin-arm64"], version);
    assert.equal(Object.keys(split.optionalDependencies).length, 5);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("release packaging rejects a missing required target before producing artifacts", () => {
  const dir = mkdtempSync(join(tmpdir(), "lars-package-missing-"));
  try {
    mkdirSync(join(dir, "native", "darwin-arm64"), { recursive: true });
    writeFileSync(join(dir, "native", "darwin-arm64", "lars.node"), "fixture");
    const result = spawnSync(process.execPath, ["ts/scripts/package-npm.mjs", "--native", join(dir, "native"),
      "--out", join(dir, "npm"), "--require-platforms", platforms.join(",")], { cwd: root, encoding: "utf8" });
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /missing required native platforms/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
