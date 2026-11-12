// Build the npm packages: one @moedex/lars-<platform>-<arch> per native addon found, and
// @moedex/lars itself, which lists them as optionalDependencies so npm installs only the one that
// matches the machine (by its os, cpu and libc fields). --github-out additionally builds a
// standalone package containing every addon, with no registry dependencies. Publishes nothing.
//
//   node scripts/package-npm.mjs --native <dir with <platform>-<arch>/lars.node> --out <dir>

import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve, relative, isAbsolute } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const ts = join(here, "..");
const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, arg, i, all) => (arg.startsWith("--") ? [...pairs, [arg.slice(2), all[i + 1]]] : pairs), []),
);
const nativeDir = resolve(args.native ?? join(ts, "native-dist"));
const out = resolve(args.out ?? join(ts, "npm-dist"));
const main = JSON.parse(readFileSync(join(ts, "package.json"), "utf8"));
const npm = process.platform === "win32" ? "npm.cmd" : "npm";
const pack = (dir, destination = out) => execFileSync(npm, ["pack", "--ignore-scripts", "--pack-destination", destination], { cwd: dir, stdio: ["ignore", "pipe", "inherit"], shell: process.platform === "win32" }).toString().trim();

const supported = ["darwin-arm64", "darwin-x64", "linux-arm64", "linux-x64", "win32-x64"];
const platforms = readdirSync(nativeDir).filter((d) => existsSync(join(nativeDir, d, "lars.node"))).sort();
if (platforms.length === 0) throw new Error(`no <platform>-<arch>/lars.node under ${nativeDir}`);
for (const triple of platforms) if (!supported.includes(triple)) throw new Error(`unsupported native platform: ${triple}`);
const missing = (args["require-platforms"] ?? "").split(",").filter((p) => p && !platforms.includes(p));
if (missing.length) throw new Error(`missing required native platforms: ${missing.join(", ")}`);
const githubOut = args["github-out"] && resolve(args["github-out"]);
const contains = (parent, child) => { const path = relative(parent, child); return !path || (!path.startsWith("..") && !isAbsolute(path)); };
if (githubOut && (contains(out, githubOut) || contains(githubOut, out))) {
  throw new Error("--github-out and --out must be separate directories");
}

const notices = args.notices && resolve(args.notices);
function copyNotices(dir) {
  if (notices) cpSync(notices, join(dir, "THIRD_PARTY_LICENSES"), { recursive: true });
}

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });
const staging = join(out, ".staging");

const optional = {};
for (const triple of platforms) {
  const [platform, arch] = triple.split("-");
  const dir = join(staging, triple);
  mkdirSync(dir, { recursive: true });
  cpSync(join(nativeDir, triple, "lars.node"), join(dir, "lars.node"));
  cpSync(join(ts, "..", "LICENSE"), join(dir, "LICENSE"));
  copyNotices(dir);
  const name = `${main.name}-${triple}`;
  writeFileSync(join(dir, "package.json"), JSON.stringify({
    name,
    version: main.version,
    description: `The native library of ${main.name} for ${triple}`,
    license: main.license,
    repository: main.repository,
    os: [platform],
    cpu: [arch],
    ...(platform === "linux" && { libc: ["glibc"] }),
    main: "lars.node",
    files: ["lars.node", "THIRD_PARTY_LICENSES"],
  }, null, 2) + "\n");
  console.log(pack(dir));
  optional[name] = main.version;
}

execFileSync(npm, ["run", "build"], { cwd: ts, stdio: "inherit", shell: process.platform === "win32" });
const dir = join(staging, "main");
mkdirSync(dir, { recursive: true });
cpSync(join(ts, "dist"), join(dir, "dist"), { recursive: true });
for (const file of ["README.md", "LICENSE"]) {
  const source = [join(ts, file), join(ts, "..", file)].find(existsSync);
  if (source) cpSync(source, join(dir, file));
}
const { devDependencies, scripts, ...published } = main;
writeFileSync(join(dir, "package.json"), JSON.stringify({ ...published, files: ["dist"], optionalDependencies: optional }, null, 2) + "\n");
console.log(pack(dir));
if (githubOut) {
  mkdirSync(githubOut, { recursive: true });
  mkdirSync(join(dir, "native"));
  for (const triple of platforms) cpSync(join(nativeDir, triple, "lars.node"), join(dir, "native", `lars.${triple}.node`));
  copyNotices(dir);
  writeFileSync(join(dir, "package.json"), JSON.stringify({ ...published, files: ["dist", "native", "THIRD_PARTY_LICENSES"] }, null, 2) + "\n");
  console.log(pack(dir, githubOut));
}
rmSync(staging, { recursive: true, force: true });
