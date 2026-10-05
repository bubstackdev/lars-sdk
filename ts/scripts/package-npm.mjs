// Build the npm packages: one @moedex/lars-<platform>-<arch> per native addon found, and
// @moedex/lars itself, which lists them as optionalDependencies so npm installs only the one that
// matches the machine (by its os, cpu and libc fields). Writes tarballs; publishes nothing.
//
//   node scripts/package-npm.mjs --native <dir with <platform>-<arch>/lars.node> --out <dir>

import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
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
const pack = (dir) => execFileSync(npm, ["pack", "--pack-destination", out], { cwd: dir, stdio: ["ignore", "pipe", "inherit"], shell: process.platform === "win32" }).toString().trim();

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });
const staging = join(out, ".staging");

const platforms = readdirSync(nativeDir).filter((d) => existsSync(join(nativeDir, d, "lars.node")));
if (platforms.length === 0) throw new Error(`no <platform>-<arch>/lars.node under ${nativeDir}`);
const optional = {};
for (const triple of platforms) {
  const [platform, arch] = triple.split("-");
  const dir = join(staging, triple);
  mkdirSync(dir, { recursive: true });
  cpSync(join(nativeDir, triple, "lars.node"), join(dir, "lars.node"));
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
    files: ["lars.node"],
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
rmSync(staging, { recursive: true, force: true });
