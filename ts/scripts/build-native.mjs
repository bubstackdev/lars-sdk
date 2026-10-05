// Build the Rust addon (crates/lars-node) and copy it to native/lars.<platform>-<arch>.node, where
// src/native.ts looks first. Packaging for npm (one optional package per platform) comes later.
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..");
const profile = process.argv.includes("--debug") ? "debug" : "release";
execFileSync("cargo", ["build", "-p", "lars-node", ...(profile === "release" ? ["--release"] : [])], {
  cwd: root,
  stdio: "inherit",
});
const library = { darwin: "liblars_node.dylib", linux: "liblars_node.so", win32: "lars_node.dll" }[process.platform];
if (!library) throw new Error(`unsupported platform ${process.platform}`);
mkdirSync(join(here, "..", "native"), { recursive: true });
const target = join(here, "..", "native", `lars.${process.platform}-${process.arch}.node`);
copyFileSync(join(root, "target", profile, library), target);
console.log(`built ${target}`);
