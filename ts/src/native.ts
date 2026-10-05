// Loading the Rust addon (crates/lars-node). Looked for, in order: LARS_NATIVE, the copy
// `npm run build:native` writes to native/, then the per-platform package
// @moedex/lars-<platform>-<arch> that published installs bring.

import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export interface NativeLars {
  evaluate(requestJson: string): Promise<string>;
  calibrate(name: string, examplesJsonl: string, maxError?: number | null): Promise<string>;
  setCalibrator(name: string, calibratorJson: string): void;
  calibrators(): string;
  status(): string;
  manifest(): string;
  close(): void;
}

export interface NativeModule {
  open(pack: string, optionsJson?: string | null): Promise<NativeLars>;
  version(): string;
}

const require = createRequire(import.meta.url);
let loaded: NativeModule | undefined;

export function native(): NativeModule {
  if (loaded) return loaded;
  const here = dirname(fileURLToPath(import.meta.url));
  const triple = `${process.platform}-${process.arch}`;
  const candidates = [process.env.LARS_NATIVE, join(here, "..", "native", `lars.${triple}.node`)];
  for (const path of candidates) {
    if (path && existsSync(path)) return (loaded = require(path) as NativeModule);
  }
  try {
    return (loaded = require(`@moedex/lars-${triple}`) as NativeModule);
  } catch {
    throw new Error(
      `@moedex/lars has no native library for ${triple}: install @moedex/lars-${triple}, ` +
        `or build one with \`npm run build:native\``,
    );
  }
}
