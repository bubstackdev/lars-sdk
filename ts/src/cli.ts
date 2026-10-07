#!/usr/bin/env node
// The `lars` command: `npx lars pull <pack>[@version] ...` downloads packs ahead of time, for
// machines that will run offline or shouldn't download on first use.

import { Lars, LarsError } from "./index.js";

const [command, ...packs] = process.argv.slice(2);
if (command === "pull" && packs.length > 0) {
  for (const pack of packs) {
    try {
      console.log(`${pack}: ${await Lars.pull(pack)}`);
    } catch (error) {
      console.error(`${pack}: ${error instanceof LarsError ? error.message : String(error)}`);
      process.exitCode = 1;
    }
  }
} else if (command === "version") {
  console.log(Lars.version());
} else {
  console.error("usage: lars pull <pack>[@version] ...\n       lars version");
  process.exitCode = 2;
}
