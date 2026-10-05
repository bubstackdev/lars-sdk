# @moedex/lars

Typed decisions as functions in your code, answered by a small language model running in your
Node or Bun process. No server, no Ollama, no network after the model pack is downloaded.

```ts
import { Lars } from "@moedex/lars";

const lars = await Lars.open({ pack: "lars-4b" });

const isPissed = lars.noul("isPissed", "The customer is angry");
const team = lars.choice("team", "Which team should handle this?", {
  billing: "Payments, charges, refunds",
  technical: "Bugs, crashes, outages",
});

if (await isPissed(message)) escalate(message);    // boolean
const t = await team(message);                      // "billing" | "technical"
```

- **`noul`** resolves to a `boolean`, **`choice`** to one of its keys (typed as a union),
  **`score`** to the expected level of an ordered scale, **`multi`** to the keys that apply.
- **`predicate.decide(input)`** gives the probabilities, the confidence (the calibrated chance the
  answer is right), and whether a certified threshold accepts the answer.
- **`lars.ask(input, { isPissed, team })`** asks several predicates in one request.
- **`predicate.calibrate(labeled, { maxError: 0.05 })`** fits a calibrator on your labels. It
  certifies a threshold, when the labels support one, so that `if (await isPissed(x))` only
  fires on answers good enough to automate.
- **Inputs** are strings, or objects sent as JSON. `{ state: (x) => ... }` picks the fields the
  model sees.

Calls queue on the model's worker thread, so inference never blocks the event loop. One process
loads each pack once, however many handles open it.

Requires Node 22+ or Bun on macOS (arm64, x64; Metal), Linux (x64, arm64; glibc 2.28+) or
Windows x64. The native library comes in an optional `@moedex/lars-<platform>-<arch>` package
that npm picks for your machine.

Docs, packs, calibration guide and limits: https://github.com/moedex/lars-sdk
