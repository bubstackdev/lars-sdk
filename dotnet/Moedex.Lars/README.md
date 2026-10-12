# Moedex.Lars

Typed decisions as functions in your code, answered by a small language model running in your
.NET process. No server, no Ollama, no network after the model pack is downloaded.

```csharp
using Moedex.Lars;

using var lars = await Lars.OpenAsync("lars-2b");

var isPissed = lars.Noul("isPissed", "The customer is angry");
var team = lars.Choice<Team>("team", "Which team should handle this?");

if (await isPissed.AskAsync(message)) Escalate(message);   // NoulDecision converts to bool
if (isPissed.Check(message)) Escalate(message);             // blocking, for scripts and console apps
Team t = await team.AskAsync(message);                      // ChoiceDecision<Team> converts to Team

public enum Team
{
    [Option("Payments, charges, refunds")] Billing,
    [Option("Bugs, crashes, outages")] Technical,
}
```

- **Primitives:** `Noul`, `Choice<TEnum>`, `Score` and `Multi<TEnum>`.
- **`AskAsync`** returns the decision: the value, the probabilities, the confidence (the
  calibrated chance the answer is right), and whether a certified threshold accepts it. It takes
  a `CancellationToken`, which cancels a queued call.
- **`lars.AskAsync(input, isPissed, team)`** asks several predicates in one request; read each
  answer with `answers.Of(isPissed)`.
- **`CalibrateAsync(labeled, maxError: 0.05)`** fits a calibrator on your labels. It certifies a
  threshold, when the labels support one, below which answers are not accepted.
- **Inputs:** strings, `JsonNode`s, or any type with a `State` selector that picks the fields
  the model sees.

Calls queue on the model's worker thread; async calls hold no thread while they wait, and
continuations never run on the model's thread.

Runs on macOS (arm64, x64; Metal), Linux (x64, arm64; glibc 2.28+) and Windows x64, with
.NET 10. The native library for each is in the package's `runtimes/` folder.

Docs, packs, calibration guide and limits: https://github.com/bubstackdev/lars-sdk
