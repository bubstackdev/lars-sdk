# Experimental image answers

LARS can answer boolean and multiple-choice questions about one JPEG or PNG using a local vision-language model. Answers come from the next-token logits of your declared option letters; no generated prose or JSON is parsed.

This is an experimental inference path. Existing text packs do not support images. A vision pack contains both compatible language weights and a vision projector. Image task accuracy has not been benchmarked broadly, and the SDK does not yet fit image predicate calibrators or certify image error thresholds. Image decisions expose `calibrated: false` unless the vision pack supplies an image calibrator bound to its inference fingerprint.

## Create a local vision pack

Download matching GGUFs from the same checkpoint and conversion. For an initial integration test, the [official Qwen3-VL-2B-Instruct GGUF repository](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF) provides `Qwen3VL-2B-Instruct-Q4_K_M.gguf` and `mmproj-Qwen3VL-2B-Instruct-F16.gguf`.

From the SDK checkout:

```sh
python3 tools/create_vision_pack.py \
  --model /path/to/Qwen3VL-2B-Instruct-Q4_K_M.gguf \
  --projector /path/to/mmproj-Qwen3VL-2B-Instruct-F16.gguf \
  --output ./models/vision-dev \
  --name lars-vision-dev
```

The output directory must not already exist. The tool copies both artifacts, computes SHA-256 hashes, and records the checkout's llama.cpp version. It does not establish model/projector compatibility or task accuracy. For custom templates or conversions, inspect the resulting manifest against the checkpoint's actual chat format. The runtime refuses a vision pack made for a different linked llama.cpp version. No vision pack name is published by this change; open your local directory.

## TypeScript

Build the addon with `cd ts && npm run build:native && npm run build`.

```ts
import { Lars, type ImageInput } from "@moedex/lars";

const lars = await Lars.open({ pack: "./models/vision-dev" });
try {
  const photo = await Lars.imageFile("./delivery.jpg");
  const damaged = lars.noul<ImageInput>(
    "damaged", "The package has visible damage",
  );
  const location = lars.choice(
    "location", "Where was the package left?",
    {
      doorstep: "At a door",
      mailbox: "Inside a mailbox",
      other: "Somewhere else or not identifiable",
    },
    { state: (image: ImageInput) => image },
  );

  const decisions = await lars.ask(photo, { damaged, location });
  console.log(decisions.damaged.value, decisions.location.value);
  console.log(decisions.damaged.calibrated);

  // A predicate call returns its plain boolean/key as usual.
  if (await damaged(photo)) flagForReview();
} finally {
  lars.close();
}
```

`Lars.imageBytes(bytes)` snapshots a buffer. `photo.withContext({ expectedLocation: "front door" })` adds explicit context. File helpers snapshot the file when called; later writes do not alter the input. Wrappers own encoded bytes but no GPU resources. Passing an image nested inside ordinary JSON state is rejected. State selectors may return an `ImageInput` directly; when grouping predicates, all selectors must resolve to the same image/context and the same request options. For a raw request use `lars.evaluateImage(photo, { questions: ... })`; its context comes from the image wrapper.

Runnable example: [image-answers.ts](../ts/examples/image-answers.ts).

## Rust

```rust,no_run
use moedex_lars::{ImageInput, Lars};

let lars = Lars::open("./models/vision-dev").wait()?;
let photo = ImageInput::from_file("./delivery.jpg")?;
let damaged = lars.noul("damaged", "The package has visible damage");
let decision = damaged.decide_image(&photo).wait()?;
if damaged.ask_image(&photo).wait()? {
    flag_for_review();
}
```

`ask_image` and `decide_image` explicitly bypass a predicate's text state selector. Context comes from `ImageInput::with_context`. Several predicates can share an image through `lars.ask_image(&photo, (&damaged, &location))`; its return shape matches `ask`. Predicates in a group must use the same request options. `evaluate_image` exposes the raw request operation.

Runnable example: `cargo run -p moedex-lars --example image_answers -- VISION_PACK IMAGE`.

## C sharp and C

```csharp
using var lars = await Lars.OpenAsync("./models/vision-dev");
var photo = await ImageInput.FromFileAsync("./delivery.jpg");
var damaged = lars.Noul("damaged", "The package has visible damage");
NoulDecision decision = await damaged.AskImageAsync(photo);
```

Choices use the same enum declarations as text, with `AskImageAsync`. `Lars.AskImageAsync` accepts an image and a list of predicates and returns `Answers`. Image calls bypass text state selectors. Raw image requests use `EvaluateImageAsync`. Byte inputs are copied and context nodes are cloned. The C ABI exposes `lars_evaluate_image_v1` and `lars_evaluate_image_async_v1`; callers pass UTF-8 request JSON plus a byte pointer/length. The async operation copies image bytes before returning. Existing text ABI functions are unchanged, and older native libraries cannot silently process an image as text.

## Limits and probability interpretation

- One still image per request; JPEG and PNG only. Animated PNG is rejected.
- Maximum encoded size: 20 MiB. Maximum decoded size: 40 million pixels. Outstanding encoded image data is limited to 64 MiB per loaded model worker.
- Preprocessing `stb-rgb-white-v1` decodes the stored raster into RGB and composites transparency on white. EXIF rotation and ICC color management are not applied; normalize phone photos before loading if those affect the task.
- The pack's `vision.max_tokens` requests a model-specific visual budget; models with fixed resolution may not honor that hint. The final multimodal token count must still fit the language context and request budget.
- Image `score`, `multi`, `explain`, hidden-state heads, external features, and named predicate calibrators are not supported. Attempts fail explicitly. The text APIs continue to support their existing features.
- Probabilities are normalized over the allowed options. A high raw probability does not establish accuracy or detect an unclear image. Include an explicit unclear/other option when appropriate, and evaluate on representative labeled images.
- Boolean threshold and acceptance behavior match text. Use decision objects when you need to distinguish a negative answer from rejection or abstention.
- Images are decoded on the worker and evaluated serially. The initial implementation re-encodes the image per question/option ordering and clears language memory between rows. Shared image embeddings, suffix branching, and cross-request vision caching are future optimizations. Text latency figures do not describe this path.
- Cancellation applies while queued. Running inference finishes normally. Close and idle unload release the projector and image resources.

Vision packs use format 2; text format 1 packs remain supported. Fingerprints cover the manifest's inference settings, the projector and model hashes, the image protocol, and SDK version. Pack download and verification include the projector. Saved calibrators from a different projector, visual budget, template, or preprocessing policy are refused. Uncalibrated packs remain useful for experiments but do not carry certified image accuracy claims.

## Validation

Regular tests cover text compatibility, image API rejection on text packs, input snapshots, and vision pack identity. Real-model tests are opt-in:

```sh
LARS_TEST_VISION_PACK=/absolute/path/to/vision-pack \
  cargo test -p lars-runtime --test images

cd ts
LARS_TEST_VISION_PACK=/absolute/path/to/vision-pack npm test
```

Set `LARS_TEST_VISION_CPU=1` for the Rust CPU run. For .NET tests, set `LARS_NATIVE` to the built C library, `LARS_TEST_VISION_PACK` to the pack, and `LARS_TEST_IMAGE_DIR` to this checkout's `fixtures/images`. The tests compare red/blue inputs, repeat image A after image B, and recover after invalid inputs. These establish integration behavior on simple fixtures; they do not replace the broader model comparison and calibration work in [the research design](image-answers-design.md).
