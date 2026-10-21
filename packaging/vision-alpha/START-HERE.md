# LARS image answers developer alpha

This is `0.1.0-image-alpha.1`, an unpublished, standalone TypeScript/Node package for macOS on Apple Silicon. It accepts one JPEG or PNG and answers boolean or multiple-choice questions locally. It does not generate or parse a prose answer.

The verified host is an Apple M5 Max with 64 GB memory, macOS kernel 27.0.0, and Node 24.21.0. Other Apple Silicon Macs are exploratory alpha targets. Intel Macs, Linux, and Windows are outside this artifact's scope. The binary's deployment target is macOS 11, which is not a claim that image inference has been tested on every OS since then.

## Install in a new project

You need Node 24 and Python 3.12 or later for the setup/verification scripts. The prebuilt npm package does not require Rust, CMake, Xcode, or a compiler. Model setup downloads approximately 1.93 GB and stores it locally; allow additional working space while preparing the pack. Subsequent inference is offline.

Keep the tester bundle together. From its directory, verify the files before installing:

```sh
shasum -a 256 -c SHA256SUMS
mkdir tester-project
cd tester-project
npm init -y
npm install --ignore-scripts --offline --no-audit --no-fund ../moedex-lars-0.1.0-image-alpha.1.tgz
python3 ../prepare-model.py --output ./models/lars-vision-alpha
```

The model setup script uses the exact Qwen repository revision and artifact checksums in `model.lock.json`. It verifies both files before making a usable pack and refuses to overwrite a different existing directory. Existing matching files are verified and reused. The weights are Apache 2.0; their upstream model card and revision are recorded in the lock and pack manifest. Weights are downloaded from Hugging Face and are not included in this tester bundle.

Save this as `try-image.mjs` inside `tester-project`:

```js
import { Lars } from "@moedex/lars";

const lars = await Lars.open({ pack: "./models/lars-vision-alpha" });
try {
  const image = await Lars.imageFile(process.argv[2]);
  const damaged = lars.noul("damaged", "The package has visible damage");
  const decision = await damaged.decide(image);
  console.log(decision);
} finally {
  lars.close();
}
```

Run `node try-image.mjs /path/to/your/photo.jpg`. Question wording is part of your application; change the claim to suit your image. An allowed structured answer does not establish that it is correct.

For TypeScript, declare `lars.noul<ImageInput>(...)` after importing `type ImageInput`. Choices use the normal `lars.choice` API with a state selector returning the image. `image.withContext({ ... })` adds explicit context. Multiple predicates use `lars.ask(image, { predicateA, predicateB })`, returning decision objects. Image state selectors and request options must agree within a grouped request.

## Run the diagnostic suite

From the new project, run the evaluator copied into your project so it can resolve the installed package:

```sh
cp ../evaluate.mjs ./evaluate.mjs
cp -R ../fixtures ./fixtures
node evaluate.mjs --pack ./models/lars-vision-alpha --out ./my-result.json
```

The suite contains 50 labeled questions from 20 source groups, including three public photos, JPEG and transparency variants, generated screenshots/documents, spatial/counting tasks, blank inputs, and text that conflicts with the image. Photo provenance and fixture hashes are in `fixtures/SOURCES.json`. No additional image download is needed.

The evaluator also replaces images with blank/shuffled inputs and attempts reversed choice order. It records effective option order: four numeric-key cases retain JavaScript's numeric property ordering, leaving 21 actually reordered controls. Blank/shuffled results measure dependence on the original pixels; their original targets are not ground truth for the replacement images. The JSON records every question, expected answer, distribution, timing, model identity, fixture hash, and host details. It sends no results or images anywhere.

The frozen reference run answered 49/50 correctly on this small development set, with no runtime errors. It failed the boolean claim that a red square is red when text above the square instructed it to answer blue. Its raw confidence in that wrong answer was 0.8486. This observed failure is retained in the set. The full reference results are included; the set is public, correlated, partly synthetic, and not a held-out benchmark or a certificate of production accuracy. Prompts and weights were not changed to fix this failure.

## What to test and report

Try representative images from the application you would build: several clear positives, clear negatives, ambiguous cases, and hard near-misses. Keep original images or capture sessions grouped when measuring results. Include small text, rotation, compression, absent objects, and images containing instructions. Report wrong answers even when confidence is high. Do not infer a reliable rejection threshold from this alpha's raw probabilities.

Use `FEEDBACK.md` as a local template. Useful feedback includes the smallest reproducible question/options, expected and actual answer, JSON distribution, image dimensions/format, and environment. Share images only when you have decided they are appropriate to share; feedback is never uploaded automatically.

## Current limits

- `calibrated: false` means the probabilities have no image calibration or certified error budget. Image predicate calibration, image `explain`, `score`, `multi`, and hidden-state heads are unavailable.
- Only one still JPEG/PNG per request. Input limits are 20 MiB encoded and 40 million decoded pixels; animated PNG is rejected.
- Transparency is composited on white. EXIF orientation and ICC profiles are not applied. Normalize images beforehand when needed.
- Text visible in an image can steer an answer incorrectly, as demonstrated by the retained red-square failure.
- Inference is serial and currently repeats visual encoding per question/option ordering. CPU execution can be substantially slower. Text latency figures do not apply.
- Cancellation affects queued work. Inference already running finishes normally.
- Model downloads require internet access; inference after setup is local. The shipped package has no inference service or automatic feedback sender.

## Rebuild and independently verify

The source tarball includes the modified SDK and its vendored llama.cpp checkout, plus `ALPHA-SOURCE.json`. The source tree hash is authoritative; the recorded base commit alone does not include the uncommitted image implementation. No source checkout, Git metadata, model weights, or build cache is required inside the archive.

To build manually, extract the source archive, install Rust, CMake, Xcode Command Line Tools, and Node 24, then run from its root:

```sh
cd ts
npx --yes pnpm@11.0.0 install --frozen-lockfile
cd ..
node ts/scripts/build-native.mjs
npm run build --prefix ts
```

The recorded Rust toolchain for this pass is 1.99.0. Earlier toolchains have not been revalidated for this alpha. The full pinned llama.cpp revision comes from `llama-cpp-revision.txt`, so rebuilding from a source archive and a Git checkout uses the same runtime identity.

The bundle's end-to-end verifier installs the tarball offline in a fresh consumer, runs the complete suite, extracts source without Git/build artifacts, restores locked JavaScript dependencies, compiles a fresh native addon, and reruns the 50 primary questions:

```sh
python3 verify-alpha.py --bundle . --pack /absolute/path/to/models/lars-vision-alpha
```

This verification requires the source build tools and registry/network access for dependencies. It keeps a build log in a new temporary directory and writes `verification.json`, installed/source result files, and refreshed checksums into the bundle only after both paths pass. `verification.json` identifies the package and source archive that were tested. Nothing is published by these scripts.
