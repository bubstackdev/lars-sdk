# Image answers alpha handoff

The tester artifact is an unpublished macOS arm64 Node package, a complete source archive, pinned model setup, the labeled development suite, and a feedback template. [START-HERE](../packaging/vision-alpha/START-HERE.md) is written for testers. Rust and C# image APIs remain in source; this artifact does not claim binary support for other platforms or distribute a NuGet alpha.

## Reproduce the model and evaluation

The alpha uses Qwen3-VL-2B-Instruct Q4_K_M and its F16 projector at the immutable repository revision in [model.lock.json](../packaging/vision-alpha/model.lock.json). Both files are checked by size and SHA-256. Setup writes a format 2 pack only after verification. To reuse prior downloads, pass `--artifacts-dir` containing `model.gguf` and `projector.gguf` or their upstream filenames.

```sh
python3 packaging/vision-alpha/prepare-model.py --output ./models/lars-vision-alpha
node ts/scripts/build-native.mjs
npm run build --prefix ts
node --test packaging/vision-alpha/evaluate.test.mjs
node packaging/vision-alpha/evaluate.mjs \
  --pack ./models/lars-vision-alpha \
  --suite fixtures/vision-alpha/suite.json \
  --sdk ts/dist/index.js \
  --out ./dist/image-evaluation.json
```

The generator, `tools/generate_vision_alpha_fixtures.py`, requires Pillow 11.3.0 only for regenerating the controlled assets. All fixture images and their checksums/provenance are included in source and the bundle. The test set is frozen as `vision-alpha-development-v1`; record a new suite version if changing labels or images. Keep its source groups intact when making splits or uncertainty estimates.

## Reference findings

The first frozen Metal run completed all 50 primary questions with no runtime errors and matched 49 labels. Documents, photos, screenshots, and spatial/counting questions matched all labels in this small set. Robustness matched 7/8. `instruction-red-45` was wrong: text telling the model to answer blue overrode a boolean question about a visible red square. Its raw confidence was 0.8486. No prompt, label, or model tuning was performed after observing this failure.

Blank replacement changed 68% of predictions and shuffled images changed 88%. All 25 choice reruns kept their original answer; 21 actually reversed option order, while four numeric-key cases retained JavaScript's numeric property ordering. The evaluator records the effective order and distinguishes those four cases. These are diagnostics of dependence and order sensitivity, not calibrated confidence or generalization guarantees. Three source photos, controlled generated images, repeated questions, and related variants are insufficient to establish production accuracy. Full per-case evidence and execution settings are in [reference-results.json](../packaging/vision-alpha/reference-results.json).

The recorded host is an M5 Max with 64 GB RAM, Node 24.21.0, Darwin 27.0.0. The initial intact-question p50/p95 were approximately 158/213 ms; those measurements are indicative for this host/run, exclude model load time, and do not describe CPU speed or another user's workload.

## Build and verify the shareable artifact

```sh
python3 packaging/vision-alpha/build-alpha.py
python3 dist/vision-alpha-0.1.0-image-alpha.1/verify-alpha.py \
  --bundle dist/vision-alpha-0.1.0-image-alpha.1 \
  --pack ./models/lars-vision-alpha
```

The builder refuses to overwrite an existing output directory. It builds an optimized native addon and TypeScript output, includes dependency notices, and packages a standalone npm tarball with `os: darwin` and `cpu: arm64`. It creates no registry release, Git tag, or network publication.

The source archive includes the working implementation and vendored llama.cpp sources without build artifacts or Git metadata. `build.json` and `ALPHA-SOURCE.json` identify its source tree independently of the base Git commit. `llama-cpp-revision.txt` fixes runtime identity across shallow/full checkouts and source archives, while a Git checkout additionally checks that its submodule matches the stamp.

The verifier checks bundle hashes, installs the standalone tarball offline in a new consumer, runs all primary/control cases, extracts clean source, installs locked JavaScript dependencies, builds a new addon, and reruns the primary cases. It refuses success if either path errors or their primary predictions disagree. After a pass, it writes `verification.json`, complete installed/source result files, and refreshed `SHA256SUMS`. Source dependency restoration may use normal registry caches and network; compilation does not reuse a previous target directory.

Share the whole verified directory or an archive containing it. The model is fetched separately from its pinned upstream revision. Include the known failure and uncalibrated status when inviting testers; ask them to try representative application images and return the local [feedback template](../packaging/vision-alpha/FEEDBACK.md) or a minimal reproduction they choose to share.
