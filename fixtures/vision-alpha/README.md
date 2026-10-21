# Image alpha development fixtures

This diagnostic suite has 50 labeled questions over 20 source groups and 22 encoded image files. It is a public development set, not a held-out benchmark, calibration set, or deployment accuracy guarantee. Several questions share images, and JPEG/transparent variants share source groups.

The three photographs come from the scikit-image v0.19.3 data directory. Chelsea is CC0 by Stefan van der Walt; coffee is CC0 by Rachel Michetti; the astronaut photograph is a NASA public-domain image. The source project documents these terms in [its data reference](https://scikit-image.org/docs/stable/api/skimage.data.html). `SOURCES.json` records URLs, authors, terms, transformations, and SHA-256 hashes. The derived coffee JPEG retains its original photo's CC0 provenance.

The other fixtures are controlled drawings, faux screenshots, and faux documents created by this repository's generator under its MIT license. Regenerate them with Pillow 11.3.0 using `tools/generate_vision_alpha_fixtures.py`; it preserves the downloaded photos. The default Pillow font is embedded by the installed library and no external font file is required. Label verification includes visual inspection of the generated assets and direct inspection of their source values.

Categories are photos, screenshots, documents, spatial/counting questions, and robustness. Boolean positives and negatives, unknown options, a contradictory uploader description, blank/transparency cases, and visible conflicting instructions are included. The first frozen run retained one failure in `instruction-red-45`; labels, prompts, and weights were not altered to remove it.

The evaluator's blank/shuffled controls preserve the original target only to measure dependence on image evidence. Those targets are not relabeled ground truth for the replacement images. Reversed-option controls preserve the actual question/image/answer. Use the complete per-case output, source groups, and category breakdowns when interpreting results.
