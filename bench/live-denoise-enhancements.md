# Live denoiser enhancement investigation

Follow-up: [implementation qualification](denoise-enhancement-implementation.md) found that neither candidate nor the bounded refinements passes all approved detail/noise gates. The production renderer remains unchanged. The measurements below are the earlier exploratory record; current test support includes the subsequent fixture enlargement.

9 September 2026. **Two non-AI candidates are worth refining, but neither is ready to replace the live renderer.** Wider chroma sampling produces a modest reduction in the supplied RAWs' color blotches. Stabilizing only the luminance edge detector reduces grain with a smaller performance cost, but increases clean fine-texture error in some synthetic cases. A more aggressive color guide fails existing color-boundary checks.

Production WGSL, automatic-estimate tables, sliders, and saved-image behavior are unchanged. The prototypes execute only in explicitly ignored GPU experiments. This investigation does not implement AI processing or modify the stock BM3D path.

## What was compared

All variants derive from the production shader at `a9c9d2ec`; checked replacement anchors fail if the source changes. The baseline shader hash, GPU/driver, RAW hashes, individual boundary failures, timings, and RAW measurements are in [summary data](data/live-denoise-enhancements-summary.json). The [864 synthetic cases](data/live-denoise-enhancements.csv) include unsuccessful and mixed results.

| Variant | Experimental change | Existing color-boundary failures | Median filter time, 1080 × 1080 |
| --- | --- | ---: | ---: |
| Baseline | Current live denoiser | 0 | 4.07 ms |
| Wider color filter | Maximum chroma radius/sigma 4 → 6; original range guide | 0 | 8.16 ms |
| Stabilized edge detector | Sobel reads the existing 3 × 3 RGB median; retain linear contrast and original bilateral filter | 0 | 4.60 ms |
| Encoded luminance guide | Median-guided Sobel and bilateral comparisons in encoded contrast | 0 | 11.94 ms |
| Encoded guide + wider color | Both changes above | 0 | 14.65 ms |
| Wider, stronger color guide | Wider filter plus separable 5 × 5 median guide | **34** | **77.72 ms** |

The encoded-guide variant deliberately explores a different contrast domain as well as guide stabilization. Its result must not be attributed solely to noise-resistant edge detection; the separate stabilized-edge candidate isolates that change.

Timings use NVIDIA GB10, Vulkan, driver 580.142, one warmup and five GPU timestamp measurements per variant at Strength 100 / Detail 50 / Chroma 100, spacing 1. Upload/readback are excluded. This square fixture is not a full-application frame-time measurement or a full-resolution 45 MP benchmark.

## RAW results

Both supplied CR3s were developed at 8192 × 5464 with the same production RAW decoder and defaults used in the earlier review: highlight compression 2.5, automatic linear mode, color NR 0.5 and sharpening 0.35. Each variant runs through the complete production renderer at Strength 100 / Detail 50 / Chroma 100; native chroma spacing is 5. Crops are taken afterward, so a small crop does not accidentally change filtering scale.

The established wall crop is `(1792,1110)`, 512 × 512 pixels. Variation is the centered standard deviation of encoded Y and RMS of centered Cb/Cr variances, in 0–255 units.

| Sample | Baseline color variation | Wider color filter | Change | Baseline luminance variation | Stabilized edge detector | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3E9A7623 | 2.767 | 2.281 | −17.6% | 13.562 | 11.241 | −17.1% |
| 3E9A7624 | 5.779 | 5.219 | −9.7% | 21.029 | 19.029 | −9.5% |

These are **surface-variation measurements, not ground-truth noise estimates or detail-recovery scores**. The photographs have no clean reference. Wider chroma sampling leaves luminance variation nearly unchanged. Its visual improvement is modest: more uniform color, with visible grain and larger mottling still present. The edge-detector candidate smooths speckling in the dark-texture crop, but that observation cannot establish preservation of actual fabric detail.

The stronger 5 × 5 color guide lowers wall color variation to 1.581 and 3.307, approximately 43% below baseline. It is rejected despite that attractive number: 34 existing boundary cases fail and GPU cost is about 19 times baseline. Stronger smoothing alone is not the acceptance criterion.

The encoded luminance guide is also not selected: the first RAW's dark-texture luminance variation rises from 4.423 to 8.766; the second rises from 9.905 to 15.781. It does not provide uniformly better real-image filtering.

A local [comparison gallery](../docs-untracked/denoise-enhancements/index.html) contains baseline/candidate crops for both photos. Open it in a browser at 100% zoom. The photos and generated PNGs remain ignored under `docs-untracked/denoise-enhancements/`; this link is specific to the local workspace. The reusable [viewer template](denoise-enhancements-viewer.html) is tracked and the RAW experiment writes it into its output directory.

## Synthetic checks and limits

Each candidate executes the same 576-case production chroma-boundary matrix, including encoded/linear input, equal-luminance color steps, strong and subtle boundaries, chroma-only and mixed noise, Chroma 50/100, and spacings 1/2/4. Existing strong-boundary limits are unchanged: maximum clean RGB error and systematic noisy bias ≤3 levels, plus useful chroma-noise reduction. Subtle-boundary cases remain diagnostics. The ignored investigation records failures rather than pretending every candidate is acceptable; it requires the production baseline to pass.

The separate exploratory grid has two seeds, three encoded brightnesses (30/128/200), chroma-noise correlation grids of 8/24 pixels, luma/color texture variants, spacings 1/5, and Detail 0/50/100: 144 cases per variant. Fine luminance noise and bilinearly interpolated coarse chroma noise are injected in encoded YCbCr, converted to float linear RGB, and compared with a clean companion. Scene bands contain a flat field, a 20-level luminance step, and an 8-pixel-period sinusoid of amplitude 6. Color-texture variants put the sinusoid in Cb. Metrics exclude 40 pixels from band/image borders to cover the widest candidate's footprint. CSV rows report centered flat noise sigma and flat/edge/texture mean squared errors for both noisy and clean filtered inputs.

The wider filter improves color residuals in this grid, but does not establish universal preservation of small colored features. At Detail 50, stabilized-edge clean luminance-texture MSE increases by up to **12.6%** relative to baseline; the encoded-guide variant increases it by up to **42.5%**. This is a real reason to refine detail protection before deployment, even where noisy-reference MSE improves.

This is exploratory evidence, not a new release calibration. In particular:

- The inherited boundary matrix uses the original 128-pixel width and 24-pixel flat margins. It is suitable for reproducing existing gates; qualifying wider support at native spacing 5 needs larger fixtures and appropriately expanded margins. The new separate blotch grid does use 40-pixel margins.
- Tests of broad real color gradients, thin colored lines, more texture frequencies, and independent held-out scenes are needed before claiming a better noise/detail tradeoff.
- Candidate preview/export parity, tile halos, full-resolution performance, and calibrated Estimate suggestions have not been qualified. The production lookup tables remain valid for the unchanged default shader only.
- Changing filter semantics affects old saved images. A production implementation needs an explicit compatibility decision along with new calibration and native-app checks.

## Recommended next implementation

Develop the wider color filter and stabilized edge detector separately. First improve edge confidence so clean fine texture does not incur the measured extra loss. For color, evaluate multiple spatial scales with explicit thin-color-detail tests; simply making the median guide larger already failed. A shared GPU guide pass is worth profiling because the prototypes repeatedly recompute neighborhood statistics inside every filter tap. Combine changes only after independent quality gates pass, then refit and validate automatic suggestions against that actual renderer.

## Reproduction

From the repository root:

```sh
REQUIRE_GPU_TESTS=1 DENOISE_ENHANCEMENT_OUT=/tmp/denoise-enhancements cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib investigate_live_denoise_enhancements -- --ignored --nocapture --test-threads=1
REQUIRE_GPU_TESTS=1 DENOISE_ENHANCEMENT_OUT=/tmp/denoise-enhancements DENOISE_RAW_IMAGES="$PWD/docs-untracked/3E9A7623.CR3:$PWD/docs-untracked/3E9A7624.CR3" cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib review_live_denoise_enhancements_raw -- --ignored --nocapture --test-threads=1
REQUIRE_GPU_TESTS=1 cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib -- --test-threads=1
```

The first experiment writes candidate WGSL and complete JSON measurements. The second writes full-resolution crops, reduced overviews, RAW crop statistics, and an offline HTML viewer. It does not save over the input RAWs.

## Verification status

Both ignored investigations completed (864 exploratory reference cases, 3,456 inherited boundary cases, and both real RAWs across six variants). Their success means the comparison completed; it does not mean the rejected candidate passed its quality gates. The final ordinary Rust library run passed **152 tests**, with **13 explicitly ignored** tests, including these two experiments.

An earlier complete library run hit an intermittent failure in the unchanged `test_gpu_denoise_step4_matches_production_function` (maximum difference 140/255). The check passed at 1/255 when rerun using the original HEAD test file, and the final full suite passed. The cause has not been established; no production shader or tolerance was changed to obtain a pass. Preserve this observation when investigating GPU stability rather than claiming every run was clean.

The modified Rust test files pass formatting checks, the comparison viewer's JavaScript passes syntax validation, every selectable local crop exists, and `git diff --check` passes.
