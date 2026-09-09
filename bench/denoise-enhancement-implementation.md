# Denoise enhancement implementation outcome

9 September 2026. **Neither proposed enhancement qualifies under the approved noise/detail gates.** Slice 1 is implemented; slices 2 and 3 completed bounded refinement and both are deferred. The approved fallback retains the production renderer, so slice 4's shader integration and recalibration were not entered.

No production WGSL, Estimate tables, source-noise measurement, settings schema, Glare, or editor behavior changed. The work adds a fixed shader reference, maintained GPU quality checks, explicit candidate measurements, and failure evidence. No acceptance tolerance was widened.

## Slice 1 — Measurement and maintained coverage

The frozen reference is `fixtures/shader-a9c9d2ec.wgsl`, SHA-256 `2eed7222e71a6da488ffd61ea369ba48077795fd0cc8e153fd780b23d8ca9a7b`. It is checked into test support and hashed by a normal test; running the suite does not require Git history. The maintained production comparison checks the actual production shader against this reference.

The actual combination is named `wide-chroma-guided-edges`; it is distinct from the earlier encoded-guide experiment called `combined`.

NVIDIA GB10 / Vulkan / driver 580.142, optimized Rust 1.96.1 tests. Strength 100, Detail 50, Chroma 100. Each GPU result is the median of five measurements after one warmup, with native spacing 1 at 1080² and 4 at 4320². Upload/readback is excluded.

| Candidate | 1080² filter | Relative to baseline | 4320² filter | Relative to baseline |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 4.04 ms | 1.00× | 69.74 ms | 1.00× |
| Wider chroma | 8.00 ms | 1.98× | 136.42 ms | 1.96× |
| Stabilized edges | 4.52 ms | 1.12× | 75.92 ms | 1.09× |
| Actual combination | 8.49 ms | 2.10× | 142.23 ms | 2.04× |

The actual combination passes the proposed 2.5× ceiling. The earlier additive estimate is superseded by these measurements.

Full production RAW render/readback times, measured once per candidate after identical default RAW development, were approximately 228/360/235/364 ms on `3E9A7623` and 225/355/230/373 ms on `3E9A7624` (baseline/wider/edges/actual combination). These single-render diagnostics are not repeated GPU-pass medians. Both files render at 8192 × 5464, native spacing 5. Crops are taken after the full render.

The color-boundary fixture now has width 256 and margin 40, with spacings 1/2/4/5. All four initial variants pass its 768 cases, including 576 strong-boundary cases each. **The existing full-pipeline integration comparisons retain their original 24-pixel margins.** Enlarging boundary fixtures did not shrink those integration checks' coverage.

The promoted flat fixtures use 53,504 and 107,008 measured pixels, an exact doubling of interior area. Across all nine tested variants, paired 1% regression decisions were stable under that doubling. Each candidate has 96 flat-noise cases (two seeds, three brightnesses, two correlation grids, two encodings, two spacings, two areas), and 648 clean-detail cases. The latter cover Detail 0/50/100, brightness 30/128/200, linear/encoded input, spacings 1/5, luminance edges, Y/Cb/Cr textures at periods 4/8/16, Cb/Cr lines of width 1/2/4, and color gradients. Clean error is evaluated per required channel and fixture; case counts do not represent distinct Rust test functions.

## Slices 2 and 3 — Bounded refinement

Noise columns below are median reductions in the corresponding approved flat-noise metric relative to baseline. Clean-detail increase is the worst per-case increase in relevant-channel MSE, in squared encoded 0–255 levels. The allowed increase is **0.25**. Failure counts include repeated settings/encodings, not distinct visual defects.

| Candidate | Relevant noise reduction | Worst clean MSE increase | Failed gates/cases | Decision |
| --- | ---: | ---: | ---: | --- |
| Chroma radius/sigma 6 | 24.3% chroma | 5.687 | 216 | Defer |
| Chroma radius/sigma 5 | 13.8% chroma | 5.008 | 138 | Defer |
| Chroma radius 5, sigma 4 | 10.5% chroma | 3.861 | 108 | Defer |
| Stabilized edges, unblended | 21.6% luminance | 1.201 | 8 | Defer |
| Edge blend 12.5% stabilized | 3.6% luminance | 0.156 | 1 | Insufficient noise improvement |
| Edge blend 25% stabilized | 7.1% luminance | 0.313 | 5 | Noise and detail fail |
| Edge blend 50% stabilized | 13.6% luminance | 0.622 | 4 | Detail fails |
| Original actual combination | 24.3% chroma / 21.6% luminance | 5.687 | 224 | Defer |

The color filters pass strong-boundary tests yet fail fine color-texture tests. For example, radius 5 / sigma 4 increases Cb texture error at period 8, brightness 30, Detail 50, spacing 1 from about 17.55 to 20.38. This confirms why the new detail coverage matters.

The edge failure survives the expanded fixtures: period-8 clean luminance texture at brightness 200, Detail 50 rises from about 9.55 to 10.75 with unblended stabilization. The tested blends use fixed weights on original/stabilized Sobel magnitudes before the existing edge-protection curve. They do not add a second Detail-dependent control mapping. The mildest blend passes the detail limit but falls short of the required 10% noise improvement; stronger blends fail detail.

No increasingly complex guide, new pass, or altered acceptance limit was introduced to force a pass. These results reject the tested bounded variants, not every conceivable non-AI denoiser. Further algorithm design would be a separate scope decision.

## Evidence and verification

- [Summary, all failures, timings, and RAW statistics](data/denoise-qualification-summary.json).
- [864 paired flat-region measurements](data/denoise-qualification-noise.csv).
- [5,832 clean-detail measurements](data/denoise-qualification-detail.csv).
- [Local full-resolution comparison gallery](../docs-untracked/denoise-enhancements/implementation/index.html), with the actual combination. Images remain ignored and this link is local to the workspace.

The ordinary library suite passed **155 tests**, with **17 optional tests ignored**, in 31.62 seconds. Added ordinary checks cover residual metrics (including constant bias versus noise variance), production/reference behavior on the promoted fixtures, and sensitivity to known detail regressions. Candidate experiments record failed qualification rather than interpreting a completed experiment as a passing candidate.

The four-way timing, initial quality, expanded boundary, RAW review, and bounded-refinement experiments were all invoked explicitly. The earlier intermittent step-4 GPU mismatch did not recur in this run; its root cause remains unestablished, and the prior evidence remains in the investigation report. Modified Rust files pass rustfmt checks; whitespace and viewer JavaScript syntax checks pass. All 24 selectable implementation crop files exist.

There is no selected replacement shader, so fitting new Estimate tables, fresh-seed release validation of a replacement, frontend build checks, and native-app acceptance of a replacement are not claimed. The current shader and its calibrated tables remain together unchanged. The recorded seeds (173/619) are qualification/tuning seeds, not independent release validation.

## Reproduction

From the repository root (set output paths as appropriate):

```sh
REQUIRE_GPU_TESTS=1 DENOISE_QUALIFICATION_OUT=/tmp/denoise-qualification cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib benchmark_denoise_actual_combination -- --ignored --nocapture --test-threads=1
REQUIRE_GPU_TESTS=1 DENOISE_QUALIFICATION_OUT=/tmp/denoise-qualification cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib qualify_initial_denoise_candidates -- --ignored --nocapture --test-threads=1
REQUIRE_GPU_TESTS=1 DENOISE_QUALIFICATION_OUT=/tmp/denoise-qualification cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib qualify_denoise_candidate_boundaries -- --ignored --nocapture --test-threads=1
REQUIRE_GPU_TESTS=1 DENOISE_QUALIFICATION_OUT=/tmp/denoise-qualification cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib qualify_denoise_refinements -- --ignored --nocapture --test-threads=1
REQUIRE_GPU_TESTS=1 DENOISE_IMPLEMENTATION_REVIEW=1 DENOISE_ENHANCEMENT_OUT=/tmp/denoise-qualification/raw DENOISE_RAW_IMAGES="$PWD/docs-untracked/3E9A7623.CR3:$PWD/docs-untracked/3E9A7624.CR3" cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib review_live_denoise_enhancements_raw -- --ignored --nocapture --test-threads=1
REQUIRE_GPU_TESTS=1 cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib -- --test-threads=1
```

Generated JSON reports preserve individual results; the checked-in CSVs flatten their `noise` and `detail` arrays. GPU timing runs should be isolated from other GPU work. Source RAWs are read only; output paths contain derived images and reports.
