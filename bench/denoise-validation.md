# Adaptive Denoise shader validation

Recorded 8 September 2026. This change addresses chroma control coupling and color-boundary protection. Noise-estimator units, the sigma-to-slider mapping, Glare's numerical budget, and the luminance range curve are unchanged.

## Control and filter contract

Strength and Detail affect luminance only. Chroma blend, tap radius, spatial sigma, and range tolerance derive from Chroma Smoothing, with image scale determining sample spacing. Review covers parameter construction, original-sample guide construction, weights, normalization, blending, and reconstruction: no chroma filtering parameter or guide consumes Strength, Detail, or filtered luminance.

For `c = Chroma Smoothing / 100`, the selected curves are:

- Blend: `c`; radius: `floor(2 + 2c)`; spatial sigma: `1 + 3c` in tap units.
- Spacing: `max(1, round(min_dimension / 1080))`.
- Chroma tolerance: `t = (4 + 10c) / 255`, expressed as Euclidean encoded Cb/Cr distance, not the RMS chroma metric used to classify fixtures.
- Chroma range weight: `exp(-(distance / t)^8)`.
- A separable 3x3 component median, at the same spacing, stabilizes the guide. Chroma averaging still uses the original linear samples. Its maximum input footprint is `(radius + 1) * spacing` per axis.
- Encoded luma range weighting uses sigma `2t` on the stabilized guide. For saturated colors, it blends toward sigma `0.4t` on original-sample encoded luminance. The blend is `smoothstep(0.04, 0.16, length(center_guide.CbCr))`. This input-color guard prevents retaining noisy luminance while smoothing chroma from clipping low channels and biasing saturated colors. All tolerance curves remain independent of Strength and Detail.

The guide extends the sRGB linear toe through negative values and extends the power branch above 1. Its power argument is nonnegative even when both branches evaluate. Constructing the guide does not clip the averaged image. The existing final nonnegative RGB reconstruction remains in place.

The selected tolerance and sharp cutoff use the measured boundary margin. A broader maximum tolerance failed the shadow-boundary bias limit; a narrow guide passed synthetic boundaries but left substantially more chroma noise in the supplied extreme-ISO RAWs. The final guide improves that tradeoff without changing acceptance thresholds. It still retains more color variation on those RAWs than the former unconditional chroma average; see the image review below.

## Automated coverage and metrics

`src-tauri/src/gpu_processing/denoise_tests.rs` appends a test entry point to the complete production WGSL and invokes `apply_denoise`. It injects scale and returns floating-point linear results. CPU code constructs fixtures and computes metrics; it does not mirror the filter. Separate tests exercise final 8-bit full-pipeline rendering, including the actual RAW rendering path.

The 576-case matrix crosses:

- Target encoded mean luminance 30/128/200; both equal-linear-luminance and unequal-luminance color pairs.
- Encoded non-RAW input and genuinely linear RAW input.
- Spacing 1/2/4 and Chroma 50/100, with Strength 0 to isolate chroma behavior.
- Near-floor separation 12.05; cases near the six-sigma class; wider color boundaries; subtle separation 4 diagnostics.
- Chroma-only noise holding linear luminance constant, and mixed noise injected into encoded channels before conversion to the declared input encoding.

Every fixture has clean/noisy companions with deterministic seeded noise, asserted headroom, and logged actual side luminances, separation, amplitude, and realized side noise. Tests reject clipping and empty measurement regions. The matrix reports 432 strong-boundary cases and 144 subtle diagnostics. Guide precision is checked separately at black, negative values, and linear highlights through 8.

In encoded 0–255 units:

```text
Y  = 0.2126 R + 0.7152 G + 0.0722 B
Cb = 0.565 (B - Y)
Cr = 0.713 (R - Y)
delta_C = sqrt((delta_Cb^2 + delta_Cr^2) / 2)
sigma_C = sqrt((variance(error_Cb) + variance(error_Cr)) / 2)
strong = delta_C >= max(12, 6 * max(sigma_left, sigma_right))
```

Clean companions inherit their noisy companion's classification. In the 32-column boundary strip, require maximum clean RGB error <=3 and maximum absolute per-column mean noisy RGB bias <=3. Flat interiors exclude 24 pixels from image and color boundaries, beyond the largest sampled footprint. Additional anti-inertness checks require chroma-only noise sigma ratios <0.8 at Chroma 50 and <0.5 at Chroma 100. Mixed-noise interiors are reported separately: unchanged linear luminance noise can reappear as encoded chroma variation in saturated colors. Both noise types retain the same strong-boundary error/bias gates.

Measured on NVIDIA GB10, Vulkan, driver 580.142, Rust 1.96.1, optimized test profile:

| Gate                                                        | Measured result                                   |
| ----------------------------------------------------------- | ------------------------------------------------- |
| Strong-boundary maximum clean RGB error                     | 0.190 / 255                                       |
| Strong-boundary maximum systematic RGB bias                 | 2.316 / 255                                       |
| Worst strong-fixture chroma-only sigma ratio, C50           | 0.554                                             |
| Worst strong-fixture chroma-only sigma ratio, C100          | 0.121                                             |
| Existing 512-pixel full-setting luminance sigma             | 6.94 → 1.39; gate <40%                            |
| Existing legacy `B - Y` sigma                               | 11.04 → 1.39; gate <25%                           |
| Existing Strength-60 luminance case                         | Passes <70% gate                                  |
| Strength 30/70/100, C60, Detail50; Detail 0/100 variants    | Float difference <=2e-6; encoded output <=1 level |
| Full-pipeline RAW/non-RAW independence                      | <=1 output level                                  |
| Disabled and both strengths zero                            | Exact identity against matching pipeline baseline |
| Real dimensions 512/2160/4320, clean tile boundaries        | <=1 output level                                  |
| 4320 RAW nonuniform render vs production function at step 4 | <=1 output level across tile seams                |

The nonuniform step-4 comparison uses a periodic scene with a halo and confirms that wrong scale produces a material difference; a uniform-image identity test cannot establish scale plumbing. The reference has an additional f16 upload before the same RAW renderer. RAW/non-RAW final bytes are not required to match. Equivalent linear samples are compared before their different downstream rendering paths.

The unchanged luminance-edge diagnostic, using a 20-level encoded gray step and Strength100/Detail50/Chroma0, retained approximately 15.42%, 66.05%, and 81.26% of adjacent-column contrast at target luminances 30, 128, and 200. These measurements demonstrate brightness dependence; they do not reproduce or validate the previously quoted 90%/54% figures.

## Supplied RAW review

The two user-provided CR3 files remain locally ignored in `docs-untracked/`. No image data or generated renders are committed.

| Sample         | Exposure metadata                                                                | SHA-256                                                            |
| -------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `3E9A7623.CR3` | 1 second, ISO 51200                                                              | `dab4c0821c5c3b42f405d5f2d0fc0924ab1030aa58dd5d136a34375b658deb72` |
| `3E9A7624.CR3` | 0.5 seconds; user reports approximately ISO 100000; current reader returns 65535 | `cde3c9fa58e00e07e2890a5506e5105facd5cfc74753c8be9052a14eb88bfa4f` |

Both decode to 8192x5464 through the production RAW developer, with no embedded-preview fallback. Review preprocessing matches loader defaults: highlight compression 2.5, linear mode auto, color NR 0.5 (inverse sigma 14), sharpening 0.35. Both baseline and changed renders use the same developed pixels and RAW pipeline. This is a controlled default-settings comparison, not a claim to reproduce the user's private editor settings.

Generated full-resolution PNG renders, 100% crops, and 1920x1281 previews at Strength/Chroma 60/60 and 100/100, Detail50. At 100%, the final filter keeps map color regions more distinct, but retains more colored noise, particularly in the higher-ISO frame. Very fine map text remains noise-obscured; dark fabric and wall luminance texture remain broadly similar. No new tile seams were apparent. These are observations without a clean real-scene reference, not measured detail-recovery claims.

For an approximately flat wall crop `(1792,1110)` of size 512x512, encoded chroma variation was:

| Sample / settings | Previous shader | Corrected shader |
| ----------------- | --------------- | ---------------- |
| 7623, 60/60       | 5.100           | 5.555            |
| 7623, 100/100     | 2.051           | 2.767            |
| 7624, 60/60       | 7.408           | 8.999            |
| 7624, 100/100     | 3.372           | 5.779            |

This variation includes any real surface variation; it is not a ground-truth noise sigma. It makes the remaining smoothing-versus-edge tradeoff visible instead of concealing it behind the synthetic tests. The 100000-ISO sample remains demanding for this local filter.

Preview/export differences already exist because previews downscale before denoising. For the higher-ISO sample, mean RGB differences between the 1920 preview and a resized full render were 11.645→11.799 levels at 60/60 and 11.034→11.084 at 100/100 (old→new). Those comparisons include resizing/render-order differences and are diagnostics, not byte-parity requirements. Full-resolution PNGs exercise export-size rendering; interactive application and export-dialog smoke testing is separate.

## Performance

One warmup followed by five measurements of the production denoise function with Strength100/Detail50/Chroma100, using the same deterministic scene and GPU. Timestamp queries measure the compute pass only; upload/readback is excluded. Baseline is the unmodified shader from `e9753cd8`.

| Input      | Minimum dimension | Previous GPU median | Corrected GPU median |
| ---------- | ----------------- | ------------------- | -------------------- |
| Encoded    | 512               | 0.236 ms            | 1.174 ms             |
| Encoded    | 2160              | 3.909 ms            | 22.433 ms            |
| Encoded    | 4320              | 17.539 ms           | 87.910 ms            |
| RAW linear | 512               | 0.201 ms            | 0.918 ms             |
| RAW linear | 2160              | 3.359 ms            | 17.661 ms            |
| RAW linear | 4320              | 15.085 ms           | 69.157 ms            |

The stabilized guide is substantially more expensive than unconditional chroma averaging. On the second supplied 8192x5464 RAW, single full-pipeline render-plus-readback measurements at 100/100 were approximately 146 ms before and 228 ms after. Those single-frame timings are illustrative, unlike the repeated GPU-pass medians. No performance ceiling was specified; these costs are an explicit limitation of this implementation.

## Reproduction and check status

From the repository root:

```sh
REQUIRE_GPU_TESTS=1 cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib test_gpu_denoise -- --nocapture --test-threads=1
REQUIRE_GPU_TESTS=1 cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib -- --nocapture --test-threads=1
REQUIRE_GPU_TESTS=1 cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib benchmark_gpu_denoise -- --ignored --nocapture --test-threads=1
```

`REQUIRE_GPU_TESTS` makes adapter absence an error for the shared GPU helper. The ordinary library run completed with 128 passing tests and 6 explicitly ignored tests. The optional benchmark and real-RAW review tests were also run explicitly and passed. The 576 matrix measurements are cases inside one test, not 576 separate Rust test functions.

For a before/after benchmark, export the earlier unmodified production shader to a temporary file with `git show e9753cd8:src-tauri/src/shaders/shader.wgsl`, then set `DENOISE_BENCH_SHADER` to its path. The ignored real-RAW review test takes `DENOISE_RAW_IMAGES` (platform path-list separator), `DENOISE_REVIEW_BASELINE`, and `DENOISE_REVIEW_OUT`. It writes only to the requested output directory. Current-session review artifacts are in `/tmp/denoise-raw-final/` and can be regenerated with those inputs.

Whole-repository formatting and strict Clippy checks currently fail on pre-existing code. Clippy reported 55 library / 104 test-target errors; none referenced the new denoise test module. Existing examples include unused Glare-test mutability, unused Rapid deconvolution members, and established argument-count/style warnings. These failures were recorded rather than mixing unrelated cleanup into this slice. The new test module is rustfmt-formatted; `git diff --check` is clean.
