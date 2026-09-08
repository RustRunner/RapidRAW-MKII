# Estimator variance revision — 8 September 2026

**The original domain-accuracy failure is fixed; production calibration remains gated by clipped-image eligibility.** Version 2 passes the 576-comparison grid that failed under Gaussian-normalized MAD. It also passes 504 comparisons on additional high-noise encoded-input fixtures, including saturated colors. Both supplied RAWs fail the existing 1% known-clipping qualification once float zero endpoints are recorded correctly. No fitted suggestions or production analysis path are enabled.

## Measurement revision

Both linear and encoded measurements now use the centered second moment of residuals after the existing robust plane fit. Linear measurement also needs this change: noise injected before decoding is not generally Gaussian after decoding. MAD remains explicitly named as `linear_mad_scale` / `encoded_mad_scale`, used for the structure diagnostic and characterized quantization flag; it is no longer mislabeled as marginal standard deviation.

Brightness is encoded Y of the spatial mean linear RGB. This estimates the local center before applying the transfer curve, reducing high-noise brightness-bucket jitter. It is not the arithmetic mean of noisy encoded pixels, nor an estimate of the hidden unclipped clean scene. Measurement version is 2.

The deterministic initial grid contains at most 16×12 native 64×64 patches. Up to 64 additional patches explore neighbors of underrepresented brightness regions, bounded by 256 total. Extra patches cannot overlap any existing patch. Sampling origins and counts are reported, and populated bins still require four qualified independent patches. Small images use their available nonoverlapping grid; insufficient bins are not silently dropped. A regression verifies deterministic independent re-sampling of a small bright region, and a separate case still rejects a region too small to provide four patches.

The module remains `cfg(test)` only. Production cache/commands, ownership, Glare's budget, shaders, saved settings, and TypeScript contracts have no changes in this revision.

## Passing reference evidence

- **576 / 576 comparisons:** the same six brightness anchors, four noise levels, gray/independent RGB, two seeds, two domains and three components from the original gate. All 96 scenes now provide qualified coverage. [Complete CSV](data/estimator-variance-domain-grid.csv); the [old failing CSV](data/estimator-mad-domain-grid.csv) is retained for comparison.
- **504 / 504 comparisons:** encoded Gaussian noise at sigma 0.008/0.032/0.064, seven neutral/saturated RGB bases, seeds 19763/88069, actual clean/noisy transformations, both domains and three components. These use the original 10% or 1e-4 absolute limit, with no widened tolerance.
- The declared separable-box correlation fixtures through radius 8 pass the original 20% or 2e-4 bound. Maximum observed relative error is 18.61%. This remains a measured limitation of finite patches and plane removal, not evidence of accuracy for arbitrary RAW covariance.
- Existing unequal/cross-channel covariance, 8/16-bit precision, half-float, negative/headroom, nonfinite/overflow, texture, and legacy-compatibility tests pass. New tests enforce independent adaptive coverage and reject black-clipped float noise.

The original domain-grid test is now part of the ordinary suite, not ignored. Only the local RAW audit remains optional among this module's tests. Fourteen ordinary measurement tests pass. Fully textured developed scenes and censored-noise qualification are not established by this result.

## Corrected clipping audit

The original prototype checked integer endpoints but omitted float zeros. This was an incomplete implementation of the agreed clipping rule. The revised audit records a pixel if any channel is exactly zero; for integer inputs it also records the maximum code. Float 1.0 is not assumed to be sensor white, and valid negative/headroom values remain valid. The audit is conservative about endpoint plateaus; it does not infer sensor-electron clipping.

The actual source path confirms lower clipping: `develop_raw_image` clamps channels at zero, and `remove_raw_artifacts_and_enhance` / its sharpening step clamp reconstructed RGB. The default preprocessing audit uses color-NR inverse sigma 14.0 and sharpening 0.35, matching the loader defaults. Simply disabling these steps does not recover an unclipped source.

| Variant | 3E9A7623: endpoint pixels | 3E9A7624: endpoint pixels |
| --- | ---: | ---: |
| Default preprocessing | 16.72% | 25.24% |
| Develop only | 51.09% | 69.73% |
| Color NR only | 9.68% | 12.29% |
| Sharpening only | 38.54% | 49.33% |
| Encoded 8-bit derivative | 17.43% | 26.06% |
| Encoded 16-bit derivative | 17.14% | 25.86% |
| JPEG quality 90 diagnostic | 9.40% | 12.89% |

Fractions describe the sampled pixels in each variant, not a full-image census. Adaptive sample coordinates can differ by variant. The first default RAW's per-patch range is 1.20–40.50%; the second is 6.52–39.28%. Thus all 256 sampled default patches in each image exceed 1%. Some NR-only/JPEG patches qualify in the first RAW, but missing qualified brightness coverage still makes every audited variant insufficient. JPEG is a separate lossy robustness diagnostic and cannot establish lossless equivalence.

[Numeric variant report](data/estimator-variance-raw-variants.json) includes hashes, dimensions, counts, fractions, timings, legacy estimates, and any qualified bins. The complete local JSON additionally includes sample origins. Encoded 8/16-bit derivatives were constructed in memory from the same developed source with explicit transfer/quantization; JPEG was encoded/decoded in memory. No source or derivative image was written to the repository or tracked.

## Remaining eligibility decision

The original rule says patches above 1% known clipping do not enter the initial qualified set. This revision preserves that rule. It fixes the sampling mechanism but cannot certify either default RAW as unclipped. A [concrete extension](estimator-clipped-noise-design.md) proposes separately validating black-clipped developed residuals with the same accuracy bounds before attempting the original table axes. Its eligibility change requires the user's decision; it is not silently enabled by this experiment.

Slider fitting, native spacing-5 suggested-setting checks, native-app Estimate/history/save/export checks, and final API/cache integration remain downstream. There is no claim that a calibrated suggestion has shipped.

## Commands and regression results

```sh
# Ordinary suite, including the formerly failing domain gate.
REQUIRE_GPU_TESTS=1 cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib -- --test-threads=1

# Export the full passing domain grid (no --ignored flag).
DENOISE_DOMAIN_REPORT=/tmp/calibration-domain-v2.json cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib audit_proposed_domain_accuracy_grid -- --nocapture

# Both RAWs and all seven variants; diagnostic reports, not acceptance assertions.
DENOISE_RAW_VARIANTS=1 DENOISE_RAW_IMAGES="$PWD/docs-untracked/3E9A7623.CR3:$PWD/docs-untracked/3E9A7624.CR3" DENOISE_MEASUREMENT_REPORT=/tmp/calibration-raw-variants-v2.json cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib audit_real_raw_measurements -- --ignored --nocapture

npm run typecheck
npm test
rustfmt +1.96.1 --edition 2021 --check src-tauri/src/noise_analysis.rs
git diff --check
```

The focused measurement run passes 14 tests (one optional RAW audit ignored); the explicit RAW/variant audit completes in 10.93 seconds. TypeScript has zero errors and all 74 frontend tests pass. One parallel full-suite run failed the existing `test_gpu_gated_blurs_bit_exact` sharpness comparison; that unchanged test passes in isolation. The full suite passes **147 tests, with 7 optional tests ignored**, in 24.32 seconds using the serial GPU mode already used in the repository's earlier validation records (NVIDIA GB10 / Vulkan / driver 580.142). No assertion or tolerance in that test was changed.
