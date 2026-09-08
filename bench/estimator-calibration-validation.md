# Historical estimator measurement experiment — 8 September 2026

This records version 1 at `ffd99746`. Its failed grid and RAW reports are retained as evidence. [Version 2](estimator-variance-validation.md) replaces the MAD variance estimate and fixes sparse sampling; the commands and known-failing test status below describe version 1.

**Not accepted for production.** The approved measurement method fails its wider encoded-domain accuracy gate and does not provide sufficient brightness coverage on both supplied RAWs. Per the [design gate](estimator-calibration-design.md), fitting stops here pending a measurement-design revision. No slider tables were fitted or enabled.

## Implementation retained

`src-tauri/src/noise_analysis.rs` and its tests implement the proposed candidate: native 64×64 patches, deterministic grid capped at 256 patches, four-iteration robust planar detrending, Gaussian-normalized residual MAD, linear/extended-sRGB Y/Cb/Cr domains, brightness grouping, and structure/correlation/quantization diagnostics. The lowest structure-score quartile is retained, with at least four qualified patches per populated bin. Candidate structure limits are 0.85 block-mean RMS / marginal scale, 0.8 lag-8 correlation, and rejection for lag-1 below −0.15 or highpass/marginal gain above 1.6. These thresholds are experimental, not validated universal texture detectors.

The module is compiled only under `cfg(test)`. Production cache types, image ownership checks, Denoise/Glare command results, TypeScript types, settings, and shaders are unchanged. The legacy estimator's RGB helper was extracted without changing its arithmetic; comments and the historical encoded-gray test name now identify its actual source-domain semantics. Glare's constant is named `LEGACY_SOURCE_SIGMA_BUDGET` and remains exactly 0.025. No production analysis overhead from the prototype is introduced.

## Explicit accuracy gate — failed

The extended audit injects linear float Gaussian noise at all six proposed brightness anchors, RGB sigma 0.001/0.008/0.032/0.064, gray and independent RGB covariance, and seeds 6113/92821. Clean scenes include the same shallow plane as the ordinary reference fixtures. Actual clean/noisy samples are transformed to extended sRGB, including valid negative values; reference sigma is the centered standard deviation of the realized residuals. No clipping or quantization is applied in this audit.

Of 96 scenes, one fails brightness coverage. The remaining 95 produce 570 domain/component comparisons under the agreed 10% or 1e-4 absolute bound:

| Result | Count |
| --- | ---: |
| Linear comparisons passing | 285 / 285 |
| Encoded comparisons passing | 232 / 285 |
| Encoded comparisons failing | 53 / 285 |
| Scene coverage failures | 1 / 96 |

Worst case: encoded Y at brightness 64/255, injected linear gray sigma 0.064, seed 92821. The candidate reports **0.15787235**, while realized residual sigma is **0.36843297**: **57.15% under-reading**. At brightness 30/255 and sigma 0.008, even an independent-RGB fixture under-reads encoded Y by 15.50%. These are not quantization errors.

The Gaussian MAD conversion is distribution-dependent. Nonlinear encoding, including its negative linear extension, changes the noise distribution. Agreement between two representations of the same source cannot establish agreement with actual transformed residual variance. No tolerance adjustment or per-photo correction factor was applied.

[Complete comparison CSV](data/estimator-mad-domain-grid.csv) includes every comparison and the one coverage failure. `audit_proposed_domain_accuracy_grid` is an explicitly ignored, **known failing** experiment which fails when requested; its ignored status is not release acceptance. It must pass after a revised method before fitting begins.

## Other reference checks — passed within their declared scope

Ten ordinary tests cover linear gray/independent RGB covariance and held-out seeds; transformed samples with noise injected in encoded RGB; unequal/correlated channels; 8/16-bit quantization; half-float round-trip precision; spatial correlation through eight pixels; integer clipping, nonfinite inputs and transfer overflow; checker texture; small images and sparse brightness bins; and unchanged legacy/Glare analysis results.

The short-range correlation fixtures use separable box filters with radii 1/2/4/8 and two seeds, normalized to marginal gray sigma 0.01. Maximum observed under-reading is about 13.32% at radius 8, within the agreed 20% or 2e-4 bound. This does not establish accuracy for arbitrary developed RAW noise.

At low 8-bit noise, MAD locks to discrete residual levels. The sigma-0.002 linear fixture at mean 0.216 is marked quantization-limited; sigma 0.01/0.03 and the tested 16-bit cases pass with a separately measured round-trip precision bound. The candidate two-code-step unresolved flag is characterized for these fixtures only. It is not a completed precision model across brightness/covariance. Half-float checks measure precision independently at linear means 0.013/0.216/0.578/−0.05/2.0.

`is_usable` in this prototype checks coverage/structure, not global accuracy acceptance. A returned bin is not evidence that its encoded scale is calibrated. Known integer endpoints are rejected; unknown RAW sensor clipping/float plateaus are not comprehensively detected. Fully textured scenes, preprocessing variants, saturation, and broader quantization cases still need qualification after the design revision.

## Local RAW audit — diagnostic, not ground truth

Both ignored CR3 files were decoded at their native 8192×5464 size using the default linear RAW mode and the existing artifact/enhancement step (14.0, 0.35). No sample image or derivative is tracked. [Numeric report](data/estimator-mad-raw-measurements.json) contains source hashes, legacy estimates, patch counts, and experimental measurements.

| Sample | Grid counts at brightness anchors 16/30/64/128/200/240 | Coverage | Analysis time |
| --- | --- | --- | ---: |
| 3E9A7623.CR3 | 51 / 39 / 162 / 4 / 0 / 0 | Passes candidate coverage | 147 ms |
| 3E9A7624.CR3 | 0 / 93 / 162 / 1 / 0 / 0 | Fails: one patch in the 128 bin | 145 ms |

Times are single warm development-build measurements including legacy analysis, excluding decode; they are not a performance benchmark. No patches were rejected by the candidate nonfinite/clipping/structure/correlation thresholds in these RAW audits. That does not prove the regions are physically flat or unclipped. The first sample's linear Y scales range from about 0.0093 to 0.0535 across accepted bins; the second has 0.0197/0.0521 in its two accepted bins. Neither source supplies a known clean reference.

Lossless/JPEG derivative audits, preprocessing interaction checks, table fitting, suggested-setting GPU checks at native spacing 5, old-setting export comparisons, and native-app Estimate/history/save/export smoke remain **unexecuted for calibration**. They are downstream of the failed measurement gate. Existing shader/ownership evidence remains in its separate validation documents.

## Reproduction and regression checks

Run from the repository root with Rust 1.96.1. Generated reports go to `/tmp`.

```sh
# Ordinary checks: 143 passed, 8 explicitly ignored; GPU tests required.
REQUIRE_GPU_TESTS=1 cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib

# Explicit calibration gate: FAILS (53 numerical failures + 1 coverage failure).
DENOISE_DOMAIN_REPORT=/tmp/calibration-domain-grid.json cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib audit_proposed_domain_accuracy_grid -- --ignored --nocapture

# Diagnostic only: runs both RAWs, reports insufficiency without asserting acceptance.
DENOISE_RAW_IMAGES="$PWD/docs-untracked/3E9A7623.CR3:$PWD/docs-untracked/3E9A7624.CR3" DENOISE_MEASUREMENT_REPORT=/tmp/calibration-raw-measurements.json cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib audit_real_raw_measurements -- --ignored --nocapture

npm run typecheck
npm test
rustfmt +1.96.1 --edition 2021 --check src-tauri/src/noise_analysis.rs
git diff --check
```

Rust regressions pass in 13.22 seconds with GPU required; 143 includes the ten new ordinary reference tests. TypeScript has zero errors; all 74 frontend tests pass. The two new ignored tests are the optional local-RAW audit and the known failing domain gate, alongside six existing optional tests. Production build/native UI acceptance is not inferred from these checks. The implementation slice is a test-only measurement experiment with failure evidence, not a completed calibration release.
