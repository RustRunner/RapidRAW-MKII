# Estimator calibration proposal

8 September 2026. **Implemented: measurement version 3 and calibrated suggestion version 1 have passed their reference, supported-setting GPU, RAW, and native-app release checks.** See [final suggestion evidence](estimator-suggestion-validation.md) for coverage, unsupported regions, fit revisions, and reproduction. The shader and ownership fixes through `94ced8f4` and TypeScript cleanup `4fa180f3` remain unchanged.

## Current outcome after the approved revision

[Version 2 evidence](estimator-variance-validation.md): centered residual second moments in both domains pass all 576 original grid comparisons and the added high-noise encoded/saturated fixtures. MAD remains separately named for robust-scale and structure diagnostics. Brightness is encoded Y of spatial mean linear RGB; deterministic sampling uses a 16×12 initial grid and up to 64 additional nonoverlapping neighbor patches, preserving the 256 cap and four-qualified-patch minimum. The full accuracy grid is now a normal regression test.

Version 2 exposed a separate release blocker: every sampled patch in both default RAWs exceeds the existing 1% clipping limit. Development/preprocessing code clamps those endpoints. The prior version incorrectly omitted them because the storage was float. That original class rejected these samples and recorded their fractions rather than claiming unclipped coverage; the separately validated version-3 class below resolves default RAW eligibility. Native decoded, NR/sharpening, encoded 8/16-bit, and JPEG variants have been audited.

The user approved the [bounded clipped-noise extension](estimator-clipped-noise-design.md). [Version 3 evidence](estimator-clipped-validation.md) validates its separate class and integrates domain measurements into the generation-owned shared cache. The original unclipped class remains limited to 1%; the new black-clipped class is bounded at 60% with structure/correlation guards. Denoise now uses the validated tables only after an explicit Estimate action. Shaders, Glare budget, and saved settings are unchanged.

## First measurement experiment and approved revision (history)

The first implementation experiment did not qualify for production. See [measurement evidence](estimator-calibration-validation.md) and the checked-in comparison data. The specification below is retained as the agreed gate, not silently loosened to accommodate the result.

- Gaussian-normalized MAD is not marginal standard deviation after a nonlinear transfer curve. Across the broader reference grid, 53 encoded comparisons miss the accuracy gate, with up to 57.15% under-reading. Linear comparisons in the same covered cases pass. Revise the encoded estimator to evaluate centered residual second moments on structure-qualified patches, retaining MAD as a separately named robust-scale diagnostic. Validate against realized transformed residual variance, with the same accuracy bounds. This is a candidate method, not yet an accepted replacement.
- Fixed brightness buckets can acquire a single patch through sparse scene coverage or noisy brightness classification. The second supplied RAW and one synthetic case fail coverage. Evaluate bounded deterministic re-sampling of underrepresented regions within the 256-patch cap, and explicitly test brightness-boundary jitter on small images. Preserve the four-independent-patch requirement and reject clipping/texture failures; do not simply drop sparse bins or count overlapping samples as independent coverage. The coverage policy needs to be resolved before either RAW can qualify for release.
- Re-run the full domain grid, selector/clipping/quantization checks, and both RAW audits before fitting any slider table. Cross-domain tests must include high-noise shadows, not just small encoded perturbations at midtones. Correlated and saturated developed-image cases remain required.

That first prototype was test-only. Its failed results remain historical evidence; the approved revisions and final release checks are documented above.

## Recommended scope

Calibrate Adaptive Denoise suggestions against the current production shader. Measure noise in explicitly named linear and encoded domains, using local brightness statistics. Preserve Glare's existing source-domain budget for this first calibration release, with its legacy measurement named explicitly. Do not change manual slider meanings, shader curves, saved strengths, or Glare settings.

Use two implementation commits after agreement: measurement/contracts/reference tests, then fitted suggestion tables/GPU validation. Do not add camera profiles, ISO-based rules, confidence controls, or a new denoising stage. The two supplied high-ISO RAWs are visual anchors; they cannot establish statistical ground truth by themselves.

## Why the present mapping is insufficient

`estimate_noise` converts the source to floating-point RGB without changing its transfer function. Its result therefore uses developed linear RGB for RAW and encoded RGB for ordinary images. Both are currently multiplied by 4500. The shader filters linear luminance/chroma, while its chroma guide and tolerance use encoded RGB. The existing `test_estimate_noise_calibration` injects approximately Gaussian gray noise into an 8-bit buffer, rounds/clips it, and checks source-domain sigma. It does not validate linear sigma or useful slider settings across brightness.

As an illustration, applying the derivative of the sRGB decode curve to a small encoded gray-noise sigma of 0.01 gives:

| Encoded gray level / 255 | Linear mean | Approximate linear sigma |
| --- | --- | --- |
| 30 | 0.012983 | 0.001805 |
| 128 | 0.215861 | 0.009302 |
| 200 | 0.577580 | 0.016516 |

These are calculated first-order examples, **not calibration ground truths**. Reference tests must transform the actual clean/noisy samples. The transfer curve follows the [ICC sRGB definition](https://registry.color.org/rgb-registry/srgb); negative/headroom handling must match the application's documented extension rather than introduce clipping.

The current luminance range curve also has a real brightness-dependent tradeoff: the existing gray-edge diagnostic retains about 15%, 66%, and 81% of adjacent-column contrast at those three brightnesses at Strength 100 / Detail 50. Calibration must select a useful compromise within that shader's capabilities. A new multiplier cannot fix a filter limitation. See [shader evidence](denoise-validation.md).

## Measurement contract

The committed image snapshot and generation ownership stay unchanged. Shared cached analysis contains source measurements only; suggestions derived from current Detail or resolution are not cached as if they were properties of the source.

| Quantity | Definition and consumer |
| --- | --- |
| `linear.sigma_y`, `sigma_cb`, `sigma_cr` | Full-resolution marginal noise estimates in the developed linear RGB domain; input features for Denoise calibration. Values may exceed 1; sigma is not a bounded pixel value. |
| `encoded.sigma_y`, `sigma_cb`, `sigma_cr` | Measurements after applying the same extended sRGB encoding used by the chroma guide; useful for guide diagnostics and evaluation, not claimed to predict the final edited output. |
| Brightness-bin statistics | Median encoded Y and accepted patch count per bin, plus the associated linear/encoded sigmas. Retain separate Cb/Cr sigmas; define a scalar `sigma_c_max = max(sigma_cb, sigma_cr)` only where the mapping calls for it. |
| Quality diagnostics | Valid/clipped fraction, spatial coverage, quantization floor, texture/scale disagreement, and correlation indicators. Runtime analysis data, not photo adjustments or a new UI control. |
| `legacy_source.sigma_y` | The exact current full-image `estimate_noise` result, in its existing source encoding. Glare alone continues to use it with 0.025. Do not derive it from the newly selected patches. |

Use `Y = 0.2126 R + 0.7152 G + 0.0722 B`, `Cb = 0.565(B-Y)`, and `Cr = 0.713(R-Y)` separately in each domain. For non-RAW input, decode each channel before forming the linear measurements. For RAW, use the developed source directly and encode a separate measurement representation. Do not infer encoding from the bit depth. Profile/color conversion must match the existing loader and GPU path; this proposal does not add color management.

Measure before editor exposure, Glare, and denoising. RAW demosaicing, decoder color NR/sharpening, and highlight handling have already occurred; this is developed-image noise, not a sensor-electron estimate. Hot-pixel correction and chromatic-aberration correction are interaction checks, not assumed equivalent to the pristine source. Source-level suggestions remain starting points when these controls are active.

The new Denoise result should expose explicit measurement names alongside `strength`/`chroma`; migrate the sigma readout and TypeScript type atomically. Do not silently redefine the old `sigma_luma` field. Keep request/return/current identity checks. Capture Detail in the request; the existing relevant-edit invalidation rejects a completion if Detail changes while it runs. Suggestions target native-resolution rendering and its source-derived chroma spacing. Preview zoom or display size must not change a repeated suggestion; previews are validated separately. The source-analysis cache remains identity-keyed.

## Proposed measurement method and limits

Keep the current 3x3 second-difference MAD as a white-noise diagnostic and as the untouched legacy Glare path. It is not an unbiased marginal sigma estimator for arbitrary correlated noise.

For new Denoise measurements, evaluate native-resolution 64x64 patches on a deterministic, evenly distributed grid, capped at 256 patches. Small images use all available nonoverlapping patches. This samples the image without resizing its pixels. Compute both representations from the same sample coordinates; avoid allocating two additional full-image buffers.

Within brightness bins, select candidate flat patches using residual structure in 8x8 block means after planar detrending. Use the lowest structure-score quartile, but require an absolute structure/estimated-noise check as well: rank alone must not declare a fully textured image flat. The absolute threshold is a fitted estimator parameter, published with its validation, not a hidden per-photo heuristic. Require at least four accepted patches per populated bin, with at least 75% usable samples each. Lack of coverage produces an insufficient-data result, not a measured zero.

Estimate marginal Y/Cb/Cr noise using the median absolute residual about a robust fitted plane, with finite-sample behavior measured on the reference fixtures. Compare against the 3x3 estimate and lag/scale diagnostics. This avoids baking the white-noise filter gain into a claim about correlated marginal variance. Plane removal can also erase broad correlated structure; test correlation lengths through 8 pixels and classify longer-scale disagreement as uncertain. Do not add a universal “RAW correction factor.”

The patch selector, robust plane fit, and diagnostic thresholds are implementation candidates subject to the accuracy gates below. If they cannot pass without selecting away noise or confusing texture with noise, revise the measurement design before fitting sliders. A single image does not guarantee separable fine texture and noise. Sensor-domain signal-dependent models also do not automatically describe a developed image: [Foi et al.](https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf) model raw sensor data and explicitly account for clipping. Here that model supplies controlled test cases, not an assertion that this decoder's output obeys it exactly.

Additional rules:

- Reject nonfinite neighborhoods. Retain valid negative values and linear headroom above 1; they are not automatically clipping.
- Record known endpoint/plateau clipping. Patches with more than 1% known clipped samples do not enter the initial accuracy-qualified set. A heavily clipped image must not receive a low-noise suggestion by discarding most of its noisy shadows.
- Characterize 8-bit and 16-bit quantization on transformed clean/noisy pairs, including half-float GPU upload. Below the measured resolution floor, report unresolved noise rather than zero physical noise. Do not blindly subtract `1/12` code-value variance from a robust estimate.
- Synthetic correlated noise uses known spatial filters and cross-channel covariance. Actual RAW validation checks preprocessing variants and texture retention; it does not label an unknown clean image as ground truth.

## Concrete suggestion-curve proposal

Replace the single 4500 multiplier for Denoise with small, versioned piecewise-linear lookup tables generated offline by the production GPU tests. Keep integer output in 0–100. This is a specified fitting/interpolation mechanism; no unmeasured table entries are proposed as release values.

- Strength table: linear `sigma_y`, encoded local mean Y, and existing Detail. Validated nodes: sigma `[0, .001, .002, .004, .008, .012, .016, .032, .064, .096, .128]`, brightness `[16, 30, 64, 96, 128, 200, 240]/255`, Detail `[0, 50, 100]`.
- Chroma table: linear `sigma_c_max`, the same brightness nodes, and the actual chroma sample spacing. Validated spacing is `[1, 2, 3, 4, 5]`; the supplied 8192x5464 RAWs use spacing 5 at native size. Detail and Strength never enter the Chroma table. Both measured chroma components and encoded diagnostics remain available for validating the table across saturated colors.
- Interpolate linearly in sigma, brightness, and Detail. Spacing is a discrete shader category. Do not extrapolate beyond validated coverage or use 100 as an automatic fallback. Expand the validated grid before claiming support for an additional spacing or noise range.
- For a photo, map each qualified brightness bin, then choose the absolutely-qualified-area-weighted median of its suggestions, before statistical ranking. No additional shadow weighting is introduced. Record bin disagreement and coverage in diagnostics. Regions outside validated coverage cannot silently be discarded to make the remaining median look reliable.

Fit by evaluating candidate slider values 0–100 in steps of 5, refining near the best candidate in steps of 1. Luma candidates are evaluated at the requested Detail with chroma disabled; chroma candidates with luminance filtering disabled, followed by combined-setting verification. Final fitting also constrains Strength candidate feasibility using mixed luma/chroma scenes and the independently fitted Chroma table; the declared reference loss and all acceptance tolerances are retained. See the final evidence for rejected interpolation corners. Use fixed clean/noisy pairs and the actual production shader, not the CPU mirror.

For each candidate, compute an encoded-domain reference loss from equal-weight flat, edge, and texture region MSEs. Luma uses Y error; chroma uses `(Cb_error² + Cr_error²)/2`. Normalize each region by its unfiltered noisy baseline error plus a floor of `max(1e-12, measured clean round-trip quantization MSE)` for that region/domain, so one large or noisy region cannot dominate. The precision floor is measured without injected noise; it is not subtracted from the estimated sigma. Include neutral, saturated, and unequal-channel-noise fixtures at each feasible node. The loss is a proposed engineering selection criterion, not a perceptual-quality claim.

Choose the lowest candidate within 5% of the minimum loss. Choose zero if the best candidate improves the total loss by less than 5% over disabled filtering or the measurement is below the validated quantization floor. Enforce the existing qualified chroma-boundary bias/clean-error limits while fitting. Fit nondecreasing suggestions along the sigma axis only where the feasible candidates permit it; revalidate after fitting and interpolation. If monotonicity or any boundary constraint cannot coexist with useful noise reduction, report that node as unsupported and revisit the design—do not hide the conflict by widening tolerances or emitting zero everywhere.

Release requires held-out validation of the resulting curve, useful reductions in moderate/high-noise fixtures, and successful suggestions on both supplied RAWs. A calibrated table is not accepted merely because two measurement representations agree.

## Glare decision and compatibility

For this release, retain `boost = clamp(0.025 / max(legacy_source.sigma_y, 1e-5), 1, ceiling)` and all existing amount/veil-size mappings. Rename/document the internal budget as a legacy source-domain heuristic. It is not a calibrated output-noise ceiling. Fixture tests must establish unchanged Glare suggestions for the same decoded source.

A later output-noise budget would need local stretch/re-encoding behavior, covariance, clipping, and interactions with denoising and subsequent exposure. That is a separate numerical/product decision. It must not inherit 0.025 by changing the input units underneath it.

Existing sidecars, presets, history snapshots, and exports retain their numeric settings and renderer semantics. There is no migration or automatic recalculation on open. New suggestions appear only after an explicit Estimate action. Calibration version, source measurements, quality diagnostics, and request state are runtime/cache metadata, not new adjustment fields. Comparing an old saved photo before/after this work must be pixel-identical until Estimate or another adjustment is applied.

## Reference and release tests

Use seeded Gaussian fixtures with independent held-out seeds. Declare the injection domain, clean scene, channel covariance, spatial correlation, precision, and clipping mask for every case.

| Reference case | Required ground truth |
| --- | --- |
| Gray noise in float linear RGB | `sigma_y = injected sigma`; chroma zero apart from arithmetic error. |
| Independent equal-variance linear RGB noise | Channel transform covariance: Y gain 0.749615, Cb gain 0.672688, Cr gain 0.760181 times the injected per-channel sigma. Also compare realized sample covariance. |
| Unequal/correlated RGB noise | Apply the stated linear channel transform to the full covariance matrix, rather than reusing gray-noise expectations. |
| Same noisy linear scene, encoded and decoded | Linear measurement agrees within separately measured encoding/quantization error. Compare actual residuals after each transform. |
| Noise injected in encoded RGB | Encoded reference uses its injection covariance; linear reference comes from transformed clean/noisy samples per brightness region. |
| Correlated, textured, clipped, quantized, negative/headroom cases | Compare with realized clean/noisy residual statistics; separately test insufficient-data outcomes and bias from sample selection. |

Initial accuracy gates proposed for review: within 10% or `1e-4` absolute sigma (whichever is larger) for unclipped float white-noise cases; within 20% or `2e-4` for the declared short-range correlated cases. Quantized cases use a separately measured precision bound in addition to these gates, not a blanket widened tolerance. Reject biased selectors using held-out scenes/seeds, not only constant patches.

GPU release checks preserve all existing identity, independence, strong-boundary, and manual-setting noise-reduction gates. Add suggested-setting checks at held-out brightness/noise values: at least 10% lower flat-region noise sigma for the qualified moderate/high-noise fixtures and at least 5% lower reference loss, alongside boundary constraints. Include actual native dimensions and preview rendering, especially spacing 5. Record fit data, table version, timing, and unsuccessful nodes. If those gates fail, the calibration is not ready to ship.

Use the two ignored CR3 files identified and hashed in [the RAW review](denoise-validation.md), plus lossless 16-bit and 8-bit encoded derivatives of the same decoded scenes and generated clean-reference fixtures. Keep lossy JPEG as a separate robustness diagnostic. A paired exposure series or static burst would strengthen real-noise validation later, but is not a prerequisite for the synthetic reference implementation.

Before enabling the fitted suggestions, complete a native-app check of both RAWs: repeat Estimate, close/reopen panels, change photos, undo/redo, save/reopen, and export. Confirm one action/history entry and unchanged saved rendering unless the new suggestion is explicitly accepted. Browser ownership smoke results already exist, but do not replace this native check.

## Implementation sequence after agreement

1. **Measurement contract and evidence:** explicit domains/quality diagnostics, shared generation-owned analysis, untouched legacy Glare measurement, new reference tests, and a measurement report. Keep production Denoise suggestions on the legacy mapping until the next commit validates a replacement.
2. **Calibrated suggestions:** fit and check in the small tables, migrate the Denoise command/readout atomically, validate suggestions through the GPU and native app, and verify old settings/export compatibility. Keep manual shader behavior and Glare suggestions unchanged.

Each implementation commit gets its own validation record and no co-author trailer. Failure of the stated measurement or fitting gates returns the proposal for revision; it does not authorize broader shader changes or relaxed acceptance criteria.
