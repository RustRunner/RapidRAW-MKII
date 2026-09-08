# Clipped developed-noise calibration extension

8 September 2026. Proposed after the measurement revision; not enabled in production.

## Why this extension is needed

The revised variance measurement passes the original 576-comparison accuracy grid and the added high-noise/saturated encoded-input fixtures. Correctly checking float black endpoints exposes a separate limit: with default preprocessing, the two RAWs have a zero endpoint in 16.72% and 25.24% of sampled pixels, respectively. Every sampled patch exceeds the approved 1% clipping threshold. RAW development clamps negative channels to zero; default color NR and sharpening also clamp reconstructed RGB. These are observed developed-image plateaus, not inferred sensor saturation.

The original calibration design excludes patches above 1% known clipping from its initial accuracy-qualified set. Merely raising that threshold would remove a protection without establishing accuracy. The proposed extension adds a separately validated black-clipped class while retaining that original class and its gates.

## Bounded experiment

1. Preserve the existing unclipped class (at most 1% observed endpoint pixels). Add a black-clipped class covering more than 1% through 60% observed zero-endpoint pixels per patch. Treat greater fractions, unknown upper clipping, insufficient independent coverage, or failed texture/correlation checks as unsupported. A zero endpoint means at least one channel is exactly zero; source negatives and float headroom remain valid.
2. Define the new class's measurement target as the centered standard deviation of **developed clean/noisy residuals**, in both linear and extended encoded domains. Do not infer pre-clipping/sensor variance from it. Retain residual mean bias as a ground-truth fixture metric: variance alone does not measure clipping bias. Keep MAD separately named as a robust-scale/selection diagnostic; its Gaussian conversion is not the variance estimator.
3. Generate paired clean/noisy neutral and saturated scenes with declared independent and unequal/cross-channel Gaussian covariance, short-range spatial correlation through eight pixels, lower clipping, and the actual CPU color NR/sharpening variants. Apply each development/preprocessing transform to both the clean and noisy member of the pair. Include clipping fractions near 5%, 15%, 30%, 45%, and 60%, with independent held-out seeds/scenes and nonconstant structure. Quantization and upper clipping remain separately classified, not silently folded into this black-clipped class.
4. Validate measurement within the existing 10% or 1e-4 absolute bound for white-noise reference cases and 20% or 2e-4 for declared correlated cases, plus separately measured precision bounds. Characterize selection bias and false qualification of clipped texture. Coverage still needs at least four independent patches per populated brightness bin. An unresolved estimate cannot become a measured zero.
5. Only after those checks pass, try the existing small table axes against both classes. Fit actual production WGSL with equal-weight flat/edge/texture encoded loss, measured clean quantization floors, and separate per-class hold-out reporting. Include residual mean bias in MSE and preserve the existing boundary error/bias limits and useful-reduction gates. Do not average a failing clipped class into a passing unclipped score. If one mapping cannot satisfy both classes, report unsupported nodes before proposing any extra table dimension.
6. Run both local RAWs, native spacing 5 and previews, and native-app Estimate/history/save/reopen/export checks before enabling suggestions. Their unknown clean scenes remain visual anchors rather than statistical ground truth.

This changes measurement eligibility and its validation scope. It does not change the shader, manual slider curves, Glare's 0.025 legacy budget, existing saved settings, or camera/ISO handling. No clipping “correction factor” or new user-facing confidence controls are proposed.

## Decision

Recommended: validate this additional class, then resume fitting only if it passes. The alternative is to retain calibration for the original unclipped class; under that scope neither supplied RAW can receive a qualified calibrated suggestion. Current production estimates remain available until a replacement completes its release gates.
