# Calibrated source-noise suggestions

8 September 2026. Calibration version 1 uses measurement version 3 and the unchanged production denoise WGSL. The two supplied RAWs now receive qualified suggestions at Detail 50. Suggestions are applied only by an explicit Estimate action; manual slider semantics, existing saved values, and Glare's legacy measurement/budget remain unchanged.

## Fit and coverage

The checked-in tables contain 616 entries, of which 70 are explicitly unsupported. Axes are linear sigma `[0, .001, .002, .004, .008, .012, .016, .032, .064, .096, .128]`, encoded brightness `[16, 30, 64, 96, 128, 200, 240]/255`, Detail `[0, 50, 100]` for Strength, and discrete native spacing `[1, 2, 3, 4, 5]` for Chroma. The added sigma/brightness/spacing points were GPU fitted, not extrapolated. The `.012` point resolves a RAW interpolation interval that previously touched an unsupported `.008` corner.

The production lookup rounds interpolated values to integers. It rejects nonfinite/out-of-range features, unsupported corners, and unvalidated spacing. Chroma's table does not depend on Strength or Detail. Native spacing matches WGSL's round-to-even behavior, including half-step dimensions. Suggestions are recomputed from cached source measurements and the captured Detail, not cached as source properties or derived from the preview size.

Each qualified brightness bin is mapped separately. The result is weighted by all absolutely qualified patch area (`represented_pixels`) before statistical ranking; `accepted_pixels` still describes the patches used for the scale estimate. This prevents retaining all clipped patches from adding an accidental shadow weight relative to an unclipped bin's selected quartile. Unsupported populated bins cannot silently disappear from the median. Linear and encoded component medians, per-bin measurements, and suggestion ranges remain runtime diagnostics, outside adjustments/sidecars/history.

## Production-GPU fitting

The fixture has three 128-pixel-wide scene/covariance variants: neutral, saturated, and unequal-channel noise. Each includes flat, edge, and texture regions. The 28-pixel margin excludes fixture boundaries at maximum spacing. Edges occupy a separate 256-row region; metrics use its interior, and qualified chroma boundaries additionally measure the 32 columns around the edge. Texture uses declared sinusoidal structure. Clean/noisy linear samples are rounded through f16 before GPU upload; the clean rounding error supplies the normalization floor. Display-domain comparison clips negative RGB in both the disabled baseline and filtered output before encoding.

The fitter evaluates the actual `apply_denoise` WGSL. It uses equal-weight normalized flat/edge/texture loss, searches controls in steps of five, then refines around the best and lowest near-best candidates. It chooses the lowest feasible value within 5% of best loss and enforces useful-reduction and boundary gates. The fit seed is 17293.

Chroma is fitted independently first. Strength candidates must also pass combined-setting constraints with independently injected luma/chroma noise, C/Y ratios .3 and 1, all three scene variants, and native spacings 1–5. The loss used to choose among feasible Strength candidates remains the declared luma reference loss. Combined constraints do not add runtime table dimensions or change Chroma's independence. Infeasible nodes remain unsupported.

The first separate-only fit passed its separate hold-out but failed 14 of 3,240 combined cases: 12 shadow chroma-reduction failures and two representations of a bright boundary failure. Lower Strength values passed the shadow counterexamples. Adding combined constraints removed those failures, while the bright Detail-0 interpolation case still measured 3.009 levels of bias against the unchanged 3-level limit. Its four interpolation corners (brightness 200/240, sigma .032/.064, Detail 0) are explicitly unsupported. These defects and Strength sweeps are retained in [failure evidence](data/estimator-suggestion-failures.json); they are not counted as passing release results. No acceptance tolerance was widened.

The final fit took 364.92 seconds on NVIDIA GB10 / Vulkan / driver 580.142. [Node results](data/estimator-suggestion-fit.csv) include the selected values, feasibility, losses, and unsupported reasons. The Rust tables include their production shader SHA-256. `generate-denoise-tables.py` serializes the fitted values; it does not perform fitting itself.

## Independent release validation

Fresh seeds 24533 and 68159 were used after fitting and declaring exclusions. They did not select candidates or exclusions. Held-out brightnesses are .10, .20, .42, .70, and .87. Separate-control sigma values are .003, .012, .024, .048, and .080; combined scenes use .012, .024, .048, and .080 with C/Y ratios .3 and 1. Detail includes 0, 25, 50, 75, and 100. Separate Chroma tests cover all five native spacings; combined tests cover 1, 3, and 5. Both unclipped and explicitly lower-clipped scenes use their realized source brightness and component sigmas for lookup. Clipped-set rows denote an applied lower clamp; bright fixtures may have no endpoint hits. Actual endpoint fractions are retained per case.

| Release set | Cases | Supported / passed | Unsupported | Failed supported |
| --- | ---: | ---: | ---: | ---: |
| Separate, unclipped | 1,500 | 1,100 | 400 | 0 |
| Separate, clipped | 1,500 | 1,058 | 442 | 0 |
| Combined, unclipped | 3,600 | 1,722 | 1,878 | 0 |
| Combined, clipped | 3,600 | 1,777 | 1,823 | 0 |

Supported moderate/high-noise domains (realized linear sigma at least .008) must reduce flat encoded noise sigma by at least 10% and reference loss by at least 5%. Qualified strong boundaries retain the 3-level clean-error and systematic-bias limits. Low-noise cases are not claimed to satisfy the moderate/high-noise reduction requirement. Unsupported cases include table gaps/range and more than 60% black endpoints; they are not assigned zero or 100 to obtain a numerical pass. Every supported separate-case lookup also checks equality between the fitted JSON and compiled Rust tables. The 10,200 per-case records are in [held-out data](data/estimator-suggestion-heldout.csv). These two release tests completed in 30.51 seconds.

## RAW and native application checks

Both original files remain ignored/untracked and retain the hashes in [the RAW review](denoise-validation.md). Default development/preprocessing and the known 8192×5464 source dimensions produce Strength 100, Chroma 100, Detail 50, native spacing 5 for both files. These are fitted results, not a saturation fallback. Color-NR-only, sharpen-only, and lossless encoded 8/16-bit derivatives also receive supported suggestions. Development without default preprocessing remains insufficient for both files; the second JPEG diagnostic is outside table coverage. JPEG is a robustness diagnostic, not a required clean reference. The full audit, including these limitations and represented-area diagnostics, is in [RAW data](data/estimator-suggestion-raw.json); required default coverage/suggestions passed in 11.31 seconds.

Native testing ran the actual Tauri app, WebKit UI, Rust commands, GPU pipeline, and GTK save dialog. A temporary Vite automation module drove the real controls and inspected editor state; no Tauri command or analysis result was mocked. A separate app identifier/config/data directory and independent RAW copies under `/tmp/native-calibration` protected the user's running app, settings, and original samples.

- Both RAWs: Estimate produced 100/100, repeat produced the same result, and each action added exactly one history entry. Undo restored the previous numeric settings; redo restored 100/100.
- Both panels were collapsed while their native request status was **pending**; completion succeeded and survived reopening. Photo switching published new generations, cleared old estimate state, and loaded saved numeric settings with a fresh history.
- A repeat at the UI's 100% zoom produced the same native suggestion as the fit-to-screen preview. Both fit previews and full-resolution exports rendered successfully.
- Hot-pixel correction, chromatic-aberration correction, +1.25 EV exposure, and Glare were enabled together in the native renderer. The source estimate remained 100/100 at native spacing 5 and a full-resolution export completed. These checks verify integration, not a calibrated final-output noise budget.
- Both saved/reopened photos retained their settings with no estimate metadata in sidecars. All exports used the actual Export Image control and GTK Save Edited Image dialog.
- The first RAW's before/after pre-Estimate PNG pixel streams are identical (IDAT SHA-256 `71fb597bb238f8d4996c089aa29ab93d6d7dbe718deed6687fe9d1eb8a6be83a`). A saved manual 35/60 setting on the second RAW reopened unchanged and exported identically across the table replacement (IDAT SHA-256 `03035335e0c03b087b7ae76334398fb24516469c9f53eedc29d8a850c133909d`). This compares complete encoded pixel streams, not only a downsampled preview.

Visual review of native exports shows less fine chroma noise, with larger colored variation still visible, especially in the higher-ISO sample. Small map text remains noise-obscured. These photos have no clean reference; this is not a claim of measured real-scene detail recovery. Native PNGs and review crops remain outside git under `/tmp/native-calibration`.

## Regression and reproduction

Rust 1.96.1: 152 library tests passed, 11 optional tests ignored in the ordinary serial run. The fitting, held-out GPU, and required RAW audits above were invoked explicitly. Frontend: 75 tests passed, TypeScript clean, production build passed, and 952 locale fallback/plural checks passed. Existing Rust dead-code/unused-mut warnings and the frontend bundle-size advisory remain unrelated to this calibration.

```sh
REQUIRE_GPU_TESTS=1 DENOISE_FIT_REPORT=/tmp/denoise-fit.json \
  cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib fit_gpu_suggestion_tables -- --ignored --nocapture --test-threads=1
python3 bench/generate-denoise-tables.py /tmp/denoise-fit.json /tmp/denoise-tables.rs
# Compare regenerated tables after rustfmt; do not replace release tables without revalidation.
REQUIRE_GPU_TESTS=1 DENOISE_VALIDATION_SEEDS=24533,68159 \
  DENOISE_FIT_REPORT=/tmp/denoise-fit.json \
  DENOISE_VALIDATION_REPORT=/tmp/denoise-heldout.json \
  DENOISE_COMBINED_REPORT=/tmp/denoise-combined.json \
  cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib validate_ -- --ignored --nocapture --test-threads=1
```

The ignored RAW audit additionally uses `DENOISE_RAW_IMAGES` (platform path-list separator), `DENOISE_RAW_VARIANTS=1`, `DENOISE_REQUIRE_RAW_COVERAGE=1`, `DENOISE_REQUIRE_RAW_SUGGESTIONS=1`, and `DENOISE_MEASUREMENT_REPORT`. Images and derived renders are never fixture assets in git.
