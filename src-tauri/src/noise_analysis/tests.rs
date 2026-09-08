use super::*;
use image::{Rgb, RgbImage};

struct Gaussian(u64);
impl Gaussian {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    }
    fn next(&mut self) -> f32 {
        ((-2.0 * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos())
            as f32
    }
}
fn fixture(mean: f32, sigma: f32, gray: bool, seed: u64) -> Rgb32FImage {
    let mut rng = Gaussian(seed);
    Rgb32FImage::from_fn(512, 512, |x, y| {
        let base = mean + 0.001 * (x as f32 / 512.0 - 0.5) + 0.0005 * (y as f32 / 512.0 - 0.5);
        let n = rng.next() * sigma;
        Rgb(if gray {
            [base + n; 3]
        } else {
            [
                base + n,
                base + rng.next() * sigma,
                base + rng.next() * sigma,
            ]
        })
    })
}
fn measured(m: &NoiseMeasurement) -> &BrightnessBin {
    assert!(m.is_usable(), "measurement not usable: {m:?}");
    m.bins.iter().max_by_key(|b| b.accepted_pixels).unwrap()
}
fn assert_sigma(actual: f32, expected: f32, relative: f32, absolute: f32, context: &str) {
    assert!(
        (actual - expected).abs() <= absolute.max(relative * expected),
        "{context}: {actual} vs {expected}"
    );
}
fn sigmas(rgb: &Rgb32FImage, clean: impl Fn(u32, u32) -> [f32; 3]) -> [f32; 3] {
    let mut sum = [0.0; 3];
    let mut square = [0.0; 3];
    for (x, y, p) in rgb.enumerate_pixels() {
        let a = ycbcr(p.0);
        let b = ycbcr(clean(x, y));
        for c in 0..3 {
            let e = a[c] - b[c];
            sum[c] += e;
            square[c] += e * e;
        }
    }
    let n = f64::from(rgb.width() * rgb.height());
    std::array::from_fn(|c| (square[c] / n - (sum[c] / n).powi(2)).max(0.0).sqrt() as f32)
}
fn clean(mean: f32, x: u32, y: u32) -> [f32; 3] {
    [mean + 0.001 * (x as f32 / 512.0 - 0.5) + 0.0005 * (y as f32 / 512.0 - 0.5); 3]
}

#[test]
fn float_linear_white_noise_matches_channel_covariance_and_held_out_seeds() {
    for seed in [1701, 81371] {
        for mean in [0.013, 0.216, 0.578, -0.05, 2.0] {
            for gray in [true, false] {
                let sigma = 0.006;
                let rgb = fixture(mean, sigma, gray, seed);
                let m = measure(&rgb, true, 0.0);
                let got = measured(&m).linear;
                let expected = if gray {
                    [sigma, 0.0, 0.0]
                } else {
                    [0.749615 * sigma, 0.672688 * sigma, 0.760181 * sigma]
                };
                for (a, b) in [got.sigma_y, got.sigma_cb, got.sigma_cr]
                    .into_iter()
                    .zip(expected)
                {
                    assert_sigma(
                        a,
                        b,
                        0.1,
                        1e-4,
                        &format!("white mean={mean} gray={gray} seed={seed}"),
                    );
                }
            }
        }
    }
}

#[test]
fn both_domains_follow_real_transformed_residuals_not_a_global_derivative() {
    for mean in [30.0 / 255.0, 128.0 / 255.0, 200.0 / 255.0] {
        let encoded = fixture(mean, 0.01, false, 9811);
        let linear =
            Rgb32FImage::from_fn(512, 512, |x, y| Rgb(encoded.get_pixel(x, y).0.map(decode)));
        let e = measure(&encoded, false, 0.0);
        let l = measure(&linear, true, 0.0);
        let eb = measured(&e);
        let lb = measured(&l);
        let linear_truth = sigmas(&linear, |x, y| clean(mean, x, y).map(decode));
        let encoded_truth = sigmas(&encoded, |x, y| clean(mean, x, y));
        for (got, truth) in [(eb.linear, linear_truth), (eb.encoded, encoded_truth)] {
            for (a, b) in [got.sigma_y, got.sigma_cb, got.sigma_cr]
                .into_iter()
                .zip(truth)
            {
                assert_sigma(a, b, 0.1, 1e-4, "transformed ground truth");
            }
        }
        for (a, b) in [eb.linear.sigma_y, eb.linear.sigma_cb, eb.linear.sigma_cr]
            .into_iter()
            .zip([lb.linear.sigma_y, lb.linear.sigma_cb, lb.linear.sigma_cr])
        {
            assert_sigma(a, b, 0.001, 1e-6, "representation agreement");
        }
    }
}

#[test]
fn unequal_and_cross_channel_noise_uses_its_covariance() {
    let mut rng = Gaussian(4397);
    let rgb = Rgb32FImage::from_fn(512, 512, |x, y| {
        let z = [rng.next(), rng.next(), rng.next()];
        let base = clean(0.2, x, y)[0];
        Rgb([
            base + 0.01 * z[0],
            base + 0.003 * z[0] + 0.005 * z[1],
            base - 0.004 * z[0] + 0.02 * z[2],
        ])
    });
    let truth = sigmas(&rgb, |x, y| clean(0.2, x, y));
    let m = measure(&rgb, true, 0.0);
    let got = measured(&m).linear;
    for (a, b) in [got.sigma_y, got.sigma_cb, got.sigma_cr]
        .into_iter()
        .zip(truth)
    {
        assert_sigma(a, b, 0.1, 1e-4, "cross-channel");
    }
}

#[test]
fn quantization_has_a_separately_measured_error_bound() {
    let base = 0.216;
    for sigma in [0.002, 0.01, 0.03] {
        let original = fixture(base, sigma, false, 4512);
        for levels in [255.0, 65535.0] {
            let quantized = Rgb32FImage::from_fn(512, 512, |x, y| {
                Rgb(original
                    .get_pixel(x, y)
                    .0
                    .map(|c| (encode(c) * levels).round() / levels))
            });
            let decoded = Rgb32FImage::from_fn(512, 512, |x, y| {
                Rgb(quantized.get_pixel(x, y).0.map(decode))
            });
            let precision = sigmas(&decoded, |x, y| original.get_pixel(x, y).0);
            let truth = sigmas(&original, |x, y| clean(base, x, y));
            let m = measure(&quantized, false, 1.0 / levels);
            let b = measured(&m);
            if levels == 255.0 && sigma == 0.002 {
                assert_eq!(
                    b.quantization_limited, [true; 3],
                    "sub-code noise must be unresolved, not claimed accurate"
                );
            } else {
                assert_eq!(b.quantization_limited, [false; 3]);
                for (c, (a, t)) in [b.linear.sigma_y, b.linear.sigma_cb, b.linear.sigma_cr]
                    .into_iter()
                    .zip(truth)
                    .enumerate()
                {
                    assert_sigma(a, t, 0.1, 1e-4 + precision[c], "precision budget");
                }
            }
            eprintln!(
                "quantization sigma={sigma} levels={levels} precision={precision:?} limited={:?}",
                b.quantization_limited
            );
        }
    }
}

fn correlated(radius: usize, seed: u64) -> Rgb32FImage {
    let n = 512usize;
    let mut rng = Gaussian(seed);
    let white: Vec<f32> = (0..n * n).map(|_| rng.next()).collect();
    let mut horizontal = vec![0.0; n * n];
    for y in 0..n {
        for x in 0..n {
            horizontal[y * n + x] = (-(radius as i32)..=radius as i32)
                .map(|d| white[y * n + ((x as i32 + d).rem_euclid(n as i32) as usize)])
                .sum::<f32>()
                / (2 * radius + 1) as f32;
        }
    }
    let mut field = vec![0.0; n * n];
    for y in 0..n {
        for x in 0..n {
            field[y * n + x] = (-(radius as i32)..=radius as i32)
                .map(|d| horizontal[((y as i32 + d).rem_euclid(n as i32) as usize) * n + x])
                .sum::<f32>()
                / (2 * radius + 1) as f32;
        }
    }
    let rms = (field.iter().map(|x| x * x).sum::<f32>() / field.len() as f32).sqrt();
    Rgb32FImage::from_fn(n as u32, n as u32, |x, y| {
        Rgb([0.2 + 0.01 * field[y as usize * n + x as usize] / rms; 3])
    })
}
#[test]
fn correlated_marginal_sigma_is_not_the_white_noise_highpass_sigma() {
    for seed in [13271, 8459] {
        for radius in [1, 2, 4, 8] {
            let rgb = correlated(radius, seed);
            let truth = sigmas(&rgb, |_, _| [0.2; 3])[0];
            let m = measure(&rgb, true, 0.0);
            let b = measured(&m);
            eprintln!("correlation radius={radius} seed={seed}: truth={truth}, measurement={b:?}");
            assert_sigma(
                b.linear.sigma_y,
                truth,
                0.2,
                2e-4,
                "correlated marginal sigma",
            );
            assert!(b.highpass_to_marginal < 0.7);
        }
    }
}
#[test]
fn clipping_nonfinite_texture_and_missing_coverage_are_not_zero_noise() {
    let clipped = Rgb32FImage::from_pixel(512, 512, Rgb([0.0; 3]));
    let m = measure(&clipped, false, 1.0 / 255.0);
    assert!(!m.is_usable());
    assert_eq!(m.quality.rejected_clipped, 64);
    let mut nonfinite = fixture(0.2, 0.01, true, 123);
    for p in nonfinite.pixels_mut() {
        p[0] = f32::NAN;
    }
    let m = measure(&nonfinite, true, 0.0);
    assert!(!m.is_usable());
    assert_eq!(m.quality.rejected_nonfinite, 64);
    let texture = Rgb32FImage::from_fn(512, 512, |x, y| {
        Rgb([0.2
            + if (x / 4 + y / 4) % 2 == 0 {
                0.03
            } else {
                -0.03
            }; 3])
    });
    assert!(!measure(&texture, true, 0.0).is_usable());
    assert!(!measure(&Rgb32FImage::new(63, 512), true, 0.0).is_usable());
    let mut mixed = fixture(0.2, 0.01, true, 891);
    for (y, row) in mixed.rows_mut().enumerate() {
        if y < 256 {
            for p in row {
                *p = Rgb([0.0; 3]);
            }
        }
    }
    let m = measure(&mixed, false, 1.0 / 255.0);
    assert!(!m.is_usable(), "cannot discard the clipped shadow bin");
}

#[test]
fn legacy_consumer_retains_exact_source_measurements_and_wire_domains_are_explicit() {
    let img = DynamicImage::ImageRgb8(RgbImage::from_fn(128, 128, |x, y| {
        Rgb([((50 + x + y) % 250) as u8; 3])
    }));
    let before = crate::denoising::estimate_noise(&img);
    let after = analyze_source(&img, false);
    assert_eq!(
        before.sigma_luma.to_bits(),
        after.legacy_source.sigma_luma.to_bits()
    );
    assert_eq!(
        before.sigma_chroma.to_bits(),
        after.legacy_source.sigma_chroma.to_bits()
    );
    let glare_before = crate::glare_recovery::estimate_glare(&img, false, before.sigma_luma);
    let glare_after =
        crate::glare_recovery::estimate_glare(&img, false, after.legacy_source.sigma_luma);
    assert_eq!(
        serde_json::to_value(glare_before).unwrap(),
        serde_json::to_value(glare_after).unwrap()
    );
    let wire = serde_json::to_value(after.measurement).unwrap();
    assert!(wire.get("quality").is_some());
}

#[test]
#[ignore = "requires local RAW samples; writes only the requested measurement report"]
fn audit_real_raw_measurements() {
    let paths = std::env::var_os("DENOISE_RAW_IMAGES").expect("DENOISE_RAW_IMAGES required");
    let mut report = Vec::new();
    for path in std::env::split_paths(&paths) {
        let bytes = std::fs::read(&path).unwrap();
        let decoded = crate::raw_processing::develop_raw_image(
            &bytes,
            false,
            2.5,
            crate::app_settings::default_linear_raw_mode(),
            None,
        )
        .unwrap();
        let mut record = |img: &DynamicImage, is_linear: bool, variant: &str| {
            let started = std::time::Instant::now();
            let analysis = analyze_source(img, is_linear);
            let elapsed = started.elapsed().as_millis();
            if variant == "default" && std::env::var_os("DENOISE_REQUIRE_RAW_COVERAGE").is_some() {
                assert!(
                    analysis.measurement.is_usable(),
                    "default RAW lacks qualified coverage: {:?}",
                    analysis.measurement.quality
                );
            }
            eprintln!("RAW {} variant={variant} usable={} ms={elapsed} clipped={:.2}% patch_range={:.2}%..{:.2}% bins={:?}",
                path.display(), analysis.measurement.is_usable(), 100.0 * analysis.measurement.quality.clipped_pixel_fraction,
                100.0 * analysis.measurement.quality.min_patch_clipped_fraction, 100.0 * analysis.measurement.quality.max_patch_clipped_fraction,
                analysis.measurement.quality.qualified_by_bin);
            let suggestion = crate::noise_calibration::suggest(&analysis.measurement,50.0,img.width(),img.height());
            if variant == "default" && std::env::var_os("DENOISE_REQUIRE_RAW_SUGGESTIONS").is_some() {
                assert!(suggestion.is_ok(), "default RAW has no calibrated suggestion: {suggestion:?}");
            }
            report.push(serde_json::json!({"path":path,"variant":variant,"is_linear":is_linear,"width":img.width(),"height":img.height(),"analysis_ms":elapsed,"legacy_source":analysis.legacy_source,"measurement":analysis.measurement,"suggestion":suggestion}));
        };
        let mut img = decoded.clone();
        crate::image_processing::remove_raw_artifacts_and_enhance(&mut img, 14.0, 0.35);
        record(&img, true, "default");
        if std::env::var_os("DENOISE_RAW_VARIANTS").is_some() {
            record(&decoded, true, "develop_only");
            for (nr, sharp, label) in [(14.0, 0.0, "color_nr_only"), (0.0, 0.35, "sharpen_only")] {
                let mut variant = decoded.clone();
                crate::image_processing::remove_raw_artifacts_and_enhance(&mut variant, nr, sharp);
                record(&variant, true, label);
            }
            let rgb = img.to_rgb32f();
            let rgb8 = RgbImage::from_fn(rgb.width(), rgb.height(), |x, y| {
                Rgb(rgb
                    .get_pixel(x, y)
                    .0
                    .map(|c| (encode(c).clamp(0.0, 1.0) * 255.0).round() as u8))
            });
            record(
                &DynamicImage::ImageRgb8(rgb8.clone()),
                false,
                "encoded_8bit",
            );
            let rgb16 = image::ImageBuffer::from_fn(rgb.width(), rgb.height(), |x, y| {
                Rgb(rgb
                    .get_pixel(x, y)
                    .0
                    .map(|c| (encode(c).clamp(0.0, 1.0) * 65535.0).round() as u16))
            });
            record(&DynamicImage::ImageRgb16(rgb16), false, "encoded_16bit");
            let mut jpeg = Vec::new();
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 90)
                .encode_image(&DynamicImage::ImageRgb8(rgb8))
                .unwrap();
            record(
                &image::load_from_memory(&jpeg).unwrap(),
                false,
                "jpeg_quality90",
            );
        }
    }
    if let Ok(path) = std::env::var("DENOISE_MEASUREMENT_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}

#[test]
fn transfer_overflow_is_missing_data_not_zero_noise() {
    let image = Rgb32FImage::from_pixel(128, 128, Rgb([f32::MAX; 3]));
    let m = measure(&image, false, 0.0);
    assert!(!m.is_usable());
    assert_eq!(m.quality.rejected_nonfinite, 4);
}

#[test]
fn half_float_upload_precision_is_measured_separately() {
    for mean in [0.013, 0.216, 0.578, -0.05, 2.0] {
        let original = fixture(mean, 0.006, false, 7883);
        let uploaded = Rgb32FImage::from_fn(512, 512, |x, y| {
            Rgb(original
                .get_pixel(x, y)
                .0
                .map(|c| half::f16::from_f32(c).to_f32()))
        });
        let precision = sigmas(&uploaded, |x, y| original.get_pixel(x, y).0);
        let truth = sigmas(&original, |x, y| clean(mean, x, y));
        let m = measure(&uploaded, true, 0.0);
        let got = measured(&m).linear;
        for (c, (a, b)) in [got.sigma_y, got.sigma_cb, got.sigma_cr]
            .into_iter()
            .zip(truth)
            .enumerate()
        {
            assert_sigma(
                a,
                b,
                0.1,
                1e-4 + precision[c],
                "half-float precision budget",
            );
        }
    }
}

// Run explicitly before fitting; failures must not be converted into relaxed
// tolerances or treated as a successful release gate.
#[test]
fn audit_proposed_domain_accuracy_grid() {
    let mut failures = Vec::new();
    let mut comparisons = 0;
    let mut records = Vec::new();
    for seed in [6113, 92821] {
        for mean_encoded in BRIGHTNESS {
            let mean = decode(mean_encoded);
            for sigma in [0.001, 0.008, 0.032, 0.064] {
                for gray in [true, false] {
                    let linear = fixture(mean, sigma, gray, seed);
                    let encoded = Rgb32FImage::from_fn(512, 512, |x, y| {
                        Rgb(linear.get_pixel(x, y).0.map(encode))
                    });
                    let m = measure(&linear, true, 0.0);
                    if !m.is_usable() {
                        records.push(serde_json::json!({"seed":seed,"mean_encoded":mean_encoded,"injected_linear_rgb_sigma":sigma,"gray":gray,"coverage_failure":m.quality}));
                        failures.push(format!("coverage mean={mean_encoded} sigma={sigma} gray={gray} seed={seed}: {:?}", m.quality));
                        continue;
                    }
                    let b = measured(&m);
                    let lt = sigmas(&linear, |x, y| clean(mean, x, y));
                    let et = sigmas(&encoded, |x, y| clean(mean, x, y).map(encode));
                    for (domain, got, truth) in
                        [("linear", b.linear, lt), ("encoded", b.encoded, et)]
                    {
                        for (c, (a, t)) in [got.sigma_y, got.sigma_cb, got.sigma_cr]
                            .into_iter()
                            .zip(truth)
                            .enumerate()
                        {
                            comparisons += 1;
                            let passed = (a - t).abs() <= (0.1 * t).max(1e-4);
                            records.push(serde_json::json!({"seed":seed,"mean_encoded":mean_encoded,"injected_linear_rgb_sigma":sigma,"gray":gray,"domain":domain,"component":(["y","cb","cr"][c]),"measured":a,"reference":t,"passed":passed}));
                            if !passed {
                                failures.push(format!("{domain}[{c}] mean={mean_encoded} sigma={sigma} gray={gray} seed={seed}: measured={a} truth={t} relative_error={}", (a-t).abs()/t));
                            }
                        }
                    }
                }
            }
        }
    }
    if let Ok(path) = std::env::var("DENOISE_DOMAIN_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&serde_json::json!({"comparisons":comparisons,"failures":failures.len(),"records":records})).unwrap()).unwrap();
    }
    eprintln!(
        "DOMAIN_GATE comparisons={comparisons} failures={}\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        failures.is_empty(),
        "proposed marginal-sigma measurement failed its accuracy gate"
    );
}

#[test]
fn a_populated_bin_with_too_few_patches_is_not_silently_discarded() {
    let rgb = Rgb32FImage::from_fn(512, 512, |x, y| {
        Rgb([if x < 64 && y < 64 { 0.8 } else { 0.2 }; 3])
    });
    let m = measure(&rgb, true, 0.0);
    assert!(!m.is_usable());
    assert_eq!(m.quality.insufficient_bins, 1);
    assert_eq!(m.bins.len(), 1);
}

#[test]
fn encoded_injection_high_noise_and_saturated_colors_match_realized_variance() {
    for seed in [19763, 88069] {
        for base in [
            [0.063; 3],
            [0.118; 3],
            [0.251; 3],
            [0.502; 3],
            [0.941; 3],
            [0.05, 0.5, 0.8],
            [0.9, 0.08, 0.3],
        ] {
            for sigma in [0.008, 0.032, 0.064] {
                let mut rng = Gaussian(seed);
                let source =
                    Rgb32FImage::from_fn(
                        512,
                        512,
                        |_, _| Rgb(base.map(|c| c + sigma * rng.next())),
                    );
                let linear = Rgb32FImage::from_fn(512, 512, |x, y| {
                    Rgb(source.get_pixel(x, y).0.map(decode))
                });
                let m = measure(&source, false, 0.0);
                let b = measured(&m);
                for (domain, got, truth) in [
                    ("linear", b.linear, sigmas(&linear, |_, _| base.map(decode))),
                    ("encoded", b.encoded, sigmas(&source, |_, _| base)),
                ] {
                    for (a, t) in [got.sigma_y, got.sigma_cb, got.sigma_cr]
                        .into_iter()
                        .zip(truth)
                    {
                        assert_sigma(a,t,0.1,1e-4,&format!("{domain} encoded injection base={base:?} sigma={sigma} seed={seed}"));
                    }
                }
            }
        }
    }
}

#[test]
fn bounded_resampling_uses_independent_pixels_and_recovers_sparse_regions() {
    let rgb = Rgb32FImage::from_fn(2048, 2048, |x, y| {
        Rgb([if x < 128 && y < 128 { 0.8 } else { 0.2 }; 3])
    });
    let a = measure(&rgb, true, 0.0);
    let b = measure(&rgb, true, 0.0);
    assert!(a.is_usable(), "{a:?}");
    assert!(a.quality.resampled_patches > 0);
    assert!(a.quality.sampled_patches <= 256);
    assert_eq!(a.quality.sampled_origins, b.quality.sampled_origins);
    for (i, p) in a.quality.sampled_origins.iter().enumerate() {
        for q in &a.quality.sampled_origins[..i] {
            assert!(p[0].abs_diff(q[0]) >= SIDE as u32 || p[1].abs_diff(q[1]) >= SIDE as u32);
        }
    }
    assert!(a.bins.iter().all(|b| b.accepted_patches >= 4));
}

#[test]
fn float_black_clipping_is_reported_instead_of_qualifying_censored_noise() {
    let mut img = fixture(0.005, 0.02, false, 9137);
    for p in img.pixels_mut() {
        p.0 = p.0.map(|v| v.max(0.0));
    }
    let m = measure(&img, true, 0.0);
    assert!(!m.is_usable());
    assert_eq!(m.quality.rejected_clipped, m.quality.sampled_patches);
    assert!(m.quality.min_patch_clipped_fraction > 0.5);
    assert!(m.bins.is_empty());
}

mod clipped;
