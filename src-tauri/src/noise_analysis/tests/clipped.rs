use super::*;

fn field(radius: usize, seed: u64) -> Vec<f32> {
    if radius == 0 {
        let mut rng = Gaussian(seed);
        (0..512 * 512).map(|_| rng.next()).collect()
    } else {
        correlated(radius, seed)
            .pixels()
            .map(|p| (p[0] - 0.2) / 0.01)
            .collect()
    }
}
fn pair(
    fields: &[Vec<f32>; 3],
    fraction: f32,
    saturated: bool,
    mixed: bool,
) -> (Rgb32FImage, Rgb32FImage) {
    let base = if saturated {
        [0.0, 0.06, 0.15]
    } else {
        [0.0; 3]
    };
    let noise: Vec<[f32; 3]> = (0..512 * 512)
        .map(|i| {
            let [a, b, c] = fields.each_ref().map(|f| f[i]);
            if mixed {
                [
                    0.02 * a,
                    0.02 * (0.4 * a + 0.9 * b),
                    0.02 * (-0.3 * a + 0.2 * b + 1.4 * c),
                ]
            } else {
                [0.02 * a; 3]
            }
        })
        .collect();
    let mut minima: Vec<f32> = noise
        .iter()
        .map(|n| (0..3).map(|c| base[c] + n[c]).fold(f32::INFINITY, f32::min))
        .collect();
    let k = (fraction * minima.len() as f32) as usize;
    let offset = -*minima.select_nth_unstable_by(k, f32::total_cmp).1;
    let clean = Rgb32FImage::from_fn(512, 512, |x, y| {
        let plane = 0.001 * (x as f32 / 511.0 - 0.5) + 0.0005 * (y as f32 / 511.0 - 0.5);
        Rgb(base.map(|c| c + offset + plane))
    });
    let noisy = Rgb32FImage::from_fn(512, 512, |x, y| {
        let b = clean.get_pixel(x, y).0;
        Rgb(std::array::from_fn(|c| {
            (b[c] + noise[(y * 512 + x) as usize][c]).max(0.0)
        }))
    });
    (
        Rgb32FImage::from_fn(512, 512, |x, y| {
            Rgb(clean.get_pixel(x, y).0.map(|c| c.max(0.0)))
        }),
        noisy,
    )
}
fn develop(rgb: Rgb32FImage, nr: f32, sharp: f32) -> Rgb32FImage {
    let mut image = DynamicImage::ImageRgb32F(rgb);
    if nr > 0.0 || sharp > 0.0 {
        crate::image_processing::remove_raw_artifacts_and_enhance(&mut image, nr, sharp);
    }
    image.into_rgb32f()
}
fn encoded_image(rgb: &Rgb32FImage) -> Rgb32FImage {
    Rgb32FImage::from_fn(rgb.width(), rgb.height(), |x, y| {
        Rgb(rgb.get_pixel(x, y).0.map(encode))
    })
}

#[test]
#[ignore = "extended clipped clean/noisy reference grid and preprocessing audit"]
fn audit_clipped_developed_accuracy() {
    let mut records = Vec::new();
    let mut failures = Vec::new();
    for seed in [14951, 98179, 73343, 54787] {
        for radius in [0, 1, 2, 4, 8] {
            let fields = [
                field(radius, seed),
                field(radius, seed + 103),
                field(radius, seed + 307),
            ];
            for saturated in [false, true] {
                for mixed in [false, true] {
                    for fraction in [0.05, 0.15, 0.30, 0.45, 0.595] {
                        let (clean, noisy) = pair(&fields, fraction, saturated, mixed);
                        for (nr, sharp, variant) in [
                            (0.0, 0.0, "lower_clip"),
                            (14.0, 0.0, "color_nr"),
                            (0.0, 0.35, "sharpen"),
                            (14.0, 0.35, "default"),
                        ] {
                            let clean = develop(clean.clone(), nr, sharp);
                            let noisy = develop(noisy.clone(), nr, sharp);
                            let m = measure(&noisy, true, 0.0);
                            let label = format!("seed={seed} radius={radius} saturated={saturated} mixed={mixed} clip={fraction} variant={variant}");
                            if !m.is_usable() {
                                // Insufficient coverage and >60% clipping are explicit
                                // exclusions. Retain their complete rejection evidence.
                                assert!(m
                                    .quality
                                    .sampled_by_bin
                                    .iter()
                                    .zip(m.quality.qualified_by_bin)
                                    .any(|(&sampled, qualified)| sampled > 0 && qualified < 4));
                                records.push(serde_json::json!({"case":label,"coverage":false,"quality":m.quality}));
                                continue;
                            }
                            let b = measured(&m);
                            let truth_linear = sigmas(&noisy, |x, y| clean.get_pixel(x, y).0);
                            let enc_clean = encoded_image(&clean);
                            let enc_noisy = encoded_image(&noisy);
                            let truth_encoded =
                                sigmas(&enc_noisy, |x, y| enc_clean.get_pixel(x, y).0);
                            let correlated_case = radius > 0 || nr > 0.0 || sharp > 0.0;
                            let (relative, absolute) = if correlated_case {
                                (0.2f32, 2e-4f32)
                            } else {
                                (0.1, 1e-4)
                            };
                            let mut bias = [0.0f64; 3];
                            for (a, b) in enc_noisy.pixels().zip(enc_clean.pixels()) {
                                let a = ycbcr(a.0);
                                let b = ycbcr(b.0);
                                for c in 0..3 {
                                    bias[c] += a[c] - b[c];
                                }
                            }
                            bias = bias.map(|v| v / (512.0 * 512.0));
                            for (domain, got, truth) in [
                                ("linear", b.linear, truth_linear),
                                ("encoded", b.encoded, truth_encoded),
                            ] {
                                for (c, (a, t)) in [got.sigma_y, got.sigma_cb, got.sigma_cr]
                                    .into_iter()
                                    .zip(truth)
                                    .enumerate()
                                {
                                    let pass = (a - t).abs() <= (relative * t).max(absolute);
                                    records.push(serde_json::json!({"case":label,"domain":domain,"component":c,"measured":a,"reference":t,"passed":pass,"encoded_residual_bias":bias,"measured_clip_fraction":b.black_clipped_fraction}));
                                    if !pass {
                                        failures.push(format!("{label} {domain}[{c}]: measured={a} truth={t} relative_error={}",(a-t).abs()/t));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if let Ok(path) = std::env::var("DENOISE_CLIPPED_REPORT") {
        std::fs::write(
            path,
            serde_json::to_vec_pretty(
                &serde_json::json!({"failures":failures.len(),"records":records}),
            )
            .unwrap(),
        )
        .unwrap();
    }
    eprintln!(
        "CLIPPED_GATE failures={}\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(failures.is_empty(), "clipped developed-noise gate failed");
}

#[test]
fn clipped_class_has_explicit_limits_and_rejects_upper_plateaus() {
    let fields = [field(0, 7139), field(0, 9137), field(0, 3179)];
    let (_, noisy) = pair(&fields, 0.30, true, true);
    let m = measure(&noisy, true, 0.0);
    assert!(m.is_usable(), "{m:?}");
    assert!(m.quality.qualified_black_clipped_patches > 0);
    assert!(m
        .bins
        .iter()
        .all(|b| b.black_clipped_fraction > 0.01 && b.black_clipped_fraction <= 0.60));
    let upper = Rgb32FImage::from_fn(512, 512, |x, y| {
        Rgb(noisy.get_pixel(x, y).0.map(|v| 1.0 - v))
    });
    assert!(!measure(&upper, true, 0.0).is_usable());
    let (_, excess) = pair(&fields, 0.75, false, true);
    assert!(!measure(&excess, true, 0.0).is_usable());
}

#[test]
fn clipped_periodic_texture_and_broad_curvature_are_not_flat_noise() {
    for period in [8.0f32, 16.0, 32.0, 96.0] {
        let mut rng = Gaussian(4417);
        let rgb = Rgb32FImage::from_fn(512, 512, |x, y| {
            let signal = 0.008 + 0.02 * (std::f32::consts::TAU * (x + y) as f32 / period).sin();
            Rgb([(signal + 0.001 * rng.next()).max(0.0); 3])
        });
        assert!(
            !measure(&rgb, true, 0.0).is_usable(),
            "period {period} texture must not qualify"
        );
    }
}

#[test]
fn clipped_quantized_pairs_preserve_developed_residual_measurement() {
    let fields = [field(0, 22147), field(0, 17489), field(0, 91939)];
    for saturated in [false, true] {
        let (clean, noisy) = pair(&fields, 0.30, saturated, true);
        for levels in [255.0f32, 65535.0] {
            let quantize = |rgb: &Rgb32FImage| {
                Rgb32FImage::from_fn(512, 512, |x, y| {
                    Rgb(rgb
                        .get_pixel(x, y)
                        .0
                        .map(|c| (encode(c).clamp(0.0, 1.0) * levels).round() / levels))
                })
            };
            let clean_e = quantize(&clean);
            let noisy_e = quantize(&noisy);
            let clean_l =
                Rgb32FImage::from_fn(512, 512, |x, y| Rgb(clean_e.get_pixel(x, y).0.map(decode)));
            let noisy_l =
                Rgb32FImage::from_fn(512, 512, |x, y| Rgb(noisy_e.get_pixel(x, y).0.map(decode)));
            let m = measure(&noisy_e, false, 1.0 / levels);
            let b = measured(&m);
            for (got, truth) in [
                (b.linear, sigmas(&noisy_l, |x, y| clean_l.get_pixel(x, y).0)),
                (
                    b.encoded,
                    sigmas(&noisy_e, |x, y| clean_e.get_pixel(x, y).0),
                ),
            ] {
                for (a, t) in [got.sigma_y, got.sigma_cb, got.sigma_cr]
                    .into_iter()
                    .zip(truth)
                {
                    assert_sigma(a, t, 0.1, 1e-4, "clipped quantized developed residual");
                }
            }
        }
    }
}
