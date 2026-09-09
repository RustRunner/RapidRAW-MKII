//! Maintained non-AI enhancement qualification against a frozen production shader.
use super::enhancements::{BASELINE, candidate, guided_edges};
use super::*;

fn widen(source: String, radius: u32) -> String {
    let mut result = source;
    for (from, to) in [
        (
            "mix(1.0, 4.0, effective_chroma / 100.0)",
            format!("mix(1.0, {radius}.0, effective_chroma / 100.0)"),
        ),
        (
            "mix(2.0, 4.0, effective_chroma / 100.0)",
            format!("mix(2.0, {radius}.0, effective_chroma / 100.0)"),
        ),
    ] {
        assert_eq!(result.matches(from).count(), 1);
        result = result.replacen(from, &to, 1);
    }
    result
}

pub(super) fn initial_variants() -> Vec<(&'static str, String)> {
    vec![
        ("baseline", BASELINE.into()),
        ("wide-chroma", candidate(false, true)),
        ("guided-edges", guided_edges()),
        ("wide-chroma-guided-edges", widen(guided_edges(), 6)),
    ]
}

#[test]
#[ignore = "matched GPU benchmarks; requires DENOISE_QUALIFICATION_OUT"]
fn benchmark_denoise_actual_combination() {
    let out = std::path::PathBuf::from(
        std::env::var_os("DENOISE_QUALIFICATION_OUT").expect("output required"),
    );
    std::fs::create_dir_all(&out).unwrap();
    let mut results = Vec::new();
    for (name, source) in initial_variants() {
        let Some(gpu) = Harness::with_source(&source) else {
            return;
        };
        for size in [1080, 4320] {
            let pixels = independence_fixture(size, true);
            let settings = [100.0, 50.0, 100.0, size as f32 / 1080.0];
            gpu.run(&pixels, size, true, settings, true);
            let mut times = Vec::new();
            for _ in 0..5 {
                gpu.run(&pixels, size, true, settings, true);
                times.push(gpu.last_gpu_ms.get().expect("GPU timestamps required"));
            }
            times.sort_by(f64::total_cmp);
            eprintln!("QUALIFICATION TIMING {name} {size}: {times:?}");
            results.push(
                serde_json::json!({"variant":name,"size":size,"gpu_ms":times,"median_ms":times[2]}),
            );
        }
    }
    std::fs::write(
        out.join("initial-timing.json"),
        serde_json::to_vec_pretty(&results).unwrap(),
    )
    .unwrap();
}

const FLAT_WIDTH: u32 = 384;
const PAD: u32 = 40;

fn from_encoded_ycc(v: [f32; 3], raw: bool) -> [f32; 4] {
    let r = v[0] + v[2] / 0.713;
    let b = v[0] + v[1] / 0.565;
    let g = (v[0] - 0.2126 * r - 0.0722 * b) / 0.7152;
    let rgb = [r, g, b].map(|c| if raw { decode(c / 255.0) } else { c / 255.0 });
    [rgb[0], rgb[1], rgb[2], 1.0]
}

// Same injection as the original experiment, now in a sufficiently large flat
// field. Noisy and clean inputs share the exact transfer/encoding convention.
fn flat(
    brightness: f32,
    seed: u32,
    correlation: u32,
    height: u32,
    raw: bool,
    noisy: bool,
) -> Vec<[f32; 4]> {
    (0..FLAT_WIDTH * height)
        .map(|i| {
            let (x, y) = (i % FLAT_WIDTH, i / FLAT_WIDTH);
            let coarse = |c| {
                let fx = (x % correlation) as f32 / correlation as f32;
                let fy = (y % correlation) as f32 / correlation as f32;
                let n = |dx, dy| noise(x / correlation + dx + seed, y / correlation + dy, c);
                ((1.0 - fx) * n(0, 0) + fx * n(1, 0)) * (1.0 - fy)
                    + ((1.0 - fx) * n(0, 1) + fx * n(1, 1)) * fy
            };
            let a = if noisy { 12.0 } else { 0.0 };
            from_encoded_ycc(
                [
                    brightness + a * noise(x + seed, y, 2),
                    a * coarse(0),
                    a * coarse(1),
                ],
                raw,
            )
        })
        .collect()
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
struct Stats {
    sigma: [f64; 3],
    mse: [f64; 3],
    count: u32,
}
impl Stats {
    fn chroma_sigma(self) -> f64 {
        ((self.sigma[1].powi(2) + self.sigma[2].powi(2)) * 0.5).sqrt()
    }
}

fn stats(output: &[[f32; 4]], clean: &[[f32; 4]], width: u32) -> Stats {
    assert_eq!(output.len(), clean.len());
    let height = output.len() as u32 / width;
    assert!(width > 2 * PAD && height > 2 * PAD);
    let (mut sum, mut squares, mut n) = ([0.0; 3], [0.0; 3], 0);
    for y in PAD..height - PAD {
        for x in PAD..width - PAD {
            let i = (y * width + x) as usize;
            let a = ycc(encoded(output[i]));
            let b = ycc(encoded(clean[i]));
            for c in 0..3 {
                let d = a[c] - b[c];
                assert!(d.is_finite());
                sum[c] += d;
                squares[c] += d * d;
            }
            n += 1;
        }
    }
    let mse = squares.map(|s| s / n as f64);
    Stats {
        sigma: std::array::from_fn(|c| (mse[c] - (sum[c] / n as f64).powi(2)).max(0.0).sqrt()),
        mse,
        count: n,
    }
}

#[derive(Clone, Copy, Debug)]
enum Scene {
    Edge,
    Texture(u32, usize),
    Line(u32, usize),
    Gradient(usize),
}
impl Scene {
    fn channels(self) -> &'static [usize] {
        match self {
            Self::Edge | Self::Texture(_, 0) => &[0],
            _ => &[1, 2],
        }
    }
}
fn scenes() -> Vec<Scene> {
    let mut result = vec![Scene::Edge];
    for channel in 0..3 {
        for period in [4, 8, 16] {
            result.push(Scene::Texture(period, channel));
        }
    }
    for channel in [1, 2] {
        for width in [1, 2, 4] {
            result.push(Scene::Line(width, channel));
        }
        result.push(Scene::Gradient(channel));
    }
    result
}
const DETAIL_WIDTH: u32 = 256;
fn clean_scene(scene: Scene, brightness: f32, raw: bool) -> Vec<[f32; 4]> {
    (0..DETAIL_WIDTH * 160)
        .map(|i| {
            let x = i % DETAIL_WIDTH;
            let mut v = [brightness, 0.0, 0.0];
            match scene {
                Scene::Edge => v[0] += if x < DETAIL_WIDTH / 2 { -10.0 } else { 10.0 },
                Scene::Texture(period, c) => {
                    v[c] += 6.0 * (x as f32 * std::f32::consts::TAU / period as f32).sin()
                }
                Scene::Line(width, c) => {
                    v[c] += if x >= DETAIL_WIDTH / 2 && x < DETAIL_WIDTH / 2 + width {
                        12.0
                    } else {
                        0.0
                    }
                }
                Scene::Gradient(c) => v[c] += 12.0 * (x as f32 / (DETAIL_WIDTH - 1) as f32 - 0.5),
            }
            from_encoded_ycc(v, raw)
        })
        .collect()
}

#[derive(serde::Serialize)]
struct QualityReport {
    name: String,
    noise: Vec<serde_json::Value>,
    detail: Vec<serde_json::Value>,
    failures: Vec<String>,
    chroma_ratio_median: f64,
    luma_ratio_median: f64,
    max_clean_mse_increase: f64,
}
fn median(mut values: Vec<f64>) -> f64 {
    assert!(!values.is_empty());
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}
fn qualify(
    name: &str,
    source: &str,
    seeds: &[u32],
    check_chroma: bool,
    check_luma: bool,
) -> Option<QualityReport> {
    let baseline = Harness::with_source(BASELINE)?;
    let gpu = Harness::with_source(source)?;
    let mut report = QualityReport {
        name: name.into(),
        noise: vec![],
        detail: vec![],
        failures: vec![],
        chroma_ratio_median: 0.0,
        luma_ratio_median: 0.0,
        max_clean_mse_increase: f64::NEG_INFINITY,
    };
    let (mut color_ratios, mut luma_ratios) = (Vec::new(), Vec::new());
    for &seed in seeds {
        for brightness in [30.0, 128.0, 200.0] {
            for correlation in [8, 24] {
                for raw in [false, true] {
                    for step in [1.0, 5.0] {
                        let mut small_decision = None;
                        for height in [256, 432] {
                            let clean_input =
                                flat(brightness, seed, correlation, height, raw, false);
                            let input = flat(brightness, seed, correlation, height, raw, true);
                            let clean =
                                baseline.run(&clean_input, FLAT_WIDTH, raw, [0.0; 4], false);
                            let settings = [100.0, 50.0, 100.0, step];
                            let before = stats(
                                &baseline.run(&input, FLAT_WIDTH, raw, settings, true),
                                &clean,
                                FLAT_WIDTH,
                            );
                            let after = stats(
                                &gpu.run(&input, FLAT_WIDTH, raw, settings, true),
                                &clean,
                                FLAT_WIDTH,
                            );
                            assert!(before.sigma[0] > 0.0 && before.chroma_sigma() > 0.0);
                            let y_ratio = after.sigma[0] / before.sigma[0];
                            let c_ratio = after.chroma_sigma() / before.chroma_sigma();
                            let decision = (c_ratio <= 1.01, y_ratio <= 1.01);
                            let label = format!(
                                "seed={seed} Y={brightness} correlation={correlation} raw={raw} step={step} height={height}"
                            );
                            if let Some(small) = small_decision {
                                if small != decision {
                                    report.failures.push(format!("unstable flat-area decision {label}: {small:?}->{decision:?}"));
                                }
                            }
                            small_decision = Some(decision);
                            if check_chroma && !decision.0 {
                                report
                                    .failures
                                    .push(format!("chroma regression {label}: {c_ratio}"));
                            }
                            if check_luma && brightness == 30.0 && !decision.1 {
                                report
                                    .failures
                                    .push(format!("shadow regression {label}: {y_ratio}"));
                            }
                            if height == 432 {
                                if step == 5.0 {
                                    color_ratios.push(c_ratio);
                                }
                                if brightness >= 128.0 {
                                    luma_ratios.push(y_ratio);
                                }
                            }
                            report.noise.push(serde_json::json!({"seed":seed,"brightness":brightness,"correlation":correlation,"raw":raw,"step":step,"height":height,"baseline":before,"candidate":after,"y_ratio":y_ratio,"c_ratio":c_ratio}));
                        }
                    }
                }
            }
        }
    }
    report.chroma_ratio_median = median(color_ratios);
    report.luma_ratio_median = median(luma_ratios);
    if check_chroma && report.chroma_ratio_median > 0.9 {
        report.failures.push(format!(
            "chroma improvement ratio {} > 0.9",
            report.chroma_ratio_median
        ));
    }
    if check_luma && report.luma_ratio_median > 0.9 {
        report.failures.push(format!(
            "luma improvement ratio {} > 0.9",
            report.luma_ratio_median
        ));
    }
    for scene in scenes() {
        for brightness in [30.0, 128.0, 200.0] {
            for raw in [false, true] {
                for step in [1.0, 5.0] {
                    for detail in [0.0, 50.0, 100.0] {
                        let input = clean_scene(scene, brightness, raw);
                        let clean = baseline.run(&input, DETAIL_WIDTH, raw, [0.0; 4], false);
                        let settings = [100.0, detail, 100.0, step];
                        let before = stats(
                            &baseline.run(&input, DETAIL_WIDTH, raw, settings, true),
                            &clean,
                            DETAIL_WIDTH,
                        );
                        let after = stats(
                            &gpu.run(&input, DETAIL_WIDTH, raw, settings, true),
                            &clean,
                            DETAIL_WIDTH,
                        );
                        for &c in scene.channels() {
                            let increase = after.mse[c] - before.mse[c];
                            report.max_clean_mse_increase =
                                report.max_clean_mse_increase.max(increase);
                            if increase > 0.25 {
                                report.failures.push(format!("clean detail {scene:?} Y={brightness} raw={raw} step={step} D={detail} channel={c}: MSE {} -> {}",before.mse[c],after.mse[c]));
                            }
                        }
                        report.detail.push(serde_json::json!({"scene":format!("{scene:?}"),"brightness":brightness,"raw":raw,"step":step,"detail":detail,"channels":scene.channels(),"baseline":before,"candidate":after}));
                    }
                }
            }
        }
    }
    Some(report)
}

fn save_reports(filename: &str, reports: &[QualityReport]) {
    let out = std::path::PathBuf::from(
        std::env::var_os("DENOISE_QUALIFICATION_OUT").expect("output required"),
    );
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(
        out.join(filename),
        serde_json::to_vec_pretty(reports).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "initial candidate qualification before tuning; requires DENOISE_QUALIFICATION_OUT"]
fn qualify_initial_denoise_candidates() {
    let mut reports = Vec::new();
    for (name, source) in initial_variants() {
        let Some(report) = qualify(
            name,
            &source,
            &[173, 619],
            name.contains("wide"),
            name.contains("edges"),
        ) else {
            return;
        };
        eprintln!(
            "QUALIFICATION {name}: {} failures; Y ratio={} C ratio={} max clean delta={}",
            report.failures.len(),
            report.luma_ratio_median,
            report.chroma_ratio_median,
            report.max_clean_mse_increase
        );
        if name == "baseline" {
            assert!(report.failures.is_empty());
        }
        reports.push(report);
    }
    save_reports("initial-quality.json", &reports);
}

#[test]
fn test_gpu_denoise_qualification_reference() {
    let Some(report) = qualify(
        "production",
        include_str!("../../shaders/shader.wgsl"),
        &[173],
        false,
        false,
    ) else {
        return;
    };
    assert!(report.failures.is_empty(), "{:?}", report.failures);
    assert_eq!(report.luma_ratio_median, 1.0);
    assert_eq!(report.chroma_ratio_median, 1.0);
    assert_eq!(report.max_clean_mse_increase, 0.0);
    assert!(
        report
            .noise
            .iter()
            .all(|r| r["baseline"]["count"].as_u64().unwrap() >= 53_504)
    );
}

#[test]
fn test_denoise_qualification_metrics() {
    use sha2::Digest;
    assert_eq!(
        hex::encode(sha2::Sha256::digest(BASELINE.as_bytes())),
        "2eed7222e71a6da488ffd61ea369ba48077795fd0cc8e153fd780b23d8ca9a7b"
    );
    let clean = vec![[decode(128.0 / 255.0); 4]; (FLAT_WIDTH * 256) as usize];
    let exact = stats(&clean, &clean, FLAT_WIDTH);
    assert_eq!(exact.mse, [0.0; 3]);
    assert_eq!(exact.sigma, [0.0; 3]);
    // Constant bias contributes to MSE, but never masquerades as noise sigma.
    let shifted = vec![[decode(130.0 / 255.0); 4]; clean.len()];
    let bias = stats(&shifted, &clean, FLAT_WIDTH);
    assert!((bias.mse[0] - 4.0).abs() < 0.001);
    assert!(bias.sigma[0] < 0.001);
    for scene in scenes() {
        assert_eq!(
            clean_scene(scene, 128.0, true).len(),
            (DETAIL_WIDTH * 160) as usize
        );
    }
}

fn edge_blend(weight: f32) -> String {
    let mut source = guided_edges().replacen(
        "fn detect_edge_strength(",
        "fn detect_edge_strength_guided(",
        1,
    );
    let start = BASELINE.find("fn detect_edge_strength(").unwrap();
    let end = start + BASELINE[start..].find("\n}\n").unwrap() + 2;
    source.push_str(&BASELINE[start..end].replacen(
        "fn detect_edge_strength(",
        "fn detect_edge_strength_original(",
        1,
    ));
    source.push_str(&format!(
        r#"
fn detect_edge_strength(coord: vec2<i32>, is_raw: u32) -> f32 {{
    return mix(detect_edge_strength_original(coord,is_raw),
        detect_edge_strength_guided(coord,is_raw), {weight});
}}
"#
    ));
    source
}

#[test]
#[ignore = "bounded candidate refinement; requires DENOISE_QUALIFICATION_OUT"]
fn qualify_denoise_refinements() {
    // Declared bounded alternatives: one smaller footprint with two spatial
    // sigmas, and three constant edge blends. No Detail-dependent curve needed
    // unless these simpler variants justify further work within the spec.
    let radius5 = widen(BASELINE.into(), 5);
    let radius5_sigma4 = radius5.replace(
        "mix(1.0, 5.0, effective_chroma / 100.0)",
        "mix(1.0, 4.0, effective_chroma / 100.0)",
    );
    let variants = vec![
        ("chroma-radius5", radius5, true, false),
        ("chroma-radius5-sigma4", radius5_sigma4, true, false),
        ("edge-blend-0.125", edge_blend(0.125), false, true),
        ("edge-blend-0.25", edge_blend(0.25), false, true),
        ("edge-blend-0.5", edge_blend(0.5), false, true),
    ];
    let mut reports = Vec::new();
    for (name, source, chroma, luma) in variants {
        let Some(report) = qualify(name, &source, &[173, 619], chroma, luma) else {
            return;
        };
        eprintln!(
            "REFINEMENT {name}: {} failures; Y ratio={} C ratio={} max clean delta={}",
            report.failures.len(),
            report.luma_ratio_median,
            report.chroma_ratio_median,
            report.max_clean_mse_increase
        );
        reports.push(report);
    }
    save_reports("refinement-quality.json", &reports);
}

#[test]
#[ignore = "expanded boundary gates for initial candidates; requires DENOISE_QUALIFICATION_OUT"]
fn qualify_denoise_candidate_boundaries() {
    let mut rows = Vec::new();
    for (name, source) in initial_variants() {
        let Some(gpu) = Harness::with_source(&source) else {
            return;
        };
        let failures = boundary_matrix_failures(&gpu);
        if name == "baseline" {
            assert!(failures.is_empty(), "baseline boundaries: {failures:?}");
        }
        eprintln!("BOUNDARY QUALIFICATION {name}: {} failures", failures.len());
        rows.push(serde_json::json!({"name":name,"failures":failures}));
    }
    let out = std::path::PathBuf::from(
        std::env::var_os("DENOISE_QUALIFICATION_OUT").expect("output required"),
    );
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(
        out.join("initial-boundaries.json"),
        serde_json::to_vec_pretty(&rows).unwrap(),
    )
    .unwrap();
}

#[test]
fn test_gpu_denoise_detail_gates_detect_known_regressions() {
    let Some(baseline) = Harness::with_source(BASELINE) else {
        return;
    };
    // These real counterexamples guard the sensitivity of the promoted detail
    // metrics: accepting clean identity or lower noise alone must not pass them.
    for (source, scene, brightness, channel) in [
        (edge_blend(0.5), Scene::Texture(8, 0), 200.0, 0),
        (
            widen(BASELINE.into(), 5).replace(
                "mix(1.0, 5.0, effective_chroma / 100.0)",
                "mix(1.0, 4.0, effective_chroma / 100.0)",
            ),
            Scene::Texture(8, 1),
            128.0,
            1,
        ),
    ] {
        let Some(gpu) = Harness::with_source(&source) else {
            return;
        };
        let input = clean_scene(scene, brightness, true);
        let settings = [100.0, 50.0, 100.0, 1.0];
        let clean = baseline.run(&input, DETAIL_WIDTH, true, [0.0; 4], false);
        let before = stats(
            &baseline.run(&input, DETAIL_WIDTH, true, settings, true),
            &clean,
            DETAIL_WIDTH,
        );
        let after = stats(
            &gpu.run(&input, DETAIL_WIDTH, true, settings, true),
            &clean,
            DETAIL_WIDTH,
        );
        assert!(
            after.mse[channel] > before.mse[channel] + 0.25,
            "known detail regression was missed: {scene:?}"
        );
    }
}
