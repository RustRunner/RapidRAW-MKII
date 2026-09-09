//! Offline, non-AI candidates. No candidate changes the production renderer or
//! invalidates its fitted suggestion tables. Run explicitly; see the bench report.
use super::*;

pub(super) const BASELINE: &str = include_str!("fixtures/shader-a9c9d2ec.wgsl");

fn replace_once(source: String, from: &str, to: &str) -> String {
    assert_eq!(source.matches(from).count(), 1, "candidate anchor: {from}");
    source.replacen(from, to, 1)
}

pub(super) fn candidate(luma: bool, wide: bool) -> String {
    let mut source = BASELINE.to_string();
    if luma {
        // Use the existing separable median guide in encoded contrast units.
        // Average original linear samples; only neighbor selection is guided.
        source = replace_once(
            source,
            "let luma = get_luma(load_linear_sample(coord + vec2<i32>(dx, dy), is_raw));",
            "let luma = denoise_local_guide(coord + vec2<i32>(dx, dy), 1, is_raw).x;",
        );
        source = replace_once(
            source,
            "let diff = abs(sample_luma - center_luma);",
            "let diff = abs(denoise_local_guide(coord + vec2<i32>(dx, dy), 1, is_raw).x - denoise_local_guide(coord, 1, is_raw).x);",
        );
    }
    if wide {
        source = replace_once(
            source,
            "let chroma_sigma = mix(1.0, 4.0, effective_chroma / 100.0);",
            "let chroma_sigma = mix(1.0, 6.0, effective_chroma / 100.0);",
        );
        source = replace_once(
            source,
            "let chroma_radius = i32(mix(2.0, 4.0, effective_chroma / 100.0));",
            "let chroma_radius = i32(mix(2.0, 6.0, effective_chroma / 100.0));",
        );
    }
    source
}

pub(super) fn guided_edges() -> String {
    let mut source = candidate(false, false);
    // Reuse the exact existing RGB median, retaining linear contrast units.
    source = replace_once(
        source,
        "fn denoise_local_guide(coord: vec2<i32>, step: i32, is_raw: u32) -> vec3<f32> {",
        "fn denoise_local_rgb(coord: vec2<i32>, step: i32, is_raw: u32) -> vec3<f32> {",
    );
    source = replace_once(
        source,
        "return denoise_guide(denoise_median3(upper, middle, lower));",
        "return denoise_median3(upper, middle, lower);",
    );
    source.push_str("\nfn denoise_local_guide(coord: vec2<i32>, step: i32, is_raw: u32) -> vec3<f32> { return denoise_guide(denoise_local_rgb(coord, step, is_raw)); }\n");
    replace_once(
        source,
        "let luma = get_luma(load_linear_sample(coord + vec2<i32>(dx, dy), is_raw));",
        "let luma = get_luma(denoise_local_rgb(coord + vec2<i32>(dx, dy), 1, is_raw));",
    )
}

fn stable_chroma() -> String {
    let mut source = candidate(false, true);
    let start = source.find("fn denoise_local_guide(").unwrap();
    let end = start + source[start..].find("\n}\n").unwrap() + 2;
    source.replace_range(start..end, r#"
fn denoise_median5(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>, e: vec3<f32>) -> vec3<f32> {
    var v = array<vec3<f32>, 5>(a,b,c,d,e);
    // Fixed sorting network, componentwise median.
    let pairs = array<vec2<u32>,9>(vec2<u32>(0,1),vec2<u32>(3,4),vec2<u32>(2,4),
        vec2<u32>(2,3),vec2<u32>(1,4),vec2<u32>(0,3),vec2<u32>(0,2),vec2<u32>(1,3),vec2<u32>(1,2));
    for (var k=0u;k<9u;k++) {
        let i=pairs[k].x; let j=pairs[k].y;
        let lo=min(v[i],v[j]); let hi=max(v[i],v[j]);
        v[i]=lo; v[j]=hi;
    }
    return v[2];
}
fn denoise_local_guide(coord: vec2<i32>, step: i32, is_raw: u32) -> vec3<f32> {
    var rows: array<vec3<f32>,5>;
    for (var y=-2;y<=2;y++) {
        rows[u32(y+2)] = denoise_median5(
            load_linear_sample(coord+vec2<i32>(-2*step,y*step),is_raw),
            load_linear_sample(coord+vec2<i32>(-step,y*step),is_raw),
            load_linear_sample(coord+vec2<i32>(0,y*step),is_raw),
            load_linear_sample(coord+vec2<i32>(step,y*step),is_raw),
            load_linear_sample(coord+vec2<i32>(2*step,y*step),is_raw));
    }
    return denoise_guide(denoise_median5(rows[0],rows[1],rows[2],rows[3],rows[4]));
}"#);
    source
}

fn variants() -> Vec<(&'static str, String)> {
    vec![
        ("baseline", candidate(false, false)),
        ("guided-luma", candidate(true, false)),
        ("wide-chroma", candidate(false, true)),
        ("combined", candidate(true, true)),
        ("guided-edges", guided_edges()),
        ("wide-stable-chroma", stable_chroma()),
    ]
}

// Encoded RGB fixture: independent fine luminance noise and bilinearly
// interpolated coarse chroma noise on an 8- or 24-pixel lattice. Both are seeded.
// Flat, 20-code-value step, and 8-pixel-period luma/color texture occupy separate bands.
fn fixture(
    brightness: f32,
    seed: u32,
    noisy: bool,
    correlation: u32,
    color_texture: bool,
) -> Vec<[f32; 4]> {
    let width = 256;
    (0..width * width)
        .map(|i| {
            let x = i % width;
            let y = i / width;
            let structure = if y < 85 {
                0.0
            } else if y < 170 {
                if x < 128 { -10.0 } else { 10.0 }
            } else {
                6.0 * (x as f32 * std::f32::consts::TAU / 8.0).sin()
            };
            let coarse = |c| {
                let fx = (x % correlation) as f32 / correlation as f32;
                let fy = (y % correlation) as f32 / correlation as f32;
                let n = |dx, dy| noise(x / correlation + dx + seed, y / correlation + dy, c);
                ((1.0 - fx) * n(0, 0) + fx * n(1, 0)) * (1.0 - fy)
                    + ((1.0 - fx) * n(0, 1) + fx * n(1, 1)) * fy
            };
            let amp = if noisy { 1.0 } else { 0.0 };
            let color_structure = if color_texture && y >= 170 {
                structure
            } else {
                0.0
            };
            let yy = brightness + structure - color_structure + amp * 12.0 * noise(x + seed, y, 2);
            let cb = color_structure + amp * 12.0 * coarse(0);
            let cr = amp * 12.0 * coarse(1);
            let r = yy + cr / 0.713;
            let b = yy + cb / 0.565;
            let g = (yy - 0.2126 * r - 0.0722 * b) / 0.7152;
            let rgb = [r, g, b].map(|v| decode(v / 255.0));
            [rgb[0], rgb[1], rgb[2], 1.0]
        })
        .collect()
}

// Exclude 40 pixels from band/image boundaries: the widest experimental
// footprint is (radius 6 + guide radius 2) * step 5.
fn metrics(output: &[[f32; 4]], clean: &[[f32; 4]]) -> serde_json::Value {
    let mut sum = [[0.0; 3]; 3];
    let mut squares = [[0.0; 3]; 3];
    let mut count = [0; 3];
    for y in 40..216 {
        let band = if y < 45 {
            0
        } else if (125..130).contains(&y) {
            1
        } else if y >= 210 {
            2
        } else {
            continue;
        };
        for x in 40..216 {
            let i = y * 256 + x;
            let a = ycc(encoded(output[i]));
            let b = ycc(encoded(clean[i]));
            for c in 0..3 {
                let e = a[c] - b[c];
                sum[band][c] += e;
                squares[band][c] += e * e;
            }
            count[band] += 1;
        }
    }
    let mse: Vec<_> = (0..3)
        .map(|b| squares[b].map(|s| s / count[b] as f64))
        .collect();
    let variance = |c: usize| (mse[0][c] - (sum[0][c] / count[0] as f64).powi(2)).max(0.0);
    serde_json::json!({"flat_sigma_y": variance(0).sqrt(),
        "flat_sigma_c": ((variance(1)+variance(2))*0.5).sqrt(),
        "flat_mse_ycc": mse[0], "edge_mse_ycc": mse[1], "texture_mse_ycc": mse[2]})
}

#[test]
#[ignore = "offline non-AI enhancement comparison; requires DENOISE_ENHANCEMENT_OUT"]
fn investigate_live_denoise_enhancements() {
    let out = std::path::PathBuf::from(
        std::env::var_os("DENOISE_ENHANCEMENT_OUT").expect("output required"),
    );
    std::fs::create_dir_all(&out).unwrap();
    let mut report = Vec::new();
    for (name, source) in variants() {
        std::fs::write(out.join(format!("{name}.wgsl")), &source).unwrap();
        let Some(gpu) = Harness::with_source(&source) else {
            return;
        };
        let failures = boundary_matrix_failures(&gpu);
        if name == "baseline" {
            assert!(
                failures.is_empty(),
                "production baseline failed: {failures:?}"
            );
        }
        let mut cases = Vec::new();
        for seed in [173, 619] {
            for brightness in [30.0, 128.0, 200.0] {
                for correlation in [8, 24] {
                    for color_texture in [false, true] {
                        let clean = fixture(brightness, seed, false, correlation, color_texture);
                        let noisy = fixture(brightness, seed, true, correlation, color_texture);
                        for step in [1.0, 5.0] {
                            for detail in [0.0, 50.0, 100.0] {
                                let settings = [100.0, detail, 100.0, step];
                                let filtered = gpu.run(&noisy, 256, true, settings, true);
                                let filtered_clean = gpu.run(&clean, 256, true, settings, true);
                                cases.push(serde_json::json!({"seed":seed,"brightness":brightness,
                            "step":step,"detail":detail,"correlation":correlation,"color_texture":color_texture,"before":metrics(&noisy,&clean),
                            "after":metrics(&filtered,&clean),"clean":metrics(&filtered_clean,&clean)}));
                            }
                        }
                    }
                }
            }
        }
        let timing_input = independence_fixture(1080, true);
        gpu.run(&timing_input, 1080, true, [100.0, 50.0, 100.0, 1.0], true);
        let mut times = Vec::new();
        for _ in 0..5 {
            gpu.run(&timing_input, 1080, true, [100.0, 50.0, 100.0, 1.0], true);
            times.push(gpu.last_gpu_ms.get().expect("GPU timestamps required"));
        }
        times.sort_by(f64::total_cmp);
        eprintln!(
            "ENHANCEMENT {name}: {} boundary failures; median GPU {} ms",
            failures.len(),
            times[2]
        );
        report.push(serde_json::json!({"name":name,"boundary_failures":failures,"gpu_ms_1080":times[2],"cases":cases}));
    }
    std::fs::write(
        out.join("synthetic.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}

#[test]
#[ignore = "local RAW enhancement crops; requires DENOISE_RAW_IMAGES and DENOISE_ENHANCEMENT_OUT"]
fn review_live_denoise_enhancements_raw() {
    use image::GenericImageView;
    let out = std::path::PathBuf::from(
        std::env::var_os("DENOISE_ENHANCEMENT_OUT").expect("output required"),
    );
    let paths = std::env::var_os("DENOISE_RAW_IMAGES").expect("RAW paths required");
    std::fs::create_dir_all(&out).unwrap();
    let implementation_review = std::env::var_os("DENOISE_IMPLEMENTATION_REVIEW").is_some();
    let mut viewer = include_str!("../../../../bench/denoise-enhancements-viewer.html").to_string();
    if implementation_review {
        viewer = viewer
            .lines()
            .filter(|line| {
                !["guided-luma", "combined", "wide-stable-chroma"]
                    .iter()
                    .any(|name| line.contains(&format!("<option value=\"{name}\"")))
            })
            .collect::<Vec<_>>()
            .join("\n");
        viewer = viewer.replace("<option value=\"guided-edges\">",
            "<option value=\"wide-chroma-guided-edges\">Wider color + stabilized edges</option>\n<option value=\"guided-edges\">");
    }
    std::fs::write(out.join("index.html"), viewer).unwrap();
    let Some(context) = test_gpu_context("enhancement RAW review") else {
        return;
    };
    let mut report = Vec::new();
    for path in std::env::split_paths(&paths) {
        let bytes = std::fs::read(&path).unwrap();
        let name = path.file_stem().unwrap().to_string_lossy();
        let mut img = crate::raw_processing::develop_raw_image(
            &bytes,
            false,
            2.5,
            crate::app_settings::default_linear_raw_mode(),
            None,
        )
        .unwrap();
        crate::image_processing::remove_raw_artifacts_and_enhance(&mut img, 14.0, 0.35);
        let (w, h) = img.dimensions();
        let input = super::super::tests::upload_rgba16f(&context, &img);
        for (variant, source) in if implementation_review {
            super::qualification::initial_variants()
        } else {
            variants()
        } {
            let processor =
                super::super::GpuProcessor::with_shader(context.clone(), w, h, &source).unwrap();
            let adjustments = crate::image_processing::get_all_adjustments_from_json(
                &serde_json::json!({
                    "denoiseEnabled":true,"denoiseStrength":100.0,"denoiseDetail":50.0,"denoiseChroma":100.0
                }),
                true,
                None,
            );
            let request = super::super::RenderRequest {
                adjustments,
                mask_bitmaps: &[],
                lut: None,
                roi: None,
            };
            let start = std::time::Instant::now();
            let (pixels, ow, oh, _, _) = processor
                .run(&input, w, h, request, false, false, None)
                .unwrap();
            let render_ms = start.elapsed().as_secs_f64() * 1000.0;
            let rendered = image::RgbaImage::from_raw(ow, oh, pixels).unwrap();
            image::imageops::resize(
                &rendered,
                1200,
                1200 * h / w,
                image::imageops::FilterType::Lanczos3,
            )
            .save(out.join(format!("{name}-{variant}-overview.png")))
            .unwrap();
            // The established wall crop plus scene center and lower-right texture.
            let mut crops = Vec::new();
            for (label, x, y) in [
                ("wall", 1792, 1110),
                ("center", w / 2 - 256, h / 2 - 256),
                ("texture", 3 * w / 4 - 256, 3 * h / 4 - 256),
            ] {
                let crop = image::imageops::crop_imm(&rendered, x, y, 512, 512).to_image();
                crop.save(out.join(format!("{name}-{variant}-{label}.png")))
                    .unwrap();
                let mut sums = [0.0; 3];
                let mut squares = [0.0; 3];
                for p in crop.pixels() {
                    let v = ycc([p[0] as f64, p[1] as f64, p[2] as f64]);
                    for c in 0..3 {
                        sums[c] += v[c];
                        squares[c] += v[c] * v[c];
                    }
                }
                let n = (512 * 512) as f64;
                let variance = |c: usize| (squares[c] / n - (sums[c] / n).powi(2)).max(0.0);
                crops.push(serde_json::json!({"crop":label,"x":x,"y":y,
                    "variation_y":variance(0).sqrt(),"variation_c":((variance(1)+variance(2))*0.5).sqrt()}));
            }
            eprintln!("RAW ENHANCEMENT {name} {variant}: {crops:?}");
            report.push(serde_json::json!({"image":name,"variant":variant,"render_ms":render_ms,"crops":crops}));
        }
    }
    std::fs::write(
        out.join("raw.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
}
