//! Denoise acceptance executes the production WGSL, with scale injected only
//! through this test entry point. CPU helpers construct fixtures and metrics;
//! they do not implement the filter.
use super::tests::test_gpu_context;
use crate::image_processing::GpuContext;
use wgpu::util::{DeviceExt, TextureDataOrder};

const ENTRY: &str = r#"
struct TestParams { settings: vec4<f32>, flags: vec4<u32> }
@group(0) @binding(14) var<storage, read> test_params: TestParams;
@group(0) @binding(15) var<storage, read_write> test_pixels: array<vec4<f32>>;
@compute @workgroup_size(8, 8)
fn test_denoise(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    if (any(id.xy >= dims)) { return; }
    let coord = vec2<i32>(id.xy);
    let rgb = load_linear_sample(coord, test_params.flags.x);
    var result = rgb;
    if (test_params.flags.y == 1u) {
        result = apply_denoise(coord, rgb, test_params.settings.x,
            test_params.settings.y, test_params.settings.z,
            test_params.settings.w, test_params.flags.x);
    }
    if (test_params.flags.y == 2u) { result = denoise_guide(rgb); }
    test_pixels[id.y * dims.x + id.x] = vec4<f32>(result, 1.0);
}
"#;

struct Harness {
    context: GpuContext,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    last_gpu_ms: std::cell::Cell<Option<f64>>,
}

impl Harness {
    fn new() -> Option<Self> {
        Self::with_source(include_str!("../shaders/shader.wgsl"))
    }

    fn with_source(source: &str) -> Option<Self> {
        // Baseline shaders predate the guide helper. The benchmark only calls
        // mode 1; omit the unused guide diagnostic branch for those sources.
        let entry = if source.contains("fn denoise_guide(") {
            ENTRY.to_string()
        } else {
            ENTRY.replace(
                "    if (test_params.flags.y == 2u) { result = denoise_guide(rgb); }",
                "",
            )
        };
        let context = test_gpu_context("denoise matrix")?;
        let device = &context.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("production denoise with test entry"),
            source: wgpu::ShaderSource::Wgsl(format!("{source}\n{entry}").into()),
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("denoise test bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                storage(14, true),
                storage(15, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("production denoise test"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("test_denoise"),
            compilation_options: Default::default(),
            cache: None,
        });
        Some(Self {
            context,
            pipeline,
            layout,
            last_gpu_ms: std::cell::Cell::new(None),
        })
    }

    fn run(
        &self,
        pixels: &[[f32; 4]],
        width: u32,
        raw: bool,
        settings: [f32; 4],
        enabled: bool,
    ) -> Vec<[f32; 4]> {
        self.run_mode(pixels, width, raw, settings, u32::from(enabled))
    }

    fn run_mode(
        &self,
        pixels: &[[f32; 4]],
        width: u32,
        raw: bool,
        settings: [f32; 4],
        mode: u32,
    ) -> Vec<[f32; 4]> {
        let device = &self.context.device;
        let height = pixels.len() as u32 / width;
        assert_eq!(pixels.len(), (width * height) as usize);
        let texture = device.create_texture_with_data(
            &self.context.queue,
            &wgpu::TextureDescriptor {
                label: Some("declared encoding fixture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            TextureDataOrder::MipMajor,
            bytemuck::cast_slice(pixels),
        );
        let view = texture.create_view(&Default::default());
        let mut params = settings.map(f32::to_bits).to_vec();
        params.extend([u32::from(raw), mode, 0, 0]);
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("injected denoise settings and scale"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let size = (pixels.len() * 16) as u64;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size + 16,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let timestamps = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| {
                device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: None,
                    ty: wgpu::QueryType::Timestamp,
                    count: 2,
                })
            });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: timestamps.as_ref().map(|query_set| {
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }
                }),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
        if let Some(queries) = &timestamps {
            let resolved = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            encoder.resolve_query_set(queries, 0..2, &resolved, 0);
            encoder.copy_buffer_to_buffer(&resolved, 0, &readback, size, 16);
        }
        self.context.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(60)),
            })
            .unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = readback.slice(..).get_mapped_range().unwrap();
        let result = bytemuck::cast_slice(&mapped[..size as usize]).to_vec();
        self.last_gpu_ms.set(timestamps.as_ref().map(|_| {
            let times: &[u64] = bytemuck::cast_slice(&mapped[size as usize..]);
            (times[1] - times[0]) as f64 * self.context.queue.get_timestamp_period() as f64 / 1e6
        }));
        drop(mapped);
        readback.unmap();
        result
    }
}

fn encode(x: f32) -> f32 {
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}
fn decode(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}
fn encoded(rgb: [f32; 4]) -> [f64; 3] {
    [0, 1, 2].map(|c| (255.0 * encode(rgb[c])) as f64)
}
fn ycc(rgb: [f64; 3]) -> [f64; 3] {
    let y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    [y, 0.565 * (rgb[2] - y), 0.713 * (rgb[0] - y)]
}
fn noise(x: u32, y: u32, c: u32) -> f32 {
    let mut h = x.wrapping_mul(0x27D4_EB2F)
        ^ y.wrapping_mul(0x1656_67B1)
        ^ (c + 1).wrapping_mul(0x9E37_79B9);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    ((h & 65535) as f32 / 65535.0 - 0.5) * 2.0
}

// Constant *linear* luminance; varying chroma on both axes. Encoding happens
// only after construction, so toggling raw really changes the supplied samples.
fn independence_fixture(width: u32, raw: bool) -> Vec<[f32; 4]> {
    (0..width * width)
        .map(|i| {
            let x = i % width;
            let y = i / width;
            let r = 0.20 + 0.06 * noise(x, y, 0);
            let b = 0.20 + 0.06 * noise(x, y, 1);
            let g = (0.20 - 0.2126 * r - 0.0722 * b) / 0.7152;
            let rgb = [r, g, b].map(|v| if raw { v } else { encode(v) });
            [rgb[0], rgb[1], rgb[2], 1.0]
        })
        .collect()
}

#[test]
fn test_gpu_denoise_independence_and_identity() {
    let Some(gpu) = Harness::new() else {
        return;
    };
    for raw in [false, true] {
        let input = independence_fixture(64, raw);
        for step in [1.0, 2.0, 4.0] {
            let disabled = gpu.run(&input, 64, raw, [100.0, 50.0, 100.0, step], false);
            let zero = gpu.run(&input, 64, raw, [0.0, 50.0, 0.0, step], true);
            assert_eq!(disabled, zero, "both strengths zero must be exactly inert");
            let baseline = gpu.run(&input, 64, raw, [30.0, 50.0, 60.0, step], true);
            for (strength, detail) in [(70.0, 50.0), (100.0, 50.0), (30.0, 0.0), (30.0, 100.0)] {
                let output = gpu.run(&input, 64, raw, [strength, detail, 60.0, step], true);
                let mut max_float = 0.0f32;
                let mut max_byte = 0.0f64;
                for (a, b) in baseline.iter().zip(&output) {
                    for c in 0..3 {
                        max_float = max_float.max((a[c] - b[c]).abs());
                        max_byte =
                            max_byte.max((encoded(*a)[c].round() - encoded(*b)[c].round()).abs());
                    }
                }
                eprintln!(
                    "independence raw={raw} step={step} strength={strength} detail={detail}: float={max_float:e} byte={max_byte}"
                );
                assert!(max_float <= 2e-6 && max_byte <= 1.0);
            }
        }
    }
}

const WIDTH: u32 = 128;
const HEIGHT: u32 = 512;
const MARGIN: u32 = 24; // Outside (radius 4 + guide radius 1) * step 4.

fn boundary_colors(target: f32, amount: f32, equal_luma: bool) -> [[f32; 4]; 2] {
    let colors = |base: f32| {
        let a = amount * base.min(1.0 - base);
        [-1.0, 1.0].map(|sign| {
            let r = base + sign * a;
            let b = base - sign * a;
            let g = if equal_luma {
                (base - 0.2126 * r - 0.0722 * b) / 0.7152
            } else {
                base
            };
            [r, g, b, 1.0]
        })
    };
    let (mut lo, mut hi) = (0.001, 0.95);
    for _ in 0..40 {
        let mid = (lo + hi) * 0.5;
        let pair = colors(mid);
        let mean = (ycc(encoded(pair[0]))[0] + ycc(encoded(pair[1]))[0]) * 0.5;
        if mean < target as f64 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    colors((lo + hi) * 0.5)
}

fn separation(pair: [[f32; 4]; 2]) -> f64 {
    let a = ycc(encoded(pair[0]));
    let b = ycc(encoded(pair[1]));
    (((a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)) / 2.0).sqrt()
}

fn boundary_fixture(
    pair: [[f32; 4]; 2],
    raw: bool,
    amplitude: f32,
    chroma_only: bool,
) -> Vec<[f32; 4]> {
    (0..WIDTH * HEIGHT)
        .map(|i| {
            let (x, y) = (i % WIDTH, i / WIDTH);
            let clean = pair[usize::from(x >= WIDTH / 2)];
            let mut linear = [0, 1, 2]
                .map(|c| decode(encode(clean[c]) + amplitude / 255.0 * noise(x, y, c as u32)));
            // Chroma-only noise: hold each region's linear luminance constant.
            // Otherwise unfiltered luminance noise can reappear as *encoded*
            // chroma variance in saturated colors, obscuring chroma-filter work.
            if chroma_only {
                linear[1] = clean[1]
                    - (0.2126 * (linear[0] - clean[0]) + 0.0722 * (linear[2] - clean[2])) / 0.7152;
            }
            assert!(
                linear.iter().all(|v| *v > 0.0 && *v < 1.0),
                "fixture clipping invalidates noise statistics"
            );
            let rgb = linear.map(|v| if raw { v } else { encode(v) });
            [rgb[0], rgb[1], rgb[2], 1.0]
        })
        .collect()
}

fn sigma(pixels: &[[f32; 4]], clean: &[[f32; 4]], side: u32) -> f64 {
    let (mut sum, mut squares, mut n) = ([0.0; 2], [0.0; 2], 0.0);
    for y in MARGIN..HEIGHT - MARGIN {
        for x in side * WIDTH / 2 + MARGIN..(side + 1) * WIDTH / 2 - MARGIN {
            let i = (y * WIDTH + x) as usize;
            let a = ycc(encoded(pixels[i]));
            let b = ycc(encoded(clean[i]));
            for c in 0..2 {
                let d = a[c + 1] - b[c + 1];
                sum[c] += d;
                squares[c] += d * d;
            }
            n += 1.0;
        }
    }
    assert!(n > 0.0);
    ((0..2)
        .map(|c| (squares[c] / n - (sum[c] / n).powi(2)).max(0.0))
        .sum::<f64>()
        / 2.0)
        .sqrt()
}

fn boundary_error(output: &[[f32; 4]], clean: &[[f32; 4]]) -> (f64, f64) {
    let (mut max_error, mut max_bias) = (0.0f64, 0.0f64);
    for x in WIDTH / 2 - 16..WIDTH / 2 + 16 {
        let mut sum = [0.0; 3];
        for y in MARGIN..HEIGHT - MARGIN {
            let i = (y * WIDTH + x) as usize;
            let a = encoded(output[i]);
            let b = encoded(clean[i]);
            for c in 0..3 {
                let d = a[c] - b[c];
                sum[c] += d;
                max_error = max_error.max(d.abs());
            }
        }
        for s in sum {
            max_bias = max_bias.max((s / (HEIGHT - 2 * MARGIN) as f64).abs());
        }
    }
    (max_error, max_bias)
}

#[test]
fn test_gpu_denoise_boundary_matrix() {
    let Some(gpu) = Harness::new() else {
        return;
    };
    let mut failures = Vec::new();
    for target in [30.0, 128.0, 200.0] {
        for equal_luma in [false, true] {
            for class in ["floor", "ratio", "wide", "subtle"] {
                let subtle = class == "subtle";
                // Include near-floor strong edges, not just very saturated colors.
                let (mut lo, mut hi) = (0.0, 0.95);
                for _ in 0..30 {
                    let mid = (lo + hi) * 0.5;
                    if separation(boundary_colors(target, mid, equal_luma))
                        < if subtle {
                            4.0
                        } else if class == "ratio" {
                            if target == 30.0 { 16.0 } else { 32.0 }
                        } else {
                            12.05
                        }
                    {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                let pair = boundary_colors(
                    target,
                    if class == "wide" {
                        0.9
                    } else {
                        (lo + hi) * 0.5
                    },
                    equal_luma,
                );
                let delta = separation(pair);
                let headroom = pair
                    .iter()
                    .flat_map(|p| encoded(*p))
                    .map(|v| v.min(255.0 - v))
                    .fold(f64::INFINITY, f64::min);
                let amplitude = if class == "wide" {
                    12.0f64.min(delta / 3.0).min(headroom * 0.75) as f32
                } else if class == "ratio" {
                    if target == 30.0 { 6.0 } else { 12.0 }
                } else {
                    3.0
                };
                for raw in [false, true] {
                    for chroma_only in [true, false] {
                        let clean_input = boundary_fixture(pair, raw, 0.0, chroma_only);
                        let noisy_input = boundary_fixture(pair, raw, amplitude, chroma_only);
                        let clean = gpu.run(&clean_input, WIDTH, raw, [0.0; 4], false);
                        let noisy = gpu.run(&noisy_input, WIDTH, raw, [0.0; 4], false);
                        let side_noise = [sigma(&noisy, &clean, 0), sigma(&noisy, &clean, 1)];
                        let before = side_noise[0].max(side_noise[1]);
                        let brightness = pair.map(|p| ycc(encoded(p))[0]);
                        assert!(
                            ((brightness[0] + brightness[1]) * 0.5 - target as f64).abs() < 0.001
                        );
                        eprintln!(
                            "fixture target={target} raw={raw} equal_luma={equal_luma} class={class} chroma_only={chroma_only}: Y={brightness:?}, amplitude={amplitude}, sigma_sides={side_noise:?}"
                        );
                        let strong = delta >= 12.0f64.max(6.0 * before);
                        assert_eq!(strong, !subtle);
                        for step in [1.0, 2.0, 4.0] {
                            for chroma in [50.0, 100.0] {
                                let settings = [0.0, 50.0, chroma, step];
                                let filtered_clean =
                                    gpu.run(&clean_input, WIDTH, raw, settings, true);
                                let filtered_noisy =
                                    gpu.run(&noisy_input, WIDTH, raw, settings, true);
                                let (error, _) = boundary_error(&filtered_clean, &clean);
                                let (_, bias) = boundary_error(&filtered_noisy, &clean);
                                let after = sigma(&filtered_noisy, &clean, 0).max(sigma(
                                    &filtered_noisy,
                                    &clean,
                                    1,
                                ));
                                let label = format!(
                                    "Y={target} equal_luma={equal_luma} raw={raw} class={class} chroma_only={chroma_only} step={step} C={chroma}"
                                );
                                eprintln!(
                                    "{label}: delta={delta:.3} sigma={before:.3}->{after:.3} clean_error={error:.3} noisy_bias={bias:.3}"
                                );
                                // Additional anti-inertness gates, declared before tuning:
                                // C50 removes >=20% sigma, C100 removes >=50%.
                                if strong
                                    && (error > 3.0
                                        || bias > 3.0
                                        || (chroma_only
                                            && after
                                                >= before * if chroma == 50.0 { 0.8 } else { 0.5 }))
                                {
                                    failures.push(label);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "boundary acceptance failures: {failures:#?}"
    );
}

#[test]
fn test_gpu_denoise_guide_headroom_and_encoding() {
    let Some(gpu) = Harness::new() else {
        return;
    };
    let values = [-0.1, -0.001, 0.0, 0.0031308, 0.2, 1.0, 2.0, 8.0];
    let input: Vec<_> = values.map(|v| [v, v, v, 1.0]).into();
    let guides = gpu.run_mode(&input, 8, true, [0.0; 4], 2);
    for (guide, value) in guides.iter().zip(values) {
        assert!(guide.iter().all(|v| v.is_finite()));
        assert!((guide[0] - encode(value)).abs() < 2e-5);
        assert!(guide[1].abs() < 2e-6 && guide[2].abs() < 2e-6);
    }
    for value in [0.0, 2.0, 8.0] {
        let input = vec![[value, value, value, 1.0]; 64];
        let output = gpu.run(&input, 8, true, [0.0, 50.0, 100.0, 4.0], true);
        assert!(output.iter().all(|p| (p[0] - value).abs() < 2e-5));
    }
    // Equivalent linear samples in two real input encodings. Compare before
    // RAW-specific downstream tone rendering, which is intentionally different.
    let linear = independence_fixture(64, true);
    let srgb: Vec<_> = linear
        .iter()
        .map(|p| [encode(p[0]), encode(p[1]), encode(p[2]), 1.0])
        .collect();
    for step in [1.0, 2.0, 4.0] {
        let a = gpu.run(&linear, 64, true, [0.0, 50.0, 100.0, step], true);
        let b = gpu.run(&srgb, 64, false, [0.0, 50.0, 100.0, step], true);
        let error = a
            .iter()
            .zip(&b)
            .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs()))
            .fold(0.0f32, f32::max);
        assert!(error < 2e-6, "encoding guide/filter mismatch: {error:e}");
    }
}

#[test]
fn test_gpu_denoise_luminance_edge_diagnostic() {
    let Some(gpu) = Harness::new() else {
        return;
    };
    for target in [30.0, 128.0, 200.0] {
        for raw in [false, true] {
            let input: Vec<_> = (0..64 * 64)
                .map(|i| {
                    let v = (target + if i % 64 < 32 { -10.0 } else { 10.0 }) / 255.0;
                    let v = if raw { decode(v) } else { v };
                    [v, v, v, 1.0]
                })
                .collect();
            let output = gpu.run(&input, 64, raw, [100.0, 50.0, 0.0, 1.0], true);
            let retention =
                (encoded(output[32 * 64 + 32])[0] - encoded(output[32 * 64 + 31])[0]) / 20.0;
            eprintln!(
                "luminance edge diagnostic Y={target} raw={raw}: adjacent-column encoded contrast retained {:.2}%",
                retention * 100.0
            );
            assert!(retention.is_finite());
        }
    }
}

fn render_pipeline(
    processor: &super::GpuProcessor,
    input: &wgpu::TextureView,
    size: u32,
    raw: bool,
    js: serde_json::Value,
) -> Vec<u8> {
    let adjustments = crate::image_processing::get_all_adjustments_from_json(&js, raw, None);
    let request = super::RenderRequest {
        adjustments,
        mask_bitmaps: &[],
        lut: None,
        roi: None,
    };
    let (pixels, w, h, _, _) = processor
        .run(input, size, size, request, false, false, None)
        .expect("full pipeline");
    assert_eq!((w, h), (size, size));
    pixels
}

#[test]
fn test_gpu_denoise_full_pipeline_independence() {
    let Some(context) = test_gpu_context("denoise full-pipeline independence") else {
        return;
    };
    const SIZE: u32 = 512;
    let processor = super::GpuProcessor::new(context.clone(), SIZE, SIZE).unwrap();
    for raw in [false, true] {
        let pixels = independence_fixture(SIZE, raw);
        let img = image::DynamicImage::ImageRgba32F(
            image::ImageBuffer::from_raw(SIZE, SIZE, pixels.into_iter().flatten().collect())
                .unwrap(),
        );
        let input = super::tests::upload_rgba16f(&context, &img);
        let mut baseline: Option<Vec<u8>> = None;
        for strength in [30.0, 70.0, 100.0] {
            let output = render_pipeline(
                &processor,
                &input,
                SIZE,
                raw,
                serde_json::json!({
                    "denoiseEnabled": true, "denoiseStrength": strength, "denoiseDetail": 50.0, "denoiseChroma": 60.0
                }),
            );
            if let Some(base) = &baseline {
                let max = output
                    .iter()
                    .zip(base)
                    .map(|(a, b)| u8::abs_diff(*a, *b))
                    .max()
                    .unwrap();
                assert!(
                    max <= 1,
                    "full-pipeline independence raw={raw} strength={strength}: {max}"
                );
            } else {
                baseline = Some(output);
            }
        }
    }
}

#[test]
fn test_gpu_denoise_full_resolution_scale_and_tiles() {
    let Some(context) = test_gpu_context("denoise real-resolution integration") else {
        return;
    };
    for (size, raw) in [(512, false), (2160, false), (4320, true)] {
        // A clean equal-linear-luminance checker with boundaries on real tile
        // seams and across both axes. The full RAW pipeline gets its own baseline.
        let pair = boundary_colors(128.0, 0.6, true);
        let img =
            image::DynamicImage::ImageRgba32F(image::ImageBuffer::from_fn(size, size, |x, y| {
                let p = pair[((x / 2048 + y / 2048) % 2) as usize];
                let rgb = [0, 1, 2].map(|c| if raw { p[c] } else { encode(p[c]) });
                image::Rgba([rgb[0], rgb[1], rgb[2], 1.0])
            }));
        let processor = super::GpuProcessor::new(context.clone(), size, size).unwrap();
        let input = super::tests::upload_rgba16f(&context, &img);
        let baseline = render_pipeline(&processor, &input, size, raw, serde_json::json!({}));
        let zero = render_pipeline(
            &processor,
            &input,
            size,
            raw,
            serde_json::json!({
                "denoiseEnabled": true, "denoiseStrength": 0.0, "denoiseDetail": 50.0, "denoiseChroma": 0.0
            }),
        );
        assert_eq!(baseline, zero, "zero strengths size={size} raw={raw}");
        let disabled = render_pipeline(
            &processor,
            &input,
            size,
            raw,
            serde_json::json!({
                "denoiseEnabled": false, "denoiseStrength": 100.0, "denoiseChroma": 100.0
            }),
        );
        assert_eq!(baseline, disabled, "disabled size={size} raw={raw}");
        for chroma in [50.0, 100.0] {
            let start = std::time::Instant::now();
            let output = render_pipeline(
                &processor,
                &input,
                size,
                raw,
                serde_json::json!({
                    "denoiseEnabled": true, "denoiseStrength": 0.0, "denoiseDetail": 50.0, "denoiseChroma": chroma
                }),
            );
            let error = output
                .iter()
                .zip(&baseline)
                .map(|(a, b)| u8::abs_diff(*a, *b))
                .max()
                .unwrap();
            eprintln!(
                "full pipeline size={size} raw={raw} C={chroma}: error={error}, render+readback={:?}",
                start.elapsed()
            );
            assert!(error <= 3, "clean tile boundary changed: {error}");
        }
    }
}

#[test]
fn test_gpu_denoise_step4_matches_production_function() {
    let Some(gpu) = Harness::new() else {
        return;
    };
    const SIZE: u32 = 4320;
    const PERIOD: u32 = 64;
    // Match production f16 upload precision before invoking the float harness.
    let pattern: Vec<_> = independence_fixture(PERIOD, true)
        .into_iter()
        .map(|p| p.map(|v| half::f16::from_f32(v).to_f32()))
        .collect();
    // Repeat a periodic scene with a halo so the harness's clamped boundaries
    // cannot influence the central period used as a reference.
    let tiled: Vec<_> = (0..(PERIOD * 3).pow(2))
        .map(|i| pattern[(((i / (PERIOD * 3)) % PERIOD) * PERIOD + i % PERIOD) as usize])
        .collect();
    let filtered = gpu.run(&tiled, PERIOD * 3, true, [0.0, 50.0, 100.0, 4.0], true);
    let wrong_scale = gpu.run(&tiled, PERIOD * 3, true, [0.0, 50.0, 100.0, 1.0], true);
    let scale_difference = filtered
        .iter()
        .zip(wrong_scale)
        .flat_map(|(a, b)| (0..3).map(move |c| (a[c] - b[c]).abs()))
        .fold(0.0f32, f32::max);
    assert!(
        scale_difference > 0.001,
        "fixture must distinguish scale 1 from 4"
    );
    let make_image = |reference: bool| {
        image::DynamicImage::ImageRgba32F(image::ImageBuffer::from_fn(SIZE, SIZE, |x, y| {
            let p = if reference {
                filtered[(((y % PERIOD + PERIOD) * PERIOD * 3) + x % PERIOD + PERIOD) as usize]
            } else {
                pattern[((y % PERIOD) * PERIOD + x % PERIOD) as usize]
            };
            image::Rgba(p)
        }))
    };
    let processor = super::GpuProcessor::new(gpu.context.clone(), SIZE, SIZE).unwrap();
    let original = super::tests::upload_rgba16f(&gpu.context, &make_image(false));
    let actual = render_pipeline(
        &processor,
        &original,
        SIZE,
        true,
        serde_json::json!({
            "denoiseEnabled": true, "denoiseStrength": 0.0, "denoiseDetail": 50.0, "denoiseChroma": 100.0
        }),
    );
    let reference = super::tests::upload_rgba16f(&gpu.context, &make_image(true));
    let expected = render_pipeline(&processor, &reference, SIZE, true, serde_json::json!({}));
    let mut max_error = 0;
    for y in MARGIN..SIZE - MARGIN {
        for x in MARGIN..SIZE - MARGIN {
            let i = ((y * SIZE + x) * 4) as usize;
            for c in 0..3 {
                max_error = max_error.max(u8::abs_diff(actual[i + c], expected[i + c]));
            }
        }
    }
    eprintln!(
        "4320 RAW production scale/tile comparison: max={max_error}/255; wrong-scale linear difference={scale_difference}"
    );
    // Reference makes one extra f16 round-trip before the same RAW renderer.
    assert!(
        max_error <= 1,
        "production scale/tile path differs from denoise function"
    );
}

/// Optional repeatable measurement of the actual denoise GPU pass. To compare
/// a baseline, supply its unmodified production shader via DENOISE_BENCH_SHADER.
#[test]
#[ignore = "large GPU timing run; run explicitly on matched hardware"]
fn benchmark_gpu_denoise() {
    let source = std::env::var("DENOISE_BENCH_SHADER")
        .ok()
        .map(|path| std::fs::read_to_string(path).unwrap())
        .unwrap_or_else(|| include_str!("../shaders/shader.wgsl").to_string());
    let Some(gpu) = Harness::with_source(&source) else {
        return;
    };
    for raw in [false, true] {
        for size in [512, 2160, 4320] {
            let pixels = independence_fixture(size, raw);
            let settings = [100.0, 50.0, 100.0, size as f32 / 1080.0];
            let _warmup = gpu.run(&pixels, size, raw, settings, true);
            let mut gpu_ms = Vec::new();
            let mut wall_ms = Vec::new();
            for _ in 0..5 {
                let start = std::time::Instant::now();
                let _output = gpu.run(&pixels, size, raw, settings, true);
                wall_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                gpu_ms.push(
                    gpu.last_gpu_ms
                        .get()
                        .expect("timestamp queries required for GPU timings"),
                );
            }
            gpu_ms.sort_by(f64::total_cmp);
            wall_ms.sort_by(f64::total_cmp);
            eprintln!(
                "BENCH raw={raw} size={size} GPU median={:.3}ms range={:.3}..{:.3}ms upload+dispatch+readback median={:.3}ms",
                gpu_ms[2], gpu_ms[0], gpu_ms[4], wall_ms[2]
            );
        }
    }
}

/// User-provided RAWs and generated renders stay outside tracked fixtures.
/// Uses the production decoder directly so embedded-preview fallback cannot
/// masquerade as RAW coverage; preprocessing matches default loader settings.
#[test]
#[ignore = "requires local RAW samples and an output directory"]
fn review_denoise_raw_images() {
    use image::GenericImageView;
    let paths = std::env::var_os("DENOISE_RAW_IMAGES").expect("DENOISE_RAW_IMAGES required");
    let out = std::path::PathBuf::from(
        std::env::var_os("DENOISE_REVIEW_OUT").expect("DENOISE_REVIEW_OUT required"),
    );
    let baseline_shader = std::fs::read_to_string(
        std::env::var_os("DENOISE_REVIEW_BASELINE").expect("DENOISE_REVIEW_BASELINE required"),
    )
    .unwrap();
    std::fs::create_dir_all(&out).unwrap();
    let Some(context) = test_gpu_context("real RAW denoise review") else {
        return;
    };
    for path in std::env::split_paths(&paths) {
        let bytes = std::fs::read(&path).unwrap();
        let name = path.file_stem().unwrap().to_string_lossy();
        eprintln!(
            "RAW {name}: sha256={} ISO={:?} exposure_seconds={:?}",
            {
                use sha2::Digest;
                hex::encode(sha2::Sha256::digest(&bytes))
            },
            crate::exif_processing::read_iso(path.to_str().unwrap(), &bytes),
            crate::exif_processing::read_exposure_time_secs(path.to_str().unwrap(), &bytes)
        );
        let mut img = crate::raw_processing::develop_raw_image(
            &bytes,
            false,
            2.5,
            crate::app_settings::default_linear_raw_mode(),
            None,
        )
        .expect("actual RAW development");
        crate::image_processing::remove_raw_artifacts_and_enhance(&mut img, 14.0, 0.35);
        let (w, h) = img.dimensions();
        eprintln!(
            "RAW {name}: developed {w}x{h}; default preprocessing color_nr=0.5 sharpening=0.35"
        );
        let processor = super::GpuProcessor::new(context.clone(), w, h).unwrap();
        let old_processor =
            super::GpuProcessor::with_shader(context.clone(), w, h, &baseline_shader).unwrap();
        let input = super::tests::upload_rgba16f(&context, &img);
        let render = |processor: &super::GpuProcessor,
                      input: &wgpu::TextureView,
                      w,
                      h,
                      strength: f32,
                      chroma: f32| {
            let adjustments = crate::image_processing::get_all_adjustments_from_json(
                &serde_json::json!({
                    "denoiseEnabled": true, "denoiseStrength": strength, "denoiseDetail": 50.0, "denoiseChroma": chroma
                }),
                true,
                None,
            );
            let request = super::RenderRequest {
                adjustments,
                mask_bitmaps: &[],
                lut: None,
                roi: None,
            };
            let (pixels, ow, oh, _, _) = processor
                .run(input, w, h, request, false, false, None)
                .unwrap();
            image::RgbaImage::from_raw(ow, oh, pixels).unwrap()
        };
        for (strength, chroma) in [(0.0, 0.0), (60.0, 60.0), (100.0, 100.0)] {
            let settings = format!("s{strength:.0}-c{chroma:.0}");
            let start = std::time::Instant::now();
            let current = render(&processor, &input, w, h, strength, chroma);
            eprintln!(
                "RAW {name} {settings}: current full render+readback {:?}",
                start.elapsed()
            );
            current
                .save(out.join(format!("{name}-{settings}-current.png")))
                .unwrap();
            image::imageops::resize(
                &current,
                960,
                960 * h / w,
                image::imageops::FilterType::Lanczos3,
            )
            .save(out.join(format!("{name}-{settings}-overview.png")))
            .unwrap();
            if strength == 0.0 {
                continue;
            }
            let start = std::time::Instant::now();
            let old = render(&old_processor, &input, w, h, strength, chroma);
            eprintln!(
                "RAW {name} {settings}: baseline full render+readback {:?}",
                start.elapsed()
            );
            old.save(out.join(format!("{name}-{settings}-baseline.png")))
                .unwrap();
            // Three 100% crops. Columns: previous shader, corrected shader.
            let mut contact = image::RgbaImage::new(1024, 1536);
            for (row, (cx, cy)) in [(w / 4, h / 4), (w / 2, h / 2), (3 * w / 4, 3 * h / 4)]
                .into_iter()
                .enumerate()
            {
                let x = cx.saturating_sub(256).min(w - 512);
                let y = cy.saturating_sub(256).min(h - 512);
                image::imageops::replace(
                    &mut contact,
                    &image::imageops::crop_imm(&old, x, y, 512, 512).to_image(),
                    0,
                    row as i64 * 512,
                );
                image::imageops::replace(
                    &mut contact,
                    &image::imageops::crop_imm(&current, x, y, 512, 512).to_image(),
                    512,
                    row as i64 * 512,
                );
            }
            contact
                .save(out.join(format!("{name}-{settings}-crops.png")))
                .unwrap();
            let preview = crate::image_processing::downscale_f32_image(&img, 1920, 1920);
            let (pw, ph) = preview.dimensions();
            let preview_processor = super::GpuProcessor::new(context.clone(), pw, ph).unwrap();
            let preview_input = super::tests::upload_rgba16f(&context, &preview);
            let rendered_preview =
                render(&preview_processor, &preview_input, pw, ph, strength, chroma);
            rendered_preview
                .save(out.join(format!("{name}-{settings}-preview.png")))
                .unwrap();
            let baseline_preview_processor =
                super::GpuProcessor::with_shader(context.clone(), pw, ph, &baseline_shader)
                    .unwrap();
            let baseline_preview = render(
                &baseline_preview_processor,
                &preview_input,
                pw,
                ph,
                strength,
                chroma,
            );
            baseline_preview
                .save(out.join(format!("{name}-{settings}-baseline-preview.png")))
                .unwrap();
            let baseline_resized =
                image::imageops::resize(&old, pw, ph, image::imageops::FilterType::Lanczos3);
            let baseline_mad = baseline_preview
                .pixels()
                .zip(baseline_resized.pixels())
                .map(|(a, b)| (0..3).map(|c| u8::abs_diff(a[c], b[c]) as f64).sum::<f64>())
                .sum::<f64>()
                / (pw * ph * 3) as f64;
            eprintln!(
                "RAW {name} {settings}: baseline preview vs resized export mean RGB difference={baseline_mad:.3}/255"
            );
            let resized_export =
                image::imageops::resize(&current, pw, ph, image::imageops::FilterType::Lanczos3);
            let mad = rendered_preview
                .pixels()
                .zip(resized_export.pixels())
                .map(|(a, b)| (0..3).map(|c| u8::abs_diff(a[c], b[c]) as f64).sum::<f64>())
                .sum::<f64>()
                / (pw * ph * 3) as f64;
            eprintln!(
                "RAW {name} {settings}: {pw}x{ph} preview vs resized export mean RGB difference={mad:.3}/255"
            );
        }
    }
}
