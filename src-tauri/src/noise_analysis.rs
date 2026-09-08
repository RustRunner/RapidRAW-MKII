//! Experimental developed-image measurements, compiled only for tests.
//! Version 2 passes the unclipped domain gates using centered residual
//! variance. Default high-ISO RAWs exceed the initial clipping limit; see
//! bench/estimator-variance-validation.md before enabling a production path.
use image::{DynamicImage, Rgb32FImage};
use rayon::prelude::*;
use serde::Serialize;

pub const VERSION: u32 = 2;
pub const BRIGHTNESS: [f32; 6] = [
    16.0 / 255.0,
    30.0 / 255.0,
    64.0 / 255.0,
    128.0 / 255.0,
    200.0 / 255.0,
    240.0 / 255.0,
];
const SIDE: usize = 64;
const MAD_NORMAL: f64 = 0.6744897501960817;
// Qualification parameters are pinned by white/correlated/structure fixtures.
const MAX_STRUCTURE_RATIO: f64 = 0.85;
const MAX_LAG8: f64 = 0.8;

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct DomainNoise {
    pub sigma_y: f32,
    pub sigma_cb: f32,
    pub sigma_cr: f32,
}
impl DomainNoise {
    fn from_array(v: [f64; 3]) -> Self {
        Self {
            sigma_y: v[0] as f32,
            sigma_cb: v[1] as f32,
            sigma_cr: v[2] as f32,
        }
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct BrightnessBin {
    pub mean_encoded_y: f32,
    pub linear: DomainNoise,
    pub encoded: DomainNoise,
    pub encoded_mad_scale: DomainNoise,
    pub linear_mad_scale: DomainNoise,
    pub sampled_patches: usize,
    pub accepted_patches: usize,
    pub accepted_pixels: usize,
    /// Below two source code steps per RGB component, a MAD scale can lock
    /// to discrete residual levels. These channels are unresolved, not zero.
    pub quantization_limited: [bool; 3],
    pub structure_ratio: f32,
    pub lag1: f32,
    pub lag8: f32,
    pub highpass_to_marginal: f32,
}
#[derive(Debug, Clone, Default, Serialize)]
pub struct Quality {
    pub sampled_patches: usize,
    pub resampled_patches: usize,
    pub sampled_origins: Vec<[u32; 2]>,
    pub sampled_by_bin: [usize; 6],
    pub qualified_by_bin: [usize; 6],
    pub rejected_nonfinite: usize,
    pub rejected_clipped: usize,
    pub clipped_pixel_fraction: f32,
    pub min_patch_clipped_fraction: f32,
    pub max_patch_clipped_fraction: f32,
    pub rejected_structure: usize,
    pub rejected_long_correlation: usize,
    pub source_code_step: f32,
    pub insufficient_bins: usize,
}
#[derive(Debug, Clone, Serialize)]
pub struct NoiseMeasurement {
    pub version: u32,
    pub bins: Vec<BrightnessBin>,
    pub quality: Quality,
}
impl NoiseMeasurement {
    pub fn is_usable(&self) -> bool {
        !self.bins.is_empty() && self.quality.insufficient_bins == 0
    }
}
#[derive(Debug)]
pub struct SourceNoiseAnalysis {
    pub legacy_source: crate::denoising::NoiseEstimate,
    pub measurement: NoiseMeasurement,
}

pub fn analyze_source(image: &DynamicImage, is_linear: bool) -> SourceNoiseAnalysis {
    let rgb = image.to_rgb32f();
    let code_step = match image.color() {
        image::ColorType::Rgb32F | image::ColorType::Rgba32F => 0.0,
        c if c.bits_per_pixel() / u16::from(c.channel_count()) == 16 => 1.0 / 65535.0,
        _ => 1.0 / 255.0,
    };
    SourceNoiseAnalysis {
        legacy_source: crate::denoising::estimate_noise_rgb(&rgb),
        measurement: measure(&rgb, is_linear, code_step),
    }
}

pub fn encode(x: f32) -> f32 {
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}
pub fn decode(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}
pub fn ycbcr(rgb: [f32; 3]) -> [f64; 3] {
    let [r, g, b] = rgb.map(f64::from);
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    [y, 0.565 * (b - y), 0.713 * (r - y)]
}
fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    *values.select_nth_unstable_by(mid, f64::total_cmp).1
}
fn mad(v: &[f64]) -> f64 {
    let center = median(v.to_vec());
    median(v.iter().map(|x| (x - center).abs()).collect()) / MAD_NORMAL
}

// Small weighted least-squares solve; coordinates normalized to [-1, 1].
fn solve(mut a: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let pivot = (col..3).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        a.swap(col, pivot);
        if a[col][col].abs() < 1e-12 {
            return None;
        }
        let d = a[col][col];
        for k in col..4 {
            a[col][k] /= d;
        }
        for row in 0..3 {
            if row == col {
                continue;
            }
            let weight = a[row][col];
            for k in col..4 {
                a[row][k] -= weight * a[col][k];
            }
        }
    }
    Some([a[0][3], a[1][3], a[2][3]])
}
fn residuals(values: &[f64]) -> Vec<f64> {
    let mut fit = [
        median(values.iter().copied().filter(|x| x.is_finite()).collect()),
        0.0,
        0.0,
    ];
    let mut scale = f64::INFINITY;
    for _ in 0..4 {
        let mut system = [[0.0; 4]; 3];
        let mut residual = Vec::new();
        for (i, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                continue;
            }
            let v = [
                1.0,
                (i % SIDE) as f64 / 31.5 - 1.0,
                (i / SIDE) as f64 / 31.5 - 1.0,
            ];
            let error = value - (0..3).map(|k| v[k] * fit[k]).sum::<f64>();
            let weight = (1.5 * scale / error.abs().max(1e-15)).min(1.0);
            for row in 0..3 {
                for col in 0..3 {
                    system[row][col] += weight * v[row] * v[col];
                }
                system[row][3] += weight * v[row] * value;
            }
            residual.push(error);
        }
        if let Some(next) = solve(system) {
            fit = next;
        }
        scale = mad(&residual).max(1e-12);
    }
    values
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            value
                - fit[0]
                - fit[1] * ((i % SIDE) as f64 / 31.5 - 1.0)
                - fit[2] * ((i / SIDE) as f64 / 31.5 - 1.0)
        })
        .collect()
}
fn lag(residual: &[f64], distance: usize) -> f64 {
    let mut cross = 0.0;
    let mut power_a = 0.0;
    let mut power_b = 0.0;
    for y in 0..SIDE - distance {
        for x in 0..SIDE - distance {
            let a = residual[y * SIDE + x];
            for b in [
                residual[y * SIDE + x + distance],
                residual[(y + distance) * SIDE + x],
            ] {
                if a.is_finite() && b.is_finite() {
                    cross += a * b;
                    power_a += a * a;
                    power_b += b * b;
                }
            }
        }
    }
    cross / (power_a * power_b).sqrt().max(1e-24)
}
fn highpass(values: &[f64]) -> f64 {
    let mut response = Vec::new();
    for y in (1..SIDE - 1).step_by(2) {
        for x in (1..SIDE - 1).step_by(2) {
            let mut value = 0.0;
            for (ky, wy) in [1.0, -2.0, 1.0].iter().enumerate() {
                for (kx, wx) in [1.0, -2.0, 1.0].iter().enumerate() {
                    value += wy * wx * values[(y + ky - 1) * SIDE + x + kx - 1];
                }
            }
            if value.is_finite() {
                response.push(value.abs());
            }
        }
    }
    median(response) / (6.0 * MAD_NORMAL)
}
#[derive(Default)]
struct Patch {
    mean: f64,
    linear: [f64; 3],
    encoded: [f64; 3],
    encoded_mad_scale: [f64; 3],
    linear_mad_scale: [f64; 3],
    valid: usize,
    clipped: usize,
    structure: f64,
    lag1: f64,
    lag8: f64,
    highpass_ratio: f64,
}
fn patch(rgb: &Rgb32FImage, x0: u32, y0: u32, is_linear: bool, code_step: f32) -> Patch {
    let mut out = Patch::default();
    let mut linear_sum = [0.0f64; 3];
    let mut domains: [Vec<[f64; 3]>; 2] = [
        Vec::with_capacity(SIDE * SIDE),
        Vec::with_capacity(SIDE * SIDE),
    ];
    for y in 0..SIDE as u32 {
        for x in 0..SIDE as u32 {
            let source = rgb.get_pixel(x0 + x, y0 + y).0;
            let linear = if is_linear {
                source
            } else {
                source.map(decode)
            };
            let encoded = if is_linear {
                source.map(encode)
            } else {
                source
            };
            // A finite encoded source can overflow during decoding. Reject
            // it in both representations instead of turning missing residuals
            // into a zero-noise result.
            let finite = linear.iter().chain(&encoded).all(|c| c.is_finite());
            if finite {
                out.valid += 1;
                for c in 0..3 {
                    linear_sum[c] += linear[c] as f64;
                }
            }
            // RAW development clamps the lower endpoint to zero, and integer
            // endpoints are known quantization limits. Record exact zero
            // plateaus in float sources too. Negative/headroom values remain
            // valid; 1.0 in a float source does not establish sensor white.
            if finite
                && source
                    .iter()
                    .any(|&c| c == 0.0 || (code_step > 0.0 && c == 1.0))
            {
                out.clipped += 1;
            }
            domains[0].push(if finite { ycbcr(linear) } else { [f64::NAN; 3] });
            domains[1].push(if finite {
                ycbcr(encoded)
            } else {
                [f64::NAN; 3]
            });
        }
    }
    // Encode the spatial mean of linear RGB, rather than a noisy encoded
    // sample median. A plane's mean is its center; this avoids transfer-curve
    // skew and reduces brightness bucket jitter in high-noise shadows.
    out.mean = ycbcr(linear_sum.map(|v| encode((v / out.valid.max(1) as f64) as f32)))[0];
    if out.valid < SIDE * SIDE * 3 / 4 || out.clipped > SIDE * SIDE / 100 {
        return out;
    }
    for (d, domain) in domains.iter().enumerate() {
        for c in 0..3 {
            let values: Vec<_> = domain.iter().map(|p| p[c]).collect();
            let residual = residuals(&values);
            let finite: Vec<_> = residual.iter().copied().filter(|x| x.is_finite()).collect();
            let sigma = mad(&finite);
            let center = finite.iter().sum::<f64>() / finite.len() as f64;
            let marginal = (finite.iter().map(|v| (v - center).powi(2)).sum::<f64>()
                / finite.len() as f64)
                .sqrt();
            if d == 0 {
                out.linear[c] = marginal;
                out.linear_mad_scale[c] = sigma;
            } else {
                out.encoded[c] = marginal;
                out.encoded_mad_scale[c] = sigma;
            }
            if d != 0 || sigma < 1e-10 {
                continue;
            }
            let mut block_means = Vec::new();
            for by in 0..8 {
                for bx in 0..8 {
                    let samples: Vec<_> = (0..8)
                        .flat_map(|dy| (0..8).map(move |dx| (by * 8 + dy) * SIDE + bx * 8 + dx))
                        .map(|i| residual[i])
                        .filter(|v| v.is_finite())
                        .collect();
                    if !samples.is_empty() {
                        block_means.push(samples.iter().sum::<f64>() / samples.len() as f64);
                    }
                }
            }
            let structure = (block_means.iter().map(|v| v * v).sum::<f64>()
                / block_means.len().max(1) as f64)
                .sqrt()
                / sigma;
            out.structure = out.structure.max(structure);
            out.lag1 = out.lag1.max(lag(&residual, 1));
            out.lag8 = out.lag8.max(lag(&residual, 8));
            out.highpass_ratio = out.highpass_ratio.max(highpass(&values) / sigma);
            // Negative lag correlations or large highpass gain are also
            // structure signals (e.g. repetitive stripes), not white noise.
            if lag(&residual, 1) < -0.15 || out.highpass_ratio > 1.6 {
                out.structure = f64::MAX;
            }
        }
    }
    out
}

fn brightness_bin(p: &Patch) -> usize {
    (0..6)
        .min_by(|&a, &b| {
            (p.mean - BRIGHTNESS[a] as f64)
                .abs()
                .total_cmp(&(p.mean - BRIGHTNESS[b] as f64).abs())
        })
        .unwrap()
}
fn qualifies(p: &Patch) -> bool {
    p.valid >= SIDE * SIDE * 3 / 4
        && p.clipped <= SIDE * SIDE / 100
        && p.lag8 <= MAX_LAG8
        && p.structure <= MAX_STRUCTURE_RATIO
}

pub fn measure(rgb: &Rgb32FImage, is_linear: bool, code_step: f32) -> NoiseMeasurement {
    let nx = (rgb.width() as usize / SIDE).min(16);
    let ny = (rgb.height() as usize / SIDE).min(12);
    let origin = |i: usize, n: usize, extent: u32| {
        if n <= 1 {
            0
        } else {
            (i * (extent as usize - SIDE) / (n - 1)) as u32
        }
    };
    let mut coords: Vec<_> = (0..ny)
        .flat_map(|y| {
            (0..nx).map(move |x| (origin(x, nx, rgb.width()), origin(y, ny, rgb.height())))
        })
        .collect();
    let mut patches: Vec<_> = coords
        .par_iter()
        .map(|&(x, y)| patch(rgb, x, y, is_linear, code_step))
        .collect();
    let initial_count = patches.len();
    // Reserve the remainder of the 256-patch budget for nonoverlapping
    // neighbors of sparsely represented brightness regions. No sampled bin
    // is discarded, and rejected neighbors never count as qualified coverage.
    let mut visited = std::collections::HashSet::new();
    while patches.len() < 256 {
        let mut counts = [0usize; 6];
        let mut qualified = [0usize; 6];
        for p in &patches {
            counts[brightness_bin(p)] += 1;
            if qualifies(p) {
                qualified[brightness_bin(p)] += 1;
            }
        }
        let mut next = None;
        'search: for (i, p) in patches.iter().enumerate() {
            let bin = brightness_bin(p);
            if counts[bin] == 0 || qualified[bin] >= 4 {
                continue;
            }
            for ring in 1i64..=4 {
                for dy in -ring..=ring {
                    for dx in -ring..=ring {
                        if dx.abs().max(dy.abs()) != ring {
                            continue;
                        }
                        let x = coords[i].0 as i64 + dx * SIDE as i64;
                        let y = coords[i].1 as i64 + dy * SIDE as i64;
                        if x < 0
                            || y < 0
                            || x + SIDE as i64 > rgb.width() as i64
                            || y + SIDE as i64 > rgb.height() as i64
                        {
                            continue;
                        }
                        let point = (x as u32, y as u32);
                        if !visited.insert(point) {
                            continue;
                        }
                        if coords.iter().any(|&(a, b)| {
                            a.abs_diff(point.0) < SIDE as u32 && b.abs_diff(point.1) < SIDE as u32
                        }) {
                            continue;
                        }
                        next = Some(point);
                        break 'search;
                    }
                }
            }
        }
        let Some((x, y)) = next else {
            break;
        };
        coords.push((x, y));
        patches.push(patch(rgb, x, y, is_linear, code_step));
    }
    let mut quality = Quality {
        sampled_patches: patches.len(),
        resampled_patches: patches.len() - initial_count,
        sampled_origins: coords.iter().map(|&(x, y)| [x, y]).collect(),
        clipped_pixel_fraction: patches.iter().map(|p| p.clipped).sum::<usize>() as f32
            / (patches.len().max(1) * SIDE * SIDE) as f32,
        min_patch_clipped_fraction: patches.iter().map(|p| p.clipped).min().unwrap_or(0) as f32
            / (SIDE * SIDE) as f32,
        max_patch_clipped_fraction: patches.iter().map(|p| p.clipped).max().unwrap_or(0) as f32
            / (SIDE * SIDE) as f32,
        source_code_step: code_step,
        ..Default::default()
    };
    let mut groups: [Vec<&Patch>; 6] = Default::default();
    let mut counts = [0; 6];
    for p in &patches {
        let bin = brightness_bin(p);
        counts[bin] += 1;
        if p.valid < SIDE * SIDE * 3 / 4 {
            quality.rejected_nonfinite += 1;
        } else if p.clipped > SIDE * SIDE / 100 {
            quality.rejected_clipped += 1;
        } else if p.lag8 > MAX_LAG8 {
            quality.rejected_long_correlation += 1;
        } else if p.structure > MAX_STRUCTURE_RATIO {
            quality.rejected_structure += 1;
        } else {
            groups[bin].push(p);
        }
    }
    quality.sampled_by_bin = counts;
    quality.qualified_by_bin = std::array::from_fn(|i| groups[i].len());
    let mut bins = Vec::new();
    for (i, group) in groups.iter_mut().enumerate() {
        if counts[i] == 0 {
            continue;
        }
        if group.len() < 4 {
            quality.insufficient_bins += 1;
            continue;
        }
        group.sort_by(|a, b| a.structure.total_cmp(&b.structure));
        group.truncate(group.len().div_ceil(4).max(4));
        let med = |get: fn(&Patch) -> f64| median(group.iter().map(|p| get(p)).collect()) as f32;
        let source_sigmas: [f64; 3] = std::array::from_fn(|c| {
            median(
                group
                    .iter()
                    .map(|p| {
                        if is_linear {
                            p.linear_mad_scale[c]
                        } else {
                            p.encoded_mad_scale[c]
                        }
                    })
                    .collect(),
            )
        });
        let gains = [0.749615, 0.672688, 0.760181];
        bins.push(BrightnessBin {
            quantization_limited: std::array::from_fn(|c| {
                code_step > 0.0 && source_sigmas[c] < 2.0 * code_step as f64 * gains[c]
            }),
            mean_encoded_y: med(|p| p.mean),
            linear: DomainNoise::from_array(std::array::from_fn(|c| {
                median(group.iter().map(|p| p.linear[c]).collect())
            })),
            encoded: DomainNoise::from_array(std::array::from_fn(|c| {
                median(group.iter().map(|p| p.encoded[c]).collect())
            })),
            encoded_mad_scale: DomainNoise::from_array(std::array::from_fn(|c| {
                median(group.iter().map(|p| p.encoded_mad_scale[c]).collect())
            })),
            linear_mad_scale: DomainNoise::from_array(std::array::from_fn(|c| {
                median(group.iter().map(|p| p.linear_mad_scale[c]).collect())
            })),
            sampled_patches: counts[i],
            accepted_patches: group.len(),
            accepted_pixels: group.iter().map(|p| p.valid).sum(),
            structure_ratio: med(|p| p.structure),
            lag1: med(|p| p.lag1),
            lag8: med(|p| p.lag8),
            highpass_to_marginal: med(|p| p.highpass_ratio),
        });
    }
    NoiseMeasurement {
        version: VERSION,
        bins,
        quality,
    }
}

#[cfg(test)]
mod tests;
