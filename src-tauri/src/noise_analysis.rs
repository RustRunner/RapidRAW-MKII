//! Developed-image source measurements in explicit linear and encoded domains.
//! Version 3 includes separately qualified black-clipped residuals. It estimates
//! developed noise, not sensor variance before clipping. Legacy Glare analysis
//! remains a separate unchanged full-image measurement.
use image::{DynamicImage, Rgb32FImage};
use rayon::prelude::*;
use serde::Serialize;

pub const VERSION: u32 = 3;
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
const MAX_CLIPPED_STRUCTURE_RATIO: f64 = 0.9;

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
    pub accepted_unclipped_patches: usize,
    pub accepted_black_clipped_patches: usize,
    pub black_clipped_fraction: f32,
    pub accepted_pixels: usize,
    pub represented_pixels: usize,
    /// Below two source code steps per RGB component, a MAD scale can lock
    /// to discrete residual levels. These channels are unresolved, not zero.
    pub quantization_limited: [bool; 3],
    pub structure_ratio: f32,
    pub lag1: f32,
    pub lag8: f32,
    pub highpass_to_marginal: f32,
    pub increment_disagreement: f32,
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
    pub qualified_unclipped_patches: usize,
    pub qualified_black_clipped_patches: usize,
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
// A centered long-separation increment removes a planar signal without
// removing all three fitted noise modes. At separations beyond the declared
// short-range correlation support, Var(X[i+d]-X[i])/2 estimates marginal
// variance. Keep this diagnostic separate from white-noise highpass gain.
fn difference_sigma(values: &[f64], distance: usize) -> f64 {
    let mut variances = Vec::new();
    for (dx, dy) in [(distance, 0), (0, distance)] {
        let mut differences = Vec::new();
        for y in 0..SIDE - dy {
            for x in 0..SIDE - dx {
                let d = values[(y + dy) * SIDE + x + dx] - values[y * SIDE + x];
                if d.is_finite() {
                    differences.push(d);
                }
            }
        }
        if differences.is_empty() {
            continue;
        }
        let mean = differences.iter().sum::<f64>() / differences.len() as f64;
        variances.push(
            differences.iter().map(|d| (d - mean).powi(2)).sum::<f64>()
                / differences.len() as f64
                / 2.0,
        );
    }
    (variances.iter().sum::<f64>() / variances.len().max(1) as f64).sqrt()
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
    black_clipped: usize,
    upper_clipped: usize,
    structure: f64,
    lag1: f64,
    lag8: f64,
    highpass_ratio: f64,
    increment_disagreement: f64,
}
fn patch(rgb: &Rgb32FImage, x0: u32, y0: u32, is_linear: bool, _code_step: f32) -> Patch {
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
            // valid. An exact 1.0 plateau is an upper-endpoint diagnostic,
            // including float preprocessing clamps; it is not sensor white.
            if finite {
                let black = source.contains(&0.0);
                let upper = source.contains(&1.0);
                out.black_clipped += usize::from(black);
                out.upper_clipped += usize::from(upper);
                out.clipped += usize::from(black || upper);
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
    if out.valid < SIDE * SIDE * 3 / 4 || clipping_class(&out).is_none() {
        return out;
    }
    for (d, domain) in domains.iter().enumerate() {
        for c in 0..3 {
            let values: Vec<_> = domain.iter().map(|p| p[c]).collect();
            let residual = residuals(&values);
            let finite: Vec<_> = residual.iter().copied().filter(|x| x.is_finite()).collect();
            let sigma = mad(&finite);
            let center = finite.iter().sum::<f64>() / finite.len() as f64;
            let mut marginal = (finite.iter().map(|v| (v - center).powi(2)).sum::<f64>()
                / finite.len() as f64)
                .sqrt();
            let local_marginal = marginal;
            let residual: Vec<_> = residual.iter().map(|v| v - center).collect();
            if clipping_class(&out) == Some(ClippingClass::BlackClipped)
                && marginal > 1e-10
                && lag(&residual, 1) > 0.3
            {
                let far = difference_sigma(&values, 32);
                let near = difference_sigma(&values, 24);
                out.increment_disagreement = out
                    .increment_disagreement
                    .max((far - near).abs() / far.max(1e-10));
                marginal = marginal.max(far);
            }
            if d == 0 {
                out.linear[c] = marginal;
                out.linear_mad_scale[c] = sigma;
            } else {
                out.encoded[c] = marginal;
                out.encoded_mad_scale[c] = sigma;
            }
            let diagnostic_scale = if clipping_class(&out) == Some(ClippingClass::BlackClipped) {
                // Structure is local residual energy. A long-separation
                // variance estimate must not make smooth curvature look flat.
                local_marginal
            } else {
                sigma
            };
            if d != 0 || diagnostic_scale < 1e-10 {
                continue;
            }
            // Robust location need not equal the mean after clipping. Lag
            // and block structure describe centered fluctuations, not bias.
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
                / diagnostic_scale;
            out.structure = out.structure.max(structure);
            out.lag1 = out.lag1.max(lag(&residual, 1));
            out.lag8 = out.lag8.max(lag(&residual, 8));
            out.highpass_ratio = out.highpass_ratio.max(highpass(&values) / diagnostic_scale);
            // Negative lag correlations or large highpass gain are also
            // structure signals (e.g. repetitive stripes), not white noise.
            if [1, 4, 8].iter().any(|&d| lag(&residual, d) < -0.15) || out.highpass_ratio > 1.6 {
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
#[derive(Debug, PartialEq, Eq)]
enum ClippingClass {
    Unclipped,
    BlackClipped,
}
fn clipping_class(p: &Patch) -> Option<ClippingClass> {
    if p.clipped <= SIDE * SIDE / 100 {
        Some(ClippingClass::Unclipped)
    } else if p.upper_clipped <= SIDE * SIDE / 100 && p.black_clipped <= SIDE * SIDE * 60 / 100 {
        Some(ClippingClass::BlackClipped)
    } else {
        None
    }
}

fn structure_limit(p: &Patch) -> f64 {
    if clipping_class(p) == Some(ClippingClass::BlackClipped) {
        MAX_CLIPPED_STRUCTURE_RATIO
    } else {
        MAX_STRUCTURE_RATIO
    }
}

fn qualifies(p: &Patch) -> bool {
    p.valid >= SIDE * SIDE * 3 / 4
        && clipping_class(p).is_some()
        && p.lag8 <= MAX_LAG8
        && p.increment_disagreement <= 0.2
        && p.structure <= structure_limit(p)
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
        } else if clipping_class(p).is_none() {
            quality.rejected_clipped += 1;
        } else if p.lag8 > MAX_LAG8 || p.increment_disagreement > 0.2 {
            quality.rejected_long_correlation += 1;
        } else if p.structure > structure_limit(p) {
            quality.rejected_structure += 1;
        } else {
            match clipping_class(p).unwrap() {
                ClippingClass::Unclipped => quality.qualified_unclipped_patches += 1,
                ClippingClass::BlackClipped => quality.qualified_black_clipped_patches += 1,
            }
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
        let represented_pixels = group.iter().map(|p| p.valid).sum();
        let has_black_clipping = group
            .iter()
            .any(|p| clipping_class(p) == Some(ClippingClass::BlackClipped));
        // Rare clipped excursions carry real variance. Ranking away those
        // patches, or taking a median of their scales, selects away noise.
        // Keep all absolutely qualified patches in the clipped class.
        if !has_black_clipping {
            group.sort_by(|a, b| a.structure.total_cmp(&b.structure));
            group.truncate(group.len().div_ceil(4).max(4));
        }
        let scale = |get: fn(&Patch) -> [f64; 3]| {
            DomainNoise::from_array(std::array::from_fn(|c| {
                if has_black_clipping {
                    (group
                        .iter()
                        .map(|p| get(p)[c].powi(2) * p.valid as f64)
                        .sum::<f64>()
                        / group.iter().map(|p| p.valid).sum::<usize>() as f64)
                        .sqrt()
                } else {
                    median(group.iter().map(|p| get(p)[c]).collect())
                }
            }))
        };
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
            linear: scale(|p| p.linear),
            encoded: scale(|p| p.encoded),
            encoded_mad_scale: DomainNoise::from_array(std::array::from_fn(|c| {
                median(group.iter().map(|p| p.encoded_mad_scale[c]).collect())
            })),
            linear_mad_scale: DomainNoise::from_array(std::array::from_fn(|c| {
                median(group.iter().map(|p| p.linear_mad_scale[c]).collect())
            })),
            sampled_patches: counts[i],
            accepted_patches: group.len(),
            accepted_unclipped_patches: group
                .iter()
                .filter(|p| clipping_class(p) == Some(ClippingClass::Unclipped))
                .count(),
            accepted_black_clipped_patches: group
                .iter()
                .filter(|p| clipping_class(p) == Some(ClippingClass::BlackClipped))
                .count(),
            black_clipped_fraction: med(|p| p.black_clipped as f64 / (SIDE * SIDE) as f64),
            accepted_pixels: group.iter().map(|p| p.valid).sum(),
            represented_pixels,
            structure_ratio: med(|p| p.structure),
            lag1: med(|p| p.lag1),
            lag8: med(|p| p.lag8),
            highpass_to_marginal: med(|p| p.highpass_ratio),
            increment_disagreement: med(|p| p.increment_disagreement),
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
