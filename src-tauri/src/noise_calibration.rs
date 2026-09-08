//! Suggestions are derived from immutable source measurements and the current
//! captured Detail/native dimensions. They never become source-cache entries.
use crate::noise_analysis::{DomainNoise, NoiseMeasurement};
use serde::Serialize;
mod tables;

#[derive(Debug, Serialize)]
pub struct CalibratedNoiseEstimate {
    pub strength: f32,
    pub chroma: f32,
    pub calibration_version: u32,
    pub linear_bin_median: DomainNoise,
    pub encoded_bin_median: DomainNoise,
    pub native_chroma_spacing: u32,
    pub detail: f32,
    pub strength_range: [u8; 2],
    pub chroma_range: [u8; 2],
    pub measurement: NoiseMeasurement,
}
fn bracket(nodes: &[f32], value: f32) -> Option<Vec<(usize, f32)>> {
    if !value.is_finite() || value < nodes[0] || value > *nodes.last()? {
        return None;
    }
    if let Some(i) = nodes.iter().position(|&v| v == value) {
        return Some(vec![(i, 1.0)]);
    }
    let hi = nodes.iter().position(|&v| v > value)?;
    let lo = hi - 1;
    let t = (value - nodes[lo]) / (nodes[hi] - nodes[lo]);
    Some(vec![(lo, 1.0 - t), (hi, t)])
}
fn interpolate<const A: usize>(
    table: &[[[Option<u8>; 11]; A]; 7],
    b: &[(usize, f32)],
    s: &[(usize, f32)],
    a: &[(usize, f32)],
) -> Option<u8> {
    let mut result = 0.0;
    for &(bi, bw) in b {
        for &(ai, aw) in a {
            for &(si, sw) in s {
                if bw * aw * sw > 0.0 {
                    result += bw * aw * sw * table[bi][ai][si]? as f32;
                }
            }
        }
    }
    Some(result.round().clamp(0.0, 100.0) as u8)
}
pub(crate) fn lookup(
    chroma: bool,
    brightness: f32,
    sigma: f32,
    detail: f32,
    step: f32,
) -> Option<u8> {
    let b = bracket(&tables::BRIGHTNESS, brightness)?;
    let s = bracket(&tables::SIGMAS, sigma)?;
    if chroma {
        let a = vec![(tables::STEPS.iter().position(|&v| v == step)?, 1.0)];
        interpolate(&tables::CHROMA, &b, &s, &a)
    } else {
        let a = bracket(&tables::DETAILS, detail)?;
        interpolate(&tables::STRENGTH, &b, &s, &a)
    }
}
pub fn native_spacing(width: u32, height: u32) -> u32 {
    // WGSL round lowers to RoundEven; preserve its half-step behavior.
    ((width.min(height) as f32 / 1080.0)
        .round_ties_even()
        .max(1.0)) as u32
}
fn weighted_median(mut values: Vec<(f32, usize)>) -> f32 {
    values.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: usize = values.iter().map(|v| v.1).sum();
    let mut sum = 0;
    for (value, weight) in values {
        sum += weight;
        if sum * 2 >= total {
            return value;
        }
    }
    unreachable!("nonempty qualified bins have positive represented area")
}
pub fn suggest(
    measurement: &NoiseMeasurement,
    detail: f32,
    width: u32,
    height: u32,
) -> Result<CalibratedNoiseEstimate, String> {
    if measurement.version != tables::MEASUREMENT_VERSION
        || !measurement.is_usable()
        || measurement.bins.iter().any(|b| b.represented_pixels == 0)
    {
        return Err("Insufficient usable source data for calibrated denoise".into());
    }
    if !detail.is_finite() || !(0.0..=100.0).contains(&detail) {
        return Err("Invalid denoise Detail".into());
    }
    let spacing = native_spacing(width, height);
    let mut strengths = Vec::new();
    let mut chromas = Vec::new();
    for bin in &measurement.bins {
        let strength = if bin.quantization_limited[0] {
            0
        } else {
            lookup(
                false,
                bin.mean_encoded_y,
                bin.linear.sigma_y,
                detail,
                spacing as f32,
            )
            .ok_or(
                "Calibration does not cover this source noise, brightness, and Detail combination",
            )?
        };
        let chroma = if bin.quantization_limited[1] && bin.quantization_limited[2] {
            0
        } else {
            lookup(true,bin.mean_encoded_y,bin.linear.sigma_cb.max(bin.linear.sigma_cr),detail,spacing as f32).ok_or("Calibration does not cover this source chroma noise, brightness, or native spacing")?
        };
        // Use all absolutely qualified area, before any statistical ranking.
        // Keeping every clipped patch must not up-weight shadows relative to
        // an unclipped bin whose scale uses a representative quartile.
        strengths.push((strength as f32, bin.represented_pixels));
        chromas.push((chroma as f32, bin.represented_pixels));
    }
    let range = |v: &[(f32, usize)]| {
        [
            v.iter().map(|v| v.0 as u8).min().unwrap(),
            v.iter().map(|v| v.0 as u8).max().unwrap(),
        ]
    };
    let summary = |encoded: bool| {
        let get = |c: usize| {
            weighted_median(
                measurement
                    .bins
                    .iter()
                    .map(|b| {
                        let d = if encoded { b.encoded } else { b.linear };
                        ([d.sigma_y, d.sigma_cb, d.sigma_cr][c], b.represented_pixels)
                    })
                    .collect(),
            )
        };
        DomainNoise {
            sigma_y: get(0),
            sigma_cb: get(1),
            sigma_cr: get(2),
        }
    };
    Ok(CalibratedNoiseEstimate {
        strength_range: range(&strengths),
        chroma_range: range(&chromas),
        strength: weighted_median(strengths),
        chroma: weighted_median(chromas),
        calibration_version: tables::VERSION,
        linear_bin_median: summary(false),
        encoded_bin_median: summary(true),
        native_chroma_spacing: spacing,
        detail,
        measurement: measurement.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lookup_rejects_unvalidated_combinations_and_chroma_ignores_detail() {
        assert!(lookup(false, 0.5, 0.15, 50.0, 5.0).is_none());
        assert!(lookup(true, 0.5, 0.02, 50.0, 6.0).is_none());
        assert!(lookup(false, f32::NAN, 0.02, 50.0, 1.0).is_none());
        assert_eq!(
            lookup(true, 0.5, 0.02, 0.0, 5.0),
            lookup(true, 0.5, 0.02, 100.0, 5.0)
        );
        assert!(lookup(false, 0.2, 0.096, 100.0, 5.0).is_none());
        assert_eq!(native_spacing(8192, 5464), 5);
        assert_eq!(native_spacing(4000, 2700), 2);
        assert_eq!(native_spacing(5000, 3780), 4);
    }
    #[test]
    fn area_weighted_median_does_not_create_an_extra_shadow_weight() {
        assert_eq!(weighted_median(vec![(95.0, 4096), (20.0, 8192)]), 20.0);
        assert_eq!(weighted_median(vec![(95.0, 16384), (20.0, 8192)]), 95.0);
    }
}
