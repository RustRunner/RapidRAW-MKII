//! Glare Recovery: removal of veiling glare (window/windshield reflections).
//!
//! The veil is modeled as a smooth additive reflection layer bounded by the
//! image's local minima, estimated per channel as a lower envelope on a small
//! thumbnail. The main shader subtracts it and re-stretches in linear light;
//! this module owns the CPU side: slider mappings, the envelope pre-pass, and
//! the exposure renormalization scalar.

use std::collections::VecDeque;

use image::DynamicImage;

use crate::image_processing::downscale_f32_image;

/// Auto/manual amount never fully crushes; fine-tune headroom is deliberate.
const AMOUNT_CEILING: f32 = 0.95;
const AMOUNT_EASE: f32 = 0.8;
const VEIL_FRAC_MIN: f32 = 0.02;
const VEIL_FRAC_MAX: f32 = 0.15;
const MAX_BOOST_CEILING: f32 = 8.0;
const REEXPOSURE_CEILING: f32 = 4.0;
const THUMB_DIM: u32 = 256;

const LUMA_COEFF: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Slider 0-100 -> subtraction strength `s`, eased so mid-travel reads as
/// "half the glare gone".
pub fn map_amount(slider: f32) -> f32 {
    AMOUNT_CEILING * (slider.clamp(0.0, 100.0) / 100.0).powf(AMOUNT_EASE)
}

/// Slider 0-100 -> envelope window as a fraction of the min image dimension,
/// log-mapped over 2-15%.
pub fn map_veil_size(slider: f32) -> f32 {
    VEIL_FRAC_MIN * (VEIL_FRAC_MAX / VEIL_FRAC_MIN).powf(slider.clamp(0.0, 100.0) / 100.0)
}

/// Slider 0-100 -> stretch clamp, log-mapped over 1-8x.
pub fn map_max_boost(slider: f32) -> f32 {
    MAX_BOOST_CEILING.powf(slider.clamp(0.0, 100.0) / 100.0)
}

pub struct VeilThumb {
    /// Linear RGB thumbnail, interleaved, w*h*3.
    pub lin: Vec<f32>,
    /// Smooth per-channel lower envelope of `lin`, same layout.
    pub veil: Vec<f32>,
    pub w: u32,
    pub h: u32,
}

/// Estimate the veil on a thumbnail of `base` (the post-transform image the
/// GPU receives). `is_linear` distinguishes raw (already linear) from encoded
/// sRGB input; the veil must live in the shader's input space.
pub fn compute_veil_thumbnail(base: &DynamicImage, is_linear: bool, window_frac: f32) -> VeilThumb {
    let start = std::time::Instant::now();

    let thumb = downscale_f32_image(base, THUMB_DIM, THUMB_DIM).to_rgb32f();
    let (w, h) = (thumb.width() as usize, thumb.height() as usize);
    let mut lin = thumb.into_raw();
    if !is_linear {
        for v in lin.iter_mut() {
            *v = srgb_component_to_linear(*v);
        }
    }

    let win = ((w.min(h) as f32 * window_frac) as usize).max(3) | 1;
    let radius = win / 2;
    let box_radius = box_radius_for_sigma(win as f32);

    let mut veil = vec![0.0f32; lin.len()];
    let mut plane = vec![0.0f32; w * h];
    let mut tmp = vec![0.0f32; w * h];
    for c in 0..3 {
        for i in 0..w * h {
            plane[i] = lin[i * 3 + c];
        }
        // Median prefilter stands in for the prototype's 2nd-percentile
        // erosion: it keeps isolated dark outliers from dragging the min.
        let med = median3x3(&plane, w, h);
        for y in 0..h {
            sliding_min_line(&med, &mut tmp, y * w, w, 1, radius);
        }
        for x in 0..w {
            sliding_min_line(&tmp, &mut plane, x, h, w, radius);
        }
        // Three box passes approximate a Gaussian with sigma ~= win.
        for _ in 0..3 {
            for y in 0..h {
                box_blur_line(&plane, &mut tmp, y * w, w, 1, box_radius);
            }
            for x in 0..w {
                box_blur_line(&tmp, &mut plane, x, h, w, box_radius);
            }
        }
        for i in 0..w * h {
            veil[i * 3 + c] = plane[i];
        }
    }

    log::info!(
        "Glare veil pre-pass: {}x{} thumb, window {}px, took {:?}",
        w,
        h,
        win,
        start.elapsed()
    );

    VeilThumb {
        lin,
        veil,
        w: w as u32,
        h: h as u32,
    }
}

/// Subtraction darkens the frame; scale so the recovered thumbnail keeps the
/// original's p99 luminance. Recomputed per Amount/Max boost change - cheap,
/// the thumbnail is already in memory.
pub fn compute_reexposure(lin: &[f32], veil: &[f32], s: f32, max_boost: f32) -> f32 {
    if s <= 0.0 {
        return 1.0;
    }
    let boost = max_boost.max(1.0);
    let n = lin.len() / 3;
    let mut pre = Vec::with_capacity(n);
    let mut post = Vec::with_capacity(n);
    for i in 0..n {
        let mut pr = 0.0f32;
        let mut po = 0.0f32;
        for c in 0..3 {
            let iv = lin[i * 3 + c];
            let v = s * veil[i * 3 + c];
            let den = (1.0 - v).clamp(1.0 / boost, 1.0);
            pr += LUMA_COEFF[c] * iv;
            po += LUMA_COEFF[c] * ((iv - v) / den).max(0.0);
        }
        pre.push(pr);
        post.push(po);
    }
    let pre99 = percentile_99(&mut pre);
    let post99 = percentile_99(&mut post);
    if post99 <= 1e-6 {
        return 1.0;
    }
    (pre99 / post99).clamp(1.0, REEXPOSURE_CEILING)
}

fn srgb_component_to_linear(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn percentile_99(values: &mut [f32]) -> f32 {
    let idx = ((values.len() - 1) as f32 * 0.99) as usize;
    values.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    values[idx]
}

fn median3x3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut vals = [0.0f32; 9];
            let mut n = 0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let sy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    let sx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    vals[n] = src[sy * w + sx];
                    n += 1;
                }
            }
            vals.sort_unstable_by(|a, b| a.total_cmp(b));
            out[y * w + x] = vals[4];
        }
    }
    out
}

/// Sliding-window minimum over one line (row or column via `stride`) using a
/// monotonic deque - O(len) regardless of window size.
fn sliding_min_line(
    src: &[f32],
    dst: &mut [f32],
    start: usize,
    len: usize,
    stride: usize,
    radius: usize,
) {
    let mut deque: VecDeque<usize> = VecDeque::new();
    let mut next = 0usize;
    for i in 0..len {
        let hi = (i + radius).min(len - 1);
        while next <= hi {
            let v = src[start + next * stride];
            while deque.back().is_some_and(|&b| src[start + b * stride] >= v) {
                deque.pop_back();
            }
            deque.push_back(next);
            next += 1;
        }
        while deque.front().is_some_and(|&f| f + radius < i) {
            deque.pop_front();
        }
        dst[start + i * stride] = src[start + deque[0] * stride];
    }
}

/// Running-sum box blur over one line, edge-replicated.
fn box_blur_line(
    src: &[f32],
    dst: &mut [f32],
    start: usize,
    len: usize,
    stride: usize,
    radius: usize,
) {
    let width = (2 * radius + 1) as f32;
    let last = len as i32 - 1;
    let mut sum = 0.0f32;
    for k in -(radius as i32)..=(radius as i32) {
        sum += src[start + k.clamp(0, last) as usize * stride];
    }
    dst[start] = sum / width;
    for i in 1..len {
        let add = (i as i32 + radius as i32).min(last) as usize;
        let sub = (i as i32 - 1 - radius as i32).max(0) as usize;
        sum += src[start + add * stride] - src[start + sub * stride];
        dst[start + i * stride] = sum / width;
    }
}

/// Radius for three iterated box passes approximating a Gaussian of `sigma`:
/// sigma^2 = 3 * ((2r+1)^2 - 1) / 12.
fn box_radius_for_sigma(sigma: f32) -> usize {
    let width = (4.0 * sigma * sigma + 1.0).sqrt();
    (((width - 1.0) / 2.0).round() as usize).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, Rgb32FImage};

    /// A dark scene with regular near-black holes under a smooth additive
    /// gradient veil: the envelope must recover the veil, not the scene.
    fn synthetic_veiled_image(w: u32, h: u32) -> (DynamicImage, Vec<f32>) {
        let mut img = Rgb32FImage::new(w, h);
        let mut veil = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = 0.10 + 0.10 * (x as f32 / w as f32);
                // Deterministic texture with a zero every 8px in each axis.
                let scene = if x % 8 == 0 || y % 8 == 0 {
                    0.0
                } else {
                    0.05 + 0.20 * (((x * 7 + y * 13) % 32) as f32 / 32.0)
                };
                let px = (0.9 * scene + v).min(1.0);
                img.put_pixel(x, y, image::Rgb([px, px, px]));
                for _ in 0..3 {
                    veil.push(v);
                }
            }
        }
        (DynamicImage::ImageRgb32F(img), veil)
    }

    #[test]
    fn envelope_recovers_smooth_veil() {
        let (img, true_veil) = synthetic_veiled_image(256, 256);
        let thumb = compute_veil_thumbnail(&img, true, map_veil_size(50.0));
        assert_eq!(thumb.veil.len(), thumb.lin.len());
        let n = thumb.veil.len();
        let mean_err: f32 = thumb
            .veil
            .iter()
            .zip(true_veil.iter().take(n))
            .map(|(e, t)| (e - t).abs())
            .sum::<f32>()
            / n as f32;
        // Envelope sits at/below the veil; the blur pulls it toward it.
        assert!(mean_err < 0.03, "mean envelope error {mean_err}");
        let over = thumb
            .veil
            .iter()
            .zip(true_veil.iter().take(n))
            .filter(|(e, t)| **e > **t + 0.02)
            .count();
        assert!(
            (over as f32) < 0.02 * n as f32,
            "envelope exceeds true veil at {over}/{n} samples"
        );
    }

    #[test]
    fn reexposure_neutral_at_zero_amount_and_bounded() {
        let (img, _) = synthetic_veiled_image(128, 128);
        let thumb = compute_veil_thumbnail(&img, true, map_veil_size(50.0));
        assert_eq!(compute_reexposure(&thumb.lin, &thumb.veil, 0.0, 6.0), 1.0);
        let scale = compute_reexposure(&thumb.lin, &thumb.veil, map_amount(85.0), 6.0);
        assert!((1.0..=4.0).contains(&scale), "scale {scale}");
    }

    #[test]
    fn slider_mappings_hit_documented_endpoints() {
        assert_eq!(map_amount(0.0), 0.0);
        assert!((map_amount(100.0) - 0.95).abs() < 1e-6);
        assert!((map_veil_size(0.0) - 0.02).abs() < 1e-6);
        assert!((map_veil_size(100.0) - 0.15).abs() < 1e-6);
        assert!((map_max_boost(0.0) - 1.0).abs() < 1e-6);
        assert!((map_max_boost(100.0) - 8.0).abs() < 1e-5);
    }
}
