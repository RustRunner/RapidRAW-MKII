//! Glare Recovery: removal of veiling glare (window/windshield reflections).
//!
//! The veil is modeled as a smooth additive reflection layer bounded by the
//! image's local minima, estimated per channel as a lower envelope on a small
//! thumbnail. The main shader subtracts it and re-stretches in linear light;
//! this module owns the CPU side: slider mappings, the envelope pre-pass, and
//! the exposure renormalization scalar.

use std::collections::VecDeque;

use image::DynamicImage;
use crate::image_identity::{EstimateError, ImageIdentity, ImageSession, OwnedEstimate};

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

pub fn slider_from_amount(s: f32) -> f32 {
    100.0 * (s.clamp(0.0, AMOUNT_CEILING) / AMOUNT_CEILING).powf(1.0 / AMOUNT_EASE)
}

/// Slider 0-100 -> envelope window as a fraction of the min image dimension,
/// log-mapped over 2-15%.
pub fn map_veil_size(slider: f32) -> f32 {
    VEIL_FRAC_MIN * (VEIL_FRAC_MAX / VEIL_FRAC_MIN).powf(slider.clamp(0.0, 100.0) / 100.0)
}

pub fn slider_from_veil_size(frac: f32) -> f32 {
    let frac = frac.clamp(VEIL_FRAC_MIN, VEIL_FRAC_MAX);
    100.0 * (frac / VEIL_FRAC_MIN).ln() / (VEIL_FRAC_MAX / VEIL_FRAC_MIN).ln()
}

/// Slider 0-100 -> stretch clamp, log-mapped over 1-8x.
pub fn map_max_boost(slider: f32) -> f32 {
    MAX_BOOST_CEILING.powf(slider.clamp(0.0, 100.0) / 100.0)
}

pub fn slider_from_max_boost(boost: f32) -> f32 {
    100.0 * boost.clamp(1.0, MAX_BOOST_CEILING).ln() / MAX_BOOST_CEILING.ln()
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

/// Windows probed by the auto-estimator, in fractions of the min dimension.
const ESTIMATE_WINDOW_FRACS: [f32; 4] = [0.03, 0.06, 0.10, 0.15];
/// Knee: the smallest window whose envelope changes by less than this
/// (mean, relative) when grown to the next probe size.
const KNEE_CHANGE_THRESHOLD: f32 = 0.10;
/// Dark-channel occupancy below which the image has real blacks and no
/// significant veil. The floor must stay above ~0.10: the prototype's
/// recovered-control test scores 0.096.
pub const GLARE_CONFIDENCE_FLOOR: f32 = 0.12;
/// Amount from occupancy: capped well short of 1.0 so auto never crushes.
const AMOUNT_FROM_RATIO: f32 = 1.9;
/// Noise level (in the file's own encoding, matching `estimate_noise`)
/// considered acceptable after the stretch; the boost budget is this over
/// the measured sigma.
const SIGMA_ACCEPTABLE: f32 = 0.025;

/// Slider suggestions for the loaded image, in the sliders' 0-100 units.
#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GlareEstimate {
    pub amount: f32,
    pub veil_size: f32,
    pub max_boost: f32,
    pub glare_ratio: f32,
    pub confident: bool,
}

/// Derive slider suggestions from the image: veil size from the envelope
/// convergence knee, amount from how much of the tonal range the veil
/// occupies, max boost from the measured noise floor.
pub fn estimate_glare(image: &DynamicImage, is_linear: bool, sigma_luma: f32) -> GlareEstimate {
    let thumbs: Vec<VeilThumb> = ESTIMATE_WINDOW_FRACS
        .iter()
        .map(|f| compute_veil_thumbnail(image, is_linear, *f))
        .collect();

    let mut chosen = ESTIMATE_WINDOW_FRACS.len() - 1;
    for i in 0..ESTIMATE_WINDOW_FRACS.len() - 1 {
        let a = &thumbs[i].veil;
        let b = &thumbs[i + 1].veil;
        let mean_a = a.iter().sum::<f32>() / a.len() as f32;
        let change = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum::<f32>()
            / a.len() as f32
            / mean_a.max(1e-4);
        if change < KNEE_CHANGE_THRESHOLD {
            chosen = i;
            break;
        }
    }
    let thumb = &thumbs[chosen];

    let luma = |buf: &[f32], i: usize| -> f32 {
        LUMA_COEFF[0] * buf[i * 3] + LUMA_COEFF[1] * buf[i * 3 + 1] + LUMA_COEFF[2] * buf[i * 3 + 2]
    };
    let n = thumb.veil.len() / 3;
    let mut veil_luma: Vec<f32> = (0..n).map(|i| luma(&thumb.veil, i)).collect();
    let mut image_luma: Vec<f32> = (0..n).map(|i| luma(&thumb.lin, i)).collect();
    let mid = veil_luma.len() / 2;
    let (_, veil_median, _) = veil_luma.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let ratio = *veil_median / percentile_99(&mut image_luma).max(1e-4);

    let s_target = (AMOUNT_FROM_RATIO * ratio).min(AMOUNT_CEILING);
    let boost = (SIGMA_ACCEPTABLE / sigma_luma.max(1e-5)).clamp(1.0, MAX_BOOST_CEILING);

    GlareEstimate {
        amount: slider_from_amount(s_target).round(),
        veil_size: slider_from_veil_size(ESTIMATE_WINDOW_FRACS[chosen]).round(),
        max_boost: slider_from_max_boost(boost).round(),
        glare_ratio: ratio,
        confident: ratio >= GLARE_CONFIDENCE_FLOOR,
    }
}

/// Tauri command: analyze the loaded source image (never the adjusted
/// preview, so repeat presses are idempotent) and suggest slider values.
/// Thumbnail-only except the noise floor, which comes from the shared
/// per-image cache.
#[tauri::command]
pub async fn estimate_glare_veil(
    expected_identity: ImageIdentity,
    state: tauri::State<'_, crate::app_state::AppState>,
) -> Result<OwnedEstimate<GlareEstimate>, EstimateError> {
    let session = ImageSession::new(&state);
    let snapshot = session.snapshot(&expected_identity)?;
    let noise = crate::denoising::measured_noise_for_snapshot(&state, &snapshot).await?;
    let image = snapshot.image.clone();
    let is_raw = snapshot.is_raw;
    let computation = tokio::task::spawn_blocking(move || estimate_glare(&image, is_raw, noise.sigma_luma)).await;
    session.snapshot(&snapshot.identity())?;
    let estimate = computation.map_err(|e| EstimateError::Failed(format!("Glare estimation task failed: {e}")))?;
    session.finish(&snapshot, estimate)
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

    #[test]
    fn mapping_inverses_round_trip() {
        for s in [0.0, 10.0, 35.0, 50.0, 75.0, 100.0] {
            assert!((slider_from_amount(map_amount(s)) - s).abs() < 0.01, "amount {s}");
            assert!((slider_from_veil_size(map_veil_size(s)) - s).abs() < 0.01, "veil {s}");
            assert!((slider_from_max_boost(map_max_boost(s)) - s).abs() < 0.01, "boost {s}");
        }
    }

    #[test]
    fn estimator_separates_veiled_from_unveiled_scene() {
        let (veiled, _) = synthetic_veiled_image(256, 256);
        let est = estimate_glare(&veiled, true, 0.004);
        assert!(est.confident, "veiled scene not flagged: {est:?}");
        assert!(est.amount > 20.0, "amount too timid: {est:?}");
        assert!(est.glare_ratio > GLARE_CONFIDENCE_FLOOR);
        // Clean noise floor allows an aggressive stretch.
        assert!(est.max_boost > 80.0, "max boost too low: {est:?}");

        // The same texture without the veil has real blacks everywhere.
        let mut img = Rgb32FImage::new(256, 256);
        for y in 0..256u32 {
            for x in 0..256u32 {
                let scene = if x % 8 == 0 || y % 8 == 0 {
                    0.0
                } else {
                    0.05 + 0.20 * (((x * 7 + y * 13) % 32) as f32 / 32.0)
                };
                img.put_pixel(x, y, image::Rgb([scene, scene, scene]));
            }
        }
        let est2 = estimate_glare(&DynamicImage::ImageRgb32F(img), true, 0.004);
        assert!(!est2.confident, "unveiled scene wrongly flagged: {est2:?}");
    }

    /// Estimator on a real glare shot. Ignored by default; run with:
    ///   GLARE_TEST_IMAGE=/path/to/glare.png cargo test --lib estimator_real -- --ignored --nocapture
    #[test]
    #[ignore]
    fn estimator_real_image() {
        let Some(path) = std::env::var_os("GLARE_TEST_IMAGE") else {
            eprintln!("skipping: GLARE_TEST_IMAGE not set");
            return;
        };
        let img = image::open(&path).expect("open GLARE_TEST_IMAGE");
        let est = estimate_glare(&img, false, 0.005);
        eprintln!("estimate on {path:?}: {est:?}");
        assert!(est.confident, "glare shot not flagged: {est:?}");
        assert!(est.amount >= 80.0, "amount below fine-tune zone: {est:?}");
    }
}
