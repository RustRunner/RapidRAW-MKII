// ============================================================================
// RAPID Processing - FFT-Based Deconvolution
// ============================================================================
//
// This module implements Regularized Pseudoinverse Deconvolution (RAPID)
// using frequency-domain Wiener filtering for blur recovery.
//
// Key components:
// - Stockham FFT algorithm for efficient 2D transforms
// - Analytical PSF generation in frequency domain
// - Wiener filter with adaptive regularization
//
// Author: RapidRAW Mod1 Team
// Date: January 2026
// ============================================================================

// Unused for now but will be needed in later phases
#[allow(unused_imports)]
use crate::cpu_fft;
#[allow(unused_imports)]
use wgpu::util::DeviceExt;

// ============================================================================
// Types and Parameters
// ============================================================================

/// Blur type for PSF generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum BlurType {
    Motion = 0,
    Defocus = 1,
    Gaussian = 2,
}

impl Default for BlurType {
    fn default() -> Self {
        BlurType::Motion
    }
}

impl From<u32> for BlurType {
    fn from(value: u32) -> Self {
        match value {
            0 => BlurType::Motion,
            1 => BlurType::Defocus,
            2 => BlurType::Gaussian,
            _ => BlurType::Motion,
        }
    }
}

/// The set of blur models composing the active PSF. Blurs that occur
/// together convolve in image space, so their transfer functions multiply in
/// the frequency domain: any subset of models forms ONE compound kernel
/// inverted by a single Wiener pass — never a sequence of deconvolutions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModeSet {
    pub motion: bool,
    pub defocus: bool,
    pub gaussian: bool,
}

impl ModeSet {
    pub const MOTION: Self = Self { motion: true, defocus: false, gaussian: false };
    pub const DEFOCUS: Self = Self { motion: false, defocus: true, gaussian: false };
    pub const GAUSSIAN: Self = Self { motion: false, defocus: false, gaussian: true };

    pub fn any(self) -> bool {
        self.motion || self.defocus || self.gaussian
    }

    /// Uniform encoding for psf_generate.wgsl: bit0 = motion, bit1 =
    /// defocus, bit2 = gaussian.
    pub fn bits(self) -> u32 {
        (self.motion as u32) | ((self.defocus as u32) << 1) | ((self.gaussian as u32) << 2)
    }
}

impl Default for ModeSet {
    fn default() -> Self {
        Self::MOTION
    }
}

/// Parameters for RAPID deconvolution
#[derive(Debug, Clone, Copy)]
pub struct RapidParams {
    /// Enable RAPID processing
    pub enabled: bool,
    /// Set of blur models composing the compound PSF (OTFs multiply)
    pub modes: ModeSet,
    /// Motion blur length in pixels
    pub motion_length: f32,
    /// Motion blur angle in degrees
    pub motion_angle: f32,
    /// Defocus blur radius in pixels
    pub defocus_radius: f32,
    /// Gaussian blur sigma
    pub gaussian_sigma: f32,
    /// Regularization parameter (noise-to-signal ratio estimate)
    pub lambda: f32,
    /// Deconvolution strength (0-1, blend with original)
    pub strength: f32,
    /// Seam-free boundary handling (reflect-101 margins + PSF-consistent
    /// border taper). Always on in production; off gives the raw zero-pad
    /// baseline for A/B tests of boundary ringing.
    pub edge_taper: bool,
    /// Minimum denominator to prevent division by zero
    pub noise_floor: f32,
    /// Adaptive per-frequency regularization. Always on in production
    /// (parse_rapid_params hardcodes it); the fixed-λ Wiener variant stays
    /// compiled as this code-level flag for A/B debugging, and tests that
    /// need deterministic fixed-λ math rely on the `false` default.
    pub adaptive: bool,
    /// OTF shape. Motion: 0 = legacy zero-free Gaussian envelope, 1 =
    /// physical hard-line OTF (signed sinc with true zeros). Defocus: 0 =
    /// legacy floored jinc, 1 = raw signed jinc (true zeros, Wiener
    /// self-limiting). The `0.0` default keeps existing tests on the exact
    /// legacy math; parse_rapid_params supplies the production values.
    pub hardness: f32,
    /// Clipped-highlight ring guard: fade the recombine gain map back to
    /// identity near (near-)saturated input pixels. Saturation records
    /// min(blur(x), 1.0), violating the linear blur model, so even a
    /// perfect filter rings around clipped sources; the guard trades their
    /// (impossible) recovery for a ring-free neighborhood. The `false`
    /// default keeps existing tests bit-exact; parse_rapid_params supplies
    /// the production value.
    pub clip_guard: bool,
}

impl Default for RapidParams {
    fn default() -> Self {
        Self {
            enabled: false,
            modes: ModeSet::MOTION,
            motion_length: 10.0,
            motion_angle: 0.0,
            defocus_radius: 5.0,
            gaussian_sigma: 2.0,
            lambda: 0.01,
            strength: 1.0,
            edge_taper: true,
            noise_floor: 1e-6,
            adaptive: false,
            hardness: 0.0,
            clip_guard: false,
        }
    }
}

impl RapidParams {
    /// Rescale the spatial kernel parameters for a working image that has been
    /// downscaled by `scale`. Lambda and strength describe frequency-domain
    /// behavior and blending, not pixel extents, so they stay unchanged. The
    /// Motion and Gaussian retain their existing floors at degenerate scales;
    /// defocus preserves the requested sub-pixel radius.
    pub fn scaled(&self, scale: f32) -> Self {
        Self {
            motion_length: (self.motion_length * scale).max(1.0),
            defocus_radius: self.defocus_radius * scale,
            gaussian_sigma: (self.gaussian_sigma * scale).max(0.3),
            ..*self
        }
    }
}

/// Rec. 709 luma coefficients — must match LUMA_COEFF in utility.wgsl and
/// shader.wgsl.
const LUMA_COEFF: [f32; 3] = [0.2126, 0.7152, 0.0722];

// ============================================================================
// Edge taper: seam-free FFT padding built CPU-side before upload
// ============================================================================
//
// FFT deconvolution is circular: the frame's left/right (and top/bottom)
// edges are neighbors. Zero-padding leaves a hard image-to-black step at the
// wrap, and the Wiener inverse amplifies exactly the frequencies where the
// PSF spectrum is near zero, ringing that step across the frame as periodic
// banding. Two mechanisms remove the amplified seam energy, chosen per axis
// at upload time:
//
// (a) Axes with pow2 headroom: continue the image into the margin with
//     reflect-101 content cosine-faded to zero, at both ends of the wrap.
//     The discontinuity moves off the visible frame and stays
//     derivative-continuous, so residual wrap-seam ringing decays before the
//     readback crop.
// (b) Axes whose margin cannot absorb the kernel extent (notably exact-pow2
//     axes with no headroom): PSF-consistent border taper (the MATLAB
//     `edgetaper` approach) — cross-fade the border strips toward a copy
//     blurred with the PSF's projection onto that axis. By the
//     projection-slice theorem the strip's spectrum along the axis is
//     pre-multiplied by the PSF spectrum, which is attenuated exactly where
//     the Wiener inverse amplifies, so the seam deconvolves to a soft step
//     instead of ringing.

/// Spatial extent of the active compound PSF in pixels: how far the circular
/// wrap can smear content across the frame boundary. The compound PSF is the
/// convolution of its member PSFs, and convolution supports add, so member
/// extents sum.
fn kernel_extent(params: &RapidParams) -> usize {
    let mut extent = 0.0f32;
    if params.modes.motion {
        extent += params.motion_length;
    }
    if params.modes.defocus {
        extent += 2.0 * params.defocus_radius;
    }
    if params.modes.gaussian {
        extent += 6.0 * params.gaussian_sigma;
    }
    (extent.ceil() as usize).max(1)
}

/// Unnormalized density components of ONE blur model's projection onto an
/// axis: (weight, half-extent, density at signed distance t from center).
/// These must match the spectra the Wiener filter divides by
/// (psf_generate.wgsl), not an idealized blur model — and the motion OTF is
/// a hardness blend of two spectra, whose projection is the same blend of
/// the two projections (a MIXTURE, not a convolution: mix() of spectra is
/// linear, so it is a mixture of PSFs).
type Density = Box<dyn Fn(f32) -> f32>;
fn mode_axis_components(params: &RapidParams, mode: BlurType, axis: usize) -> Vec<(f32, f32, Density)> {
    match mode {
        BlurType::Motion => {
            // Soft arm: the zero-free Gaussian envelope with sigma_freq = 1/L
            // (motion_blur_spectrum), i.e. spatially a Gaussian of
            // sigma = L/(2π) along the motion direction, which projects onto
            // an axis as a Gaussian of sigma·|cos| / sigma·|sin|. Hard arm:
            // the line segment itself, a box over the projected extent.
            let dir = params.motion_angle.to_radians();
            let along = if axis == 0 { dir.cos() } else { dir.sin() };
            let proj = (params.motion_length * along).abs();
            let s = proj / (2.0 * std::f32::consts::PI);
            let half_box = proj / 2.0;
            let h = params.hardness.clamp(0.0, 1.0);
            vec![
                (
                    1.0 - h,
                    3.0 * s,
                    Box::new(move |t: f32| (-t * t / (2.0 * s * s).max(1e-6)).exp()) as Density,
                ),
                (
                    h,
                    half_box,
                    Box::new(move |t: f32| if t.abs() <= half_box { 1.0f32 } else { 0.0 }) as Density,
                ),
            ]
        }
        BlurType::Defocus => {
            // jinc spectrum = uniform disk, which projects as its chord length.
            let r = params.defocus_radius.max(0.0);
            vec![(1.0, r, Box::new(move |t: f32| (r * r - t * t).max(0.0).sqrt()) as Density)]
        }
        BlurType::Gaussian => {
            // gaussian_blur_spectrum caps effective sigma at 8; mirror that.
            let s = params.gaussian_sigma.clamp(1e-3, 8.0);
            vec![(1.0, 3.0 * s, Box::new(move |t: f32| (-t * t / (2.0 * s * s)).exp()) as Density)]
        }
    }
}

/// Integrates weighted density components into a normalized tap vector;
/// `[1.0]` when the projection is sub-pixel (identity).
fn integrate_axis_components(components: Vec<(f32, f32, Density)>) -> Vec<f32> {
    let radius = components
        .iter()
        .filter(|(weight, _, _)| *weight > 0.0)
        .map(|(_, half_extent, _)| half_extent.ceil() as i32)
        .max()
        .unwrap_or(0);
    if radius < 1 {
        return vec![1.0];
    }
    // Midpoint-integrate each density over the tap cells [i-0.5, i+0.5] and
    // normalize it to unit mass before weighting, so the blend keeps the
    // requested mass split regardless of each density's raw scale.
    let mut kernel = vec![0.0f32; (2 * radius + 1) as usize];
    for (weight, _, density) in &components {
        if *weight <= 0.0 {
            continue;
        }
        let taps: Vec<f32> = (-radius..=radius)
            .map(|i| {
                (0..4)
                    .map(|k| density(i as f32 - 0.5 + (k as f32 + 0.5) / 4.0))
                    .sum::<f32>()
            })
            .collect();
        let sum: f32 = taps.iter().sum();
        if sum <= f32::EPSILON {
            continue;
        }
        for (out, tap) in kernel.iter_mut().zip(&taps) {
            *out += weight * tap / sum;
        }
    }
    let sum: f32 = kernel.iter().sum();
    if sum <= f32::EPSILON {
        return vec![1.0];
    }
    for w in &mut kernel {
        *w /= sum;
    }
    kernel
}

/// Full discrete convolution of two unit-mass tap vectors, renormalized to
/// absorb float rounding (support = a + b - 1).
fn convolve_taps(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; a.len() + b.len() - 1];
    for (i, &av) in a.iter().enumerate() {
        for (j, &bv) in b.iter().enumerate() {
            out[i + j] += av * bv;
        }
    }
    let sum: f32 = out.iter().sum();
    if sum > f32::EPSILON {
        for w in &mut out {
            *w /= sum;
        }
    }
    out
}

/// Projection of the active compound PSF onto one axis (0 = x, 1 = y),
/// normalized to sum 1; `[1.0]` when the projection is sub-pixel (identity).
/// The compound PSF is the convolution of its member PSFs, and the
/// projection of a convolution is the convolution of the projections — so
/// member vectors convolve here, while each member's hardness-blend arms
/// stay a mixture INSIDE its own vector (see `mode_axis_components`). A
/// single-member set takes the integration path untouched, bit-identical to
/// the pre-compound implementation.
fn psf_axis_projection(params: &RapidParams, axis: usize) -> Vec<f32> {
    let mut projection: Option<Vec<f32>> = None;
    for (active, mode) in [
        (params.modes.motion, BlurType::Motion),
        (params.modes.defocus, BlurType::Defocus),
        (params.modes.gaussian, BlurType::Gaussian),
    ] {
        if !active {
            continue;
        }
        let taps = integrate_axis_components(mode_axis_components(params, mode, axis));
        projection = Some(match projection {
            None => taps,
            Some(acc) => convolve_taps(&acc, &taps),
        });
    }
    projection.unwrap_or_else(|| vec![1.0])
}

/// PSF-consistent border taper along one axis: cross-fade each border strip
/// toward a copy blurred with the PSF's projection onto that axis, sampling
/// circularly (as MATLAB `edgetaper` does) so the blurred border mixes both
/// sides of the wrap seam — the fade then replaces the hard seam step with a
/// PSF-smooth transition. Only valid on axes where the frame edges really
/// are FFT neighbors (no pad headroom). The interior beyond `taper` px of
/// the edges is untouched.
fn edge_taper_axis(
    pixels: &mut [f32],
    width: usize,
    height: usize,
    axis: usize,
    taper: usize,
    kernel: &[f32],
) {
    use rayon::prelude::*;

    let len = if axis == 0 { width } else { height };
    let lines = if axis == 0 { height } else { width };
    let taper = taper.min(len / 2);
    if taper == 0 || kernel.len() <= 1 {
        return;
    }
    let radius = (kernel.len() / 2) as isize;

    // Flat component index of (position d along the axis, line l across it).
    let idx = |d: usize, l: usize| -> usize {
        if axis == 0 { (l * width + d) * 4 } else { (d * width + l) * 4 }
    };

    // Pass 1 (read-only, parallel over lines): blend each border-strip sample
    // toward its blurred value. Layout per line: 2 sides × taper × RGB.
    let blended: Vec<Vec<f32>> = (0..lines)
        .into_par_iter()
        .map(|l| {
            let mut out = Vec::with_capacity(taper * 6);
            for side in 0..2 {
                for d in 0..taper {
                    let pos = if side == 0 { d } else { len - 1 - d };
                    // Cosine ramp: fully blurred at the edge, original again
                    // at the interior end of the strip.
                    let alpha =
                        0.5 - 0.5 * (std::f32::consts::PI * d as f32 / taper as f32).cos();
                    for c in 0..3 {
                        let mut blurred = 0.0f32;
                        for (j, &w) in kernel.iter().enumerate() {
                            let t =
                                (pos as isize + j as isize - radius).rem_euclid(len as isize);
                            blurred += w * pixels[idx(t as usize, l) + c];
                        }
                        let orig = pixels[idx(pos, l) + c];
                        out.push(alpha * orig + (1.0 - alpha) * blurred);
                    }
                }
            }
            out
        })
        .collect();

    // Pass 2: write back.
    for (l, vals) in blended.iter().enumerate() {
        let mut it = vals.iter();
        for side in 0..2 {
            for d in 0..taper {
                let pos = if side == 0 { d } else { len - 1 - d };
                let base = idx(pos, l);
                for c in 0..3 {
                    pixels[base + c] = *it.next().unwrap();
                }
            }
        }
    }
}

/// Per-axis map from padded index to (source index, fade weight): identity
/// inside the frame, reflect-101 content cosine-faded to zero through the
/// margin at both ends of the circular wrap, zero weight elsewhere.
fn mirror_fade_map(size: usize, padded: usize, margin: usize) -> Vec<(usize, f32)> {
    let mut map = vec![(0usize, 0.0f32); padded];
    for (i, entry) in map.iter_mut().enumerate().take(size) {
        *entry = (i, 1.0);
    }
    for k in 0..margin {
        // ~1 adjacent to the frame, ~0 approaching the zero fill.
        let w =
            0.5 + 0.5 * ((k as f32 + 1.0) * std::f32::consts::PI / (margin as f32 + 1.0)).cos();
        // Just past the last sample: reflect-101 about size-1.
        map[size + k] = (size - 2 - k, w);
        // Just before wrapping back to sample 0: reflect-101 about 0.
        map[padded - 1 - k] = (1 + k, w);
    }
    map
}

/// Build the pow2-padded RGBA f32 upload buffer for `deconvolve_linear_image`.
/// With `params.edge_taper` off this is a plain zero-pad — the A/B baseline
/// for boundary-ringing tests.
fn build_padded_input(
    rgba: &image::Rgba32FImage,
    padded_w: u32,
    padded_h: u32,
    params: &RapidParams,
) -> Vec<f32> {
    use rayon::prelude::*;

    let (width, height) = (rgba.width() as usize, rgba.height() as usize);
    let (padded_w, padded_h) = (padded_w as usize, padded_h as usize);
    let mut pixels = rgba.as_raw().clone();

    let extent = kernel_extent(params);
    let (margin_x, margin_y) = if params.edge_taper {
        // Mirror margins claim at most half the headroom per end of the wrap.
        (
            ((padded_w - width) / 2).min(extent),
            ((padded_h - height) / 2).min(extent),
        )
    } else {
        (0, 0)
    };

    if params.edge_taper {
        // (b) on axes with no mirror margin, where the frame edges are
        // direct FFT neighbors (notably exact-pow2 axes with no headroom).
        // The cross-fade spans 2x the kernel extent, matching the support of
        // the PSF autocorrelation that MATLAB's edgetaper fades over.
        if margin_x == 0 {
            edge_taper_axis(&mut pixels, width, height, 0, 2 * extent, &psf_axis_projection(params, 0));
        }
        if margin_y == 0 {
            edge_taper_axis(&mut pixels, width, height, 1, 2 * extent, &psf_axis_projection(params, 1));
        }
    }

    // (a) separable reflect-101 + fade fill through the margins; identity
    // weight inside the frame, zero weight beyond the margins.
    let map_x = mirror_fade_map(width, padded_w, margin_x);
    let map_y = mirror_fade_map(height, padded_h, margin_y);

    let mut padded = vec![0.0f32; padded_w * padded_h * 4];
    padded
        .par_chunks_exact_mut(padded_w * 4)
        .enumerate()
        .for_each(|(y, row)| {
            let (sy, wy) = map_y[y];
            if wy == 0.0 {
                return;
            }
            let src_row = &pixels[sy * width * 4..(sy + 1) * width * 4];
            for (x, out) in row.chunks_exact_mut(4).enumerate() {
                let (sx, wx) = map_x[x];
                let w = wy * wx;
                if w == 0.0 {
                    continue;
                }
                let src = &src_row[sx * 4..sx * 4 + 4];
                out[0] = src[0] * w;
                out[1] = src[1] * w;
                out[2] = src[2] * w;
                out[3] = src[3];
            }
        });
    padded
}

// ============================================================================
// Clipped-highlight ring guard
// ============================================================================
//
// Saturated pixels record min(blur(x), 1.0), not blur(x): the linear model
// the Wiener filter inverts is wrong there, and each clipped source radiates
// a ring train the width of the restoration filter's impulse response — no
// linear filter can avoid it. The guard detects (near-)clipped input pixels,
// builds a feathered distance field from them, and fades the recombine gain
// map back to identity nearby, so unclipped content keeps full recovery.

/// Working-scale value at which a channel counts as clipped. Max-channel,
/// not luma: a blown red tail light clips R while luma stays ~0.25, and
/// per-channel saturation is what breaks the linear model.
const CLIP_GUARD_SAT: f32 = 0.98;
/// Full suppression within this many kernel extents of a clipped pixel
/// (the reach of the clipped sample under the blur)...
const CLIP_GUARD_D0_EXTENTS: f32 = 1.0;
/// ...normally fading to zero by this many extents.
const CLIP_GUARD_D1_EXTENTS: f32 = 3.0;
/// A defocus inverse has a much longer jinc ring train. Real clipped-sky
/// captures retain visible dark rings well beyond the generic three-extent
/// feather, so keep identity influence through the measured tail. Motion and
/// Gaussian modes retain the narrower guard.
const CLIP_GUARD_DEFOCUS_D1_EXTENTS: f32 = 16.0;

fn clip_guard_d1_pixels(params: &RapidParams, compound_extent: usize) -> f32 {
    let compact_d1 = CLIP_GUARD_D1_EXTENTS * compound_extent as f32;
    if params.modes.defocus {
        let defocus_extent = (2.0 * params.defocus_radius).ceil().max(1.0);
        compact_d1.max(CLIP_GUARD_DEFOCUS_D1_EXTENTS * defocus_extent)
    } else {
        compact_d1
    }
}

/// Chamfer 3x3 distance transform: per-pixel distance in pixels to the
/// nearest set pixel, capped at `cap`. Two raster scans (forward, then
/// backward) with weights 1/sqrt(2); overestimates Euclidean distance by at
/// most ~8%, exact enough for a feathered mask. O(n), one Vec<f32>.
fn chamfer_distance(mask: &[bool], width: usize, height: usize, cap: f32) -> Vec<f32> {
    const DIAG: f32 = std::f32::consts::SQRT_2;
    let mut dist = vec![cap; width * height];
    for (i, &m) in mask.iter().enumerate() {
        if m {
            dist[i] = 0.0;
        }
    }
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let mut d = dist[i];
            if x > 0 {
                d = d.min(dist[i - 1] + 1.0);
            }
            if y > 0 {
                d = d.min(dist[i - width] + 1.0);
                if x > 0 {
                    d = d.min(dist[i - width - 1] + DIAG);
                }
                if x + 1 < width {
                    d = d.min(dist[i - width + 1] + DIAG);
                }
            }
            dist[i] = d;
        }
    }
    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let i = y * width + x;
            let mut d = dist[i];
            if x + 1 < width {
                d = d.min(dist[i + 1] + 1.0);
            }
            if y + 1 < height {
                d = d.min(dist[i + width] + 1.0);
                if x + 1 < width {
                    d = d.min(dist[i + width + 1] + DIAG);
                }
                if x > 0 {
                    d = d.min(dist[i + width - 1] + DIAG);
                }
            }
            dist[i] = d;
        }
    }
    dist
}

/// Per-pixel guard weight: 1 (reproduce the input) within D0 of a clipped
/// pixel, 0 (full recovery) beyond D1, smoothstep between. None when nothing
/// clips, so the recombine loop stays untouched at zero cost.
fn clip_guard_weights(
    rgba: &image::Rgba32FImage,
    extent: usize,
    clip_guard_sat_linear: f32,
    d1_pixels: f32,
) -> Option<Vec<f32>> {
    let (width, height) = (rgba.width() as usize, rgba.height() as usize);
    let mut mask = vec![false; width * height];
    let mut any = false;
    for (i, p) in rgba.pixels().enumerate() {
        if p[0].max(p[1]).max(p[2]) >= clip_guard_sat_linear {
            mask[i] = true;
            any = true;
        }
    }
    if !any {
        return None;
    }
    let d0 = CLIP_GUARD_D0_EXTENTS * extent as f32;
    let d1 = d1_pixels.max(d0);
    let dist = chamfer_distance(&mask, width, height, d1);
    Some(
        dist.iter()
            .map(|&d| {
                let t = ((d1 - d) / (d1 - d0).max(1e-6)).clamp(0.0, 1.0);
                t * t * (3.0 - 2.0 * t)
            })
            .collect(),
    )
}

// ============================================================================
// GPU Uniform Structs (must match shader)
// ============================================================================

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FFTParams {
    size: u32,
    pass_num: u32,   // Current pass (0 to log2(size)-1) - named to avoid WGSL keyword
    direction: i32,  // 1 = forward, -1 = inverse
    is_horizontal: u32,
    width: u32,
    height: u32,
    _pad: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BitRevParams {
    width: u32,
    height: u32,
    log2_width: u32,
    log2_height: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PSFParams {
    width: u32,
    height: u32,
    active_modes: u32,
    motion_length: f32,
    motion_angle: f32,
    defocus_radius: f32,
    gaussian_sigma: f32,
    hardness: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct WienerParams {
    width: u32,
    height: u32,
    lambda: f32,
    strength: f32,
    noise_floor: f32,
    adaptive: u32,
    gaussian_active: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct UtilityParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    normalize_factor: f32,
    channel: u32,  // 0=R, 1=G, 2=B
    _pad: [u32; 2],
}

// ============================================================================
// Frequency Domain Textures
// ============================================================================

/// Collection of textures for frequency domain processing. Deconvolution is
/// luma-only: a single Y channel goes through the FFT/Wiener round-trip and
/// chroma is reapplied from the original image at readback via a gain map.
struct FrequencyTextures {
    /// Luma channel frequency data
    freq_y: wgpu::Texture,
    /// Ping-pong buffer for FFT passes
    freq_temp: wgpu::Texture,
    /// PSF frequency data
    psf_freq: wgpu::Texture,
}

struct FrequencyTextureViews {
    freq_y: wgpu::TextureView,
    freq_temp: wgpu::TextureView,
    psf_freq: wgpu::TextureView,
}

impl FrequencyTextures {
    fn create_views(&self) -> FrequencyTextureViews {
        FrequencyTextureViews {
            freq_y: self.freq_y.create_view(&Default::default()),
            freq_temp: self.freq_temp.create_view(&Default::default()),
            psf_freq: self.psf_freq.create_view(&Default::default()),
        }
    }
}

// ============================================================================
// RAPID Deconvolver
// ============================================================================

/// Normalization parameters for IFFT (must match shader)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NormalizeParams {
    width: u32,
    height: u32,
    scale: f32,
    _pad: u32,
}

/// GPU-based FFT deconvolution processor
pub struct RapidDeconvolver {
    // FFT Compute pipelines
    fft_horizontal_pipeline: wgpu::ComputePipeline,
    fft_vertical_pipeline: wgpu::ComputePipeline,
    fft_normalize_pipeline: wgpu::ComputePipeline,

    // Bit-reversal pipeline for Cooley-Tukey FFT
    bit_reverse_pipeline: wgpu::ComputePipeline,

    // PSF generation pipeline
    psf_generate_pipeline: wgpu::ComputePipeline,

    // Wiener filter pipelines
    wiener_pipeline: wgpu::ComputePipeline,
    wiener_adaptive_pipeline: wgpu::ComputePipeline,

    // Utility pipelines
    real_to_complex_pipeline: wgpu::ComputePipeline,
    complex_to_real_pipeline: wgpu::ComputePipeline,

    // Bind group layouts
    fft_bgl: wgpu::BindGroupLayout,
    bitrev_bgl: wgpu::BindGroupLayout,
    psf_bgl: wgpu::BindGroupLayout,
    wiener_bgl: wgpu::BindGroupLayout,
    utility_bgl: wgpu::BindGroupLayout,

    // Parameter buffers
    fft_params_buffer: wgpu::Buffer,
    bitrev_params_buffer: wgpu::Buffer,
    normalize_params_buffer: wgpu::Buffer,
    psf_params_buffer: wgpu::Buffer,
    wiener_params_buffer: wgpu::Buffer,
    utility_params_buffer: wgpu::Buffer,

    // Reusable frequency textures (allocated on demand)
    freq_textures: Option<FrequencyTextures>,
    freq_views: Option<FrequencyTextureViews>,

    // Current allocation size
    allocated_width: u32,
    allocated_height: u32,

    // GPU capabilities
    max_texture_size: u32,
}

impl RapidDeconvolver {
    /// Check if the GPU supports RAPID deconvolution
    pub fn check_gpu_support(
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
    ) -> Result<(), String> {
        let limits = device.limits();

        // Check texture dimension limits
        if limits.max_texture_dimension_2d < 4096 {
            return Err(format!(
                "GPU max texture dimension {} is below minimum 4096 for RAPID",
                limits.max_texture_dimension_2d
            ));
        }

        // Check compute workgroup limits
        if limits.max_compute_workgroup_size_x < 256 {
            return Err(format!(
                "GPU max workgroup size {} is below required 256 for FFT",
                limits.max_compute_workgroup_size_x
            ));
        }

        // Check storage texture binding limits
        if limits.max_storage_textures_per_shader_stage < 2 {
            return Err(
                "GPU does not support enough storage textures per stage for RAPID".to_string()
            );
        }

        // Verify Rg32Float format support for storage textures
        let format_features = adapter.get_texture_format_features(wgpu::TextureFormat::Rg32Float);
        if !format_features
            .allowed_usages
            .contains(wgpu::TextureUsages::STORAGE_BINDING)
        {
            return Err(
                "GPU does not support Rg32Float storage textures required for RAPID".to_string(),
            );
        }

        let info = adapter.get_info();
        log::info!(
            "RAPID GPU check passed: {} ({:?}, {:?}), max texture: {}",
            info.name,
            info.backend,
            info.device_type,
            limits.max_texture_dimension_2d
        );

        Ok(())
    }

    /// Create a new RAPID deconvolver
    pub fn new(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Result<Self, String> {
        // Validate GPU support
        Self::check_gpu_support(adapter, device)?;

        let limits = device.limits();

        // Create bind group layouts
        let fft_bgl = Self::create_fft_bgl(device);
        let psf_bgl = Self::create_psf_bgl(device);
        let wiener_bgl = Self::create_wiener_bgl(device);
        let utility_bgl = Self::create_utility_bgl(device);

        // Load FFT shader module
        let fft_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RAPID FFT Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/rapid/fft_stockham.wgsl").into(),
            ),
        });

        // Create FFT pipeline layout
        let fft_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RAPID FFT Pipeline Layout"),
            bind_group_layouts: &[Some(&fft_bgl)],
            immediate_size: 0,
        });

        // Create FFT horizontal pipeline
        let fft_horizontal_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RAPID FFT Horizontal"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_shader,
                entry_point: Some("fft_horizontal"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create FFT vertical pipeline
        let fft_vertical_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RAPID FFT Vertical"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_shader,
                entry_point: Some("fft_vertical"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create FFT normalize pipeline
        let fft_normalize_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RAPID FFT Normalize"),
                layout: Some(&fft_pipeline_layout),
                module: &fft_shader,
                entry_point: Some("fft_normalize"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create bit-reversal bind group layout (same structure as FFT)
        let bitrev_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RAPID BitRev BGL"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bit-reversal pipeline layout
        let bitrev_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RAPID BitRev Pipeline Layout"),
            bind_group_layouts: &[Some(&bitrev_bgl)],
            immediate_size: 0,
        });

        // Create bit-reversal pipeline
        let bit_reverse_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RAPID Bit Reverse"),
                layout: Some(&bitrev_pipeline_layout),
                module: &fft_shader,
                entry_point: Some("bit_reverse_2d"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Load PSF shader module
        let psf_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RAPID PSF Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/rapid/psf_generate.wgsl").into(),
            ),
        });

        // Create PSF pipeline layout
        let psf_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RAPID PSF Pipeline Layout"),
            bind_group_layouts: &[Some(&psf_bgl)],
            immediate_size: 0,
        });

        // Create PSF generation pipeline
        let psf_generate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RAPID PSF Generate"),
                layout: Some(&psf_pipeline_layout),
                module: &psf_shader,
                entry_point: Some("generate_psf_spectrum"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Load Wiener shader module
        let wiener_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RAPID Wiener Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/rapid/wiener_filter.wgsl").into(),
            ),
        });

        // Create Wiener pipeline layout
        let wiener_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RAPID Wiener Pipeline Layout"),
            bind_group_layouts: &[Some(&wiener_bgl)],
            immediate_size: 0,
        });

        // Create Wiener filter pipeline
        let wiener_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RAPID Wiener"),
                layout: Some(&wiener_pipeline_layout),
                module: &wiener_shader,
                entry_point: Some("wiener_deconvolve"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create adaptive Wiener filter pipeline
        let wiener_adaptive_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RAPID Wiener Adaptive"),
                layout: Some(&wiener_pipeline_layout),
                module: &wiener_shader,
                entry_point: Some("wiener_adaptive"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Load utility shader module
        let utility_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RAPID Utility Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/rapid/utility.wgsl").into(),
            ),
        });

        // Create utility pipeline layout
        let utility_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RAPID Utility Pipeline Layout"),
            bind_group_layouts: &[Some(&utility_bgl)],
            immediate_size: 0,
        });

        // Create real to complex pipeline
        let real_to_complex_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RAPID Real to Complex"),
                layout: Some(&utility_pipeline_layout),
                module: &utility_shader,
                entry_point: Some("real_to_complex_pad"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create complex to real pipeline
        let complex_to_real_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RAPID Complex to Real"),
                layout: Some(&utility_pipeline_layout),
                module: &utility_shader,
                entry_point: Some("complex_to_real_crop"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Create parameter buffers
        let fft_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RAPID FFT Params"),
            size: std::mem::size_of::<FFTParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bitrev_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RAPID BitRev Params"),
            size: std::mem::size_of::<BitRevParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let normalize_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RAPID Normalize Params"),
            size: std::mem::size_of::<NormalizeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let psf_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RAPID PSF Params"),
            size: std::mem::size_of::<PSFParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let wiener_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RAPID Wiener Params"),
            size: std::mem::size_of::<WienerParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let utility_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RAPID Utility Params"),
            size: std::mem::size_of::<UtilityParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        log::info!("RAPID deconvolver initialized with all pipelines");

        Ok(Self {
            fft_horizontal_pipeline,
            fft_vertical_pipeline,
            fft_normalize_pipeline,
            bit_reverse_pipeline,
            psf_generate_pipeline,
            wiener_pipeline,
            wiener_adaptive_pipeline,
            real_to_complex_pipeline,
            complex_to_real_pipeline,
            fft_bgl,
            bitrev_bgl,
            psf_bgl,
            wiener_bgl,
            utility_bgl,
            fft_params_buffer,
            bitrev_params_buffer,
            normalize_params_buffer,
            psf_params_buffer,
            wiener_params_buffer,
            utility_params_buffer,
            freq_textures: None,
            freq_views: None,
            allocated_width: 0,
            allocated_height: 0,
            max_texture_size: limits.max_texture_dimension_2d,
        })
    }

    /// Estimated peak device memory for a width×height deconvolution: the
    /// pow2-padded Rgba32Float input (16 B/px), three Rg32Float frequency
    /// textures (24 B/px), and the readback staging buffer (8 B/px), with
    /// 20% headroom. Everything is sized at the padded extent.
    pub fn required_vram_mb(width: u32, height: u32) -> u64 {
        let (padded_w, padded_h) = Self::get_padded_dimensions(width, height);
        let bytes = padded_w as u64 * padded_h as u64 * (16 + 24 + 8);
        (bytes * 12 / 10) / (1024 * 1024)
    }

    /// Largest working scale in (0, 1] whose padded pipeline fits both the
    /// max texture dimension and the VRAM budget. Walks power-of-two
    /// boundaries downward, since the padded allocations only change there.
    pub fn max_scale_for_vram(
        width: u32,
        height: u32,
        budget_mb: u64,
        max_texture_size: u32,
    ) -> f32 {
        let mut scale = 1.0f32;
        for _ in 0..20 {
            let w = ((width as f32 * scale).round() as u32).max(1);
            let h = ((height as f32 * scale).round() as u32).max(1);
            let (padded_w, padded_h) = Self::get_padded_dimensions(w, h);
            let fits_texture = padded_w <= max_texture_size && padded_h <= max_texture_size;
            if fits_texture && Self::required_vram_mb(w, h) <= budget_mb {
                return scale;
            }
            // Shrink the dominant padded axis under its next boundary.
            scale = if padded_w >= padded_h {
                (padded_w / 2) as f32 / width as f32
            } else {
                (padded_h / 2) as f32 / height as f32
            };
        }
        scale
    }

    /// Ensure frequency textures are allocated for the given image size
    fn ensure_textures(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let padded_width = width.next_power_of_two();
        let padded_height = height.next_power_of_two();

        // Check if already allocated large enough
        if padded_width <= self.allocated_width && padded_height <= self.allocated_height {
            return;
        }

        log::info!(
            "RAPID: Allocating frequency textures {}x{} for image {}x{}",
            padded_width,
            padded_height,
            width,
            height
        );

        let create_freq_texture = |label: &str| -> wgpu::Texture {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: padded_width,
                    height: padded_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };

        let textures = FrequencyTextures {
            freq_y: create_freq_texture("RAPID freq_y"),
            freq_temp: create_freq_texture("RAPID freq_temp"),
            psf_freq: create_freq_texture("RAPID psf_freq"),
        };

        let views = textures.create_views();

        self.freq_textures = Some(textures);
        self.freq_views = Some(views);
        self.allocated_width = padded_width;
        self.allocated_height = padded_height;

        let memory_mb = (padded_width as u64 * padded_height as u64 * 8 * 3) / (1024 * 1024);
        log::info!("RAPID: Allocated {} MB for frequency textures", memory_mb);
    }

    /// Get the padded dimensions for FFT (next power of 2)
    pub fn get_padded_dimensions(width: u32, height: u32) -> (u32, u32) {
        (width.next_power_of_two(), height.next_power_of_two())
    }

    /// Calculate number of FFT passes needed for a dimension
    pub fn get_fft_passes(size: u32) -> u32 {
        (size as f32).log2() as u32
    }

    // ========================================================================
    // Bind Group Layout Creation
    // ========================================================================

    /// Create bind group layout for FFT passes
    fn create_fft_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RAPID FFT BGL"),
            entries: &[
                // Input texture (read)
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
                // Output texture (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Parameters uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create bind group layout for PSF generation
    fn create_psf_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RAPID PSF BGL"),
            entries: &[
                // Output texture (PSF spectrum)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Parameters uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create bind group layout for Wiener filter
    fn create_wiener_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RAPID Wiener BGL"),
            entries: &[
                // Image frequency data (read)
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
                // PSF frequency data (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Output frequency data (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Parameters uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    /// Create bind group layout for utility operations (real<->complex, windowing)
    fn create_utility_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RAPID Utility BGL"),
            entries: &[
                // Input texture (RGBA or complex)
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
                // Output texture (complex or RGBA)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // Parameters uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    // ========================================================================
    // FFT Operations (Phase 2)
    // ========================================================================

    /// Perform 1D FFT along rows (horizontal)
    ///
    /// Uses ping-pong buffers: after each pass, the result alternates between
    /// the two textures. Returns true if final result is in `texture_a`.
    ///
    /// # Arguments
    /// * `encoder` - Command encoder to record commands
    /// * `device` - GPU device
    /// * `queue` - GPU queue for buffer writes
    /// * `texture_a` - First texture (input/output)
    /// * `texture_b` - Second texture (ping-pong buffer)
    /// * `width` - FFT size (must be power of 2)
    /// * `height` - Number of rows to process
    /// * `forward` - true for forward FFT, false for inverse
    ///
    /// # Returns
    /// (new_encoder, result_in_a) - New encoder and whether result is in texture_a
    pub fn encode_fft_horizontal(
        &self,
        encoder: wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _texture_a: &wgpu::Texture,
        view_a: &wgpu::TextureView,
        _texture_b: &wgpu::Texture,
        view_b: &wgpu::TextureView,
        width: u32,
        height: u32,
        forward: bool,
    ) -> (wgpu::CommandEncoder, bool) {
        let num_passes = Self::get_fft_passes(width);
        let direction: i32 = if forward { 1 } else { -1 };

        let mut current_encoder = encoder;

        for pass in 0..num_passes {
            // Determine source and destination for this pass
            let (src_view, dst_view) = if pass % 2 == 0 {
                (view_a, view_b)
            } else {
                (view_b, view_a)
            };

            // Update FFT parameters
            let params = FFTParams {
                size: width,
                pass_num: pass,
                direction,
                is_horizontal: 1,
                width,
                height,
                _pad: [0; 2],
            };
            queue.write_buffer(&self.fft_params_buffer, 0, bytemuck::bytes_of(&params));

            // Create bind group for this pass
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("FFT H Pass {} BG", pass)),
                layout: &self.fft_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(dst_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.fft_params_buffer.as_entire_binding(),
                    },
                ],
            });

            // Dispatch compute shader
            // Each thread handles one butterfly, we have width/2 butterflies per row
            let workgroups_x = (width / 2 + 255) / 256;
            let workgroups_y = height;

            {
                let mut cpass = current_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("FFT H Pass {}", pass)),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.fft_horizontal_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            // Submit after each pass to ensure params buffer is read before next write
            queue.submit(std::iter::once(current_encoder.finish()));
            current_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("FFT H Pass {} Encoder", pass + 1)),
            });
        }

        // Return new encoder and whether result is in texture_a (even number of passes)
        (current_encoder, num_passes % 2 == 0)
    }

    /// Perform 1D FFT along columns (vertical)
    ///
    /// # Returns
    /// (new_encoder, result_in_a) - New encoder and whether result is in texture_a
    pub fn encode_fft_vertical(
        &self,
        encoder: wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _texture_a: &wgpu::Texture,
        view_a: &wgpu::TextureView,
        _texture_b: &wgpu::Texture,
        view_b: &wgpu::TextureView,
        width: u32,
        height: u32,
        forward: bool,
    ) -> (wgpu::CommandEncoder, bool) {
        let num_passes = Self::get_fft_passes(height);
        let direction: i32 = if forward { 1 } else { -1 };

        let mut current_encoder = encoder;

        for pass in 0..num_passes {
            // Determine source and destination for this pass
            let (src_view, dst_view) = if pass % 2 == 0 {
                (view_a, view_b)
            } else {
                (view_b, view_a)
            };

            // Update FFT parameters
            let params = FFTParams {
                size: height,
                pass_num: pass,
                direction,
                is_horizontal: 0,
                width,
                height,
                _pad: [0; 2],
            };
            queue.write_buffer(&self.fft_params_buffer, 0, bytemuck::bytes_of(&params));

            // Create bind group for this pass
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("FFT V Pass {} BG", pass)),
                layout: &self.fft_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(dst_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.fft_params_buffer.as_entire_binding(),
                    },
                ],
            });

            // Dispatch compute shader
            // Each thread handles one butterfly, we have height/2 butterflies per column
            let workgroups_x = width;
            let workgroups_y = (height / 2 + 255) / 256;

            {
                let mut cpass = current_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("FFT V Pass {}", pass)),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.fft_vertical_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            // Submit after each pass to ensure params buffer is read before next write
            queue.submit(std::iter::once(current_encoder.finish()));
            current_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("FFT V Pass {} Encoder", pass + 1)),
            });
        }

        // Return new encoder and whether result is in texture_a (even number of passes)
        (current_encoder, num_passes % 2 == 0)
    }

    /// Perform full 2D FFT (rows then columns)
    ///
    /// # Arguments
    /// * `encoder` - Command encoder (takes ownership)
    /// * `device` - GPU device
    /// * `queue` - GPU queue
    /// * `input` - Input texture with complex data
    /// * `input_view` - View of input texture
    /// * `temp` - Temporary texture for ping-pong
    /// * `temp_view` - View of temp texture
    /// * `width` - Image width (must be power of 2)
    /// * `height` - Image height (must be power of 2)
    /// * `forward` - true for forward FFT, false for inverse
    ///
    /// # Returns
    /// (new_encoder, result_texture) - New encoder and reference to texture containing result
    pub fn encode_fft_2d<'a>(
        &self,
        encoder: wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &'a wgpu::Texture,
        input_view: &'a wgpu::TextureView,
        temp: &'a wgpu::Texture,
        temp_view: &'a wgpu::TextureView,
        width: u32,
        height: u32,
        forward: bool,
    ) -> (wgpu::CommandEncoder, &'a wgpu::Texture) {
        // First do horizontal FFT (along rows)
        let (encoder_after_h, result_in_input) = self.encode_fft_horizontal(
            encoder, device, queue,
            input, input_view, temp, temp_view,
            width, height, forward,
        );

        // Determine which textures to use for vertical FFT based on where row FFT result is
        let (col_input, col_input_view, col_temp, col_temp_view) = if result_in_input {
            (input, input_view, temp, temp_view)
        } else {
            (temp, temp_view, input, input_view)
        };

        // Then do vertical FFT (along columns)
        let (encoder_after_v, final_in_col_input) = self.encode_fft_vertical(
            encoder_after_h, device, queue,
            col_input, col_input_view, col_temp, col_temp_view,
            width, height, forward,
        );

        // Return new encoder and the texture containing the final result
        if final_in_col_input {
            (encoder_after_v, col_input)
        } else {
            (encoder_after_v, col_temp)
        }
    }

    /// Apply IFFT normalization (divide by N = width * height)
    pub fn encode_fft_normalize(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        let scale = 1.0 / (width * height) as f32;

        let params = NormalizeParams {
            width,
            height,
            scale,
            _pad: 0,
        };
        queue.write_buffer(&self.normalize_params_buffer, 0, bytemuck::bytes_of(&params));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FFT Normalize BG"),
            layout: &self.fft_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.normalize_params_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups_x = (width + 15) / 16;
        let workgroups_y = (height + 15) / 16;

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FFT Normalize"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.fft_normalize_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    /// Apply bit-reversal permutation for Cooley-Tukey FFT
    ///
    /// This reorders data from natural order to bit-reversed order (pre-FFT)
    /// or from bit-reversed to natural order (post-FFT).
    pub fn encode_bit_reverse(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        let log2_width = (width as f32).log2() as u32;
        let log2_height = (height as f32).log2() as u32;

        let params = BitRevParams {
            width,
            height,
            log2_width,
            log2_height,
        };
        queue.write_buffer(&self.bitrev_params_buffer, 0, bytemuck::bytes_of(&params));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BitRev BG"),
            layout: &self.bitrev_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.bitrev_params_buffer.as_entire_binding(),
                },
            ],
        });

        let workgroups_x = (width + 15) / 16;
        let workgroups_y = (height + 15) / 16;

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Bit Reverse"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.bit_reverse_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    /// Perform forward 2D FFT with result guaranteed in output texture
    ///
    /// This is a convenience method that handles buffer management and ensures
    /// the result ends up in a predictable location. Uses Cooley-Tukey DIT with
    /// bit-reversal to produce natural-order frequency output.
    ///
    /// # Returns
    /// New encoder after FFT operations complete
    pub fn forward_fft_2d(
        &self,
        mut encoder: wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data_texture: &wgpu::Texture,
        data_view: &wgpu::TextureView,
        temp_texture: &wgpu::Texture,
        temp_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> wgpu::CommandEncoder {
        // Step 1: Bit-reverse the input (data -> temp)
        // This prepares data for Cooley-Tukey DIT which produces natural-order output
        self.encode_bit_reverse(&mut encoder, device, queue, data_view, temp_view, width, height);

        // Submit to ensure bit-reversal completes before FFT
        queue.submit(std::iter::once(encoder.finish()));
        encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FFT Forward After BitRev"),
        });

        // Step 2: Run FFT passes (starting from temp which has bit-reversed data)
        let (mut encoder, result) = self.encode_fft_2d(
            encoder, device, queue,
            temp_texture, temp_view, data_texture, data_view,
            width, height, true,
        );

        // Step 3: Ensure result ends up in data_texture
        if !std::ptr::eq(result, data_texture) {
            encoder.copy_texture_to_texture(
                temp_texture.as_image_copy(),
                data_texture.as_image_copy(),
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }

        encoder
    }

    /// Perform inverse 2D FFT with normalization, result in data texture
    ///
    /// Uses Cooley-Tukey DIT with bit-reversal. Input should be in natural
    /// frequency order (matching forward FFT output).
    ///
    /// # Returns
    /// New encoder after FFT operations complete
    pub fn inverse_fft_2d(
        &self,
        mut encoder: wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data_texture: &wgpu::Texture,
        data_view: &wgpu::TextureView,
        temp_texture: &wgpu::Texture,
        temp_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> wgpu::CommandEncoder {
        // Step 1: Bit-reverse the input (data -> temp)
        self.encode_bit_reverse(&mut encoder, device, queue, data_view, temp_view, width, height);

        // Submit to ensure bit-reversal completes before FFT
        queue.submit(std::iter::once(encoder.finish()));
        encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FFT Inverse After BitRev"),
        });

        // Step 2: Run inverse FFT passes (starting from temp which has bit-reversed data)
        let (mut encoder, result) = self.encode_fft_2d(
            encoder, device, queue,
            temp_texture, temp_view, data_texture, data_view,
            width, height, false,
        );

        // Step 3: Apply normalization
        let (norm_input, norm_output) = if std::ptr::eq(result, data_texture) {
            (data_view, temp_view)
        } else {
            (temp_view, data_view)
        };

        self.encode_fft_normalize(&mut encoder, device, queue, norm_input, norm_output, width, height);

        // Step 4: Copy to data_texture if needed
        if std::ptr::eq(result, data_texture) {
            // Normalization wrote to temp, copy back
            encoder.copy_texture_to_texture(
                temp_texture.as_image_copy(),
                data_texture.as_image_copy(),
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
        // else: normalization already wrote to data_texture

        encoder
    }

    // ========================================================================
    // PSF Generation (Phase 3)
    // ========================================================================

    /// Generate PSF spectrum in frequency domain
    ///
    /// Creates the frequency-domain representation of the blur kernel
    /// directly using analytical formulas (no spatial FFT needed).
    pub fn encode_psf_generation(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        params: &RapidParams,
    ) {
        // Set up PSF parameters
        let psf_params = PSFParams {
            width,
            height,
            active_modes: params.modes.bits(),
            motion_length: params.motion_length,
            motion_angle: params.motion_angle,
            defocus_radius: params.defocus_radius,
            gaussian_sigma: params.gaussian_sigma,
            hardness: params.hardness,
        };
        queue.write_buffer(&self.psf_params_buffer, 0, bytemuck::bytes_of(&psf_params));

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PSF Generate BG"),
            layout: &self.psf_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.psf_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let workgroups_x = (width + 15) / 16;
        let workgroups_y = (height + 15) / 16;

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("PSF Generate"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.psf_generate_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    // ========================================================================
    // Wiener Filter (Phase 3)
    // ========================================================================

    /// Apply Wiener deconvolution filter
    ///
    /// Computes F̂ = G · H* / (|H|² + λ) where:
    /// - G is the degraded image spectrum
    /// - H is the PSF spectrum
    /// - λ is the regularization parameter
    pub fn encode_wiener_filter(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image_freq_view: &wgpu::TextureView,
        psf_freq_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        params: &RapidParams,
    ) {
        // Set up Wiener parameters
        let wiener_params = WienerParams {
            width,
            height,
            lambda: params.lambda,
            strength: params.strength,
            noise_floor: params.noise_floor,
            adaptive: if params.adaptive { 1 } else { 0 },
            gaussian_active: params.modes.gaussian as u32,
            _pad: 0,
        };
        queue.write_buffer(&self.wiener_params_buffer, 0, bytemuck::bytes_of(&wiener_params));

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Wiener Filter BG"),
            layout: &self.wiener_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(image_freq_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(psf_freq_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.wiener_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let workgroups_x = (width + 15) / 16;
        let workgroups_y = (height + 15) / 16;

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Wiener Filter"),
            timestamp_writes: None,
        });

        // Use adaptive or standard Wiener based on params
        if params.adaptive {
            cpass.set_pipeline(&self.wiener_adaptive_pipeline);
        } else {
            cpass.set_pipeline(&self.wiener_pipeline);
        }

        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    // ========================================================================
    // Utility Operations (Phase 3)
    // ========================================================================

    /// Convert a single channel from RGBA to complex
    ///
    /// Extracts one channel (R, G, or B) and zero-pads to the destination
    /// size. Edge tapering happens CPU-side before upload, so in the normal
    /// pipeline the source already spans the destination.
    pub fn encode_real_to_complex(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        channel: u32,
    ) {
        // Set up utility parameters
        let utility_params = UtilityParams {
            src_width,
            src_height,
            dst_width,
            dst_height,
            normalize_factor: 1.0,
            channel,
            _pad: [0; 2],
        };
        queue.write_buffer(&self.utility_params_buffer, 0, bytemuck::bytes_of(&utility_params));

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Real to Complex BG"),
            layout: &self.utility_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.utility_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let workgroups_x = (dst_width + 15) / 16;
        let workgroups_y = (dst_height + 15) / 16;

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Real to Complex"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.real_to_complex_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    /// Convert complex back to real, crop padding, and normalize
    ///
    /// Takes the real part of the complex spectrum after inverse FFT,
    /// crops to original dimensions, and clips to [0, 1].
    pub fn encode_complex_to_real(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        normalize_factor: f32,
    ) {
        // Set up utility parameters
        let utility_params = UtilityParams {
            src_width: width,
            src_height: height,
            dst_width: width,
            dst_height: height,
            normalize_factor,
            channel: 0,
            _pad: [0; 2],
        };
        queue.write_buffer(&self.utility_params_buffer, 0, bytemuck::bytes_of(&utility_params));

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Complex to Real BG"),
            layout: &self.utility_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.utility_params_buffer.as_entire_binding(),
                },
            ],
        });

        // Dispatch
        let workgroups_x = (width + 15) / 16;
        let workgroups_y = (height + 15) / 16;

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Complex to Real"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.complex_to_real_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    // ========================================================================
    // Public API
    // ========================================================================

    /// Execute RAPID deconvolution on an image
    ///
    /// This is the main entry point for RAPID processing.
    /// Returns Ok(()) if successful, Err with message if failed.
    ///
    /// Note: Full implementation will be added in Phase 4.
    pub fn deconvolve(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _input_view: &wgpu::TextureView,
        _output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        params: &RapidParams,
    ) -> Result<(), String> {
        if !params.enabled {
            return Ok(());
        }

        // Ensure frequency textures are allocated
        self.ensure_textures(device, width, height);

        let (padded_w, padded_h) = Self::get_padded_dimensions(width, height);
        let row_passes = Self::get_fft_passes(padded_w);
        let col_passes = Self::get_fft_passes(padded_h);

        log::info!(
            "RAPID: Processing {}x{} (padded {}x{}), {} row passes, {} col passes",
            width,
            height,
            padded_w,
            padded_h,
            row_passes,
            col_passes
        );

        // TODO: Implement full pipeline in Phase 4
        // For now, just log that we would process
        log::info!(
            "RAPID: Would deconvolve {:?} blur (length={}, angle={}, radius={}, sigma={})",
            params.modes,
            params.motion_length,
            params.motion_angle,
            params.defocus_radius,
            params.gaussian_sigma
        );

        Err("RAPID deconvolution not yet implemented (Phase 2-4)".to_string())
    }

    /// GPU core for a linear DynamicImage. Returns linear ImageRgba32F and
    /// receives the clip threshold explicitly so provenance stays outside.
    pub fn deconvolve_linear_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &image::DynamicImage,
        params: &RapidParams,
        clip_guard_sat_linear: f32,
    ) -> Result<image::DynamicImage, String> {
        use image::{GenericImageView, Rgba};

        if !params.enabled {
            return Ok(image.clone());
        }

        let (width, height) = image.dimensions();

        // Check size limits
        if width > self.max_texture_size || height > self.max_texture_size {
            return Err(format!(
                "Image {}x{} exceeds max texture size {}",
                width, height, self.max_texture_size
            ));
        }

        // Log processing info
        log::info!(
            "RAPID deconvolve_image: {}x{} image, {:?} blur (L={:.1}, A={:.1}°, R={:.1}, σ={:.1}), λ={:.4}, strength={:.1}%",
            width, height,
            params.modes,
            params.motion_length,
            params.motion_angle,
            params.defocus_radius,
            params.gaussian_sigma,
            params.lambda,
            params.strength * 100.0
        );

        let start_time = std::time::Instant::now();

        // DEBUG: Test levels
        // 0 = Full pipeline
        // 1 = Bypass everything, return original
        // 2 = Test real_to_complex only (no FFT)
        // 3 = Test real_to_complex + forward FFT + inverse FFT (no Wiener)
        const DEBUG_LEVEL: u32 = 0; // Full pipeline enabled

        if DEBUG_LEVEL == 1 {
            log::info!("RAPID DEBUG: Bypassing FFT, returning original image");
            return Ok(image.clone());
        }

        // Get padded dimensions for FFT
        let (padded_w, padded_h) = Self::get_padded_dimensions(width, height);

        log::debug!(
            "RAPID: estimated peak VRAM ~{} MB for {}x{} (padded {}x{})",
            Self::required_vram_mb(width, height),
            width,
            height,
            padded_w,
            padded_h
        );

        // Ensure frequency textures are allocated
        self.ensure_textures(device, width, height);

        // Step 1: Build the seam-free padded buffer CPU-side (edge taper)
        // and upload it at the full FFT extent.
        let rgba_image = image.to_rgba32f();
        let padded_pixels = build_padded_input(&rgba_image, padded_w, padded_h, params);
        let input_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RAPID Input RGBA"),
            size: wgpu::Extent3d {
                width: padded_w,
                height: padded_h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload padded image data to texture
        queue.write_texture(
            input_texture.as_image_copy(),
            bytemuck::cast_slice(&padded_pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_w * 16), // 4 channels * 4 bytes per f32
                rows_per_image: Some(padded_h),
            },
            wgpu::Extent3d {
                width: padded_w,
                height: padded_h,
                depth_or_array_layers: 1,
            },
        );

        let input_view = input_texture.create_view(&Default::default());

        // Get frequency texture views
        let freq_textures = self.freq_textures.as_ref()
            .ok_or("Frequency textures not allocated")?;
        let freq_views = self.freq_views.as_ref()
            .ok_or("Frequency texture views not created")?;

        // Create command encoder for the entire pipeline
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RAPID Deconvolution"),
        });

        // Step 2: Extract Rec. 709 luma to complex. Deconvolution is
        // luma-only: chroma is reapplied from the original image at readback
        // via a gain map, which suppresses the per-channel divergence that
        // showed up as red/blue fringing on recovered edges.
        log::debug!("RAPID: Converting luma to frequency domain...");
        self.encode_real_to_complex(
            &mut encoder, device, queue,
            &input_view, &freq_views.freq_y,
            padded_w, padded_h, padded_w, padded_h,
            3, // Rec. 709 luma
        );

        // DEBUG_LEVEL 2: Skip FFT, PSF, Wiener - just test real_to_complex + readback
        let mut encoder = if DEBUG_LEVEL == 2 {
            log::info!("RAPID DEBUG: Testing real_to_complex only (no FFT)");
            // Skip directly to readback - freq_y contains the padded luma
            encoder
        } else {
            // Forward FFT (takes ownership and returns a new encoder)
            let mut encoder = self.forward_fft_2d(
                encoder, device, queue,
                &freq_textures.freq_y, &freq_views.freq_y,
                &freq_textures.freq_temp, &freq_views.freq_temp,
                padded_w, padded_h,
            );

            // Step 3: Generate PSF in frequency domain
            log::debug!("RAPID: Generating PSF in frequency domain...");
            self.encode_psf_generation(
                &mut encoder, device, queue,
                &freq_views.psf_freq,
                padded_w, padded_h,
                params,
            );

            // Step 4: Apply Wiener filter. freq_temp is the output since
            // input and output can't be the same texture in storage binding;
            // copy back afterwards.
            log::debug!("RAPID: Applying Wiener deconvolution filter...");
            self.encode_wiener_filter(
                &mut encoder, device, queue,
                &freq_views.freq_y, &freq_views.psf_freq, &freq_views.freq_temp,
                padded_w, padded_h, params,
            );
            encoder.copy_texture_to_texture(
                freq_textures.freq_temp.as_image_copy(),
                freq_textures.freq_y.as_image_copy(),
                wgpu::Extent3d { width: padded_w, height: padded_h, depth_or_array_layers: 1 },
            );

            // Step 5: Inverse FFT (includes normalization)
            log::debug!("RAPID: Transforming back to spatial domain...");
            self.inverse_fft_2d(
                encoder, device, queue,
                &freq_textures.freq_y, &freq_views.freq_y,
                &freq_textures.freq_temp, &freq_views.freq_temp,
                padded_w, padded_h,
            )
        }; // end else (DEBUG_LEVEL != 2)

        // Step 6: Read back the deconvolved luma from GPU
        let bytes_per_row = padded_w * 8; // 2 channels (RG) * 4 bytes per f32
        let aligned_bytes_per_row = (bytes_per_row + 255) & !255; // Align to 256 bytes
        let buffer_size = (aligned_bytes_per_row * padded_h) as u64;

        let staging_y = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RAPID Staging Y"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_texture_to_buffer(
            freq_textures.freq_y.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_y,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(padded_h),
                },
            },
            wgpu::Extent3d { width: padded_w, height: padded_h, depth_or_array_layers: 1 },
        );

        // Submit all GPU work
        queue.submit(std::iter::once(encoder.finish()));

        // Map and read back the buffer
        let (tx, rx) = std::sync::mpsc::channel();
        staging_y.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(60)),
        })
        .map_err(|e| format!("RAPID device poll failed: {e}"))?;

        rx.recv()
            .map_err(|e| format!("Channel receive error: {}", e))?
            .map_err(|e| format!("Buffer map error for luma: {:?}", e))?;

        let y_data: Vec<f32> = {
            let view = staging_y.slice(..).get_mapped_range().map_err(|e| format!("Failed to map staging buffer: {e:?}"))?;
            let data: &[f32] = bytemuck::cast_slice(&view);
            // Extract just the real part (every other value) with proper row alignment
            let mut result = Vec::with_capacity((padded_w * padded_h) as usize);
            let f32_per_aligned_row = aligned_bytes_per_row as usize / 4;
            for y in 0..padded_h as usize {
                for x in 0..padded_w as usize {
                    let idx = y * f32_per_aligned_row + x * 2; // *2 because RG format
                    result.push(data[idx]);
                }
            }
            result
        };
        staging_y.unmap();

        // Step 7: Recombine (crop to original size). The deconvolved luma is
        // applied as a gain map over the original RGB: hue and saturation
        // survive exactly, and per-channel divergence cannot occur. The gain
        // clamp and denominator floor keep near-black pixels from exploding.
        // The clip guard fades the gain back to identity near saturated
        // input pixels; at w = 1 the input pixel is reproduced exactly.
        let guard = if params.clip_guard {
            let extent = kernel_extent(params);
            clip_guard_weights(
                &rgba_image,
                extent,
                clip_guard_sat_linear,
                clip_guard_d1_pixels(params, extent),
            )
        } else {
            None
        };
        let mut output = image::Rgba32FImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let idx = (y * padded_w + x) as usize;
                let src = rgba_image.get_pixel(x, y);
                let y_in =
                    src[0] * LUMA_COEFF[0] + src[1] * LUMA_COEFF[1] + src[2] * LUMA_COEFF[2];
                let mut gain = (y_data[idx] / y_in.max(1e-4)).clamp(0.0, 4.0);
                if let Some(w) = &guard {
                    let t = w[(y * width + x) as usize];
                    gain = gain * (1.0 - t) + t;
                }
                let r = (src[0] * gain).clamp(0.0, 1.0);
                let g = (src[1] * gain).clamp(0.0, 1.0);
                let b = (src[2] * gain).clamp(0.0, 1.0);
                output.put_pixel(x, y, Rgba([r, g, b, src[3]]));
            }
        }

        let elapsed = start_time.elapsed();
        log::info!("RAPID deconvolution completed in {:.2?}", elapsed);

        Ok(image::DynamicImage::ImageRgba32F(output))
    }

    /// Check if GPU supports RAPID without needing device (simpler check)
    pub fn check_gpu_support_simple(adapter: &wgpu::Adapter) -> bool {
        let format_features = adapter.get_texture_format_features(wgpu::TextureFormat::Rg32Float);
        format_features
            .allowed_usages
            .contains(wgpu::TextureUsages::STORAGE_BINDING)
    }
}

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// Pipeline integration: full-image blur-recovery pre-pass
// ============================================================================

struct RapidGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    deconvolver: RapidDeconvolver,
    /// Set by the device-lost callback; a poisoned RAPID device is dropped
    /// and recreated on the next get_rapid_gpu call.
    poisoned: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Adapter-aware budget computed once at init; see rapid_vram_budget_mb.
    vram_budget_mb: u64,
}

enum RapidGpuSlot {
    Untried,
    Unavailable,
    Ready(std::sync::Arc<std::sync::Mutex<RapidGpu>>),
}

static RAPID_GPU: std::sync::Mutex<RapidGpuSlot> = std::sync::Mutex::new(RapidGpuSlot::Untried);

/// Lazily creates a dedicated wgpu device for blur recovery. Kept separate
/// from the main GpuContext so the pre-pass needs no plumbing through the
/// tiled pipeline; returns None (and logs) when the GPU is unsupported. A
/// device lost at runtime is torn down and recreated on the next call.
fn get_rapid_gpu() -> Option<std::sync::Arc<std::sync::Mutex<RapidGpu>>> {
    let mut slot = RAPID_GPU.lock().unwrap();

    if let RapidGpuSlot::Ready(gpu) = &*slot {
        let poisoned = gpu
            .lock()
            .unwrap()
            .poisoned
            .load(std::sync::atomic::Ordering::SeqCst);
        if poisoned {
            log::warn!("RAPID device was lost; recreating");
            *slot = RapidGpuSlot::Untried;
        }
    }

    match &*slot {
        RapidGpuSlot::Ready(gpu) => Some(gpu.clone()),
        RapidGpuSlot::Unavailable => None,
        RapidGpuSlot::Untried => match build_rapid_gpu() {
            Some(gpu) => {
                let gpu = std::sync::Arc::new(std::sync::Mutex::new(gpu));
                *slot = RapidGpuSlot::Ready(gpu.clone());
                Some(gpu)
            }
            None => {
                *slot = RapidGpuSlot::Unavailable;
                None
            }
        },
    }
}

fn build_rapid_gpu() -> Option<RapidGpu> {
    let mut instance_desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    instance_desc.memory_budget_thresholds = wgpu::MemoryBudgetThresholds {
        for_resource_creation: Some(75),
        for_device_loss: Some(95),
    };
    // Mirror the main context's Windows restriction so both devices land on
    // the same backend family instead of RAPID defaulting to Backends::all().
    #[cfg(target_os = "windows")]
    if std::env::var("WGPU_BACKEND").is_err() {
        instance_desc.backends = wgpu::Backends::PRIMARY;
    }
    let instance = wgpu::Instance::new(instance_desc);
    let adapter = match pollster::block_on(instance.request_adapter(
        &wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        },
    )) {
        Ok(a) => a,
        Err(e) => {
            log::warn!("RAPID: no GPU adapter available ({e}); blur recovery disabled");
            return None;
        }
    };
    let (device, queue) = match pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("RAPID Deconvolution Device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        },
    )) {
        Ok(dq) => dq,
        Err(e) => {
            log::warn!("RAPID: failed to create device ({e}); blur recovery disabled");
            return None;
        }
    };

    // Deliberately no crash-flag write here: a loss confined to the RAPID
    // device (e.g. its own budget kill) should not rewrite the app's
    // backend; a real TDR also fires the main device's callback, which does.
    let poisoned = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let poisoned = poisoned.clone();
        device.set_device_lost_callback(move |reason, message| {
            if matches!(reason, wgpu::DeviceLostReason::Destroyed) {
                return;
            }
            log::error!("RAPID device lost ({:?}): {}", reason, message);
            poisoned.store(true, std::sync::atomic::Ordering::SeqCst);
        });
    }
    device.on_uncaptured_error(std::sync::Arc::new(|error: wgpu::Error| {
        log::error!("Uncaptured wgpu error on the RAPID device: {}", error);
    }));

    let info = adapter.get_info();
    let is_integrated = matches!(
        info.device_type,
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu
    );
    let vram_budget_mb = rapid_vram_budget_mb(is_integrated);
    log::info!(
        "RAPID VRAM budget: {} MB ({:?})",
        vram_budget_mb,
        info.device_type
    );

    match RapidDeconvolver::new(&adapter, &device) {
        Ok(deconvolver) => Some(RapidGpu {
            device,
            queue,
            deconvolver,
            poisoned,
            vram_budget_mb,
        }),
        Err(e) => {
            log::warn!("RAPID: unsupported GPU ({e}); blur recovery disabled");
            None
        }
    }
}

/// Parses blur-recovery params from the frontend adjustment JSON.
/// A mode contributes iff its enable toggle is on AND its kernel parameter
/// is positive; the stage runs iff any mode contributes and strength is
/// positive. Three sidecar generations parse without mutation:
/// - gen 0 (pre-revamp): explicit `rapidEnabled: false` vetoes everything
///   (the "original"/before preview override injects it), explicit true
///   activates the saved `rapidBlurType` mode with the old kernel defaults
///   for keys the sidecar never stored;
/// - gen 1 (kernel-gated interim): no toggles saved — the saved
///   `rapidBlurType` mode is on iff its kernel is positive, exactly the
///   gate that era shipped;
/// - gen 2: explicit per-mode toggles. Detection is OBJECT-WIDE: if any
///   toggle key is present, absent siblings mean false — falling back
///   per-key would let a legacy-selected mode reactivate inside partially
///   keyed gen-2 JSON.
/// Absent `rapidStrength`/`rapidHardness` fall back to the legacy 100s
/// (not the new-edit defaults of 50): pre-gen-2 sidecars must keep their
/// full-strength hard-line renders.
pub fn parse_rapid_params(adjustments: &serde_json::Value) -> Option<RapidParams> {
    let legacy_enabled = adjustments["rapidEnabled"].as_bool();
    if legacy_enabled == Some(false) {
        return None;
    }
    let legacy_on = legacy_enabled == Some(true);
    let legacy_mode = adjustments["rapidBlurType"].as_str().unwrap_or("motion");
    // Persisted sidecars are untrusted numeric input. Resolve legacy
    // defaults first, then cap only values above the reachable UI rails;
    // negative kernels remain inert under the existing positive-mode gate.
    let motion_length = (adjustments["rapidLength"]
        .as_f64()
        .unwrap_or(if legacy_on { 10.0 } else { 0.0 }) as f32)
        .min(200.0);
    let defocus_radius = (adjustments["rapidRadius"]
        .as_f64()
        .unwrap_or(if legacy_on { 5.0 } else { 0.0 }) as f32)
        .min(50.0);
    let gaussian_sigma = (adjustments["rapidSigma"]
        .as_f64()
        .unwrap_or(if legacy_on { 2.0 } else { 0.0 }) as f32)
        .min(8.0);
    let lambda = (adjustments["rapidLambda"].as_f64().unwrap_or(0.01) as f32)
        .clamp(0.001, 0.1);
    let strength = (adjustments["rapidStrength"].as_f64().unwrap_or(100.0) as f32 / 100.0)
        .clamp(0.0, 1.0);
    let gen2 = ["rapidMotionEnabled", "rapidDefocusEnabled", "rapidGaussianEnabled"]
        .iter()
        .any(|k| adjustments[*k].is_boolean());
    let mode_on = |key: &str, name: &str, kernel: f32| {
        adjustments[key]
            .as_bool()
            .unwrap_or(!gen2 && legacy_mode == name && (legacy_on || kernel > 0.0))
    };
    // An on-but-zero-kernel mode is inert by design, and all-zero parameters
    // are not an inert pass (the FFT round trip quantizes and gain-shifts
    // the frame, and scaled() floors kernels), so each member needs a
    // positive kernel and inactivity must skip the stage entirely.
    let modes = ModeSet {
        motion: mode_on("rapidMotionEnabled", "motion", motion_length) && motion_length > 0.0,
        defocus: mode_on("rapidDefocusEnabled", "defocus", defocus_radius)
            && defocus_radius > 0.0,
        gaussian: mode_on("rapidGaussianEnabled", "gaussian", gaussian_sigma)
            && gaussian_sigma > 0.0,
    };
    if !modes.any() || strength <= 0.0 {
        return None;
    }
    Some(RapidParams {
        enabled: true,
        modes,
        motion_length,
        motion_angle: adjustments["rapidAngle"].as_f64().unwrap_or(0.0) as f32,
        defocus_radius,
        gaussian_sigma,
        lambda,
        strength,
        // Always on in production since the toggle was demoted; stale
        // rapidAdaptive keys in old sidecars are ignored.
        adaptive: true,
        // Sidecars saved before this key exist get the hard-line model on
        // their next render: the ghosting it fixes is a defect, not a look.
        // The slider value passes through for every mode — the defocus
        // component pins its own hardness to the raw jinc inside
        // psf_generate.wgsl, where one uniform can serve motion's slider
        // and that pin simultaneously in a compound set.
        hardness: (adjustments["rapidHardness"].as_f64().unwrap_or(100.0) as f32 / 100.0)
            .clamp(0.0, 1.0),
        // Always on in production, no UI toggle: the light-centered rings it
        // removes are a defect, not a look.
        clip_guard: true,
        ..Default::default()
    })
}

/// True when blur recovery would actually run for these adjustments.
pub fn is_rapid_active(adjustments: &serde_json::Value) -> bool {
    parse_rapid_params(adjustments).is_some()
}

/// Migrates the recovery keys of a FULL adjustments record (a sidecar's
/// complete JSON) to the current schema, in place. In a full record,
/// absence within a present subsystem is unambiguous legacy, so this
/// synthesizes the per-mode toggles and pins the legacy defaults the new
/// INITIAL values no longer provide. Every arm is presence-guarded: a
/// record that never touched a subsystem comes out untouched (a
/// curves-only object gains nothing). Twin of `migrateLegacyRecoveryState`
/// in src/utils/adjustments.ts — change both or neither. Used where the
/// backend merges partial adjustments into raw sidecar JSON (batch paste);
/// render paths instead keep honoring legacy keys via `parse_rapid_params`.
pub fn migrate_legacy_recovery_state(adjustments: &mut serde_json::Value) {
    migrate_rapid_state(adjustments);
    migrate_glare_state(adjustments);
    migrate_lowlight_state(adjustments);
}

fn migrate_rapid_state(adjustments: &mut serde_json::Value) {
    const TOGGLES: [&str; 3] = [
        "rapidMotionEnabled",
        "rapidDefocusEnabled",
        "rapidGaussianEnabled",
    ];
    // Any toggle present means the record is already current — only a stray
    // hand-edited rapidEnabled needs cleaning up.
    if TOGGLES.iter().any(|k| adjustments[*k].is_boolean()) {
        if let Some(obj) = adjustments.as_object_mut() {
            obj.remove("rapidEnabled");
        }
        return;
    }
    const LEGACY_KEYS: [&str; 9] = [
        "rapidEnabled",
        "rapidBlurType",
        "rapidLength",
        "rapidAngle",
        "rapidRadius",
        "rapidSigma",
        "rapidLambda",
        "rapidHardness",
        "rapidStrength",
    ];
    if LEGACY_KEYS.iter().all(|k| adjustments[*k].is_null()) {
        return;
    }
    let legacy_enabled = adjustments["rapidEnabled"].as_bool();
    let legacy_mode = adjustments["rapidBlurType"]
        .as_str()
        .unwrap_or("motion")
        .to_string();
    let kernel_len = adjustments["rapidLength"].as_f64().unwrap_or(0.0);
    let kernel_rad = adjustments["rapidRadius"].as_f64().unwrap_or(0.0);
    let kernel_sig = adjustments["rapidSigma"].as_f64().unwrap_or(0.0);
    let Some(obj) = adjustments.as_object_mut() else {
        return;
    };
    let mut set_toggles = |motion: bool, defocus: bool, gaussian: bool| {
        obj.insert("rapidMotionEnabled".to_string(), serde_json::json!(motion));
        obj.insert("rapidDefocusEnabled".to_string(), serde_json::json!(defocus));
        obj.insert(
            "rapidGaussianEnabled".to_string(),
            serde_json::json!(gaussian),
        );
    };
    match legacy_enabled {
        // Explicit false: the stage was off; stored kernel values are
        // abandoned state, zeroed as before, and every toggle comes out
        // explicit false.
        Some(false) => {
            set_toggles(false, false, false);
            for key in ["rapidLength", "rapidRadius", "rapidSigma"] {
                obj.insert(key.to_string(), serde_json::json!(0.0));
            }
        }
        // Explicit true: the saved mode was active; pin the old kernel
        // defaults into keys the sidecar never stored so the render
        // survives the flag's removal.
        Some(true) => {
            set_toggles(
                legacy_mode == "motion",
                legacy_mode == "defocus",
                legacy_mode == "gaussian",
            );
            for (key, default) in [
                ("rapidLength", 10.0),
                ("rapidRadius", 5.0),
                ("rapidSigma", 2.0),
            ] {
                obj.entry(key).or_insert_with(|| serde_json::json!(default));
            }
        }
        // Gen 1 (kernel-gated interim): only the saved mode could be
        // active, iff its kernel was positive.
        None => {
            set_toggles(
                legacy_mode == "motion" && kernel_len > 0.0,
                legacy_mode == "defocus" && kernel_rad > 0.0,
                legacy_mode == "gaussian" && kernel_sig > 0.0,
            );
        }
    }
    // Pin the legacy taste defaults: pre-gen-2 records rendered absent
    // strength/hardness at 100, and the new INITIALs (50) must not
    // reinterpret them.
    obj.entry("rapidStrength")
        .or_insert_with(|| serde_json::json!(100.0));
    obj.entry("rapidHardness")
        .or_insert_with(|| serde_json::json!(100.0));
    obj.remove("rapidEnabled");
}

fn migrate_glare_state(adjustments: &mut serde_json::Value) {
    if adjustments["glareEnabled"].is_boolean() {
        return;
    }
    const GLARE_KEYS: [&str; 4] = [
        "glareAmount",
        "glareVeilSize",
        "glareMaxBoost",
        "glareShowVeil",
    ];
    if GLARE_KEYS.iter().all(|k| adjustments[*k].is_null()) {
        return;
    }
    // Deliberate exception: glareShowVeil is ignored even though the legacy
    // gate honored it — it is a transient estimate-flash flag, and a stale
    // true must not enable the stage.
    let active = adjustments["glareAmount"].as_f64().unwrap_or(0.0) > 0.0;
    if let Some(obj) = adjustments.as_object_mut() {
        obj.insert("glareEnabled".to_string(), serde_json::json!(active));
    }
}

fn migrate_lowlight_state(adjustments: &mut serde_json::Value) {
    // Pin the old defaults for sections that were enabled before the
    // INITIAL changes (threshold 50 -> 100, strengths 50 -> 0) can
    // reinterpret absent keys. Guarded on the enabled flags themselves:
    // a record that never touched Low-Light comes out untouched.
    let hot_pixels = adjustments["hotPixelEnabled"].as_bool() == Some(true);
    let denoise = adjustments["denoiseEnabled"].as_bool() == Some(true);
    let Some(obj) = adjustments.as_object_mut() else {
        return;
    };
    if hot_pixels {
        obj.entry("hotPixelThreshold")
            .or_insert_with(|| serde_json::json!(50.0));
    }
    if denoise {
        for key in ["denoiseStrength", "denoiseDetail", "denoiseChroma"] {
            obj.entry(key).or_insert_with(|| serde_json::json!(50.0));
        }
    }
}

/// Full-image FFT deconvolution pre-pass. Runs before geometry transforms so
/// the PSF stays defined in sensor pixel space. No-ops (with a warning) when
/// the GPU is unavailable or processing fails.
///
/// With `rapid_scale < 1.0` the deconvolution runs on a proportionally
/// downscaled copy (with kernel parameters rescaled to match) and the result
/// is resampled back to the input dimensions - a fast approximation for
/// interactive previews; exact output requires 1.0. Output dimensions always
/// equal input dimensions either way, so downstream geometry (crop/rotation
/// coordinates) is unaffected.
/// VRAM budget for the deconvolution pipeline, computed once at RAPID device
/// init. wgpu's AdapterInfo exposes no memory size on any backend, so the
/// default is flat on discrete GPUs and derived from available system RAM on
/// integrated ones, where the GPU and CPU drain the same pool. The
/// RAPID_VRAM_MB environment variable overrides both (useful for exercising
/// the scaled fallback).
fn rapid_vram_budget_mb(is_integrated: bool) -> u64 {
    let env_override = std::env::var("RAPID_VRAM_MB")
        .ok()
        .and_then(|v| v.parse().ok());
    let available_ram_mb = if env_override.is_none() && is_integrated {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        sys.available_memory() / (1024 * 1024)
    } else {
        0
    };
    rapid_vram_budget_from(env_override, is_integrated, available_ram_mb)
}

fn rapid_vram_budget_from(
    env_override: Option<u64>,
    is_integrated: bool,
    available_ram_mb: u64,
) -> u64 {
    if let Some(mb) = env_override {
        return mb;
    }
    if is_integrated {
        (available_ram_mb / 4).min(4096)
    } else {
        4096
    }
}

/// Run blur recovery in linear light while retaining the original image for
/// failure. Encoded sources are decoded before any resize and encoded once
/// after the final resize; `source_linear` is original-source provenance.
pub fn apply_blur_recovery_scaled<'a>(
    image: std::borrow::Cow<'a, image::DynamicImage>,
    adjustments: &serde_json::Value,
    rapid_scale: f32,
    source_linear: bool,
) -> std::borrow::Cow<'a, image::DynamicImage> {
    let Some(params) = parse_rapid_params(adjustments) else {
        return image;
    };
    let Some(gpu) = get_rapid_gpu() else {
        return image;
    };
    let mut gpu = gpu.lock().unwrap();
    let RapidGpu {
        device,
        queue,
        deconvolver,
        vram_budget_mb,
        ..
    } = &mut *gpu;
    let start = std::time::Instant::now();

    // VRAM budget: cap the working scale so the padded pipeline fits.
    // Preview and export share the same cap, so a machine that can't run
    // full resolution still renders both identically.
    let (w, h) = (image.width(), image.height());
    let budget_mb = *vram_budget_mb;
    let vram_scale =
        RapidDeconvolver::max_scale_for_vram(w, h, budget_mb, deconvolver.max_texture_size);
    if vram_scale < rapid_scale {
        log::warn!(
            "RAPID: {}x{} needs ~{} MB against a {} MB budget; capping working scale at {:.3}",
            w,
            h,
            RapidDeconvolver::required_vram_mb(w, h),
            budget_mb,
            vram_scale
        );
    }
    let scale = rapid_scale.min(vram_scale);

    // The threshold follows original provenance even though the GPU core
    // always receives a linear working buffer.
    let clip_guard_sat_linear = if source_linear {
        CLIP_GUARD_SAT
    } else {
        crate::image_processing::srgb_channel_to_linear(CLIP_GUARD_SAT)
    };
    let linear = image::DynamicImage::ImageRgba32F(image.as_ref().to_rgba32f());
    let linear = if source_linear {
        linear
    } else {
        crate::image_processing::apply_srgb_to_linear(linear)
    };

    if scale < 0.999 {
        let small_w = ((w as f32 * scale).round() as u32).max(1);
        let small_h = ((h as f32 * scale).round() as u32).max(1);
        let small = linear.resize_exact(small_w, small_h, image::imageops::FilterType::Triangle);
        return match deconvolver.deconvolve_linear_image(
            device,
            queue,
            &small,
            &params.scaled(scale),
            clip_guard_sat_linear,
        ) {
            Ok(out) => {
                let restored = out.resize_exact(w, h, image::imageops::FilterType::Triangle);
                let restored = if source_linear {
                    restored
                } else {
                    crate::image_processing::apply_linear_to_srgb(restored)
                };
                log::info!(
                    "RAPID: scaled blur recovery pre-pass ({}x{} @ {:.3}) took {:?}",
                    small_w,
                    small_h,
                    scale,
                    start.elapsed()
                );
                std::borrow::Cow::Owned(restored)
            }
            Err(e) => {
                log::warn!("RAPID: scaled blur recovery failed ({e}); using original image");
                image
            }
        };
    }

    match deconvolver.deconvolve_linear_image(
        device,
        queue,
        &linear,
        &params,
        clip_guard_sat_linear,
    ) {
        Ok(out) => {
            let out = if source_linear {
                out
            } else {
                crate::image_processing::apply_linear_to_srgb(out)
            };
            log::info!("RAPID: blur recovery pre-pass took {:?}", start.elapsed());
            std::borrow::Cow::Owned(out)
        }
        Err(e) => {
            log::warn!("RAPID: blur recovery failed ({e}); using original image");
            image
        }
    }
}

// ============================================================================
// Blur-kernel estimation (cepstral analysis)
// ============================================================================

/// Result of cepstral blur-kernel estimation. `length` is in full-resolution
/// pixels; `angle` follows psf_generate.wgsl's motion_angle convention
/// (degrees, 0-180, image-space Y-down). `confident` applies the gate so the
/// threshold lives in one place; `confidence` is the raw score (how many
/// standard deviations the cepstral peak sits below the search-region mean).
/// `hardness` is the fitted motion-OTF shape (0 = Gaussian envelope, 1 =
/// hard line) and `lambda` a suggested Wiener regularization from the
/// measured noise-to-signal ratio; both are meaningful only when
/// `confident` (0 and the 0.01 default otherwise).
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct BlurEstimate {
    pub length: f32,
    pub angle: f32,
    pub confidence: f32,
    pub confident: bool,
    pub hardness: f32,
    pub lambda: f32,
}

/// Estimates below this score are reported as not confident: the deepest
/// negative excursion of pure noise over a ~100k-sample search region already
/// reaches ~4-5 sigma, so a real cepstral peak must clear that comfortably.
const BLUR_CONFIDENCE_GATE: f32 = 6.0;

/// Cap on observed log-depression depth (shared by the motion hardness fit
/// and the defocus ring matcher): a true spectral zero has unbounded log
/// depth while observed notches saturate at the noise floor, so one
/// accidental plunge must not dominate a median.
const DEPTH_CAP: f32 = 6.0;

/// Fit the motion-OTF hardness from spectral notch depth. The captured
/// ln(eps+|F|) plane is profiled along the blur direction (a wedge of
/// near-axis bins bucketed by f_along), detrended with a boxcar one notch
/// period wide, and the mean depression at the notch frequencies k/L is
/// matched against the identical measurement of the analytic OTF blend over
/// a 17-point hardness grid. Depression is measured at the notches only —
/// an anti-notch reference would let the signed sinc cancel the Gaussian
/// arm near h ≈ 0.6 and destroy monotonicity — capped, because a true zero
/// has unbounded log depth while observed notches saturate at the noise
/// floor, and combined as the *median* over k so one accidental plunge
/// (e.g. the spectrum's noise-floor knee) cannot impersonate a comb. The
/// model's power is pre-smeared with the Hann window's spectral kernel:
/// the observed notches are filled by window leakage (decisive once the
/// notch period nears the ~2-bin main lobe, i.e. long blurs), and an
/// unsmeared model would overpromise depth and fit soft. Ties resolve
/// toward the harder (physical) model. The model gets an epsilon on its
/// own unit scale; the observed epsilon guards only against log(0) and
/// cancels in the detrended depression.
fn fit_motion_hardness(
    spectrum_ln: &[f32],
    pw: usize,
    ph: usize,
    l_work: f32,
    angle_deg: f32,
) -> f32 {
    const PERP_MAX: f32 = 0.05;
    let nb = pw / 2;
    let (cos_a, sin_a) = (angle_deg.to_radians().cos(), angle_deg.to_radians().sin());

    // Wedge profile: mean ln|F| bucketed by |f_along| over [0, 0.5), using
    // per-axis normalized frequencies (the psf_generate.wgsl convention).
    let mut sums = vec![0.0f64; nb];
    let mut counts = vec![0u32; nb];
    for y in 0..ph {
        let mut v = y as f32 / ph as f32;
        if v > 0.5 {
            v -= 1.0;
        }
        for x in 0..pw {
            let mut u = x as f32 / pw as f32;
            if u > 0.5 {
                u -= 1.0;
            }
            let f_along = (u * cos_a + v * sin_a).abs();
            let f_perp = (-u * sin_a + v * cos_a).abs();
            if f_perp > PERP_MAX || f_along >= 0.5 {
                continue;
            }
            let b = ((f_along * 2.0 * nb as f32) as usize).min(nb - 1);
            sums[b] += spectrum_ln[y * pw + x] as f64;
            counts[b] += 1;
        }
    }
    let profile: Vec<Option<f32>> = sums
        .iter()
        .zip(&counts)
        .map(|(&s, &c)| (c > 0).then(|| (s / c as f64) as f32))
        .collect();

    // Boxcar detrend one notch period wide, then median capped depression
    // over the frequencies (k + offset)/L that fit in the usable band.
    // offset 0 samples the notches; offset 1/2 samples the anti-notch
    // controls. `spread` widens the sample to the deepest residual of the
    // immediate neighborhood — wanted at notches, which may straddle a
    // bucket boundary, but not at controls, which would otherwise pick up
    // notch flanks once the period shrinks toward a few buckets.
    let half_period = (((pw as f32 / l_work).round() as usize).max(3)) / 2;
    let depth_of = |profile: &[Option<f32>], offset: f32, spread: usize| -> Option<f32> {
        let n = profile.len();
        let residual_at = |i: usize| -> Option<f32> {
            let p = profile[i]?;
            let lo = i.saturating_sub(half_period);
            let hi = (i + half_period).min(n - 1);
            let vals: Vec<f32> = (lo..=hi).filter_map(|j| profile[j]).collect();
            if vals.len() < (hi - lo) / 2 + 1 {
                return None;
            }
            Some(p - vals.iter().sum::<f32>() / vals.len() as f32)
        };
        let mut depths = Vec::new();
        for k in 1..=4 {
            let f = (k as f32 + offset) / l_work;
            if f > 0.45 {
                break;
            }
            let b = (f * 2.0 * nb as f32) as usize;
            if b < spread.max(1) || b + spread.max(1) >= n {
                break;
            }
            let d = (b - spread..=b + spread)
                .filter_map(residual_at)
                .fold(f32::INFINITY, f32::min);
            if d.is_finite() {
                depths.push((-d).min(DEPTH_CAP));
            }
        }
        if depths.len() < 2 {
            return None;
        }
        depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = depths.len() / 2;
        Some(if depths.len() % 2 == 1 {
            depths[mid]
        } else {
            (depths[mid - 1] + depths[mid]) / 2.0
        })
    };

    // Sparse wedge or too few usable notches: fall back to the physical
    // prior rather than inventing a soft fit from nothing — a confident
    // cepstral peak is itself comb evidence.
    let Some(d_obs) = depth_of(&profile, 0.0, 1) else {
        return 1.0;
    };
    // A real comb is deep at k/L and flat at (k+1/2)/L; scene ripple and
    // spurious cepstral hits score both alike. Depth must clear an absolute
    // floor and double the control to count as shape evidence — otherwise
    // report soft and leave the zero-free legacy inverse in charge.
    let d_ctrl = depth_of(&profile, 0.5, 0).unwrap_or(0.0);
    if d_obs < 0.3 || d_obs < 2.0 * d_ctrl.max(0.0) {
        return 0.0;
    }

    // Invert the model depth curve. Raw model depth is non-monotone in h
    // (near h ≈ 0.6 the signed sinc cancels the Gaussian arm at the
    // anti-notches and drags the detrend baseline down), so fit against
    // the running max: the smallest hardness whose model comb is at least
    // as deep as the observed one. Observed deeper than even the full
    // line model means h = 1.
    let mut d_iso = f32::NEG_INFINITY;
    for i in 0..=16 {
        let h = i as f32 / 16.0;
        let magnitude: Vec<f32> = (0..nb)
            .map(|b| {
                let f = (b as f32 + 0.5) * 0.5 / nb as f32;
                // Mirror motion_blur_spectrum: floored Gaussian envelope
                // (MAGNITUDE_FLOOR) blended with the raw signed sinc.
                let gauss = (-0.5 * (f * l_work) * (f * l_work)).exp().max(0.15);
                let x = std::f32::consts::PI * l_work * f;
                let line = if x.abs() < 1e-6 { 1.0 } else { x.sin() / x };
                (1.0 - h) * gauss + h * line
            })
            .collect();
        // Hann leakage fill: the window's 3-tap spectral kernel smears
        // incoherent content in power, [1, 4, 1]/6 at 1-bin (= 1-bucket)
        // spacing. Without this the model's notches stay deeper than any
        // observed notch can be and every real blur fits soft.
        let model: Vec<Option<f32>> = (0..nb)
            .map(|b| {
                let sq = |v: f32| v * v;
                let lo = sq(magnitude[b.saturating_sub(1)]);
                let hi = sq(magnitude[(b + 1).min(nb - 1)]);
                let power = (lo + 4.0 * sq(magnitude[b]) + hi) / 6.0;
                Some((1e-6 + power.sqrt()).ln())
            })
            .collect();
        if let Some(d_mod) = depth_of(&model, 0.0, 1) {
            d_iso = d_iso.max(d_mod);
            if d_iso >= d_obs {
                return h;
            }
        }
    }
    1.0
}

/// Shared spectral prep for the blur estimators: luma of a working copy
/// (downscaled to <=1024 px), mean-subtracted, Hann-windowed over the image
/// extent (so the frame boundary doesn't dominate and the zero-pad stays
/// continuous), zero-padded into a pow2 grid and forward-FFT'd.
struct WorkingSpectrum {
    /// The complex spectrum; the motion cepstrum consumes it in place.
    data: Vec<cpu_fft::Complex>,
    /// ln(eps + |F|) snapshot for the shape fits. The eps-log keeps the
    /// decomposition ln|G| = ln|F_img| + ln|H| additive, so notch/ring
    /// depth is the model's own, uncorrupted by image brightness — the
    /// cepstrum's 1+|F| compresses depth scale-dependently and cannot be
    /// reused for this.
    spectrum_ln: Vec<f32>,
    pw: usize,
    ph: usize,
    w: usize,
    h: usize,
    /// working / full-res
    scale: f32,
}

fn working_spectrum(image: &image::DynamicImage, source_linear: bool) -> WorkingSpectrum {
    use image::GenericImageView;

    let (full_w, full_h) = image.dimensions();
    // Convert and decode before interpolation: estimators must inspect
    // luma-of-linear, never a resized encoded buffer.
    let linear = image::DynamicImage::ImageRgb32F(image.to_rgb32f());
    let linear = if source_linear {
        linear
    } else {
        crate::image_processing::apply_srgb_to_linear(linear)
    };
    let scale = (1024.0 / full_w.max(full_h).max(1) as f32).min(1.0);
    let working;
    let working_ref = if scale < 1.0 {
        let w = ((full_w as f32 * scale).round() as u32).max(1);
        let h = ((full_h as f32 * scale).round() as u32).max(1);
        working = linear.resize_exact(w, h, image::imageops::FilterType::Triangle);
        &working
    } else {
        &linear
    };
    let rgb = working_ref.to_rgb32f();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let (pw, ph) = (w.next_power_of_two(), h.next_power_of_two());

    let raw = rgb.as_raw();
    let mut mean = 0.0f64;
    let mut luma = vec![0.0f32; w * h];
    for (i, px) in raw.chunks_exact(3).enumerate() {
        let y = px[0] * LUMA_COEFF[0] + px[1] * LUMA_COEFF[1] + px[2] * LUMA_COEFF[2];
        luma[i] = y;
        mean += y as f64;
    }
    let mean = (mean / (w * h) as f64) as f32;

    let mut data = vec![cpu_fft::Complex::new(0.0, 0.0); pw * ph];
    for y in 0..h {
        let hann_y = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * y as f32 / h as f32).cos();
        for x in 0..w {
            let hann_x = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * x as f32 / w as f32).cos();
            data[y * pw + x] =
                cpu_fft::Complex::new((luma[y * w + x] - mean) * hann_x * hann_y, 0.0);
        }
    }

    cpu_fft::fft_2d(&mut data, pw, ph, true);
    let max_mag = data.iter().map(|v| v.magnitude()).fold(0.0f32, f32::max);
    let eps = (1e-6 * max_mag).max(f32::MIN_POSITIVE);
    let spectrum_ln: Vec<f32> = data.iter().map(|v| (eps + v.magnitude()).ln()).collect();

    WorkingSpectrum { data, spectrum_ln, pw, ph, w, h, scale }
}

/// Fit a coupled quadratic surface to a 3x3 cepstral neighborhood and
/// return its stationary point when it is a trustworthy, bounded minimum.
/// Rows are y and columns are x: `s[j][i]` is offset `(i - 1, j - 1)`.
fn refine_peak_2d(s: [[f32; 3]; 3]) -> Option<(f32, f32)> {
    let center = s[1][1] as f64;
    if !center.is_finite() {
        return None;
    }

    let (mut sum, mut sum_x, mut sum_y) = (0.0f64, 0.0f64, 0.0f64);
    let (mut sum_xx, mut sum_yy, mut sum_xy) = (0.0f64, 0.0f64, 0.0f64);
    for (j, row) in s.iter().enumerate() {
        let y = j as f64 - 1.0;
        for (i, &raw) in row.iter().enumerate() {
            if !raw.is_finite() {
                return None;
            }
            let x = i as f64 - 1.0;
            let value = raw as f64 - center;
            sum += value;
            sum_x += x * value;
            sum_y += y * value;
            sum_xx += x * x * value;
            sum_yy += y * y * value;
            sum_xy += x * y * value;
        }
    }

    let b = sum_x / 6.0;
    let c = sum_y / 6.0;
    let d = sum_xx / 2.0 - sum / 3.0;
    let e = sum_xy / 4.0;
    let g = sum_yy / 2.0 - sum / 3.0;
    if ![b, c, d, e, g].into_iter().all(f64::is_finite) {
        return None;
    }

    // Eigenvalues of the Hessian [[2d, e], [e, 2g]]. The relative guard
    // is scale-independent and rejects flat/ridge-like noisy fits.
    let root = ((d - g) * (d - g) + e * e).sqrt();
    let lambda_plus = d + g + root;
    let lambda_minus = d + g - root;
    if !lambda_plus.is_finite()
        || !lambda_minus.is_finite()
        || lambda_plus <= 0.0
        || lambda_minus <= 0.0
        || lambda_minus / lambda_plus < 1e-3
    {
        return None;
    }

    let determinant = 4.0 * d * g - e * e;
    if !determinant.is_finite() || determinant <= 0.0 {
        return None;
    }
    let dx = (e * c - 2.0 * g * b) / determinant;
    let dy = (e * b - 2.0 * d * c) / determinant;
    if !dx.is_finite() || !dy.is_finite() || dx.abs() > 0.5 || dy.abs() > 0.5 {
        return None;
    }

    Some((dx as f32, dy as f32))
}

/// Gather the production 3x3 neighborhood and retain the legacy separable
/// parabola as the exact fallback when the coupled fit is not trustworthy.
fn refine_cepstral_peak<F>(peak_dx: i32, peak_dy: i32, sample: &F) -> (f32, f32)
where
    F: Fn(i32, i32) -> f32,
{
    let mut s = [[0.0f32; 3]; 3];
    for (j, dy) in (-1..=1).enumerate() {
        for (i, dx) in (-1..=1).enumerate() {
            s[j][i] = sample(peak_dx + dx, peak_dy + dy);
        }
    }

    refine_peak_2d(s).unwrap_or_else(|| {
        let refine_axis = |c_m: f32, c_0: f32, c_p: f32| -> f32 {
            let curvature = c_m - 2.0 * c_0 + c_p;
            if curvature <= 1e-12 {
                0.0
            } else {
                (0.5 * (c_m - c_p) / curvature).clamp(-0.5, 0.5)
            }
        };
        (
            refine_axis(s[1][0], s[1][1], s[1][2]),
            refine_axis(s[0][1], s[1][1], s[2][1]),
        )
    })
}

/// Estimate linear motion blur via cepstral analysis.
///
/// The luma of a working copy (downscaled to <=1024 px, Hann-windowed so the
/// frame boundary doesn't dominate) goes through 2D FFT -> log(1 + |F|) ->
/// inverse 2D FFT. A linear motion blur multiplies the spectrum by a comb of
/// near-zeros at 1/L spacing, which the log turns into an additive periodic
/// component: the real cepstrum shows a negative peak pair at distance L
/// along the blur direction. Search radius 3-250 working px; the detection
/// floor is ~2-3 working px, so short blurs on large images come back
/// low-confidence rather than wrong.
pub fn estimate_blur(image: &image::DynamicImage, source_linear: bool) -> BlurEstimate {
    let WorkingSpectrum { mut data, spectrum_ln, pw, ph, w, h, scale } =
        working_spectrum(image, source_linear);

    // Real cepstrum: log magnitude -> inverse FFT (the forward FFT already
    // ran in working_spectrum).
    for v in data.iter_mut() {
        *v = cpu_fft::Complex::new((1.0 + v.magnitude()).ln(), 0.0);
    }
    cpu_fft::fft_2d(&mut data, pw, ph, false);

    // Search the half-plane (the cepstrum of a real signal is symmetric) for
    // the most negative peak in the valid radius band.
    let r_min = 3.0f32;
    let r_max = 250.0f32.min(w.min(h) as f32 / 2.0 - 1.0);
    let sample = |dx: i32, dy: i32| -> f32 {
        let sx = (dx as isize).rem_euclid(pw as isize) as usize;
        let sy = (dy as isize).rem_euclid(ph as isize) as usize;
        data[sy * pw + sx].re
    };
    let mut peak_val = f32::INFINITY;
    let mut peak_dx = 0i32;
    let mut peak_dy = 0i32;
    let max_r = r_max.ceil() as i32;
    for dy in 0..=max_r {
        for dx in -max_r..=max_r {
            if dy == 0 && dx <= 0 {
                continue;
            }
            let r = ((dx * dx + dy * dy) as f32).sqrt();
            if !(r_min..=r_max).contains(&r) {
                continue;
            }
            let v = sample(dx, dy);
            if v < peak_val {
                peak_val = v;
                peak_dx = dx;
                peak_dy = dy;
            }
        }
    }

    // Local noise floor (the spec's confidence definition): statistics over
    // the annulus at the peak's own radius, excluding the peak's immediate
    // neighborhood. The cepstrum's magnitude decays steeply with radius, so
    // a global z-score would let the smooth near-origin envelope of any
    // sharp image masquerade as a deep peak; against same-radius neighbors
    // only a genuine spectral comb stands out.
    let r_peak = ((peak_dx * peak_dx + peak_dy * peak_dy) as f32).sqrt();
    let band = (r_peak * 0.15).max(3.0);
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count = 0u64;
    for dy in 0..=max_r {
        for dx in -max_r..=max_r {
            if dy == 0 && dx <= 0 {
                continue;
            }
            let r = ((dx * dx + dy * dy) as f32).sqrt();
            if !(r_min..=r_max).contains(&r) || (r - r_peak).abs() > band {
                continue;
            }
            let ex = dx - peak_dx;
            let ey = dy - peak_dy;
            if ex * ex + ey * ey <= 9 {
                continue;
            }
            let v = sample(dx, dy) as f64;
            sum += v;
            sum_sq += v * v;
            count += 1;
        }
    }

    if count < 16 {
        // Image too small to search meaningfully.
        return BlurEstimate {
            length: 0.0,
            angle: 0.0,
            confidence: 0.0,
            confident: false,
            hardness: 0.0,
            lambda: 0.01,
        };
    }
    let n = count as f64;
    let mean_c = sum / n;
    let std_c = ((sum_sq / n - mean_c * mean_c).max(1e-20)).sqrt();
    let confidence = ((mean_c - peak_val as f64) / std_c) as f32;

    // Sub-bin refinement: the cepstral minimum is quantized to integer
    // working pixels, and length rescales by full/working resolution — on
    // a large frame that is several full-res pixels of length error, enough
    // to misalign the far notches the hard inverse depends on. Fit the full
    // 3x3 neighborhood so diagonal curvature is represented; guarded fits
    // fall back exactly to the prior separable parabolas.
    let (offset_x, offset_y) = refine_cepstral_peak(peak_dx, peak_dy, &sample);
    let dxf = peak_dx as f32 + offset_x;
    let dyf = peak_dy as f32 + offset_y;

    let r_refined = (dxf * dxf + dyf * dyf).sqrt();
    let length = r_refined / scale;
    let mut angle = dyf.atan2(dxf).to_degrees();
    if angle < 0.0 {
        angle += 180.0;
    }
    if angle >= 180.0 {
        angle -= 180.0;
    }

    let confident = confidence >= BLUR_CONFIDENCE_GATE;
    let hardness = if !confident {
        0.0
    } else if r_refined < 6.0 {
        // Under ~6 working px fewer than 3 notch periods fit below Nyquist —
        // too few to fit a shape; assume the physical prior (hard line).
        1.0
    } else {
        fit_motion_hardness(&spectrum_ln, pw, ph, r_refined, angle)
    };
    let lambda = if confident { suggest_lambda(&spectrum_ln, pw, ph, angle) } else { 0.01 };

    BlurEstimate { length, angle, confidence, confident, hardness, lambda }
}

/// Suggest a Wiener regularization (the UI's "Artifact suppression") from
/// the captured spectrum. Lambda in the shipped filter competes with
/// |H|² ≈ 1, so it is dimensionless and the textbook choice is the image's
/// noise-to-signal power ratio — and both sides are measurable here. The
/// wedge along the motion axis past the sinc rolloff carries noise only
/// (the blur zeroed the signal there), while the axis perpendicular to the
/// motion is unblurred, so its mid band is the scene's surviving signal
/// power. Log-domain medians keep both robust to the notch comb and to
/// outliers. The 4x factor biases toward over-suppression: the working
/// copy's downscale averages away part of the full-res noise floor, and
/// suppressing too little rings while too much merely softens. For the
/// same reason the result is floored at the suppression slider's midpoint
/// (0.01, i.e. 50): real-frame QA showed the working-scale measurement
/// still landing far below where recovery looks right, so the estimate
/// only ever raises the starting point above 50, never below. This is a
/// starting point for the slider, not a verdict.
fn suggest_lambda(spectrum_ln: &[f32], pw: usize, ph: usize, angle_deg: f32) -> f32 {
    let (cos_a, sin_a) = (angle_deg.to_radians().cos(), angle_deg.to_radians().sin());
    let mut noise_ln = Vec::new();
    let mut signal_ln = Vec::new();
    for y in 0..ph {
        let mut v = y as f32 / ph as f32;
        if v > 0.5 {
            v -= 1.0;
        }
        for x in 0..pw {
            let mut u = x as f32 / pw as f32;
            if u > 0.5 {
                u -= 1.0;
            }
            let f_along = (u * cos_a + v * sin_a).abs();
            let f_perp = (-u * sin_a + v * cos_a).abs();
            if f_along > 0.30 && f_perp < 0.05 {
                noise_ln.push(spectrum_ln[y * pw + x]);
            } else if f_along < 0.05 && (0.05..0.25).contains(&f_perp) {
                signal_ln.push(spectrum_ln[y * pw + x]);
            }
        }
    }
    if noise_ln.len() < 64 || signal_ln.len() < 64 {
        return 0.01;
    }
    let median = |v: &mut Vec<f32>| -> f32 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let nsr = (2.0 * (median(&mut noise_ln) - median(&mut signal_ln))).exp();
    (4.0 * nsr).clamp(0.01, 0.1)
}

/// Tauri command: estimate the motion-blur kernel of the currently loaded
/// image from its cepstrum. Returns full-resolution length, angle in the PSF
/// convention, and the confidence score/gate; the frontend leaves the sliders
/// untouched when `confident` is false.
#[tauri::command]
pub async fn estimate_blur_kernel(
    state: tauri::State<'_, crate::app_state::AppState>,
) -> Result<BlurEstimate, String> {
    let (image, source_linear) = {
        let guard = state.original_image.lock().unwrap();
        guard
            .as_ref()
            .map(|loaded| (loaded.image.clone(), loaded.is_raw))
            .ok_or("No image loaded")?
    };
    let start = std::time::Instant::now();
    let estimate = tokio::task::spawn_blocking(move || estimate_blur(&image, source_linear))
        .await
        .map_err(|e| format!("Blur estimation task failed: {e}"))?;
    log::info!(
        "RAPID: blur estimate L={:.1}px A={:.1}° H={:.2} λ={:.4} confidence={:.1} ({}confident) in {:?}",
        estimate.length,
        estimate.angle,
        estimate.hardness,
        estimate.lambda,
        estimate.confidence,
        if estimate.confident { "" } else { "not " },
        start.elapsed()
    );
    Ok(estimate)
}

// ============================================================================
// Defocus and gaussian estimators
// ============================================================================

/// CPU port of psf_generate.wgsl's bessel_j1 (same rational approximation for
/// |x| < 8, same asymptotic expansion beyond), so CPU-side OTF model
/// evaluations match the spectra the shader divides by.
fn bessel_j1(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let ans1 = x * (72362614232.0
            + y * (-7895059235.0
                + y * (242396853.1
                    + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        let ans2 = 144725228442.0
            + y * (2300535178.0
                + y * (18583304.74 + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
        ans1 / ans2
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356194491; // ax - 3*pi/4
        let ans1 = 1.0
            + y * (0.183105e-2
                + y * (-0.3516396496e-4 + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        let ans2 = 0.04687499995
            + y * (-0.2002690873e-3
                + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        let ans = (0.636619772 / ax).sqrt() * (xx.cos() * ans1 - z * xx.sin() * ans2);
        if x < 0.0 { -ans } else { ans }
    }
}

/// jinc(x) = 2·J1(x)/x with jinc(0) = 1 — the disc OTF's radial profile
/// (defocus_blur_spectrum in psf_generate.wgsl).
fn jinc(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        return 1.0;
    }
    2.0 * bessel_j1(x) / x
}

/// Radial frequency ρ = √(u² + v²) in cycles/pixel for a bin of the pow2
/// spectrum, with per-axis normalized frequencies (the psf_generate.wgsl
/// convention shared by fit_motion_hardness).
fn bin_rho(x: usize, y: usize, pw: usize, ph: usize) -> f32 {
    let mut u = x as f32 / pw as f32;
    if u > 0.5 {
        u -= 1.0;
    }
    let mut v = y as f32 / ph as f32;
    if v > 0.5 {
        v -= 1.0;
    }
    (u * u + v * v).sqrt()
}

/// Mean ln|F| bucketed by ρ over [0, 0.5), nb = pw/2 buckets; empty buckets
/// are None (the fit_motion_hardness wedge-profile shape, radialized).
fn radial_profile_mean(spectrum_ln: &[f32], pw: usize, ph: usize) -> Vec<Option<f32>> {
    let nb = pw / 2;
    let mut sums = vec![0.0f64; nb];
    let mut counts = vec![0u32; nb];
    for y in 0..ph {
        for x in 0..pw {
            let rho = bin_rho(x, y, pw, ph);
            if rho >= 0.5 {
                continue;
            }
            let b = ((rho * 2.0 * nb as f32) as usize).min(nb - 1);
            sums[b] += spectrum_ln[y * pw + x] as f64;
            counts[b] += 1;
        }
    }
    sums.iter()
        .zip(&counts)
        .map(|(&s, &c)| (c > 0).then(|| (s / c as f64) as f32))
        .collect()
}

/// Median ln|F| bucketed by ρ — robust to the scene's bright spectral lines,
/// which matters for the gaussian falloff fit where a single streaky edge
/// would drag a mean profile.
fn radial_profile_median(spectrum_ln: &[f32], pw: usize, ph: usize) -> Vec<Option<f32>> {
    let nb = pw / 2;
    let mut buckets: Vec<Vec<f32>> = vec![Vec::new(); nb];
    for y in 0..ph {
        for x in 0..pw {
            let rho = bin_rho(x, y, pw, ph);
            if rho >= 0.5 {
                continue;
            }
            let b = ((rho * 2.0 * nb as f32) as usize).min(nb - 1);
            buckets[b].push(spectrum_ln[y * pw + x]);
        }
    }
    buckets
        .into_iter()
        .map(|mut v| {
            if v.is_empty() {
                None
            } else {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Some(v[v.len() / 2])
            }
        })
        .collect()
}

/// Radial analog of suggest_lambda for isotropic blurs: the unblurred
/// perpendicular axis does not exist, so the signal band is the low-ρ
/// annulus compensated by the fitted model's own ln-attenuation at the band
/// center — capped at 3 nats, because near a jinc zero the correction
/// explodes, and over-suppression is the preferred failure (see
/// suggest_lambda's comment). Same 4x bias, clamp and sparse-band bail as
/// the motion version.
fn suggest_lambda_radial(
    spectrum_ln: &[f32],
    pw: usize,
    ph: usize,
    otf_ln_at: impl Fn(f32) -> f32,
) -> f32 {
    let mut noise_ln = Vec::new();
    let mut signal_ln = Vec::new();
    for y in 0..ph {
        for x in 0..pw {
            let rho = bin_rho(x, y, pw, ph);
            if rho > 0.35 {
                noise_ln.push(spectrum_ln[y * pw + x]);
            } else if (0.05..=0.15).contains(&rho) {
                signal_ln.push(spectrum_ln[y * pw + x]);
            }
        }
    }
    if noise_ln.len() < 64 || signal_ln.len() < 64 {
        return 0.01;
    }
    let median = |v: &mut Vec<f32>| -> f32 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    // The observed signal band is post-blur; add back the model's own
    // attenuation at the band center to compare pre-blur signal power
    // against the (unattenuated, sensor-side) noise floor.
    let comp = (-otf_ln_at(0.10)).clamp(0.0, 3.0);
    let nsr = (2.0 * (median(&mut noise_ln) - median(&mut signal_ln) - comp)).exp();
    (4.0 * nsr).clamp(0.01, 0.1)
}

/// Estimates below this score are reported as not confident. The score is a
/// z-score of the best candidate's matched-ring contrast against the rest of
/// the radius grid, so it shares BLUR_CONFIDENCE_GATE's shape but competes
/// against structured scene spectra rather than cepstral noise; the sharp
/// and gaussian-blurred synthetic negatives calibrate the value.
const DEFOCUS_CONFIDENCE_GATE: f32 = 5.0;

/// Absolute ring-contrast floor (nats) accompanying the z-score gate. The
/// z-score alone is fragile when the whole grid scores near zero with tiny
/// variance — a gaussian blur's smooth knee then z-spikes at a small radius
/// despite carrying no ring comb. Synthetic calibration: real discs score
/// 0.51 (R=14) to 1.59 (R=4); the gaussian false lock 0.26; sharp 0.11.
const DEFOCUS_MIN_RING_CONTRAST: f32 = 0.35;

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct DefocusEstimate {
    pub radius: f32,
    pub confidence: f32,
    pub confident: bool,
    pub lambda: f32,
}

/// Canonical not-confident result: every field finite (serde_json rejects
/// NaN/inf), lambda at the slider midpoint like the motion bail.
const DEFOCUS_NOT_CONFIDENT: DefocusEstimate = DefocusEstimate {
    radius: 0.0,
    confidence: 0.0,
    confident: false,
    lambda: 0.01,
};

/// Estimate defocus (disc) blur radius by matched analysis of the jinc
/// OTF's zero rings.
///
/// A disc of radius R multiplies the spectrum by jinc(2πRρ), whose zeros
/// sit at the J1 roots — anharmonically spaced (first gap 0.61/R vs the
/// asymptotic 0.5/R), which smears a cepstral ring impulse at exactly the
/// small radii the UI covers. So instead each candidate R is scored
/// directly: median capped depression of the boxcar-detrended radial
/// profile at its predicted zero radii, minus the same measurement at
/// inter-zero midpoint controls (a real ring comb is deep at the zeros and
/// flat between them; broadband scene texture scores both alike).
/// R_work ∈ [2.5, 25]: the floor is where the second J1 zero leaves the
/// ρ ≤ 0.45 band (two zeros minimum for a comb), so small blurs on large
/// frames come back low-confidence rather than wrong — the motion
/// estimator's documented limitation, shared.
pub fn estimate_defocus(image: &image::DynamicImage, source_linear: bool) -> DefocusEstimate {
    const R_MIN: f32 = 2.5;
    const R_MAX: f32 = 25.0;
    const R_STEP: f32 = 0.1;
    const RHO_MAX: f32 = 0.45;

    let ws = working_spectrum(image, source_linear);
    let profile = radial_profile_mean(&ws.spectrum_ln, ws.pw, ws.ph);
    let nb = ws.pw / 2;

    // J1 roots via McMahon, x_k ≈ β − 3/(8β), β = (k + 1/4)π: absolute
    // error < 4e-4 for k ≥ 1, far below a profile bucket in ρ. Generated
    // out to the band edge at the largest candidate (R = 25 consumes 22).
    let max_roots = (2.0 * std::f32::consts::PI * R_MAX * RHO_MAX / std::f32::consts::PI).ceil()
        as usize
        + 2;
    let j1_roots: Vec<f32> = (1..=max_roots)
        .map(|k| {
            let beta = (k as f32 + 0.25) * std::f32::consts::PI;
            beta - 3.0 / (8.0 * beta)
        })
        .collect();

    let median = |mut v: Vec<f32>| -> Option<f32> {
        if v.is_empty() {
            return None;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = v.len() / 2;
        Some(if v.len() % 2 == 1 { v[mid] } else { (v[mid - 1] + v[mid]) / 2.0 })
    };

    // Score every candidate radius on the shared profile.
    let mut scores: Vec<(f32, f32)> = Vec::new();
    let steps = ((R_MAX - R_MIN) / R_STEP).round() as usize;
    for i in 0..=steps {
        let r = R_MIN + i as f32 * R_STEP;
        // Boxcar detrend one ring period (≈ 0.5/R in ρ) wide, exactly the
        // fit_motion_hardness treatment.
        let half_period = ((ws.pw as f32 / (2.0 * r)).round() as usize).max(3) / 2;
        let residual_at = |i: usize| -> Option<f32> {
            let p = profile[i]?;
            let lo = i.saturating_sub(half_period);
            let hi = (i + half_period).min(nb - 1);
            let vals: Vec<f32> = (lo..=hi).filter_map(|j| profile[j]).collect();
            if vals.len() < (hi - lo) / 2 + 1 {
                return None;
            }
            Some(p - vals.iter().sum::<f32>() / vals.len() as f32)
        };
        // Depression at a radius: deepest residual of the bucket ± spread,
        // negated and capped. Zeros get spread 1 (they may straddle a
        // bucket boundary); controls get spread 0 so they don't pick up
        // zero flanks once the ring period shrinks.
        let depression_at = |rho: f32, spread: usize| -> Option<f32> {
            let b = (rho * 2.0 * nb as f32) as usize;
            if b < spread.max(1) || b + spread.max(1) >= nb {
                return None;
            }
            let d = (b - spread..=b + spread)
                .filter_map(residual_at)
                .fold(f32::INFINITY, f32::min);
            d.is_finite().then(|| (-d).min(DEPTH_CAP))
        };

        let two_pi_r = 2.0 * std::f32::consts::PI * r;
        let mut zero_depths = Vec::new();
        let mut ctl_depths = Vec::new();
        for pair in j1_roots.windows(2) {
            let rho_zero = pair[0] / two_pi_r;
            if rho_zero > RHO_MAX {
                break;
            }
            if let Some(d) = depression_at(rho_zero, 1) {
                zero_depths.push(d);
            }
            let rho_mid = (pair[0] + pair[1]) / 2.0 / two_pi_r;
            if rho_mid <= RHO_MAX
                && let Some(d) = depression_at(rho_mid, 0)
            {
                ctl_depths.push(d);
            }
        }
        if zero_depths.len() < 2 {
            continue;
        }
        let (Some(z), Some(c)) = (median(zero_depths), median(ctl_depths)) else {
            continue;
        };
        scores.push((r, z - c));
    }

    let Some(&(best_r_grid, best_score)) = scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    else {
        return DEFOCUS_NOT_CONFIDENT;
    };

    // Confidence: z-score of the best score against the rest of the grid,
    // excluding ±0.5 px around the peak (the annulus-statistics analog of
    // the cepstral confidence).
    let pool: Vec<f32> = scores
        .iter()
        .filter(|(r, _)| (r - best_r_grid).abs() > 0.5)
        .map(|&(_, s)| s)
        .collect();
    if pool.len() < 16 {
        return DEFOCUS_NOT_CONFIDENT;
    }
    let n = pool.len() as f64;
    let mean = pool.iter().map(|&s| s as f64).sum::<f64>() / n;
    let var = pool.iter().map(|&s| (s as f64 - mean).powi(2)).sum::<f64>() / n;
    let std = var.max(1e-20).sqrt();
    let confidence = ((best_score as f64 - mean) / std) as f32;

    // 3-point parabolic refine over grid neighbors, only when the peak is
    // interior to a contiguous stretch of the grid; a boundary best keeps
    // its raw value.
    let mut r_best = best_r_grid;
    if let Some(i) = scores.iter().position(|&(r, _)| r == best_r_grid)
        && i > 0
        && i + 1 < scores.len()
        && (scores[i + 1].0 - scores[i - 1].0 - 2.0 * R_STEP).abs() < 1e-4
    {
        let (s_m, s_0, s_p) = (scores[i - 1].1, scores[i].1, scores[i + 1].1);
        let curvature = s_m - 2.0 * s_0 + s_p;
        if curvature < -1e-12 {
            r_best += (0.5 * (s_m - s_p) / curvature).clamp(-0.5, 0.5) * R_STEP;
        }
    }

    let radius = r_best / ws.scale;
    let confident = confidence.is_finite()
        && confidence >= DEFOCUS_CONFIDENCE_GATE
        && best_score >= DEFOCUS_MIN_RING_CONTRAST;
    if !confident {
        return DefocusEstimate {
            radius: if radius.is_finite() { radius } else { 0.0 },
            confidence: if confidence.is_finite() { confidence } else { 0.0 },
            confident: false,
            lambda: 0.01,
        };
    }
    let lambda = suggest_lambda_radial(&ws.spectrum_ln, ws.pw, ws.ph, |rho| {
        jinc(2.0 * std::f32::consts::PI * r_best * rho).abs().max(1e-6).ln()
    });
    if !radius.is_finite() || !lambda.is_finite() {
        return DEFOCUS_NOT_CONFIDENT;
    }
    DefocusEstimate { radius, confidence, confident: true, lambda }
}

/// The t-statistic of the fitted spectral curvature must clear this before
/// a gaussian estimate is reported confident. Deliberately strict: an
/// over-strict gate degrades to "set manually" (the pre-revamp UX for this
/// tab), a lax one writes wrong sigmas.
const GAUSSIAN_CONFIDENCE_GATE: f32 = 8.0;

/// Below this working-scale sigma a "fit" is AA-filter/demosaic rolloff,
/// not photographic blur.
const GAUSSIAN_SIGMA_WORK_FLOOR: f32 = 0.5;

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct GaussianEstimate {
    pub sigma: f32,
    pub confidence: f32,
    pub confident: bool,
    pub lambda: f32,
}

/// Canonical not-confident result — every field finite.
const GAUSSIAN_NOT_CONFIDENT: GaussianEstimate = GaussianEstimate {
    sigma: 0.0,
    confidence: 0.0,
    confident: false,
    lambda: 0.01,
};

/// Estimate isotropic gaussian blur sigma from the spectrum's radial
/// falloff.
///
/// The gaussian OTF is exp(−2π²σ²ρ²) (gaussian_blur_spectrum), so
/// ln|G(ρ)| = ln|F_scene(ρ)| − 2π²σ²ρ², and natural scenes are
/// approximately power-law in ρ. A 3-parameter least squares of
/// y = c + a·lnρ − s·ρ² over the usable band separates the scene slope
/// (a, free — it absorbs the power law) from the blur curvature s;
/// σ_work = √(s/2π²). Fit in f64 on centered predictors; confidence is
/// the t-statistic of s.
pub fn estimate_gaussian(image: &image::DynamicImage, source_linear: bool) -> GaussianEstimate {
    let ws = working_spectrum(image, source_linear);
    let profile = radial_profile_median(&ws.spectrum_ln, ws.pw, ws.ph);
    let nb = ws.pw / 2;
    let rho_of = |b: usize| (b as f32 + 0.5) / (2.0 * nb as f32);

    // Noise floor from the outermost annulus; the usable band keeps only
    // buckets clearly above it, so the flat floor cannot bias sigma low.
    let floor_samples: Vec<f32> = (0..nb)
        .filter(|&b| rho_of(b) >= 0.46)
        .filter_map(|b| profile[b])
        .collect();
    if floor_samples.is_empty() {
        return GAUSSIAN_NOT_CONFIDENT;
    }
    let mut floor_sorted = floor_samples;
    floor_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let noise_floor = floor_sorted[floor_sorted.len() / 2];

    // Band start 0.02: well past the Hann main lobe (~2 bins), and at the
    // large-σ end the attenuation knee crosses the noise floor early enough
    // that the extra low-ρ buckets are what keep the fit determined
    // (σ_work = 6 has ~21 usable buckets above 0.04, ~31 above 0.02).
    let pts: Vec<(f64, f64, f64)> = (0..nb)
        .filter(|&b| {
            let rho = rho_of(b);
            (0.02..=0.42).contains(&rho)
        })
        .filter_map(|b| {
            let y = profile[b]?;
            (y >= noise_floor + 0.5).then(|| {
                let rho = rho_of(b) as f64;
                (rho.ln(), rho * rho, y as f64)
            })
        })
        .collect();
    if pts.len() < 24 {
        return GAUSSIAN_NOT_CONFIDENT;
    }

    // Centered 3×3 normal equations (intercept eliminated by centering).
    let n = pts.len() as f64;
    let m1 = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let m2 = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let my = pts.iter().map(|p| p.2).sum::<f64>() / n;
    let (mut s11, mut s12, mut s22, mut s1y, mut s2y, mut syy) =
        (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for &(t1, t2, y) in &pts {
        let (d1, d2, dy) = (t1 - m1, t2 - m2, y - my);
        s11 += d1 * d1;
        s12 += d1 * d2;
        s22 += d2 * d2;
        s1y += d1 * dy;
        s2y += d2 * dy;
        syy += dy * dy;
    }
    let det = s11 * s22 - s12 * s12;
    // Scale-invariant rank check: over a short band lnρ and ρ² are nearly
    // collinear and the system degenerates.
    if s11 <= 0.0 || s22 <= 0.0 || det <= 1e-12 * s11 * s22 {
        return GAUSSIAN_NOT_CONFIDENT;
    }
    let a = (s22 * s1y - s12 * s2y) / det;
    let b2 = (s11 * s2y - s12 * s1y) / det; // coefficient on ρ²; s = −b2
    let s_curv = -b2;
    if s_curv <= 0.0 {
        return GAUSSIAN_NOT_CONFIDENT;
    }

    // OLS covariance: t = s / stderr(s), stderr² = σ̂²·S11/det with
    // σ̂² = RSS/(n − 3).
    let rss = (syy - a * s1y - b2 * s2y).max(0.0);
    let sigma2 = rss / (n - 3.0);
    let var_s = sigma2 * s11 / det;
    if !var_s.is_finite() || var_s <= 0.0 {
        return GAUSSIAN_NOT_CONFIDENT;
    }
    let t = (s_curv / var_s.sqrt()) as f32;

    let sigma_work = (s_curv / (2.0 * std::f64::consts::PI * std::f64::consts::PI)).sqrt() as f32;
    let sigma = sigma_work / ws.scale;
    if !t.is_finite() || !sigma.is_finite() {
        return GAUSSIAN_NOT_CONFIDENT;
    }
    let confident = t >= GAUSSIAN_CONFIDENCE_GATE && sigma_work >= GAUSSIAN_SIGMA_WORK_FLOOR;
    let lambda = if confident {
        let two_pi_sq = 2.0 * std::f32::consts::PI * std::f32::consts::PI;
        suggest_lambda_radial(&ws.spectrum_ln, ws.pw, ws.ph, |rho| {
            -two_pi_sq * sigma_work * sigma_work * rho * rho
        })
    } else {
        0.01
    };
    if !lambda.is_finite() {
        return GAUSSIAN_NOT_CONFIDENT;
    }
    GaussianEstimate { sigma, confidence: t, confident, lambda }
}

/// Tauri command: estimate the defocus-blur radius of the currently loaded
/// image from its spectrum's jinc zero rings. Full-resolution radius; the
/// frontend leaves the sliders untouched when `confident` is false.
#[tauri::command]
pub async fn estimate_defocus_kernel(
    state: tauri::State<'_, crate::app_state::AppState>,
) -> Result<DefocusEstimate, String> {
    let (image, source_linear) = {
        let guard = state.original_image.lock().unwrap();
        guard
            .as_ref()
            .map(|loaded| (loaded.image.clone(), loaded.is_raw))
            .ok_or("No image loaded")?
    };
    let start = std::time::Instant::now();
    let estimate = tokio::task::spawn_blocking(move || estimate_defocus(&image, source_linear))
        .await
        .map_err(|e| format!("Defocus estimation task failed: {e}"))?;
    log::info!(
        "RAPID: defocus estimate R={:.1}px λ={:.4} confidence={:.1} ({}confident) in {:?}",
        estimate.radius,
        estimate.lambda,
        estimate.confidence,
        if estimate.confident { "" } else { "not " },
        start.elapsed()
    );
    Ok(estimate)
}

/// Tauri command: estimate the gaussian blur sigma of the currently loaded
/// image from its spectral falloff. Full-resolution sigma; the frontend
/// leaves the sliders untouched when `confident` is false.
#[tauri::command]
pub async fn estimate_gaussian_kernel(
    state: tauri::State<'_, crate::app_state::AppState>,
) -> Result<GaussianEstimate, String> {
    let (image, source_linear) = {
        let guard = state.original_image.lock().unwrap();
        guard
            .as_ref()
            .map(|loaded| (loaded.image.clone(), loaded.is_raw))
            .ok_or("No image loaded")?
    };
    let start = std::time::Instant::now();
    let estimate = tokio::task::spawn_blocking(move || estimate_gaussian(&image, source_linear))
        .await
        .map_err(|e| format!("Gaussian estimation task failed: {e}"))?;
    log::info!(
        "RAPID: gaussian estimate σ={:.2}px λ={:.4} confidence={:.1} ({}confident) in {:?}",
        estimate.sigma,
        estimate.lambda,
        estimate.confidence,
        if estimate.confident { "" } else { "not " },
        start.elapsed()
    );
    Ok(estimate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_fft::{fft_1d, fft_2d, Complex};
    use std::f32::consts::PI;

    // ========================================================================
    // Unit Tests
    // ========================================================================

    #[test]
    fn test_blur_type_conversion() {
        assert_eq!(BlurType::from(0), BlurType::Motion);
        assert_eq!(BlurType::from(1), BlurType::Defocus);
        assert_eq!(BlurType::from(2), BlurType::Gaussian);
        assert_eq!(BlurType::from(99), BlurType::Motion); // Default fallback
    }

    #[test]
    fn test_rapid_vram_budget() {
        use super::rapid_vram_budget_from;

        // The env override always wins.
        assert_eq!(rapid_vram_budget_from(Some(512), true, 32768), 512);
        assert_eq!(rapid_vram_budget_from(Some(8192), false, 0), 8192);
        // Discrete GPUs keep the flat default regardless of RAM.
        assert_eq!(rapid_vram_budget_from(None, false, 4096), 4096);
        assert_eq!(rapid_vram_budget_from(None, false, 262144), 4096);
        // Integrated: a quarter of available RAM, capped at the flat default.
        assert_eq!(rapid_vram_budget_from(None, true, 32768), 4096);
        assert_eq!(rapid_vram_budget_from(None, true, 16384), 4096);
        assert_eq!(rapid_vram_budget_from(None, true, 12288), 3072);
        assert_eq!(rapid_vram_budget_from(None, true, 8192), 2048);
    }

    #[test]
    fn test_rapid_params_default() {
        let params = RapidParams::default();
        assert!(!params.enabled);
        assert_eq!(params.modes, ModeSet::MOTION);
        assert!((params.lambda - 0.01).abs() < 0.001);
        assert!((params.strength - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_padded_dimensions() {
        assert_eq!(RapidDeconvolver::get_padded_dimensions(1920, 1080), (2048, 2048));
        assert_eq!(RapidDeconvolver::get_padded_dimensions(2048, 2048), (2048, 2048));
        assert_eq!(RapidDeconvolver::get_padded_dimensions(2049, 1080), (4096, 2048));
        assert_eq!(RapidDeconvolver::get_padded_dimensions(4096, 2160), (4096, 4096));
    }

    #[test]
    fn test_fft_passes() {
        assert_eq!(RapidDeconvolver::get_fft_passes(256), 8);
        assert_eq!(RapidDeconvolver::get_fft_passes(1024), 10);
        assert_eq!(RapidDeconvolver::get_fft_passes(2048), 11);
        assert_eq!(RapidDeconvolver::get_fft_passes(4096), 12);
        assert_eq!(RapidDeconvolver::get_fft_passes(8192), 13);
    }

    #[test]
    fn test_uniform_struct_sizes() {
        // Ensure structs are properly sized for GPU alignment
        assert_eq!(std::mem::size_of::<FFTParams>(), 32);
        assert_eq!(std::mem::size_of::<PSFParams>(), 32);
        assert_eq!(std::mem::size_of::<WienerParams>(), 32);
        assert_eq!(std::mem::size_of::<UtilityParams>(), 32);
        assert_eq!(std::mem::size_of::<NormalizeParams>(), 16);
    }

    /// CPU lock for the Gaussian-active adaptive-lambda policy in
    /// wiener_filter.wgsl. Lambda itself must interpolate smoothly between
    /// the dead-band and live postures without creating a transfer step.
    #[test]
    fn test_gaussian_gate_curve_continuous_and_bounded() {
        const GATE_LOW_POWER: f32 = 0.01;
        const GATE_HIGH_POWER: f32 = 0.04;

        for &base_lambda in &[0.01f32, 0.05] {
            for &snr_clamped in &[0.1f32, 1.0, 10.0] {
                let lambda_dead = base_lambda / snr_clamped.min(1.0);
                let lambda_live = base_lambda / snr_clamped;
                let lambda_at_power = |h2: f32| -> f32 {
                    let t = ((h2 - GATE_LOW_POWER)
                        / (GATE_HIGH_POWER - GATE_LOW_POWER))
                        .clamp(0.0, 1.0);
                    let trust = t * t * (3.0 - 2.0 * t);
                    lambda_dead + (lambda_live - lambda_dead) * trust
                };

                assert!(
                    (lambda_at_power(GATE_LOW_POWER) - lambda_dead).abs() <= 1e-7,
                    "low endpoint drifted for base lambda {base_lambda}, SNR {snr_clamped}"
                );
                assert!(
                    (lambda_at_power(GATE_HIGH_POWER) - lambda_live).abs() <= 1e-7,
                    "high endpoint drifted for base lambda {base_lambda}, SNR {snr_clamped}"
                );

                let lambda_min = lambda_dead.min(lambda_live);
                let lambda_max = lambda_dead.max(lambda_live);
                let mut previous_transfer: Option<f32> = None;
                let mut max_gain = 0.0f32;
                for step in 0..=100_000u32 {
                    let h = step as f32 * 1e-5;
                    let h2 = h * h;
                    let lambda = lambda_at_power(h2);
                    assert!(
                        lambda >= lambda_min - 1e-7 && lambda <= lambda_max + 1e-7,
                        "lambda {lambda} escaped [{lambda_min}, {lambda_max}] for base lambda \
                         {base_lambda}, SNR {snr_clamped}, H {h}"
                    );
                    let transfer = h2 / (h2 + lambda);
                    if let Some(previous) = previous_transfer {
                        assert!(
                            (transfer - previous).abs() < 0.01,
                            "restored transfer stepped from {previous} to {transfer} for base \
                             lambda {base_lambda}, SNR {snr_clamped}, H {h}"
                        );
                    }
                    previous_transfer = Some(transfer);
                    max_gain = max_gain.max(h / (h2 + lambda));
                }

                let transfer_at = |h: f32| -> f32 {
                    let h2 = h * h;
                    h2 / (h2 + lambda_at_power(h2))
                };
                assert!(
                    (transfer_at(0.15001) - transfer_at(0.14999)).abs() < 0.01,
                    "former H=0.15 boundary is discontinuous for base lambda {base_lambda}, \
                     SNR {snr_clamped}"
                );

                if snr_clamped == 10.0 {
                    let bound = if base_lambda == 0.01 { 5.22 } else { 4.49 };
                    assert!(
                        max_gain < bound,
                        "gain {max_gain} exceeds {bound} for base lambda {base_lambda}"
                    );
                }
            }
        }
    }

    // ========================================================================
    // FFT Reference Tests
    // ========================================================================

    #[test]
    fn test_fft_1d_impulse() {
        // FFT of impulse [1, 0, 0, 0] should be [1, 1, 1, 1]
        let mut data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        fft_1d(&mut data, true);

        for (i, x) in data.iter().enumerate() {
            assert!(
                (x.re - 1.0).abs() < 1e-5 && x.im.abs() < 1e-5,
                "FFT of impulse failed at index {}: got ({}, {})",
                i, x.re, x.im
            );
        }
    }

    #[test]
    fn test_fft_1d_dc() {
        // FFT of constant [1, 1, 1, 1] should be [4, 0, 0, 0]
        let mut data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

        fft_1d(&mut data, true);

        assert!((data[0].re - 4.0).abs() < 1e-5, "DC component should be 4");
        for i in 1..4 {
            assert!(
                data[i].magnitude() < 1e-5,
                "Non-DC component should be 0 at index {}",
                i
            );
        }
    }

    #[test]
    fn test_fft_1d_roundtrip() {
        // FFT followed by IFFT should recover original signal
        let original = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(8.0, 0.0),
        ];

        let mut data = original.clone();
        fft_1d(&mut data, true);
        fft_1d(&mut data, false);

        for (i, (orig, result)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (orig.re - result.re).abs() < 1e-4 && (orig.im - result.im).abs() < 1e-4,
                "Roundtrip failed at index {}: expected ({}, {}), got ({}, {})",
                i, orig.re, orig.im, result.re, result.im
            );
        }
    }

    #[test]
    fn test_fft_1d_sine() {
        // FFT of a single sine wave should have two peaks
        let n = 8;
        let freq = 1; // One cycle
        let mut data: Vec<Complex> = (0..n)
            .map(|i| {
                let angle = 2.0 * PI * freq as f32 * i as f32 / n as f32;
                Complex::new(angle.sin(), 0.0)
            })
            .collect();

        fft_1d(&mut data, true);

        // For a sine wave, energy should be at indices 1 and n-1
        let peak1 = data[freq].magnitude();
        let peak2 = data[n - freq].magnitude();

        assert!(peak1 > 3.0, "Peak at index {} should be significant", freq);
        assert!(peak2 > 3.0, "Peak at index {} should be significant", n - freq);

        // Other frequencies should be near zero
        assert!(data[0].magnitude() < 1e-5, "DC should be zero for sine");
    }

    #[test]
    fn test_fft_2d_roundtrip() {
        // 2D FFT roundtrip test
        let width = 4;
        let height = 4;
        let original: Vec<Complex> = (0..width * height)
            .map(|i| Complex::new((i + 1) as f32, 0.0))
            .collect();

        let mut data = original.clone();
        fft_2d(&mut data, width, height, true);
        fft_2d(&mut data, width, height, false);

        for (i, (orig, result)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (orig.re - result.re).abs() < 1e-3 && (orig.im - result.im).abs() < 1e-3,
                "2D roundtrip failed at index {}: expected ({}, {}), got ({}, {})",
                i, orig.re, orig.im, result.re, result.im
            );
        }
    }

    #[test]
    fn test_fft_2d_separable() {
        // Test that 2D FFT is separable (row FFT then col FFT = 2D FFT)
        let width = 4;
        let height = 4;
        let original: Vec<Complex> = (0..width * height)
            .map(|i| Complex::new(((i * 7) % 13) as f32, 0.0))
            .collect();

        // Method 1: Direct 2D FFT
        let mut data1 = original.clone();
        fft_2d(&mut data1, width, height, true);

        // Method 2: Row FFT then column FFT (which is what fft_2d does)
        // This test verifies the implementation is correct by checking consistency
        let mut data2 = original.clone();

        // Row transforms
        for row in 0..height {
            let start = row * width;
            let mut row_data: Vec<Complex> = data2[start..start + width].to_vec();
            fft_1d(&mut row_data, true);
            data2[start..start + width].copy_from_slice(&row_data);
        }

        // Column transforms
        for col in 0..width {
            let mut col_data: Vec<Complex> = (0..height).map(|row| data2[row * width + col]).collect();
            fft_1d(&mut col_data, true);
            for (row, &val) in col_data.iter().enumerate() {
                data2[row * width + col] = val;
            }
        }

        // Results should match
        for (i, (v1, v2)) in data1.iter().zip(data2.iter()).enumerate() {
            assert!(
                (v1.re - v2.re).abs() < 1e-4 && (v1.im - v2.im).abs() < 1e-4,
                "2D FFT methods differ at index {}: ({}, {}) vs ({}, {})",
                i, v1.re, v1.im, v2.re, v2.im
            );
        }
    }

    #[test]
    fn test_parseval_theorem() {
        // Parseval's theorem: sum of |x|^2 = (1/N) * sum of |X|^2
        let original = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
            Complex::new(6.0, 0.0),
            Complex::new(7.0, 0.0),
            Complex::new(8.0, 0.0),
        ];

        let n = original.len();
        let time_energy: f32 = original.iter().map(|x| x.re * x.re + x.im * x.im).sum();

        let mut freq = original.clone();
        fft_1d(&mut freq, true);
        let freq_energy: f32 = freq.iter().map(|x| x.re * x.re + x.im * x.im).sum();

        // time_energy should equal freq_energy / N
        let expected_freq_energy = time_energy * n as f32;
        assert!(
            (freq_energy - expected_freq_energy).abs() < 1e-3,
            "Parseval's theorem failed: time={}, freq={}, expected freq={}",
            time_energy, freq_energy, expected_freq_energy
        );
    }

    /// Spike gate for the MKII port: prove the full GPU pipeline
    /// (upload -> real_to_complex -> FFT -> PSF -> Wiener -> IFFT -> readback)
    /// runs end to end on this machine against the current wgpu.
    #[test]
    fn test_gpu_deconvolve_end_to_end() {
        use image::GenericImageView;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("skipping GPU spike test: no adapter ({e})");
                return;
            }
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("RAPID spike test device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("failed to create device");

        let mut deconv = RapidDeconvolver::new(&adapter, &device).expect("failed to create deconvolver");

        // Mid-gray field with a bright square: enough structure to survive
        // a Wiener round-trip recognizably.
        let mut img = image::RgbaImage::from_pixel(160, 120, image::Rgba([90, 90, 90, 255]));
        for y in 40..80 {
            for x in 60..100 {
                img.put_pixel(x, y, image::Rgba([230, 230, 230, 255]));
            }
        }
        let input = image::DynamicImage::ImageRgba8(img);

        let params = RapidParams {
            enabled: true,
            modes: ModeSet::GAUSSIAN,
            gaussian_sigma: 1.5,
            lambda: 0.01,
            strength: 1.0,
            ..Default::default()
        };

        let out = deconv
            .deconvolve_linear_image(&device, &queue, &input, &params, CLIP_GUARD_SAT)
            .expect("deconvolve_image failed");
        assert_eq!(out.dimensions(), input.dimensions());

        let rgb = out.to_rgb8();
        let (mut lo, mut hi) = (255u8, 0u8);
        for p in rgb.pixels() {
            for &c in &p.0 {
                lo = lo.min(c);
                hi = hi.max(c);
            }
        }
        assert!(hi > lo, "output is a constant image");
        let center = rgb.get_pixel(80, 60);
        assert!(center[0] > 150, "bright square lost after deconvolution: {:?}", center);
        let outside = rgb.get_pixel(15, 15);
        assert!(outside[0] < 150, "background blown out after deconvolution: {:?}", outside);
    }

    /// Adapter + device + deconvolver for GPU tests, or None to skip on
    /// machines without a usable adapter (mirrors the spike test's skip).
    fn gpu_test_context(label: &str) -> Option<(wgpu::Device, wgpu::Queue, RapidDeconvolver)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("skipping {label}: no adapter ({e})");
                return None;
            }
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("RAPID GPU test device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("failed to create device");
        let deconv = RapidDeconvolver::new(&adapter, &device).expect("failed to create deconvolver");
        Some((device, queue, deconv))
    }

    /// The blur-recovery pre-pass must not collapse its linear f32 result to
    /// 8-bit, and recombination must preserve source alpha exactly.
    #[test]
    fn test_deconvolve_output_is_f32_unquantized() {
        let Some((device, queue, mut deconv)) = gpu_test_context("f32 output GPU test") else {
            return;
        };

        const WIDTH: u32 = 1024;
        const HEIGHT: u32 = 64;
        let source = image::Rgba32FImage::from_fn(WIDTH, HEIGHT, |x, y| {
            let t = x as f32 / (WIDTH - 1) as f32;
            let value = 0.02 + 0.23 * t;
            let alpha_step = ((x + 3 * y) % 37) as f32 / 36.0;
            image::Rgba([value, value, value, 0.1 + 0.8 * alpha_step])
        });
        let input = image::DynamicImage::ImageRgba32F(source.clone());
        let params = RapidParams {
            enabled: true,
            modes: ModeSet::DEFOCUS,
            defocus_radius: 3.0,
            lambda: 0.01,
            strength: 1.0,
            adaptive: true,
            clip_guard: true,
            ..Default::default()
        };

        let out = deconv
            .deconvolve_linear_image(&device, &queue, &input, &params, CLIP_GUARD_SAT)
            .expect("deconvolve_image failed");
        let output = match out {
            image::DynamicImage::ImageRgba32F(output) => output,
            other => panic!("expected ImageRgba32F, got {:?}", other.color()),
        };

        let scan_y = HEIGHT / 2;
        let values: std::collections::BTreeSet<u32> = (0..WIDTH)
            .map(|x| {
                let value = output.get_pixel(x, scan_y)[0];
                assert!(value.is_finite(), "non-finite ramp value at x={x}");
                value.to_bits()
            })
            .collect();
        assert!(
            values.len() > 512,
            "f32 ramp retained only {} distinct values",
            values.len()
        );

        for (src, dst) in source.pixels().zip(output.pixels()) {
            assert_eq!(src[3].to_bits(), dst[3].to_bits(), "alpha changed during recombination");
        }
    }

    /// Square-on-gray fixture shared by the compound-mode GPU tests.
    fn gpu_test_image() -> image::DynamicImage {
        let mut img = image::RgbaImage::from_pixel(160, 120, image::Rgba([90, 90, 90, 255]));
        for y in 40..80 {
            for x in 60..100 {
                img.put_pixel(x, y, image::Rgba([230, 230, 230, 255]));
            }
        }
        image::DynamicImage::ImageRgba8(img)
    }

    /// A compound mode set must run as ONE pass and produce a result that is
    /// finite and distinct from either member alone — the product OTF is a
    /// different filter than each factor.
    #[test]
    fn test_gpu_deconvolve_compound_modes() {
        use image::GenericImageView;

        let Some((device, queue, mut deconv)) = gpu_test_context("compound GPU test") else {
            return;
        };
        let input = gpu_test_image();
        let run = |deconv: &mut RapidDeconvolver, modes: ModeSet| -> image::DynamicImage {
            let params = RapidParams {
                enabled: true,
                modes,
                motion_length: 12.0,
                motion_angle: 0.0,
                gaussian_sigma: 1.5,
                lambda: 0.01,
                strength: 1.0,
                ..Default::default()
            };
            deconv
                .deconvolve_linear_image(&device, &queue, &input, &params, CLIP_GUARD_SAT)
                .expect("deconvolve_image failed")
        };
        let compound = run(&mut deconv, ModeSet { motion: true, defocus: false, gaussian: true });
        assert_eq!(compound.dimensions(), input.dimensions());
        for p in compound.to_rgb32f().pixels() {
            assert!(p.0.iter().all(|c| c.is_finite()), "non-finite pixel in compound output");
        }
        let motion_only = run(&mut deconv, ModeSet::MOTION);
        let gaussian_only = run(&mut deconv, ModeSet::GAUSSIAN);
        let differs = |a: &image::DynamicImage, b: &image::DynamicImage| -> bool {
            a.to_rgb8()
                .pixels()
                .zip(b.to_rgb8().pixels())
                .any(|(pa, pb)| pa.0.iter().zip(&pb.0).any(|(&ca, &cb)| ca.abs_diff(cb) > 1))
        };
        assert!(differs(&compound, &motion_only), "compound output equals motion-only");
        assert!(differs(&compound, &gaussian_only), "compound output equals gaussian-only");
    }

    /// The defocus component's hardness is pinned to 1.0 inside the shader
    /// (the slider only exists in the motion UI), so a defocus-only render
    /// must be invariant under the hardness parameter.
    #[test]
    fn test_gpu_defocus_hardness_invariance() {
        let Some((device, queue, mut deconv)) = gpu_test_context("defocus hardness GPU test") else {
            return;
        };
        let input = gpu_test_image();
        let run = |deconv: &mut RapidDeconvolver, hardness: f32| -> image::RgbaImage {
            let params = RapidParams {
                enabled: true,
                modes: ModeSet::DEFOCUS,
                defocus_radius: 3.0,
                lambda: 0.01,
                strength: 1.0,
                hardness,
                ..Default::default()
            };
            deconv
                .deconvolve_linear_image(&device, &queue, &input, &params, CLIP_GUARD_SAT)
                .expect("deconvolve_image failed")
                .to_rgba8()
        };
        let soft = run(&mut deconv, 0.0);
        let hard = run(&mut deconv, 1.0);
        let max_diff = soft
            .pixels()
            .zip(hard.pixels())
            .flat_map(|(a, b)| a.0.iter().zip(&b.0).map(|(&ca, &cb)| ca.abs_diff(cb)))
            .max()
            .unwrap_or(0);
        assert!(
            max_diff <= 1,
            "defocus render varied with hardness (max channel diff {max_diff})"
        );
    }

    /// Banding metric: Hann-windowed spectral power of the red channel over
    /// an interior span of `n` columns starting at `x0`, averaged across
    /// rows, summed over bins [k_lo, k_hi]. The window isolates the span
    /// from its own endpoints, so what remains in the band is periodic
    /// banding that reached the frame interior — the artifact the edge taper
    /// kills — and not the (legitimate, localized) softened border strips.
    fn interior_band_energy(
        img: &image::DynamicImage,
        x0: usize,
        n: usize,
        k_lo: usize,
        k_hi: usize,
    ) -> f64 {
        let rgb = img.to_rgb32f();
        let h = rgb.height() as usize;
        let mut energy = 0.0f64;
        for y in 0..h {
            let mut row: Vec<Complex> = (0..n)
                .map(|i| {
                    let hann = 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos();
                    Complex::new(rgb.get_pixel((x0 + i) as u32, y as u32)[0] * hann, 0.0)
                })
                .collect();
            fft_1d(&mut row, true);
            for v in &row[k_lo..=k_hi] {
                energy += (v.re as f64).powi(2) + (v.im as f64).powi(2);
            }
        }
        energy / h as f64
    }

    /// Exit test body: the edge taper must collapse FFT wrap-seam banding by
    /// an order of magnitude versus raw zero-pad. Width is an exact power of
    /// two (no pad headroom -> PSF-consistent border taper) while height pads
    /// 200 -> 256 (reflect-101 mirror margins), so both mechanisms run.
    /// Parameterized over the motion-OTF hardness: h = 0 is the legacy
    /// Gaussian envelope, h = 1 the hard-line OTF whose sinc zeros are
    /// exactly the amplification the Gaussian was introduced to avoid.
    /// `drift_gate` bounds interior drift: the scene is sharp, so the h = 1
    /// inverse legitimately rings the content steps at the frame edges with
    /// echoes at multiples of L, which reach the interior window — filter
    /// physics on unblurred input, not a taper defect, hence a looser gate.
    fn assert_edge_taper_reduces_banding(hardness: f32, drift_gate: f32) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("skipping GPU banding test: no adapter ({e})");
                return;
            }
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("RAPID banding test device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("failed to create device");
        let mut deconv = RapidDeconvolver::new(&adapter, &device).expect("failed to create deconvolver");

        // Bright-edge image: flat interior with a dark block on the left
        // frame edge and a bright block on the right. The circular wrap seam
        // is a 0.9 step; the interior is constant, so any periodic energy
        // that shows up there is boundary ringing, not content.
        let (width, height) = (256u32, 200u32);
        let img = image::RgbaImage::from_fn(width, height, |x, _| {
            let v: f32 = if x < 8 {
                0.05
            } else if x >= width - 8 {
                0.95
            } else {
                0.4
            };
            let b = (v * 255.0).round() as u8;
            image::Rgba([b, b, b, 255])
        });
        let input = image::DynamicImage::ImageRgba8(img);

        let blur_len = 32usize;
        let base_params = RapidParams {
            enabled: true,
            modes: ModeSet::MOTION,
            motion_length: blur_len as f32,
            motion_angle: 0.0,
            lambda: 0.002,
            strength: 1.0,
            edge_taper: false,
            hardness,
            ..Default::default()
        };
        let tapered_params = RapidParams { edge_taper: true, ..base_params };

        let base = deconv
            .deconvolve_linear_image(&device, &queue, &input, &base_params, CLIP_GUARD_SAT)
            .expect("baseline deconvolve failed");
        let tapered = deconv
            .deconvolve_linear_image(&device, &queue, &input, &tapered_params, CLIP_GUARD_SAT)
            .expect("tapered deconvolve failed");

        // The motion OTF is H(k) = exp(-0.5·(k·L/N)²); the Wiener amplitude
        // gain H/(H²+λ) peaks ~11x near k=19 at N=256, L=32, λ=0.002. Over
        // the central 128 columns that band maps to bins ~[5, 14] (periods
        // 9-26 px). The baseline's seam ringing reaches the interior; the
        // tapered output must not.
        let e_base = interior_band_energy(&base, 64, 128, 5, 14);
        let e_tapered = interior_band_energy(&tapered, 64, 128, 5, 14);
        eprintln!(
            "edge taper banding reduction: {:.1}x (baseline {:.3e}, tapered {:.3e})",
            e_base / e_tapered,
            e_base,
            e_tapered
        );
        assert!(
            e_base > 10.0 * e_tapered,
            "edge taper reduced banding only {:.1}x (baseline {:.3e}, tapered {:.3e})",
            e_base / e_tapered,
            e_base,
            e_tapered
        );

        // The taper must not disturb the interior: the center of a smooth
        // blur-consistent ramp should survive deconvolution nearly unchanged.
        let in_rgb = input.to_rgb32f();
        let out_rgb = tapered.to_rgb32f();
        let y = height / 2;
        let mut max_dev = 0.0f32;
        let mut max_x = 0u32;
        for x in 64..192 {
            let d = (out_rgb.get_pixel(x, y)[0] - in_rgb.get_pixel(x, y)[0]).abs();
            if d > max_dev {
                max_dev = d;
                max_x = x;
            }
        }
        eprintln!("interior drift: {max_dev:.4} at x={max_x}");
        assert!(
            max_dev < drift_gate,
            "interior drifted by {max_dev} after tapered deconvolution"
        );
    }

    #[test]
    fn test_gpu_edge_taper_reduces_banding() {
        assert_edge_taper_reduces_banding(0.0, 0.05);
    }

    /// "The edge taper makes the zeros safe" made falsifiable: the h = 1
    /// line OTF has true sinc zeros, the worst case for seam amplification.
    #[test]
    fn test_gpu_edge_taper_reduces_banding_at_full_hardness() {
        assert_edge_taper_reduces_banding(1.0, 0.10);
    }

    /// The hardness blend is the ghost fix: deconvolving hard-line-blurred
    /// data with the legacy Gaussian-envelope OTF factors into a sharpening
    /// kernel convolved with the *unmodeled* L-px box, so every feature
    /// reconstructs as a bright copy at each end of the smear — the double
    /// image at ±L/2 around the feature, separation L. The h = 1 line OTF
    /// absorbs the box into the model and reconstructs one centered copy.
    /// Measured as peak *positive* profile deviation in the copy windows
    /// relative to the principal reconstruction: the positive part excludes
    /// the dark notch-loss dips at ±L (spectral lines the blur truly
    /// zeroed), which no model shape can restore.
    #[test]
    fn test_gpu_hardness_collapses_ghost_pair() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("skipping GPU ghost test: no adapter ({e})");
                return;
            }
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("RAPID ghost test device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("failed to create device");
        let mut deconv = RapidDeconvolver::new(&adapter, &device).expect("failed to create deconvolver");

        // Isolated bright bar on mid-gray, hard-line blurred: the ground
        // truth has a single feature, so anything at ±L is a model artifact.
        // The bar sits clear of the border-taper strips (2·L from each edge)
        // and the gray levels leave headroom for ringing in both directions.
        // Odd blur length: motion_blur_line then hits exact integer taps, a
        // true 31-px box whose spectral zeros sit at k/31 where the h = 1
        // model puts them — an even length rounds the half-integer taps into
        // a holed kernel whose zeros match no sinc.
        let (width, height) = (256u32, 200u32);
        let bar_center = 128i32;
        let blur_len = 31i32;
        let sharp = image::RgbaImage::from_fn(width, height, |x, _| {
            let v: f32 = if (x as i32 - bar_center).abs() <= 2 { 0.9 } else { 0.4 };
            let b = (v * 255.0).round() as u8;
            image::Rgba([b, b, b, 255])
        });
        let input = image::DynamicImage::ImageRgba8(motion_blur_line(&sharp, blur_len as f32, 0.0));

        let soft_params = RapidParams {
            enabled: true,
            modes: ModeSet::MOTION,
            motion_length: blur_len as f32,
            motion_angle: 0.0,
            lambda: 0.002,
            strength: 1.0,
            ..Default::default()
        };
        let hard_params = RapidParams { hardness: 1.0, ..soft_params };

        // Peak positive deviation from the far-field background of the
        // row-averaged column profile over an inclusive column window. The
        // bar is 5 px, so the smear ends — the ghost copies — sit near
        // ±(L + 5)/2 = ±18; windows [10, 22] cover them with margin while
        // staying clear of the principal window and the ±L echo dips.
        let profile_pos_peak = |img: &image::DynamicImage, x_lo: i32, x_hi: i32| -> f32 {
            let rgb = img.to_rgb32f();
            let col = |x: i32| -> f32 {
                (0..rgb.height()).map(|y| rgb.get_pixel(x as u32, y)[0]).sum::<f32>()
                    / rgb.height() as f32
            };
            let bg: f32 = (70..86).chain(170..186).map(col).sum::<f32>() / 32.0;
            (x_lo..=x_hi).map(|x| col(x) - bg).fold(0.0, f32::max)
        };
        let ghost_ratio = |img: &image::DynamicImage| -> f32 {
            let principal = profile_pos_peak(img, bar_center - 4, bar_center + 4);
            let left = profile_pos_peak(img, bar_center - 22, bar_center - 10);
            let right = profile_pos_peak(img, bar_center + 10, bar_center + 22);
            left.max(right) / principal.max(1e-6)
        };

        let soft = deconv
            .deconvolve_linear_image(&device, &queue, &input, &soft_params, CLIP_GUARD_SAT)
            .expect("soft deconvolve failed");
        let hard = deconv
            .deconvolve_linear_image(&device, &queue, &input, &hard_params, CLIP_GUARD_SAT)
            .expect("hard deconvolve failed");

        let r_soft = ghost_ratio(&soft);
        let r_hard = ghost_ratio(&hard);
        eprintln!(
            "ghost ratio: soft {r_soft:.3}, hard {r_hard:.3}, reduction {:.1}x",
            r_soft / r_hard
        );
        assert!(
            r_soft > 3.0 * r_hard,
            "hard-line OTF reduced the ghost pair only {:.1}x (soft {r_soft:.3}, hard {r_hard:.3})",
            r_soft / r_hard
        );
    }

    /// The defocus analog of the ghost-pair test, run at production posture
    /// (adaptive λ, default λ = 0.01, sensor-style noise). The raw jinc with
    /// the dead-band λ gate in wiener_adaptive keeps true zeros and caps the
    /// dead-band noise gain; this bounds flat-field ripple and the feature
    /// neighborhood absolutely. Historically an A/B against the floored jinc
    /// (hardness 0), but the shader now pins the defocus component to the
    /// raw jinc unconditionally, so the floored baseline is unreachable —
    /// the ceilings below are calibrated from the measured production values
    /// (flat_var 2.8e-4, near_mse 2.0e-3) with margin, and the regressions
    /// they guard sit far above: the floored jinc measured ~10x the ripple
    /// (~2.8e-3), and h = 1 *without* the dead-band gate measured ~25x
    /// (~7e-3; the adaptive estimator reads noise-only dead-band bins as
    /// high-SNR, drops λ_eff to λ/10, and the unfloored Wiener's
    /// 1/(2·sqrt(λ_eff)) gain peak rings harder than the floor ever did).
    #[test]
    fn test_gpu_defocus_dead_band_ripple_bounded() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("skipping GPU defocus ring test: no adapter ({e})");
                return;
            }
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("RAPID defocus ring test device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("failed to create device");
        let mut deconv = RapidDeconvolver::new(&adapter, &device).expect("failed to create deconvolver");

        // Bright square on mid-gray, disc-blurred: the ground truth is flat
        // away from the square, so any periodic energy out there is OTF
        // artifact, not content.
        let (width, height) = (256usize, 200usize);
        let radius = 8.0f32;
        let field: Vec<f32> = (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as i32, (i / width) as i32);
                if (x - 128).abs() <= 3 && (y - 100).abs() <= 3 { 0.9 } else { 0.4 }
            })
            .collect();
        let blurred = disc_blur_field(&field, width, height, radius);
        let hash2 = |x: u32, y: u32| -> u32 {
            let mut h = x.wrapping_mul(0x27D4_EB2F) ^ y.wrapping_mul(0x1656_67B1);
            h ^= h >> 16;
            h = h.wrapping_mul(0x7FEB_352D);
            h ^= h >> 15;
            h
        };

        let flat_var = |img: &image::DynamicImage| -> f64 {
            let rgb = img.to_rgb32f();
            let (x0, x1, y0, y1) = (40u32, 88u32, 68u32, 132u32);
            let n = ((x1 - x0) * (y1 - y0)) as f64;
            let (mut sum, mut sum2) = (0.0f64, 0.0f64);
            for y in y0..y1 {
                for x in x0..x1 {
                    let v = rgb.get_pixel(x, y)[0] as f64;
                    sum += v;
                    sum2 += v * v;
                }
            }
            let mean = sum / n;
            (sum2 / n - mean * mean).max(0.0)
        };
        let near_mse = |img: &image::DynamicImage| -> f64 {
            let rgb = img.to_rgb32f();
            let (x0, x1, y0, y1) = (96u32, 160u32, 84u32, 116u32);
            let mut sum = 0.0f64;
            for y in y0..y1 {
                for x in x0..x1 {
                    let truth = if (x as i32 - 128).abs() <= 3 && (y as i32 - 100).abs() <= 3 {
                        0.9f32
                    } else {
                        0.4f32
                    };
                    let d = (rgb.get_pixel(x, y)[0] - truth) as f64;
                    sum += d * d;
                }
            }
            sum / ((x1 - x0) * (y1 - y0)) as f64
        };

        // Sensor-style noise added after the blur: real captures always
        // carry it, and it is what dead-band gain turns into ripple. Without
        // it the metric floor is u8 quantization, ~0.3 levels of std, and
        // the comparison measures nothing visible.
        let noisy: Vec<f32> = blurred
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let (x, y) = ((i % width) as u32, (i / width) as u32);
                v + ((hash2(x, y) & 0xff) as f32 / 255.0 - 0.5) * 0.03
            })
            .collect();
        let input = gray_image(&noisy, width as u32, height as u32);

        let params = RapidParams {
            enabled: true,
            modes: ModeSet::DEFOCUS,
            defocus_radius: radius,
            lambda: 0.01,
            strength: 1.0,
            adaptive: true,
            ..Default::default()
        };
        let out = deconv
            .deconvolve_linear_image(&device, &queue, &input, &params, CLIP_GUARD_SAT)
            .expect("defocus deconvolve failed");

        let (v, m) = (flat_var(&out), near_mse(&out));
        eprintln!("defocus dead-band posture: flat_var {v:.3e}, near_mse {m:.3e}");
        assert!(
            v < 8e-4,
            "flat-field ripple {v:.3e} exceeds the dead-band ceiling (production ~2.8e-4; \
             floored jinc ~2.8e-3, ungated λ ~7e-3)"
        );
        assert!(
            m < 2.7e-3,
            "feature neighborhood MSE {m:.3e} exceeds the ceiling (production ~2.0e-3)"
        );
    }

    /// Gaussian-floor comparison fixture. Floored-shader baselines captured
    /// 23AUG26 with the adapter printed by the test:
    ///
    /// - fixed lambda=0.01: flat_var 1.571030e-3, near_mse 2.423333e-3
    /// - adaptive lambda=0.01: flat_var 2.697270e-3, near_mse 3.483825e-3
    /// - adaptive lambda=0.05: flat_var 1.816453e-3, near_mse 2.721453e-3
    ///
    /// Unfloored measurements on the same NVIDIA GB10/Vulkan posture:
    ///
    /// - fixed lambda=0.01: flat_var 1.415709e-4, near_mse 1.082751e-3
    /// - adaptive lambda=0.01: flat_var 1.485348e-4, near_mse 1.059713e-3
    /// - adaptive lambda=0.05: flat_var 5.162276e-5, near_mse 1.096019e-3
    ///
    /// The 1.5e-3 absolute MSE ceiling retains about 36% margin above the
    /// worst accepted candidate measurement. Comparative gates independently
    /// require each posture to beat its own floored baseline.
    #[test]
    fn test_gpu_gaussian_unflooring_noise_bounded() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("skipping GPU gaussian unflooring test: no adapter ({e})");
                return;
            }
        };
        eprintln!("gaussian unflooring adapter: {:?}", adapter.get_info());
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("RAPID gaussian unflooring test device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("failed to create device");
        let mut deconv = RapidDeconvolver::new(&adapter, &device).expect("failed to create deconvolver");

        let (width, height) = (256usize, 200usize);
        let sigma = 2.0f32;
        let field: Vec<f32> = (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as i32, (i / width) as i32);
                if (x - 128).abs() <= 3 && (y - 100).abs() <= 3 { 0.9 } else { 0.4 }
            })
            .collect();
        let blurred = gaussian_blur_field(&field, width, height, sigma);
        let hash2 = |x: u32, y: u32| -> u32 {
            let mut h = x.wrapping_mul(0x27D4_EB2F) ^ y.wrapping_mul(0x1656_67B1);
            h ^= h >> 16;
            h = h.wrapping_mul(0x7FEB_352D);
            h ^= h >> 15;
            h
        };

        let flat_var = |img: &image::DynamicImage| -> f64 {
            let rgb = img.to_rgb32f();
            let (x0, x1, y0, y1) = (40u32, 88u32, 68u32, 132u32);
            let n = ((x1 - x0) * (y1 - y0)) as f64;
            let (mut sum, mut sum2) = (0.0f64, 0.0f64);
            for y in y0..y1 {
                for x in x0..x1 {
                    let v = rgb.get_pixel(x, y)[0] as f64;
                    sum += v;
                    sum2 += v * v;
                }
            }
            let mean = sum / n;
            (sum2 / n - mean * mean).max(0.0)
        };
        let near_mse = |img: &image::DynamicImage| -> f64 {
            let rgb = img.to_rgb32f();
            let (x0, x1, y0, y1) = (96u32, 160u32, 84u32, 116u32);
            let mut sum = 0.0f64;
            for y in y0..y1 {
                for x in x0..x1 {
                    let truth = if (x as i32 - 128).abs() <= 3 && (y as i32 - 100).abs() <= 3 {
                        0.9f32
                    } else {
                        0.4f32
                    };
                    let d = (rgb.get_pixel(x, y)[0] - truth) as f64;
                    sum += d * d;
                }
            }
            sum / ((x1 - x0) * (y1 - y0)) as f64
        };

        let noisy: Vec<f32> = blurred
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let (x, y) = ((i % width) as u32, (i / width) as u32);
                v + ((hash2(x, y) & 0xff) as f32 / 255.0 - 0.5) * 0.03
            })
            .collect();
        let input = gray_image(&noisy, width as u32, height as u32);

        let postures = [
            ("fixed lambda=0.01", 0.01f32, false, 1.571030e-3f64, 2.423333e-3f64),
            ("adaptive lambda=0.01", 0.01f32, true, 2.697270e-3f64, 3.483825e-3f64),
            ("adaptive lambda=0.05", 0.05f32, true, 1.816453e-3f64, 2.721453e-3f64),
        ];
        let mut observed_variance = [0.0f64; 3];
        for (index, &(label, lambda, adaptive, floored_var, floored_mse)) in
            postures.iter().enumerate()
        {
            let params = RapidParams {
                enabled: true,
                modes: ModeSet::GAUSSIAN,
                gaussian_sigma: sigma,
                lambda,
                strength: 1.0,
                adaptive,
                ..Default::default()
            };
            let out = deconv
                .deconvolve_linear_image(&device, &queue, &input, &params, CLIP_GUARD_SAT)
                .expect("gaussian deconvolve failed");
            let (v, m) = (flat_var(&out), near_mse(&out));
            observed_variance[index] = v;
            eprintln!("gaussian unfloored {label}: flat_var {v:.6e}, near_mse {m:.6e}");
            assert!(
                v <= 0.8 * floored_var,
                "{label}: flat variance {v:.6e} did not improve at least 20% from floored \
                 baseline {floored_var:.6e}"
            );
            assert!(
                m <= 1.25 * floored_mse,
                "{label}: near MSE {m:.6e} exceeds 1.25x floored baseline \
                 {floored_mse:.6e}"
            );
            assert!(
                m <= 1.5e-3,
                "{label}: near MSE {m:.6e} exceeds the absolute 1.5e-3 ceiling"
            );
        }
        assert!(
            observed_variance[2] < observed_variance[1],
            "adaptive lambda=0.05 variance {:.6e} must stay below lambda=0.01 {:.6e}",
            observed_variance[2],
            observed_variance[1]
        );
    }

    /// Chamfer distances against brute-force Euclidean on a small grid: the
    /// 3x3 1/sqrt(2) transform never undershoots and overestimates by at
    /// most ~8% before the cap.
    #[test]
    fn test_chamfer_distance_transform() {
        let (w, h) = (17usize, 11usize);
        let mut mask = vec![false; w * h];
        let seeds = [(3usize, 2usize), (13usize, 8usize)];
        for &(x, y) in &seeds {
            mask[y * w + x] = true;
        }
        let cap = 8.0f32;
        let dist = chamfer_distance(&mask, w, h, cap);
        for y in 0..h {
            for x in 0..w {
                let exact = seeds
                    .iter()
                    .map(|&(sx, sy)| {
                        let (dx, dy) = (x as f32 - sx as f32, y as f32 - sy as f32);
                        (dx * dx + dy * dy).sqrt()
                    })
                    .fold(f32::INFINITY, f32::min)
                    .min(cap);
                let got = dist[y * w + x];
                assert!(
                    got + 1e-3 >= exact && got <= exact * 1.09 + 1e-3,
                    "chamfer {got:.3} vs euclidean {exact:.3} at ({x},{y})"
                );
            }
        }
    }

    /// Defocus keeps the guard active across its measured long inverse-jinc
    /// tail. In a compound kernel only the defocus support receives that
    /// multiplier; unrelated motion/Gaussian support retains three extents.
    #[test]
    fn test_defocus_clip_guard_uses_long_feather() {
        let motion = RapidParams {
            modes: ModeSet::MOTION,
            motion_length: 10.0,
            ..Default::default()
        };
        let defocus_params = RapidParams {
            modes: ModeSet::DEFOCUS,
            defocus_radius: 5.0,
            ..Default::default()
        };
        let compound = RapidParams {
            modes: ModeSet {
                motion: true,
                defocus: true,
                gaussian: false,
            },
            motion_length: 200.0,
            defocus_radius: 10.0,
            ..Default::default()
        };
        let motion_d1 = clip_guard_d1_pixels(&motion, kernel_extent(&motion));
        let defocus_d1 =
            clip_guard_d1_pixels(&defocus_params, kernel_extent(&defocus_params));
        let compound_d1 = clip_guard_d1_pixels(&compound, kernel_extent(&compound));
        assert_eq!(motion_d1, 30.0);
        assert_eq!(defocus_d1, 160.0);
        assert_eq!(kernel_extent(&compound), 220);
        assert_eq!(compound_d1, 660.0);

        let rgba = image::Rgba32FImage::from_fn(170, 1, |x, _| {
            let value = if x == 0 { 1.0 } else { 0.2 };
            image::Rgba([value, value, value, 1.0])
        });
        let compact = clip_guard_weights(&rgba, 10, CLIP_GUARD_SAT, motion_d1).unwrap();
        let defocus =
            clip_guard_weights(&rgba, 10, CLIP_GUARD_SAT, defocus_d1).unwrap();
        assert_eq!(compact[100], 0.0);
        assert!(defocus[100] > 0.3);
        assert_eq!(defocus[160], 0.0);
    }

    /// Clipped-highlight guard: an overbright disc saturates after disc
    /// blur (recording min(blur, 1.0)), so even the h = 1 filter rings
    /// around it — a model violation, not an OTF defect. With the guard the
    /// disc's neighborhood must come back near-input while a bar pattern
    /// ("text") beyond D1 still sharpens.
    #[test]
    fn test_gpu_clip_guard_suppresses_highlight_rings() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("skipping GPU clip guard test: no adapter ({e})");
                return;
            }
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("RAPID clip guard test device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("failed to create device");
        let mut deconv = RapidDeconvolver::new(&adapter, &device).expect("failed to create deconvolver");

        // Overbright disc (4.0, sensor-clipped to 1.0 after blur) at (70,100)
        // on mid-gray; 5 px bars at x in [370, 430). The clipped set reaches
        // x ~ 81, so with kernel extent 16 the defocus guard's D1 = 256 ends
        // near x = 337 and the bars keep full recovery weight.
        let (width, height) = (512usize, 200usize);
        let radius = 8.0f32;
        let field: Vec<f32> = (0..width * height)
            .map(|i| {
                let (x, y) = ((i % width) as i32, (i / width) as i32);
                let dx = x - 70;
                let dy = y - 100;
                if ((dx * dx + dy * dy) as f32).sqrt() <= 6.0 {
                    4.0
                } else if (370..430).contains(&x) {
                    if (x - 370) % 10 < 5 { 0.75 } else { 0.15 }
                } else {
                    0.4
                }
            })
            .collect();
        let blurred = disc_blur_field(&field, width, height, radius);
        let input = gray_image(&blurred, width as u32, height as u32);

        let unguarded = RapidParams {
            enabled: true,
            modes: ModeSet::DEFOCUS,
            defocus_radius: radius,
            lambda: 0.002,
            strength: 1.0,
            hardness: 1.0,
            ..Default::default()
        };
        let guarded = RapidParams { clip_guard: true, ..unguarded };

        // (a) Ring energy next to the disc, measured against the input over
        // a flat window right of the clipped set (inside the guard's feather).
        let near_disc_mse = |img: &image::DynamicImage| -> f64 {
            let (rgb, inp) = (img.to_rgb32f(), input.to_rgb32f());
            let (x0, x1, y0, y1) = (85u32, 125u32, 70u32, 130u32);
            let mut sum = 0.0f64;
            for y in y0..y1 {
                for x in x0..x1 {
                    let d = (rgb.get_pixel(x, y)[0] - inp.get_pixel(x, y)[0]) as f64;
                    sum += d * d;
                }
            }
            sum / ((x1 - x0) * (y1 - y0)) as f64
        };
        // (b) Bar contrast: max-min of the row-averaged profile.
        let bar_amplitude = |img: &image::DynamicImage| -> f32 {
            let rgb = img.to_rgb32f();
            let col = |x: u32| -> f32 {
                (80..120).map(|y| rgb.get_pixel(x, y)[0]).sum::<f32>() / 40.0
            };
            let profile: Vec<f32> = (372..428).map(col).collect();
            profile.iter().fold(f32::MIN, |a, &b| a.max(b))
                - profile.iter().fold(f32::MAX, |a, &b| a.min(b))
        };

        let off = deconv
            .deconvolve_linear_image(&device, &queue, &input, &unguarded, CLIP_GUARD_SAT)
            .expect("unguarded deconvolve failed");
        let on = deconv
            .deconvolve_linear_image(&device, &queue, &input, &guarded, CLIP_GUARD_SAT)
            .expect("guarded deconvolve failed");

        let (mse_off, mse_on) = (near_disc_mse(&off), near_disc_mse(&on));
        let (amp_in, amp_on) = (bar_amplitude(&input), bar_amplitude(&on));
        eprintln!(
            "clip guard: near-disc MSE off {mse_off:.3e} -> on {mse_on:.3e} ({:.1}x), \
             bar amplitude in {amp_in:.3} -> on {amp_on:.3} ({:.1}x)",
            mse_off / mse_on,
            amp_on / amp_in
        );
        assert!(
            mse_off > 3.0 * mse_on,
            "guard reduced near-disc ring energy only {:.1}x (off {mse_off:.3e}, on {mse_on:.3e})",
            mse_off / mse_on
        );
        assert!(
            amp_on > 1.5 * amp_in,
            "bar pattern did not sharpen under the guard (in {amp_in:.3}, on {amp_on:.3})"
        );
    }

    /// Without clipped pixels the guard must be a bit-exact no-op: the empty
    /// mask skips the distance transform and the recombine loop untouched.
    #[test]
    fn test_gpu_clip_guard_noop_without_clipping() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("skipping GPU clip guard no-op test: no adapter ({e})");
                return;
            }
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("RAPID clip guard no-op test device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("failed to create device");
        let mut deconv = RapidDeconvolver::new(&adapter, &device).expect("failed to create deconvolver");

        // The clip-guard scene minus the disc: nothing reaches CLIP_GUARD_SAT.
        let (width, height) = (256usize, 200usize);
        let radius = 8.0f32;
        let field: Vec<f32> = (0..width * height)
            .map(|i| {
                let x = (i % width) as i32;
                if (150..210).contains(&x) {
                    if (x - 150) % 10 < 5 { 0.75 } else { 0.15 }
                } else {
                    0.4
                }
            })
            .collect();
        let blurred = disc_blur_field(&field, width, height, radius);
        let input = gray_image(&blurred, width as u32, height as u32);

        let unguarded = RapidParams {
            enabled: true,
            modes: ModeSet::DEFOCUS,
            defocus_radius: radius,
            lambda: 0.002,
            strength: 1.0,
            hardness: 1.0,
            ..Default::default()
        };
        let guarded = RapidParams { clip_guard: true, ..unguarded };

        let off = deconv
            .deconvolve_linear_image(&device, &queue, &input, &unguarded, CLIP_GUARD_SAT)
            .expect("unguarded deconvolve failed");
        let on = deconv
            .deconvolve_linear_image(&device, &queue, &input, &guarded, CLIP_GUARD_SAT)
            .expect("guarded deconvolve failed");
        assert!(
            off.as_bytes() == on.as_bytes(),
            "clip guard changed output bytes on a clip-free image"
        );
    }

    /// Luma-only deconvolution with gain-map recombine must preserve chroma
    /// even when the source carries per-channel misregistration (chromatic
    /// aberration) — the per-channel pipeline amplified that into red/blue
    /// fringes. Luma edge energy must still increase (detail recovered).
    #[test]
    fn test_gpu_luma_deconvolve_preserves_chroma() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("skipping GPU chroma test: no adapter ({e})");
                return;
            }
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("RAPID chroma test device"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .expect("failed to create device");
        let mut deconv = RapidDeconvolver::new(&adapter, &device).expect("failed to create deconvolver");

        // Vertical bars with ~2 px per-channel misregistration (R shifted
        // left, B right — simulated CA), then gaussian-blurred so the
        // deconvolution has real detail to recover.
        let (width, height) = (192u32, 160u32);
        let bar = |x: i64| -> f32 {
            if (x.rem_euclid(48)) < 24 { 0.25 } else { 0.75 }
        };
        let sharp = image::RgbaImage::from_fn(width, height, |x, _| {
            let r = bar(x as i64 - 2);
            let g = bar(x as i64);
            let b = bar(x as i64 + 2);
            image::Rgba([
                (r * 255.0).round() as u8,
                (g * 255.0).round() as u8,
                (b * 255.0).round() as u8,
                255,
            ])
        });
        let input = image::DynamicImage::ImageRgba8(sharp).blur(1.5);

        let params = RapidParams {
            enabled: true,
            modes: ModeSet::GAUSSIAN,
            gaussian_sigma: 1.5,
            lambda: 0.01,
            strength: 1.0,
            ..Default::default()
        };
        let out = deconv
            .deconvolve_linear_image(&device, &queue, &input, &params, CLIP_GUARD_SAT)
            .expect("deconvolve_image failed");

        // Chroma: rg-chromaticity is scale-invariant, so the gain map must
        // leave it untouched outside clipped pixels (where chroma loss is
        // expected and legitimate).
        let (i_rgb, o_rgb) = (input.to_rgb32f(), out.to_rgb32f());
        let mut max_chroma_delta = 0.0f32;
        let mut judged = 0u32;
        for y in 4..height - 4 {
            for x in 4..width - 4 {
                let ip = i_rgb.get_pixel(x, y);
                let op = o_rgb.get_pixel(x, y);
                let is = ip[0] + ip[1] + ip[2];
                let os = op[0] + op[1] + op[2];
                if os < 0.2 || op.0.iter().any(|&c| c > 0.98) {
                    continue;
                }
                let dr = (ip[0] / is - op[0] / os).abs();
                let db = (ip[2] / is - op[2] / os).abs();
                max_chroma_delta = max_chroma_delta.max(dr.max(db));
                judged += 1;
            }
        }
        assert!(judged > 1000, "too few unclipped pixels to judge chroma ({judged})");
        assert!(
            max_chroma_delta < 0.02,
            "chroma drifted by {max_chroma_delta} after luma-only deconvolution"
        );

        // Luma edge energy must increase against the blurred input.
        let edge_energy = |img: &image::Rgb32FImage| -> f64 {
            let luma = |p: &image::Rgb<f32>| -> f64 {
                (p[0] * LUMA_COEFF[0] + p[1] * LUMA_COEFF[1] + p[2] * LUMA_COEFF[2]) as f64
            };
            let mut e = 0.0f64;
            for y in 0..height {
                for x in 1..width {
                    let d = luma(img.get_pixel(x, y)) - luma(img.get_pixel(x - 1, y));
                    e += d * d;
                }
            }
            e
        };
        let (e_in, e_out) = (edge_energy(&i_rgb), edge_energy(&o_rgb));
        eprintln!(
            "luma deconvolve: chroma delta {:.4}, edge energy {:.2}x ({} pixels judged)",
            max_chroma_delta,
            e_out / e_in,
            judged
        );
        assert!(
            e_out > 1.15 * e_in,
            "no luma detail recovered: in={e_in:.3}, out={e_out:.3}"
        );
    }

    #[test]
    fn test_rapid_params_scaled() {
        let p = RapidParams {
            motion_length: 100.0,
            motion_angle: 35.0,
            defocus_radius: 40.0,
            gaussian_sigma: 4.0,
            lambda: 0.02,
            strength: 0.8,
            ..Default::default()
        };

        let s = p.scaled(0.25);
        assert!((s.motion_length - 25.0).abs() < 1e-6);
        assert!((s.defocus_radius - 10.0).abs() < 1e-6);
        assert!((s.gaussian_sigma - 1.0).abs() < 1e-6);
        // Non-spatial parameters must be untouched by scaling.
        assert_eq!(s.lambda, p.lambda);
        assert_eq!(s.strength, p.strength);
        assert_eq!(s.motion_angle, p.motion_angle);

        // Motion and Gaussian retain their kernel floors at degenerate scales.
        let tiny = p.scaled(0.001);
        assert!(tiny.motion_length >= 1.0);
        assert!(tiny.gaussian_sigma >= 0.3);

        // Defocus preserves an honest sub-pixel working radius.
        let subpixel = RapidParams { defocus_radius: 2.0, ..p }.scaled(0.17);
        assert!((subpixel.defocus_radius - 0.34).abs() < 1e-6);

        let unit = p.scaled(1.0);
        assert!((unit.motion_length - p.motion_length).abs() < 1e-6);
        assert!((unit.defocus_radius - p.defocus_radius).abs() < 1e-6);
    }

    #[test]
    fn test_psf_axis_projection() {
        // Horizontal motion: box along x, identity along y.
        let p = RapidParams {
            modes: ModeSet::MOTION,
            motion_length: 32.0,
            motion_angle: 0.0,
            ..Default::default()
        };
        let kx = psf_axis_projection(&p, 0);
        assert!(kx.len() >= 31, "x projection should span the blur length, got {}", kx.len());
        assert!((kx.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        let ky = psf_axis_projection(&p, 1);
        assert_eq!(ky.len(), 1, "vertical projection of a horizontal line must be identity");

        // Defocus and gaussian project onto both axes.
        for p in [
            RapidParams { modes: ModeSet::DEFOCUS, defocus_radius: 5.0, ..Default::default() },
            RapidParams { modes: ModeSet::GAUSSIAN, gaussian_sigma: 2.0, ..Default::default() },
        ] {
            for axis in 0..2 {
                let k = psf_axis_projection(&p, axis);
                assert!(k.len() > 1);
                assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn test_mode_set_bits() {
        assert_eq!(ModeSet::MOTION.bits(), 1);
        assert_eq!(ModeSet::DEFOCUS.bits(), 2);
        assert_eq!(ModeSet::GAUSSIAN.bits(), 4);
        let all = ModeSet { motion: true, defocus: true, gaussian: true };
        assert_eq!(all.bits(), 7);
        assert!(all.any());
        let none = ModeSet { motion: false, defocus: false, gaussian: false };
        assert_eq!(none.bits(), 0);
        assert!(!none.any());
    }

    #[test]
    fn test_kernel_extent_compound() {
        // Single modes keep their pre-compound extents.
        let motion = RapidParams { modes: ModeSet::MOTION, motion_length: 10.0, ..Default::default() };
        assert_eq!(kernel_extent(&motion), 10);
        let defocus = RapidParams { modes: ModeSet::DEFOCUS, defocus_radius: 5.0, ..Default::default() };
        assert_eq!(kernel_extent(&defocus), 10);
        let gaussian = RapidParams { modes: ModeSet::GAUSSIAN, gaussian_sigma: 2.0, ..Default::default() };
        assert_eq!(kernel_extent(&gaussian), 12);
        // Compound supports add (convolution support is the sum).
        let compound = RapidParams {
            modes: ModeSet { motion: true, defocus: false, gaussian: true },
            motion_length: 10.0,
            gaussian_sigma: 2.0,
            ..Default::default()
        };
        assert_eq!(kernel_extent(&compound), 22);
        // An empty set floors at 1 like a sub-pixel kernel.
        let empty = RapidParams {
            modes: ModeSet { motion: false, defocus: false, gaussian: false },
            ..Default::default()
        };
        assert_eq!(kernel_extent(&empty), 1);
    }

    /// Bit-for-bit parity of single-mode projections with the pre-compound
    /// implementation: goldens captured from the last commit before the
    /// ModeSet refactor. A single-member set must take the integration path
    /// untouched — any drift here means the refactor was not pure code
    /// motion.
    #[test]
    fn test_psf_axis_projection_single_mode_parity() {
        const MOTION20_H04_AX0: [f32; 19] = [
            6.156384014e-3, 2.419506386e-2, 2.641135640e-2, 3.114545345e-2, 3.982412070e-2,
            5.334763229e-2, 7.095774263e-2, 8.947129548e-2, 1.038440987e-1, 1.092937514e-1,
            1.038440987e-1, 8.947130293e-2, 7.095774263e-2, 5.334763229e-2, 3.982412070e-2,
            3.114545345e-2, 2.641135640e-2, 2.419506386e-2, 6.156384014e-3,
        ];
        const MOTION20_H04_AX1: [f32; 11] = [
            2.123314701e-2, 4.692157730e-2, 6.645793468e-2, 1.089163870e-1, 1.623771340e-1,
            1.881877035e-1, 1.623771340e-1, 1.089163944e-1, 6.645793468e-2, 4.692157730e-2,
            2.123314701e-2,
        ];
        const DEFOCUS5_AX0: [f32; 11] = [
            1.914434321e-2, 7.534631342e-2, 1.013437882e-1, 1.162930802e-1, 1.243876815e-1,
            1.269696504e-1, 1.243876815e-1, 1.162930802e-1, 1.013437882e-1, 7.534631342e-2,
            1.914434321e-2,
        ];
        const GAUSSIAN2_AX0: [f32; 13] = [
            2.393511590e-3, 9.225073270e-3, 2.781469934e-2, 6.561533362e-2, 1.211178526e-1,
            1.749509126e-1, 1.977652311e-1, 1.749508977e-1, 1.211178526e-1, 6.561533362e-2,
            2.781470306e-2, 9.225073270e-3, 2.393511357e-3,
        ];
        let motion = RapidParams {
            modes: ModeSet::MOTION,
            motion_length: 20.0,
            motion_angle: 30.0,
            hardness: 0.4,
            ..Default::default()
        };
        let defocus = RapidParams { modes: ModeSet::DEFOCUS, defocus_radius: 5.0, ..Default::default() };
        let gaussian = RapidParams { modes: ModeSet::GAUSSIAN, gaussian_sigma: 2.0, ..Default::default() };
        let cases: [(&str, &RapidParams, usize, &[f32]); 4] = [
            ("motion ax0", &motion, 0, &MOTION20_H04_AX0),
            ("motion ax1", &motion, 1, &MOTION20_H04_AX1),
            ("defocus ax0", &defocus, 0, &DEFOCUS5_AX0),
            ("gaussian ax0", &gaussian, 0, &GAUSSIAN2_AX0),
        ];
        for (name, p, axis, golden) in cases {
            let k = psf_axis_projection(p, axis);
            assert_eq!(k.len(), golden.len(), "{name}: support width changed");
            for (i, (got, want)) in k.iter().zip(golden).enumerate() {
                assert!(
                    (got - want).abs() < 1e-8,
                    "{name}[{i}]: got {got:.9e}, golden {want:.9e}"
                );
            }
        }
    }

    #[test]
    fn test_psf_axis_projection_compound() {
        // Motion at 30° + defocus: support = sum of member supports
        // (19 + 11 - 1), unit mass, symmetric (both members are symmetric).
        let p = RapidParams {
            modes: ModeSet { motion: true, defocus: true, gaussian: false },
            motion_length: 20.0,
            motion_angle: 30.0,
            hardness: 0.4,
            defocus_radius: 5.0,
            ..Default::default()
        };
        let k = psf_axis_projection(&p, 0);
        assert_eq!(k.len(), 29, "compound support must be the sum of member supports");
        assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        for i in 0..k.len() / 2 {
            assert!(
                (k[i] - k[k.len() - 1 - i]).abs() < 1e-6,
                "compound of symmetric members must be symmetric at tap {i}"
            );
        }

        // A member whose projection is the identity leaves the other member
        // unchanged: horizontal motion projects onto y as [1.0], so
        // motion+defocus on axis 1 equals defocus alone.
        let horiz = RapidParams {
            modes: ModeSet { motion: true, defocus: true, gaussian: false },
            motion_length: 20.0,
            motion_angle: 0.0,
            defocus_radius: 5.0,
            ..Default::default()
        };
        let compound_y = psf_axis_projection(&horiz, 1);
        let defocus_only = psf_axis_projection(
            &RapidParams { modes: ModeSet::DEFOCUS, defocus_radius: 5.0, ..Default::default() },
            1,
        );
        assert_eq!(compound_y.len(), defocus_only.len());
        for (a, b) in compound_y.iter().zip(&defocus_only) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mirror_fade_map() {
        // 200 -> 256 with a 24 px margin: identity inside the frame, mirrored
        // fading content at both ends of the wrap, zeros between.
        let map = mirror_fade_map(200, 256, 24);
        assert_eq!(map[0], (0, 1.0));
        assert_eq!(map[199], (199, 1.0));
        // Just past the frame: reflect-101 of row 198, weight near 1.
        assert_eq!(map[200].0, 198);
        assert!(map[200].1 > 0.9);
        // The near margin fades toward zero.
        assert!(map[223].1 < 0.1);
        // Just before wrapping to row 0: reflect-101 of row 1, weight near 1.
        assert_eq!(map[255].0, 1);
        assert!(map[255].1 > 0.9);
        assert!(map[232].1 < 0.1);
        // Zero fill between the margins.
        for i in 224..232 {
            assert_eq!(map[i].1, 0.0);
        }
        // Zero margin = plain zero pad.
        let plain = mirror_fade_map(200, 256, 0);
        assert!(plain[200..].iter().all(|&(_, w)| w == 0.0));
    }

    /// Deterministic broadband scene (hash noise over smooth blobs) for the
    /// cepstral estimator: enough spectral content at all frequencies that
    /// the blur's spectral comb is observable. The hash needs full avalanche
    /// — a linear-congruential mix leaves lattice periodicities that the
    /// cepstrum (correctly) detects as structure.
    fn synthetic_scene(w: u32, h: u32) -> image::RgbaImage {
        let hash2 = |x: u32, y: u32| -> u32 {
            let mut h = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77);
            h ^= h >> 16;
            h = h.wrapping_mul(0x7FEB_352D);
            h ^= h >> 15;
            h = h.wrapping_mul(0x846C_A68B);
            h ^= h >> 16;
            h
        };
        image::RgbaImage::from_fn(w, h, |x, y| {
            let noise = (hash2(x, y) & 0xff) as f32 / 255.0;
            let blob = ((x as f32 / 37.0).sin() + (y as f32 / 29.0).cos()) * 0.25 + 0.5;
            let v = (0.35 * noise + 0.65 * blob).clamp(0.0, 1.0);
            let b = (v * 255.0) as u8;
            image::Rgba([b, b, b, 255])
        })
    }

    /// Convolve with a true line PSF (uniform box along the given direction,
    /// image-space Y-down) — the real-world blur the estimator must detect,
    /// as opposed to the engine's Gaussian-envelope deconvolution model.
    fn motion_blur_line(img: &image::RgbaImage, length: f32, angle_deg: f32) -> image::RgbaImage {
        let (w, h) = (img.width(), img.height());
        let dir = angle_deg.to_radians();
        let (dx, dy) = (dir.cos(), dir.sin());
        let n = length.ceil().max(1.0) as i32;
        image::RgbaImage::from_fn(w, h, |x, y| {
            let mut acc = [0.0f32; 3];
            let mut count = 0.0f32;
            for i in 0..n {
                let t = i as f32 - (n as f32 - 1.0) / 2.0;
                let sx = (x as f32 + t * dx).round() as i32;
                let sy = (y as f32 + t * dy).round() as i32;
                if sx >= 0 && sx < w as i32 && sy >= 0 && sy < h as i32 {
                    let p = img.get_pixel(sx as u32, sy as u32);
                    for c in 0..3 {
                        acc[c] += p[c] as f32;
                    }
                    count += 1.0;
                }
            }
            image::Rgba([
                (acc[0] / count).round() as u8,
                (acc[1] / count).round() as u8,
                (acc[2] / count).round() as u8,
                255,
            ])
        })
    }

    /// Convolve a grayscale f32 field with a true pillbox PSF (uniform disk)
    /// — the real-world defocus blur whose jinc spectrum the deconvolution
    /// divides by. Border taps average over the in-frame subset, like
    /// motion_blur_line. Row prefix sums make each disc row an O(1) span
    /// (the naive tap loop makes the 2048² scale-mapping test cost ~3.3 G
    /// visits); disc membership is the same √(dx²+dy²) ≤ radius rule.
    fn disc_blur_field(field: &[f32], w: usize, h: usize, radius: f32) -> Vec<f32> {
        let ri = radius.ceil() as i32;
        let half_widths: Vec<i32> = (-ri..=ri)
            .map(|dy| {
                let mut wdy = -1;
                for dx in 0..=ri {
                    if ((dx * dx + dy * dy) as f32).sqrt() <= radius {
                        wdy = dx;
                    } else {
                        break;
                    }
                }
                wdy
            })
            .collect();
        let mut prefix = vec![0.0f64; (w + 1) * h];
        for y in 0..h {
            let row = y * (w + 1);
            for x in 0..w {
                prefix[row + x + 1] = prefix[row + x] + field[y * w + x] as f64;
            }
        }
        let mut out = vec![0.0f32; w * h];
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let mut acc = 0.0f64;
                let mut count = 0i64;
                for dy in -ri..=ri {
                    let wdy = half_widths[(dy + ri) as usize];
                    let sy = y + dy;
                    if wdy < 0 || sy < 0 || sy >= h as i32 {
                        continue;
                    }
                    let x0 = (x - wdy).max(0);
                    let x1 = (x + wdy).min(w as i32 - 1);
                    if x0 > x1 {
                        continue;
                    }
                    let row = sy as usize * (w + 1);
                    acc += prefix[row + x1 as usize + 1] - prefix[row + x0 as usize];
                    count += (x1 - x0 + 1) as i64;
                }
                out[(y * w as i32 + x) as usize] = (acc / count as f64) as f32;
            }
        }
        out
    }

    /// Disc-blur an RGBA test image via its luma field. The estimators read
    /// luma only, so the gray result loses nothing.
    fn disc_blur(img: &image::RgbaImage, radius: f32) -> image::DynamicImage {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let field: Vec<f32> = img.pixels().map(|p| p[0] as f32 / 255.0).collect();
        let blurred = disc_blur_field(&field, w, h, radius);
        gray_image(&blurred, w as u32, h as u32)
    }

    /// Isotropic gaussian blur, separable over a scalar field with clamped
    /// borders. The normalized discrete kernel is truncated at +/-3 sigma;
    /// at sigma=2 its transfer differs from the analytic shader OTF by about
    /// 0.4% near the former |H|=0.15 crossing, which the comparative fixture
    /// bounds rather than concealing with a different reference blur.
    fn gaussian_blur_field(
        field: &[f32],
        width: usize,
        height: usize,
        sigma: f32,
    ) -> Vec<f32> {
        let (w, h) = (width as i32, height as i32);
        let radius = (3.0 * sigma).ceil() as i32;
        let weights: Vec<f32> = (-radius..=radius)
            .map(|i| (-((i * i) as f32) / (2.0 * sigma * sigma)).exp())
            .collect();
        let wsum: f32 = weights.iter().sum();
        let mut tmp = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0f32;
                for (j, wt) in weights.iter().enumerate() {
                    let sx = (x + j as i32 - radius).clamp(0, w - 1);
                    acc += wt * field[(y * w + sx) as usize];
                }
                tmp[(y * w + x) as usize] = acc / wsum;
            }
        }
        let mut out = vec![0.0f32; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0f32;
                for (j, wt) in weights.iter().enumerate() {
                    let sy = (y + j as i32 - radius).clamp(0, h - 1);
                    acc += wt * tmp[(sy * w + x) as usize];
                }
                out[(y * w + x) as usize] = acc / wsum;
            }
        }
        out
    }

    /// Quantizing wrapper for the real-world blur whose spectrum
    /// estimate_gaussian fits.
    fn gaussian_blur_iso(img: &image::RgbaImage, sigma: f32) -> image::DynamicImage {
        let (w, h) = (img.width() as usize, img.height() as usize);
        let field: Vec<f32> = img.pixels().map(|p| p[0] as f32 / 255.0).collect();
        let blurred = gaussian_blur_field(&field, w, h, sigma);
        gray_image(&blurred, w as u32, h as u32)
    }

    /// Quantize a grayscale f32 field to an sRGB-range u8 image, clamping to
    /// [0, 1] — the sensor's saturation step for overbright field values.
    fn gray_image(field: &[f32], w: u32, h: u32) -> image::DynamicImage {
        image::DynamicImage::ImageRgba8(image::RgbaImage::from_fn(w, h, |x, y| {
            let v = (field[(y * w + x) as usize].clamp(0.0, 1.0) * 255.0).round() as u8;
            image::Rgba([v, v, v, 255])
        }))
    }

    #[test]
    fn test_refine_peak_2d_recovers_rotated_minimum() {
        let (peak_x, peak_y) = (7i32, -11i32);
        let mut worst_legacy_error = 0.0f32;
        for ix in -4..=4 {
            for iy in -4..=4 {
                let expected_x = ix as f32 * 0.1;
                let expected_y = iy as f32 * 0.1;
                let target_x = peak_x as f32 + expected_x;
                let target_y = peak_y as f32 + expected_y;
                let sample = |x: i32, y: i32| -> f32 {
                    let rx = x as f32 - target_x;
                    let ry = y as f32 - target_y;
                    37.0 + rx * rx + 1.5 * rx * ry + 2.0 * ry * ry
                };

                let (got_x, got_y) = refine_cepstral_peak(peak_x, peak_y, &sample);
                assert!(
                    (got_x - expected_x).abs() <= 0.02
                        && (got_y - expected_y).abs() <= 0.02,
                    "2D fit recovered ({got_x:.4}, {got_y:.4}), expected \
                     ({expected_x:.4}, {expected_y:.4})"
                );

                let refine_axis = |c_m: f32, c_0: f32, c_p: f32| -> f32 {
                    let curvature = c_m - 2.0 * c_0 + c_p;
                    if curvature <= 1e-12 {
                        0.0
                    } else {
                        (0.5 * (c_m - c_p) / curvature).clamp(-0.5, 0.5)
                    }
                };
                let center = sample(peak_x, peak_y);
                let legacy_x = refine_axis(
                    sample(peak_x - 1, peak_y),
                    center,
                    sample(peak_x + 1, peak_y),
                );
                let legacy_y = refine_axis(
                    sample(peak_x, peak_y - 1),
                    center,
                    sample(peak_x, peak_y + 1),
                );
                worst_legacy_error = worst_legacy_error
                    .max((legacy_x - expected_x).abs())
                    .max((legacy_y - expected_y).abs());
            }
        }
        assert!(
            worst_legacy_error > 0.1,
            "legacy separable fit's worst error was only {worst_legacy_error:.4}"
        );
    }

    #[test]
    fn test_refine_peak_2d_degenerate_falls_back() {
        assert_eq!(refine_peak_2d([[1.0; 3]; 3]), None);

        let saddle = std::array::from_fn(|j| {
            std::array::from_fn(|i| {
                let x = i as f32 - 1.0;
                let y = j as f32 - 1.0;
                x * x - y * y
            })
        });
        assert_eq!(refine_peak_2d(saddle), None);

        let mut non_finite = [[0.0f32; 3]; 3];
        non_finite[0][2] = f32::NAN;
        assert_eq!(refine_peak_2d(non_finite), None);
        non_finite[0][2] = f32::INFINITY;
        assert_eq!(refine_peak_2d(non_finite), None);

        let ridge = std::array::from_fn(|j| {
            std::array::from_fn(|i| {
                let x = i as f32 - 1.0;
                let y = j as f32 - 1.0;
                x * x + 0.0005 * y * y
            })
        });
        assert_eq!(refine_peak_2d(ridge), None);

        let (peak_x, peak_y) = (5i32, -7i32);
        let target_x = peak_x as f32 + 0.7;
        let target_y = peak_y as f32 - 0.4;
        let sample = |x: i32, y: i32| -> f32 {
            let rx = x as f32 - target_x;
            let ry = y as f32 - target_y;
            rx * rx + 1.5 * rx * ry + 2.0 * ry * ry
        };
        let rail_samples = std::array::from_fn(|j| {
            std::array::from_fn(|i| sample(peak_x + i as i32 - 1, peak_y + j as i32 - 1))
        });
        let center = rail_samples[1][1];
        assert!(
            rail_samples.iter().flatten().all(|&value| center <= value),
            "rail fixture must keep the center as the best integer sample"
        );
        assert_eq!(refine_peak_2d(rail_samples), None);

        let refine_axis = |c_m: f32, c_0: f32, c_p: f32| -> f32 {
            let curvature = c_m - 2.0 * c_0 + c_p;
            if curvature <= 1e-12 {
                0.0
            } else {
                (0.5 * (c_m - c_p) / curvature).clamp(-0.5, 0.5)
            }
        };
        let expected = (
            refine_axis(rail_samples[1][0], center, rail_samples[1][2]),
            refine_axis(rail_samples[0][1], center, rail_samples[2][1]),
        );
        let got = refine_cepstral_peak(peak_x, peak_y, &sample);
        assert_eq!(got.0.to_bits(), expected.0.to_bits());
        assert_eq!(got.1.to_bits(), expected.1.to_bits());
    }

    /// Convention gate for the estimator: synthetic line blurs at five angles
    /// must come back with the right length and angle in psf_generate.wgsl's
    /// motion_angle convention (degrees, 0-180, image-space Y-down), and a
    /// sharp image must fail the confidence gate rather than invent a blur.
    #[test]
    fn test_estimate_blur_five_angles() {
        let scene = synthetic_scene(512, 512);
        let blur_len = 25.0f32;
        for &angle in &[0.0f32, 30.0, 45.0, 90.0, 135.0] {
            let blurred =
                image::DynamicImage::ImageRgba8(motion_blur_line(&scene, blur_len, angle));
            let est = estimate_blur(&blurred, true);
            eprintln!(
                "blur estimate at {angle}°: L={:.1} A={:.1} H={:.2} confidence={:.1}",
                est.length, est.angle, est.hardness, est.confidence
            );
            assert!(
                est.confident,
                "estimator not confident at {angle}° (confidence {:.1})",
                est.confidence
            );
            assert!(
                (est.length - blur_len).abs() <= 2.0,
                "length off at {angle}°: got {:.1}, expected {blur_len}",
                est.length
            );
            let diff = (est.angle - angle).abs();
            let angular_error = diff.min(180.0 - diff);
            assert!(
                angular_error <= 2.0,
                "angle off at {angle}°: got {:.1}",
                est.angle
            );
            assert!(
                est.hardness >= 0.7,
                "hard-line blur fitted too soft at {angle}°: hardness {:.2}",
                est.hardness
            );
        }

        // Negative control: no blur -> no confident estimate.
        let sharp = estimate_blur(&image::DynamicImage::ImageRgba8(scene), true);
        eprintln!(
            "sharp-image: L={:.1} A={:.1} confidence={:.1}",
            sharp.length, sharp.angle, sharp.confidence
        );
        assert!(
            !sharp.confident,
            "estimator hallucinated a blur on a sharp image (confidence {:.1})",
            sharp.confidence
        );
    }

    /// Soft negative control for the hardness fit: a directional Gaussian
    /// (the engine's h = 0 model, sigma = L/2π along the axis) produces no
    /// spectral notch comb. Usually that means no confident cepstral peak
    /// at all; if one does clear the gate, the fitted hardness must land at
    /// the soft end rather than prescribing the hard-line inverse.
    #[test]
    fn test_estimate_blur_directional_gaussian_soft() {
        let scene = synthetic_scene(512, 512);
        let sigma = 25.0f32 / (2.0 * PI);
        let radius = (3.0 * sigma).ceil() as i32;
        let weights: Vec<f32> =
            (-radius..=radius).map(|i| (-((i * i) as f32) / (2.0 * sigma * sigma)).exp()).collect();
        let wsum: f32 = weights.iter().sum();
        let blurred = image::RgbaImage::from_fn(512, 512, |x, y| {
            let mut acc = 0.0f32;
            for (j, w) in weights.iter().enumerate() {
                let sx = (x as i32 + j as i32 - radius).clamp(0, 511);
                acc += w * scene.get_pixel(sx as u32, y)[0] as f32;
            }
            let b = (acc / wsum).round() as u8;
            image::Rgba([b, b, b, 255])
        });
        let est = estimate_blur(&image::DynamicImage::ImageRgba8(blurred), true);
        eprintln!(
            "directional gaussian: L={:.1} A={:.1} H={:.2} confidence={:.1} ({}confident)",
            est.length,
            est.angle,
            est.hardness,
            est.confidence,
            if est.confident { "" } else { "not " }
        );
        if est.confident {
            assert!(
                est.hardness <= 0.3,
                "gaussian blur fitted too hard: hardness {:.2}",
                est.hardness
            );
        }
    }

    /// The suggested suppression must track the actual noise floor: the
    /// same blurred scene with added sensor-style noise must suggest a
    /// higher lambda, and both must stay inside the slider's range.
    #[test]
    fn test_estimate_blur_lambda_tracks_noise() {
        let scene = synthetic_scene(512, 512);
        let blurred = motion_blur_line(&scene, 25.0, 0.0);
        let hash2 = |x: u32, y: u32| -> u32 {
            let mut h = x.wrapping_mul(0x27D4_EB2F) ^ y.wrapping_mul(0x1656_67B1);
            h ^= h >> 16;
            h = h.wrapping_mul(0x7FEB_352D);
            h ^= h >> 15;
            h
        };
        let noisy = image::RgbaImage::from_fn(512, 512, |x, y| {
            let p = blurred.get_pixel(x, y);
            let n = ((hash2(x, y) & 0xff) as f32 / 255.0 - 0.5) * 12.0;
            let b = (p[0] as f32 + n).clamp(0.0, 255.0).round() as u8;
            image::Rgba([b, b, b, 255])
        });

        let est_clean = estimate_blur(&image::DynamicImage::ImageRgba8(blurred), true);
        let est_noisy = estimate_blur(&image::DynamicImage::ImageRgba8(noisy), true);
        eprintln!(
            "lambda suggestion: clean {:.4} (confidence {:.1}), noisy {:.4} (confidence {:.1})",
            est_clean.lambda, est_clean.confidence, est_noisy.lambda, est_noisy.confidence
        );
        assert!(est_clean.confident && est_noisy.confident);
        for l in [est_clean.lambda, est_noisy.lambda] {
            assert!(
                (0.01..=0.1).contains(&l),
                "suggested lambda {l} outside the floored slider range"
            );
        }
        assert!(
            est_noisy.lambda > est_clean.lambda,
            "noise did not raise the suggested lambda ({:.4} vs {:.4})",
            est_noisy.lambda,
            est_clean.lambda
        );
    }

    /// Blurs past the 200 px UI rail must still estimate correctly: the
    /// estimator searches up to 250 working px and the backend reports the
    /// unclamped length (the log line shows it) even though the UI caps
    /// applied values at 200 — recoveries above that only amplify ringing.
    #[test]
    fn test_estimate_blur_beyond_old_rail() {
        let scene = synthetic_scene(1024, 1024);
        let blur_len = 221.0f32;
        let blurred = image::DynamicImage::ImageRgba8(motion_blur_line(&scene, blur_len, 0.0));
        let est = estimate_blur(&blurred, true);
        eprintln!(
            "long blur: L={:.1} A={:.1} H={:.2} confidence={:.1}",
            est.length, est.angle, est.hardness, est.confidence
        );
        assert!(
            est.confident,
            "estimator not confident on a {blur_len} px blur (confidence {:.1})",
            est.confidence
        );
        assert!(
            (est.length - blur_len).abs() <= 8.0,
            "length off: got {:.1}, expected {blur_len}",
            est.length
        );
        assert!(
            est.hardness >= 0.7,
            "hard-line blur fitted too soft: hardness {:.2}",
            est.hardness
        );
    }

    /// The ported CPU jinc must agree with the shader model: zero at the
    /// J1 roots, sign flip between them, unity at the origin.
    #[test]
    fn test_cpu_jinc_matches_zeros() {
        assert!((jinc(0.0) - 1.0).abs() < 1e-6);
        for root in [3.8317f32, 7.0156] {
            assert!(
                jinc(root).abs() < 5e-3,
                "jinc({root}) = {} should be ~0",
                jinc(root)
            );
        }
        assert!(jinc(5.4) < 0.0, "jinc must be negative between the first two zeros");
    }

    /// Disc blurs across the UI's radius range must come back confident and
    /// within a pixel; a sharp scene must fail the gate rather than invent
    /// a defocus.
    #[test]
    fn test_estimate_defocus_radii() {
        let scene = synthetic_scene(512, 512);
        for &radius in &[4.0f32, 8.0, 14.0] {
            let blurred = disc_blur(&scene, radius);
            let est = estimate_defocus(&blurred, true);
            eprintln!(
                "defocus estimate at R={radius}: R={:.2} confidence={:.1} λ={:.4} ({}confident)",
                est.radius,
                est.confidence,
                est.lambda,
                if est.confident { "" } else { "not " }
            );
            assert!(
                est.confident,
                "estimator not confident at R={radius} (confidence {:.1})",
                est.confidence
            );
            assert!(
                (est.radius - radius).abs() <= 1.0,
                "radius off at R={radius}: got {:.2}",
                est.radius
            );
        }

        let sharp = estimate_defocus(&image::DynamicImage::ImageRgba8(scene), true);
        eprintln!(
            "sharp-image defocus: R={:.2} confidence={:.1}",
            sharp.radius, sharp.confidence
        );
        assert!(
            !sharp.confident,
            "estimator hallucinated a defocus on a sharp image (confidence {:.1})",
            sharp.confidence
        );
    }

    /// A 2048² frame runs the estimator at working scale 0.5: the reported
    /// radius must map back to full resolution.
    #[test]
    fn test_estimate_defocus_scale_mapping() {
        let scene = synthetic_scene(2048, 2048);
        let blurred = disc_blur(&scene, 16.0);
        let est = estimate_defocus(&blurred, true);
        eprintln!(
            "defocus scale mapping: R={:.2} confidence={:.1} ({}confident)",
            est.radius,
            est.confidence,
            if est.confident { "" } else { "not " }
        );
        assert!(est.confident, "not confident (confidence {:.1})", est.confidence);
        assert!(
            (est.radius - 16.0).abs() <= 2.0,
            "full-res radius off: got {:.2}, expected 16",
            est.radius
        );
    }

    /// A gaussian blur has no spectral zeros, so the matched ring comb has
    /// nothing to lock onto — the physically adjacent cross-mode negative
    /// that calibrates the defocus gate.
    #[test]
    fn test_estimate_defocus_rejects_gaussian() {
        let scene = synthetic_scene(512, 512);
        let blurred = gaussian_blur_iso(&scene, 3.0);
        let est = estimate_defocus(&blurred, true);
        eprintln!(
            "defocus-on-gaussian: R={:.2} confidence={:.1}",
            est.radius, est.confidence
        );
        assert!(
            !est.confident,
            "defocus estimator locked onto a gaussian blur (confidence {:.1})",
            est.confidence
        );
    }

    /// Gaussian sigmas across the UI range must fit within 25% relative
    /// error with an in-range suggested lambda; a sharp scene must fail
    /// the gate.
    #[test]
    fn test_estimate_gaussian_sigmas() {
        let scene = synthetic_scene(512, 512);
        for &sigma in &[1.5f32, 3.0, 6.0] {
            let blurred = gaussian_blur_iso(&scene, sigma);
            let est = estimate_gaussian(&blurred, true);
            eprintln!(
                "gaussian estimate at σ={sigma}: σ={:.2} t={:.1} λ={:.4} ({}confident)",
                est.sigma,
                est.confidence,
                est.lambda,
                if est.confident { "" } else { "not " }
            );
            assert!(
                est.confident,
                "estimator not confident at σ={sigma} (t={:.1})",
                est.confidence
            );
            assert!(
                (est.sigma - sigma).abs() / sigma <= 0.25,
                "sigma off at σ={sigma}: got {:.2}",
                est.sigma
            );
            assert!(
                (0.01..=0.1).contains(&est.lambda),
                "suggested lambda out of slider range: {:.4}",
                est.lambda
            );
        }

        let sharp = estimate_gaussian(&image::DynamicImage::ImageRgba8(scene), true);
        eprintln!(
            "sharp-image gaussian: σ={:.2} t={:.1}",
            sharp.sigma, sharp.confidence
        );
        assert!(
            !sharp.confident,
            "estimator hallucinated a gaussian blur on a sharp image (t={:.1})",
            sharp.confidence
        );
    }

    /// A 2048² frame runs at working scale 0.5: the reported sigma must map
    /// back to full resolution.
    #[test]
    fn test_estimate_gaussian_scale_mapping() {
        let scene = synthetic_scene(2048, 2048);
        let blurred = gaussian_blur_iso(&scene, 4.0);
        let est = estimate_gaussian(&blurred, true);
        eprintln!(
            "gaussian scale mapping: σ={:.2} t={:.1} ({}confident)",
            est.sigma,
            est.confidence,
            if est.confident { "" } else { "not " }
        );
        assert!(est.confident, "not confident (t={:.1})", est.confidence);
        assert!(
            (est.sigma - 4.0).abs() <= 1.0,
            "full-res sigma off: got {:.2}, expected 4",
            est.sigma
        );
    }

    /// The jinc ring dips of a disc blur must not least-squares-fit as
    /// spurious positive gaussian curvature past the t-gate — the symmetric
    /// cross-mode negative to test_estimate_defocus_rejects_gaussian.
    #[test]
    fn test_estimate_gaussian_rejects_defocus() {
        let scene = synthetic_scene(512, 512);
        let blurred = disc_blur(&scene, 8.0);
        let est = estimate_gaussian(&blurred, true);
        eprintln!(
            "gaussian-on-defocus: σ={:.2} t={:.1}",
            est.sigma, est.confidence
        );
        assert!(
            !est.confident,
            "gaussian estimator locked onto a disc blur (t={:.1})",
            est.confidence
        );
    }

    /// A linear motion blur attenuates one axis only; the radial median
    /// profile must suppress its single-axis sinc dips rather than fit
    /// them as isotropic curvature.
    #[test]
    fn test_estimate_gaussian_rejects_motion() {
        let scene = synthetic_scene(512, 512);
        let blurred = image::DynamicImage::ImageRgba8(motion_blur_line(&scene, 25.0, 0.0));
        let est = estimate_gaussian(&blurred, true);
        eprintln!(
            "gaussian-on-motion: σ={:.2} t={:.1}",
            est.sigma, est.confidence
        );
        assert!(
            !est.confident,
            "gaussian estimator locked onto a motion blur (t={:.1})",
            est.confidence
        );
    }

    /// Degenerate inputs must produce finite, honest results — no panic,
    /// no NaN reaching the serialized structs, no invented blur.
    #[test]
    fn test_estimator_degenerate_inputs() {
        let flat = image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
            64,
            64,
            image::Rgba([128, 128, 128, 255]),
        ));
        let tiny = image::DynamicImage::ImageRgba8(synthetic_scene(8, 8));
        let strip = image::DynamicImage::ImageRgba8(synthetic_scene(1024, 128));
        for (name, img) in [("flat", &flat), ("tiny", &tiny), ("strip", &strip)] {
            let d = estimate_defocus(img, true);
            let g = estimate_gaussian(img, true);
            eprintln!(
                "degenerate {name}: defocus R={:.2} c={:.1}, gaussian σ={:.2} t={:.1}",
                d.radius, d.confidence, g.sigma, g.confidence
            );
            for v in [d.radius, d.confidence, d.lambda, g.sigma, g.confidence, g.lambda] {
                assert!(v.is_finite(), "{name}: non-finite field {v}");
            }
            assert!(!d.confident, "{name}: defocus estimator invented a blur");
            assert!(!g.confident, "{name}: gaussian estimator invented a blur");
        }
    }

    #[test]
    fn test_vram_estimator_and_scale_cap() {
        // Motivating 45 MP case: 8192x5464 pads to 8192x8192; input (16 B/px)
        // + 3 freq textures (24 B/px) + staging (8 B/px), x1.2 headroom
        // = ~3686 MB, inside the 4 GB default budget thanks to the luma-only
        // pipeline.
        let mb = RapidDeconvolver::required_vram_mb(8192, 5464);
        assert!((3600..3750).contains(&mb), "unexpected estimate: {mb} MB");
        assert_eq!(RapidDeconvolver::max_scale_for_vram(8192, 5464, 4096, 8192), 1.0);

        // A tighter budget steps the working scale down to the next
        // power-of-two boundary that fits.
        let capped = RapidDeconvolver::max_scale_for_vram(8192, 5464, 1000, 8192);
        assert!((capped - 0.5).abs() < 1e-6, "expected 0.5, got {capped}");
        assert!(RapidDeconvolver::required_vram_mb(4096, 2732) <= 1000);

        // A small max texture dimension caps the scale even with VRAM to
        // spare.
        let tex_capped = RapidDeconvolver::max_scale_for_vram(8192, 5464, u64::MAX, 4096);
        assert!((tex_capped - 0.5).abs() < 1e-6, "expected 0.5, got {tex_capped}");

        // Small images pass through untouched.
        assert_eq!(RapidDeconvolver::max_scale_for_vram(1920, 1080, 4096, 8192), 1.0);
    }

    /// Old sidecars may still carry rapidAdaptive; the key is ignored and
    /// adaptive regularization is always on, while stored lambda keeps its
    /// raw meaning (the log-scale "Artifact suppression" slider is a UI-only
    /// view over it).
    #[test]
    fn test_parse_rapid_params_ignores_stale_adaptive_key() {
        let adjustments = serde_json::json!({
            "rapidEnabled": true,
            "rapidBlurType": "motion",
            "rapidLength": 200.0,
            "rapidAngle": 0.0,
            "rapidLambda": 0.076,
            "rapidStrength": 100.0,
            "rapidAdaptive": false,
        });
        let params = parse_rapid_params(&adjustments).expect("params should parse");
        assert!(params.adaptive, "adaptive must be always-on regardless of stale sidecar keys");
        assert!((params.lambda - 0.076).abs() < 1e-6, "lambda must stay raw");
    }

    #[test]
    fn test_parse_rapid_params_clamps_pathological_sidecar() {
        let mut adjustments = serde_json::json!({
            "rapidMotionEnabled": true,
            "rapidDefocusEnabled": true,
            "rapidGaussianEnabled": true,
            "rapidLength": 5000.0,
            "rapidAngle": 721.0,
            "rapidRadius": 500.0,
            "rapidSigma": 40.0,
            "rapidLambda": 5.0,
            "rapidStrength": 100.0,
        });
        let params = parse_rapid_params(&adjustments).expect("params should parse");
        assert_eq!(params.motion_length, 200.0);
        assert_eq!(params.defocus_radius, 50.0);
        assert_eq!(params.gaussian_sigma, 8.0);
        assert_eq!(params.lambda, 0.1);
        assert_eq!(params.motion_angle, 721.0);

        adjustments["rapidLambda"] = serde_json::json!(1e-9);
        let params = parse_rapid_params(&adjustments).expect("params should parse");
        assert_eq!(params.lambda, 0.001);

        adjustments["rapidMotionEnabled"] = serde_json::json!(false);
        adjustments["rapidGaussianEnabled"] = serde_json::json!(false);
        let params = parse_rapid_params(&adjustments).expect("defocus should parse");
        assert!(!params.modes.motion);
        assert!(params.modes.defocus);
        assert!(!params.modes.gaussian);
        assert_eq!(kernel_extent(&params), 100);
    }

    /// The hardness slider passes through parse unchanged for every mode:
    /// the defocus component pins its own raw jinc inside psf_generate.wgsl
    /// (guarded by test_gpu_defocus_hardness_invariance), so parse must not
    /// mask the slider — in a compound set the same uniform serves motion.
    /// The parse must also hardcode the clip guard on (no UI toggle).
    #[test]
    fn test_parse_hardness_slider_passthrough() {
        let mut adjustments = serde_json::json!({
            "rapidEnabled": true,
            "rapidBlurType": "defocus",
            "rapidRadius": 8.0,
            "rapidLambda": 0.01,
            "rapidStrength": 100.0,
            "rapidHardness": 40.0,
        });
        let params = parse_rapid_params(&adjustments).expect("params should parse");
        assert!(
            (params.hardness - 0.4).abs() < 1e-6,
            "defocus must pass the slider through (the pin lives in the shader), got {}",
            params.hardness
        );
        assert!(params.clip_guard, "clip guard must be always-on in production");

        adjustments["rapidBlurType"] = serde_json::json!("motion");
        let params = parse_rapid_params(&adjustments).expect("params should parse");
        assert!(
            (params.hardness - 0.4).abs() < 1e-6,
            "motion must keep the parsed hardness, got {}",
            params.hardness
        );
    }

    /// Explicit gen-2 toggles all off veto the stage even with kernels set:
    /// toggling a mode off preserves its kernel values in the sidecar.
    #[test]
    fn test_parse_toggles_all_off() {
        let adjustments = serde_json::json!({
            "rapidMotionEnabled": false,
            "rapidDefocusEnabled": false,
            "rapidGaussianEnabled": false,
            "rapidLength": 50.0,
            "rapidRadius": 8.0,
            "rapidStrength": 100.0,
        });
        assert!(parse_rapid_params(&adjustments).is_none());
    }

    /// An on-but-zero-kernel mode is inert by design (zero-start sliders):
    /// the toggle alone must not run the stage.
    #[test]
    fn test_parse_toggle_on_zero_kernel_inert() {
        let adjustments = serde_json::json!({
            "rapidDefocusEnabled": true,
            "rapidRadius": 0.0,
            "rapidStrength": 100.0,
        });
        assert!(parse_rapid_params(&adjustments).is_none());
    }

    /// Any subset of modes composes into one set; each member needs its
    /// toggle AND a positive kernel.
    #[test]
    fn test_parse_compound_modes() {
        let adjustments = serde_json::json!({
            "rapidMotionEnabled": true,
            "rapidDefocusEnabled": false,
            "rapidGaussianEnabled": true,
            "rapidLength": 24.0,
            "rapidRadius": 8.0,
            "rapidSigma": 1.5,
            "rapidStrength": 100.0,
        });
        let params = parse_rapid_params(&adjustments).expect("compound set should parse");
        assert_eq!(
            params.modes,
            ModeSet { motion: true, defocus: false, gaussian: true }
        );
    }

    /// Gen-2 detection is object-wide: one toggle present means absent
    /// siblings are false — the legacy blurType/kernel fallback must not
    /// reactivate a mode inside partially keyed gen-2 JSON.
    #[test]
    fn test_parse_gen2_siblings_default_false() {
        let adjustments = serde_json::json!({
            "rapidDefocusEnabled": true,
            "rapidRadius": 8.0,
            "rapidBlurType": "motion",
            "rapidLength": 50.0,
            "rapidStrength": 100.0,
        });
        let params = parse_rapid_params(&adjustments).expect("defocus should parse");
        assert_eq!(params.modes, ModeSet::DEFOCUS);
    }

    /// Pre-gen-2 records render absent strength/hardness at the legacy
    /// 100s, not the new-edit defaults of 50.
    #[test]
    fn test_parse_legacy_strength_hardness_defaults() {
        let adjustments = serde_json::json!({
            "rapidBlurType": "motion",
            "rapidLength": 50.0,
        });
        let params = parse_rapid_params(&adjustments).expect("gen-1 record should parse");
        assert!((params.strength - 1.0).abs() < 1e-6);
        assert!((params.hardness - 1.0).abs() < 1e-6);
    }

    /// The before-view override and pre-revamp sidecars saved with the
    /// toggle off both carry an explicit rapidEnabled: false — it must
    /// keep vetoing the stage even with live kernel values present.
    #[test]
    fn test_parse_gate_explicit_false_wins() {
        let adjustments = serde_json::json!({
            "rapidEnabled": false,
            "rapidBlurType": "motion",
            "rapidLength": 50.0,
            "rapidStrength": 100.0,
        });
        assert!(parse_rapid_params(&adjustments).is_none());
    }

    /// Without the legacy flag the stage is gated on the active mode's own
    /// kernel: the other modes' stored values must not activate it.
    #[test]
    fn test_parse_gate_active_mode_kernel() {
        let mk = |blur_type: &str, length: f64, radius: f64, sigma: f64| {
            serde_json::json!({
                "rapidBlurType": blur_type,
                "rapidLength": length,
                "rapidRadius": radius,
                "rapidSigma": sigma,
                "rapidStrength": 100.0,
            })
        };
        assert!(parse_rapid_params(&mk("motion", 0.0, 5.0, 2.0)).is_none());
        assert!(parse_rapid_params(&mk("motion", 50.0, 0.0, 0.0)).is_some());
        assert!(parse_rapid_params(&mk("defocus", 50.0, 0.0, 2.0)).is_none());
        assert!(parse_rapid_params(&mk("defocus", 0.0, 8.0, 0.0)).is_some());
        assert!(parse_rapid_params(&mk("gaussian", 50.0, 8.0, 0.0)).is_none());
        assert!(parse_rapid_params(&mk("gaussian", 0.0, 0.0, 2.5)).is_some());
    }

    #[test]
    fn test_parse_gate_zero_strength() {
        let adjustments = serde_json::json!({
            "rapidBlurType": "motion",
            "rapidLength": 50.0,
            "rapidStrength": 0.0,
        });
        assert!(parse_rapid_params(&adjustments).is_none());
    }

    /// A legacy sidecar can be as sparse as {"rapidEnabled": true}: the old
    /// kernel defaults must come back so the image renders as it always did.
    #[test]
    fn test_parse_legacy_enabled_defaults() {
        let adjustments = serde_json::json!({ "rapidEnabled": true });
        let params =
            parse_rapid_params(&adjustments).expect("legacy-enabled sidecar must stay active");
        assert_eq!(params.modes, ModeSet::MOTION);
        assert!((params.motion_length - 10.0).abs() < 1e-6);
        assert!((params.defocus_radius - 5.0).abs() < 1e-6);
        assert!((params.gaussian_sigma - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_migrate_gen0_rapid_state() {
        // Explicit false: kernels zeroed, toggles explicit false, flag
        // removed, other keys intact, taste pins added.
        let mut off = serde_json::json!({
            "rapidEnabled": false, "rapidLength": 50.0, "exposure": 1.0,
        });
        migrate_legacy_recovery_state(&mut off);
        assert!(off.get("rapidEnabled").is_none());
        assert_eq!(off["rapidLength"], serde_json::json!(0.0));
        assert_eq!(off["rapidRadius"], serde_json::json!(0.0));
        assert_eq!(off["rapidSigma"], serde_json::json!(0.0));
        assert_eq!(off["rapidMotionEnabled"], serde_json::json!(false));
        assert_eq!(off["rapidDefocusEnabled"], serde_json::json!(false));
        assert_eq!(off["rapidGaussianEnabled"], serde_json::json!(false));
        assert_eq!(off["rapidStrength"], serde_json::json!(100.0));
        assert_eq!(off["rapidHardness"], serde_json::json!(100.0));
        assert_eq!(off["exposure"], serde_json::json!(1.0));

        // Explicit true: absent kernels filled with the old defaults,
        // present values kept verbatim, saved mode toggled on.
        let mut on = serde_json::json!({ "rapidEnabled": true, "rapidLength": 120.0 });
        migrate_legacy_recovery_state(&mut on);
        assert!(on.get("rapidEnabled").is_none());
        assert_eq!(on["rapidLength"], serde_json::json!(120.0));
        assert_eq!(on["rapidRadius"], serde_json::json!(5.0));
        assert_eq!(on["rapidSigma"], serde_json::json!(2.0));
        assert_eq!(on["rapidMotionEnabled"], serde_json::json!(true));
        assert_eq!(on["rapidDefocusEnabled"], serde_json::json!(false));
        assert_eq!(on["rapidGaussianEnabled"], serde_json::json!(false));
    }

    /// Gen-1 records (kernel-gated interim, no toggles): the saved mode's
    /// activity is synthesized from its kernel, and the taste defaults that
    /// era rendered for absent keys are pinned before the new INITIALs can
    /// reinterpret them.
    #[test]
    fn test_migrate_gen1_synthesizes_toggles() {
        let mut active = serde_json::json!({ "rapidBlurType": "defocus", "rapidRadius": 8.0 });
        migrate_legacy_recovery_state(&mut active);
        assert_eq!(active["rapidDefocusEnabled"], serde_json::json!(true));
        assert_eq!(active["rapidMotionEnabled"], serde_json::json!(false));
        assert_eq!(active["rapidGaussianEnabled"], serde_json::json!(false));
        assert_eq!(active["rapidStrength"], serde_json::json!(100.0));
        assert_eq!(active["rapidHardness"], serde_json::json!(100.0));

        // A non-saved mode's kernel must not activate it (gen 1 was
        // single-mode: only the saved blurType could run).
        let mut cross = serde_json::json!({ "rapidBlurType": "motion", "rapidRadius": 8.0 });
        migrate_legacy_recovery_state(&mut cross);
        assert_eq!(cross["rapidMotionEnabled"], serde_json::json!(false));
        assert_eq!(cross["rapidDefocusEnabled"], serde_json::json!(false));

        // A clean gen-1 record comes out with everything off.
        let mut clean = serde_json::json!({ "rapidBlurType": "motion", "rapidLength": 0.0 });
        migrate_legacy_recovery_state(&mut clean);
        assert_eq!(clean["rapidMotionEnabled"], serde_json::json!(false));
    }

    #[test]
    fn test_migrate_gen2_passthrough() {
        // A current record is untouched (no pins, no toggle rewrites) —
        // only a stray hand-edited rapidEnabled is cleaned up.
        let mut modern = serde_json::json!({
            "rapidMotionEnabled": true, "rapidLength": 80.0, "rapidStrength": 40.0,
        });
        let before = modern.clone();
        migrate_legacy_recovery_state(&mut modern);
        assert_eq!(modern, before);

        let mut stray = serde_json::json!({
            "rapidGaussianEnabled": false, "rapidEnabled": true, "rapidLength": 80.0,
        });
        migrate_legacy_recovery_state(&mut stray);
        assert!(stray.get("rapidEnabled").is_none());
        assert_eq!(stray["rapidLength"], serde_json::json!(80.0));
        assert!(stray.get("rapidStrength").is_none(), "gen-2 records take no pins");
    }

    /// The presence guards: an object that never touched a subsystem gains
    /// nothing from migration — a curves-only preset spread into live state
    /// must not disable or retune recovery.
    #[test]
    fn test_migrate_skips_absent_subsystems() {
        let mut unrelated = serde_json::json!({
            "exposure": 1.0, "curves": { "luma": [] }, "contrast": 12.0,
        });
        let before = unrelated.clone();
        migrate_legacy_recovery_state(&mut unrelated);
        assert_eq!(unrelated, before);
    }

    #[test]
    fn test_migrate_glare_state() {
        // Legacy active glare: toggle synthesized on.
        let mut active = serde_json::json!({ "glareAmount": 60.0, "glareVeilSize": 40.0 });
        migrate_legacy_recovery_state(&mut active);
        assert_eq!(active["glareEnabled"], serde_json::json!(true));

        // Legacy inactive glare (touched but amount 0): explicit off. A
        // stale glareShowVeil is deliberately NOT honored — it is a
        // transient estimate-flash flag, not user intent.
        let mut veil_only = serde_json::json!({ "glareAmount": 0.0, "glareShowVeil": true });
        migrate_legacy_recovery_state(&mut veil_only);
        assert_eq!(veil_only["glareEnabled"], serde_json::json!(false));

        // An explicit toggle is passthrough.
        let mut modern = serde_json::json!({ "glareEnabled": true, "glareAmount": 0.0 });
        let before = modern.clone();
        migrate_legacy_recovery_state(&mut modern);
        assert_eq!(modern, before);
    }

    #[test]
    fn test_migrate_lowlight_pins() {
        // Enabled sections pin the old 50 defaults into absent value keys.
        let mut enabled = serde_json::json!({ "hotPixelEnabled": true, "denoiseEnabled": true });
        migrate_legacy_recovery_state(&mut enabled);
        assert_eq!(enabled["hotPixelThreshold"], serde_json::json!(50.0));
        assert_eq!(enabled["denoiseStrength"], serde_json::json!(50.0));
        assert_eq!(enabled["denoiseDetail"], serde_json::json!(50.0));
        assert_eq!(enabled["denoiseChroma"], serde_json::json!(50.0));

        // Present values are kept; disabled sections take no pins.
        let mut tuned = serde_json::json!({
            "hotPixelEnabled": false, "denoiseEnabled": true, "denoiseStrength": 30.0,
        });
        migrate_legacy_recovery_state(&mut tuned);
        assert_eq!(tuned["denoiseStrength"], serde_json::json!(30.0));
        assert!(tuned.get("hotPixelThreshold").is_none());
    }

    /// Batch paste onto an unopened legacy-disabled sidecar: after the
    /// write-path migration + merge, the pasted recovery must be active and
    /// the legacy flag gone — it would otherwise veto the paste in
    /// thumbnails/exports and zero it on the next open. The payload models
    /// the current frontend, which sends recovery groups whole (toggles
    /// included).
    #[test]
    fn test_paste_merge_activates_legacy_disabled_target() {
        let mut sidecar = serde_json::json!({
            "rapidEnabled": false,
            "rapidLength": 50.0,
        });
        let pasted = serde_json::json!({
            "rapidMotionEnabled": true,
            "rapidDefocusEnabled": false,
            "rapidGaussianEnabled": false,
            "rapidBlurType": "motion",
            "rapidLength": 80.0,
            "rapidStrength": 100.0,
        });
        migrate_legacy_recovery_state(&mut sidecar);
        let target = sidecar.as_object_mut().unwrap();
        for (k, v) in pasted.as_object().unwrap() {
            target.insert(k.clone(), v.clone());
        }
        assert!(sidecar.get("rapidEnabled").is_none());
        let params = parse_rapid_params(&sidecar).expect("pasted recovery must be active");
        assert!((params.motion_length - 80.0).abs() < 1e-6);
        assert_eq!(params.modes, ModeSet::MOTION);
    }

    fn rgb_error(
        actual: &image::DynamicImage,
        expected: &image::DynamicImage,
    ) -> (f32, f32) {
        assert_eq!(
            (actual.width(), actual.height()),
            (expected.width(), expected.height())
        );
        let actual = actual.to_rgb32f();
        let expected = expected.to_rgb32f();
        let mut max_error = 0.0f32;
        let mut squared_error = 0.0f64;
        let mut samples = 0usize;
        for (a, b) in actual.as_raw().iter().zip(expected.as_raw()) {
            let error = (*a - *b).abs();
            max_error = max_error.max(error);
            squared_error += (error as f64) * (error as f64);
            samples += 1;
        }
        (max_error, (squared_error / samples as f64).sqrt() as f32)
    }

    #[test]
    fn test_deconvolve_linear_matches_encoded_flag() {
        if get_rapid_gpu().is_none() {
            eprintln!("skipping encoded provenance GPU test: no adapter");
            return;
        }

        let linear = image::DynamicImage::ImageRgba32F(image::Rgba32FImage::from_fn(
            192,
            128,
            |x, y| {
                let wave =
                    0.5 + 0.25 * (x as f32 / 11.0).sin() + 0.25 * (y as f32 / 17.0).cos();
                let stroke = if (x / 13 + y / 19) % 5 == 0 { 0.12 } else { 0.0 };
                let value = (0.08 + 0.58 * wave + stroke).clamp(0.02, 0.82);
                let alpha = 0.35 + 0.6 * ((x + 3 * y) % 31) as f32 / 30.0;
                image::Rgba([value, 0.84 * value + 0.03, 0.68 * value + 0.07, alpha])
            },
        ));
        let encoded = crate::image_processing::apply_linear_to_srgb(linear.clone());
        let adjustments = serde_json::json!({
            "rapidMotionEnabled": false,
            "rapidDefocusEnabled": true,
            "rapidGaussianEnabled": false,
            "rapidRadius": 3.5,
            "rapidLambda": 0.02,
            "rapidStrength": 80.0,
        });

        let linear_out = apply_blur_recovery_scaled(
            std::borrow::Cow::Owned(linear),
            &adjustments,
            1.0,
            true,
        )
        .into_owned();
        let encoded_out = apply_blur_recovery_scaled(
            std::borrow::Cow::Owned(encoded),
            &adjustments,
            1.0,
            false,
        )
        .into_owned();
        let decoded_out = crate::image_processing::apply_srgb_to_linear(encoded_out);
        let (max_error, rms_error) = rgb_error(&decoded_out, &linear_out);
        eprintln!(
            "linear/encoded wrapper agreement: max {max_error:.3e}, rms {rms_error:.3e}"
        );
        assert!(
            max_error < 1e-5 && rms_error < 1e-6,
            concat!(
                "encoded provenance diverged after decoding: ",
                "max {:.3e}, rms {:.3e}"
            ),
            max_error, rms_error
        );
    }

    /// A 1152px source forces the estimator resize branch. Decode-before-resize
    /// measured RMS 7.96e-5 on the reference host; the intentionally wrong
    /// encoded-as-linear flag measured 3.72e-1 (about 4,670x worse).
    #[test]
    fn test_working_spectrum_linearizes() {
        const WIDTH: usize = 1152;
        const HEIGHT: usize = 768;
        const RADIUS: f32 = 7.0;
        let scene: Vec<f32> = (0..WIDTH * HEIGHT)
            .map(|i| {
                let (x, y) = ((i % WIDTH) as u32, (i / WIDTH) as u32);
                let mut hash = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77);
                hash ^= hash >> 16;
                hash = hash.wrapping_mul(0x7FEB_352D);
                hash ^= hash >> 15;
                let noise = (hash & 0xffff) as f32 / 65535.0;
                let wave =
                    0.5 + 0.25 * (x as f32 / 37.0).sin() + 0.25 * (y as f32 / 29.0).cos();
                let stroke = if (x / 47 + y / 61) % 7 == 0 { 0.1 } else { 0.0 };
                (0.04 + 0.28 * noise + 0.48 * wave + stroke).clamp(0.02, 0.86)
            })
            .collect();
        let blurred = disc_blur_field(&scene, WIDTH, HEIGHT, RADIUS);
        let linear = image::DynamicImage::ImageRgb32F(image::Rgb32FImage::from_fn(
            WIDTH as u32,
            HEIGHT as u32,
            |x, y| {
                let value = blurred[y as usize * WIDTH + x as usize];
                image::Rgb([value, value, value])
            },
        ));
        let encoded = crate::image_processing::apply_linear_to_srgb(linear.clone());

        let linear_ws = working_spectrum(&linear, true);
        let encoded_ws = working_spectrum(&encoded, false);
        let wrong_ws = working_spectrum(&encoded, true);
        assert_eq!(
            (linear_ws.w, linear_ws.h, linear_ws.pw, linear_ws.ph),
            (encoded_ws.w, encoded_ws.h, encoded_ws.pw, encoded_ws.ph)
        );
        assert_eq!(linear_ws.scale.to_bits(), encoded_ws.scale.to_bits());

        let spectrum_error = |candidate: &WorkingSpectrum| -> (f32, f32) {
            let mut max_error = 0.0f32;
            let mut squared_error = 0.0f64;
            for (a, b) in linear_ws.spectrum_ln.iter().zip(&candidate.spectrum_ln) {
                let error = (*a - *b).abs();
                max_error = max_error.max(error);
                squared_error += (error as f64) * (error as f64);
            }
            (
                max_error,
                (squared_error / linear_ws.spectrum_ln.len() as f64).sqrt() as f32,
            )
        };
        let (matched_max, matched_rms) = spectrum_error(&encoded_ws);
        let (_, wrong_rms) = spectrum_error(&wrong_ws);
        eprintln!(
            concat!(
                "working spectrum transfer posture: matched max {:.3e}, ",
                "matched rms {:.3e}, wrong-flag rms {:.3e}"
            ),
            matched_max, matched_rms, wrong_rms
        );
        assert!(
            matched_max < 1.2e-2 && matched_rms < 2e-4,
            "linearized spectrum mismatch: max {matched_max:.3e}, rms {matched_rms:.3e}"
        );
        assert!(
            wrong_rms > 100.0 * matched_rms.max(1e-6) && wrong_rms > 0.2,
            concat!(
                "wrong provenance was not detectably worse: ",
                "matched {:.3e}, wrong {:.3e}"
            ),
            matched_rms, wrong_rms
        );

        let linear_estimate = estimate_defocus(&linear, true);
        let encoded_estimate = estimate_defocus(&encoded, false);
        assert!(linear_estimate.confident && encoded_estimate.confident);
        assert!(
            (linear_estimate.radius - encoded_estimate.radius).abs() <= 0.2,
            "radius changed with encoding: linear {:.2}, encoded {:.2}",
            linear_estimate.radius,
            encoded_estimate.radius
        );
        assert!(
            (linear_estimate.lambda - encoded_estimate.lambda).abs() <= 0.002,
            "lambda changed with encoding: linear {:.4}, encoded {:.4}",
            linear_estimate.lambda,
            encoded_estimate.lambda
        );
    }

    #[test]
    fn test_blur_recovery_scaled_linearizes_before_resize() {
        let Some(gpu) = get_rapid_gpu() else {
            eprintln!("skipping scaled linearization GPU test: no adapter");
            return;
        };

        const WIDTH: u32 = 320;
        const HEIGHT: u32 = 240;
        const SCALE: f32 = 0.5;
        let encoded = image::DynamicImage::ImageRgba32F(image::Rgba32FImage::from_fn(
            WIDTH,
            HEIGHT,
            |x, y| {
                let dx = x as i32 - 76;
                let dy = y as i32 - 92;
                let highlight = dx * dx + dy * dy <= 36;
                let strokes = (132..260).contains(&x) && ((x - 132) / 3) % 2 == 0;
                let diagonal = ((x as i32 - y as i32 - 35).abs() <= 2)
                    || ((x as i32 + y as i32 - 360).abs() <= 2);
                let value: f32 = if highlight {
                    0.985
                } else if strokes || diagonal {
                    0.88
                } else {
                    0.08 + 0.2 * (x as f32 / 23.0).sin().abs()
                };
                image::Rgba([
                    value,
                    (0.9 * value + 0.02).min(0.985),
                    (0.72 * value + 0.04).min(0.985),
                    1.0,
                ])
            },
        ));
        let adjustments = serde_json::json!({
            "rapidMotionEnabled": false,
            "rapidDefocusEnabled": true,
            "rapidGaussianEnabled": false,
            "rapidRadius": 4.0,
            "rapidLambda": 0.02,
            "rapidStrength": 85.0,
        });
        let wrapper = apply_blur_recovery_scaled(
            std::borrow::Cow::Owned(encoded.clone()),
            &adjustments,
            SCALE,
            false,
        )
        .into_owned();

        let params = parse_rapid_params(&adjustments).unwrap();
        let small_w = (WIDTH as f32 * SCALE).round() as u32;
        let small_h = (HEIGHT as f32 * SCALE).round() as u32;
        let decoded =
            crate::image_processing::apply_srgb_to_linear(image::DynamicImage::ImageRgba32F(
                encoded.to_rgba32f(),
            ));
        let correct_small =
            decoded.resize_exact(small_w, small_h, image::imageops::FilterType::Triangle);
        let encoded_small =
            encoded.resize_exact(small_w, small_h, image::imageops::FilterType::Triangle);
        let wrong_order_small = crate::image_processing::apply_srgb_to_linear(encoded_small);
        let decoded_threshold =
            crate::image_processing::srgb_channel_to_linear(CLIP_GUARD_SAT);
        let scaled_params = params.scaled(SCALE);

        let (reference_linear, wrong_order_linear, wrong_threshold_linear) = {
            let mut gpu = gpu.lock().unwrap();
            let RapidGpu {
                device,
                queue,
                deconvolver,
                ..
            } = &mut *gpu;
            let reference = deconvolver
                .deconvolve_linear_image(
                    device,
                    queue,
                    &correct_small,
                    &scaled_params,
                    decoded_threshold,
                )
                .expect("reference deconvolution failed");
            let wrong_order = deconvolver
                .deconvolve_linear_image(
                    device,
                    queue,
                    &wrong_order_small,
                    &scaled_params,
                    decoded_threshold,
                )
                .expect("wrong-order deconvolution failed");
            let wrong_threshold = deconvolver
                .deconvolve_linear_image(
                    device,
                    queue,
                    &correct_small,
                    &scaled_params,
                    CLIP_GUARD_SAT,
                )
                .expect("wrong-threshold deconvolution failed");
            (reference, wrong_order, wrong_threshold)
        };
        let finish = |linear: image::DynamicImage| {
            let restored =
                linear.resize_exact(WIDTH, HEIGHT, image::imageops::FilterType::Triangle);
            crate::image_processing::apply_linear_to_srgb(restored)
        };
        let reference = finish(reference_linear);
        let wrong_order = finish(wrong_order_linear);
        let wrong_threshold = finish(wrong_threshold_linear);

        let (reference_max, reference_rms) = rgb_error(&wrapper, &reference);
        let (_, wrong_order_rms) = rgb_error(&wrapper, &wrong_order);
        let (_, wrong_threshold_rms) = rgb_error(&wrapper, &wrong_threshold);
        eprintln!(
            concat!(
                "scaled linearization posture: reference max {:.3e}, ",
                "rms {:.3e}; encoded-first rms {:.3e}; raw-threshold rms {:.3e}"
            ),
            reference_max, reference_rms, wrong_order_rms, wrong_threshold_rms
        );
        assert!(
            reference_max < 2e-5 && reference_rms < 2e-6,
            concat!(
                "scaled wrapper missed explicit linear reference: ",
                "max {:.3e}, rms {:.3e}"
            ),
            reference_max, reference_rms
        );
        assert!(
            wrong_order_rms > 2e-2 && wrong_order_rms > 20.0 * reference_rms.max(1e-7),
            "resize-encoded-first path was not detectably different: {wrong_order_rms:.3e}"
        );
        assert!(
            wrong_threshold_rms > 5e-3
                && wrong_threshold_rms > 20.0 * reference_rms.max(1e-7),
            "wrapper did not preserve encoded threshold provenance: {wrong_threshold_rms:.3e}"
        );
    }

    #[test]
    fn test_clip_guard_threshold_tracks_encoding() {
        let encoded_highlight = 0.985;
        let decoded_highlight =
            crate::image_processing::srgb_channel_to_linear(encoded_highlight);
        let rgba = image::Rgba32FImage::from_fn(7, 7, |x, y| {
            let value = if x == 3 && y == 3 {
                decoded_highlight
            } else {
                0.2
            };
            image::Rgba([value, value, value, 1.0])
        });
        let decoded_threshold =
            crate::image_processing::srgb_channel_to_linear(CLIP_GUARD_SAT);
        let d1_pixels = CLIP_GUARD_DEFOCUS_D1_EXTENTS * 2.0;
        let guard = clip_guard_weights(
            &rgba,
            2,
            decoded_threshold,
            d1_pixels,
        )
        .expect("decoded 0.985 highlight must trip the encoded-source threshold");
        assert!(guard.iter().any(|&weight| weight > 0.0));
        assert!(
            clip_guard_weights(
                &rgba,
                2,
                CLIP_GUARD_SAT,
                d1_pixels,
            )
            .is_none(),
            "decoded 0.985 highlight must not trip the raw linear threshold"
        );
    }

    #[test]
    fn test_blur_recovery_scaled_preserves_dimensions() {
        // Output dimensions must equal input dimensions so downstream
        // geometry (crop/rotation coordinates) is unaffected. Holds both
        // with a GPU (downscale-deconvolve-upscale) and without (pass-through).
        let img = image::DynamicImage::ImageRgba8(image::ImageBuffer::from_fn(200, 150, |x, y| {
            image::Rgba([((x * 7 + y * 13) % 255) as u8, 128, 64, 255])
        }));
        let adjustments = serde_json::json!({
            "rapidEnabled": true,
            "rapidBlurType": "motion",
            "rapidLength": 20.0,
            "rapidAngle": 0.0,
            "rapidLambda": 0.01,
            "rapidStrength": 100.0,
        });
        let out = apply_blur_recovery_scaled(
            std::borrow::Cow::Owned(img),
            &adjustments,
            0.5,
            true,
        );
        assert_eq!((out.width(), out.height()), (200, 150));
    }
}
