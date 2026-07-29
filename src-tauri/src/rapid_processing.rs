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
use std::sync::Arc;
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

/// Parameters for RAPID deconvolution
#[derive(Debug, Clone, Copy)]
pub struct RapidParams {
    /// Enable RAPID processing
    pub enabled: bool,
    /// Type of blur to deconvolve
    pub blur_type: BlurType,
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
    /// Use adaptive regularization based on local variance
    pub adaptive: bool,
}

impl Default for RapidParams {
    fn default() -> Self {
        Self {
            enabled: false,
            blur_type: BlurType::Motion,
            motion_length: 10.0,
            motion_angle: 0.0,
            defocus_radius: 5.0,
            gaussian_sigma: 2.0,
            lambda: 0.01,
            strength: 1.0,
            edge_taper: true,
            noise_floor: 1e-6,
            adaptive: false,
        }
    }
}

impl RapidParams {
    /// Create RapidParams from adjustment values (from frontend)
    pub fn from_adjustments(
        enabled: bool,
        blur_type: u32,
        motion_length: f32,
        motion_angle: f32,
        defocus_radius: f32,
        gaussian_sigma: f32,
        lambda: f32,
        strength: f32,
        noise_floor: f32,
        adaptive: bool,
    ) -> Self {
        Self {
            enabled,
            blur_type: BlurType::from(blur_type),
            motion_length,
            motion_angle,
            defocus_radius,
            gaussian_sigma,
            lambda,
            strength: strength / 100.0, // Convert 0-100 to 0-1
            edge_taper: true,
            noise_floor,
            adaptive,
        }
    }

    /// Rescale the spatial kernel parameters for a working image that has been
    /// downscaled by `scale`. Lambda and strength describe frequency-domain
    /// behavior and blending, not pixel extents, so they stay unchanged. The
    /// floors keep degenerate scales from collapsing the PSF to a no-op.
    pub fn scaled(&self, scale: f32) -> Self {
        Self {
            motion_length: (self.motion_length * scale).max(1.0),
            defocus_radius: (self.defocus_radius * scale).max(0.5),
            gaussian_sigma: (self.gaussian_sigma * scale).max(0.3),
            ..*self
        }
    }
}

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

/// Spatial extent of the active PSF in pixels: how far the circular wrap can
/// smear content across the frame boundary.
fn kernel_extent(params: &RapidParams) -> usize {
    let extent = match params.blur_type {
        BlurType::Motion => params.motion_length,
        BlurType::Defocus => 2.0 * params.defocus_radius,
        BlurType::Gaussian => 6.0 * params.gaussian_sigma,
    };
    (extent.ceil() as usize).max(1)
}

/// Projection of the active PSF onto one axis (0 = x, 1 = y), normalized to
/// sum 1; `[1.0]` when the projection is sub-pixel (identity).
fn psf_axis_projection(params: &RapidParams, axis: usize) -> Vec<f32> {
    // Half-extent and unnormalized density of the projected PSF at signed
    // distance t from its center. These must match the spectra the Wiener
    // filter divides by (psf_generate.wgsl), not an idealized blur model.
    let (half_extent, density): (f32, Box<dyn Fn(f32) -> f32>) = match params.blur_type {
        BlurType::Motion => {
            // The engine's motion OTF is a zero-free Gaussian envelope with
            // sigma_freq = 1/L (motion_blur_spectrum), i.e. spatially a
            // Gaussian of sigma = L/(2π) along the motion direction, which
            // projects onto an axis as a Gaussian of sigma·|cos| / sigma·|sin|.
            let dir = params.motion_angle.to_radians();
            let along = if axis == 0 { dir.cos() } else { dir.sin() };
            let s = (params.motion_length * along).abs() / (2.0 * std::f32::consts::PI);
            (3.0 * s, Box::new(move |t: f32| (-t * t / (2.0 * s * s).max(1e-6)).exp()))
        }
        BlurType::Defocus => {
            // jinc spectrum = uniform disk, which projects as its chord length.
            let r = params.defocus_radius.max(0.0);
            (r, Box::new(move |t: f32| (r * r - t * t).max(0.0).sqrt()))
        }
        BlurType::Gaussian => {
            // gaussian_blur_spectrum caps effective sigma at 8; mirror that.
            let s = params.gaussian_sigma.clamp(1e-3, 8.0);
            (3.0 * s, Box::new(move |t: f32| (-t * t / (2.0 * s * s)).exp()))
        }
    };

    let radius = half_extent.ceil() as i32;
    if radius < 1 {
        return vec![1.0];
    }
    // Midpoint-integrate the density over each tap cell [i-0.5, i+0.5].
    let mut kernel: Vec<f32> = (-radius..=radius)
        .map(|i| {
            (0..4)
                .map(|k| density(i as f32 - 0.5 + (k as f32 + 0.5) / 4.0))
                .sum::<f32>()
        })
        .collect();
    let sum: f32 = kernel.iter().sum();
    if sum <= f32::EPSILON {
        return vec![1.0];
    }
    for w in &mut kernel {
        *w /= sum;
    }
    kernel
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

/// Build the pow2-padded RGBA f32 upload buffer for `deconvolve_image`.
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
    blur_type: u32,
    motion_length: f32,
    motion_angle: f32,
    defocus_radius: f32,
    gaussian_sigma: f32,
    _pad: f32,
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
    _pad: [f32; 2],
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

/// Collection of textures for frequency domain processing
struct FrequencyTextures {
    /// Red channel frequency data
    freq_r: wgpu::Texture,
    /// Green channel frequency data
    freq_g: wgpu::Texture,
    /// Blue channel frequency data
    freq_b: wgpu::Texture,
    /// Ping-pong buffer for FFT passes
    freq_temp: wgpu::Texture,
    /// PSF frequency data (shared across channels)
    psf_freq: wgpu::Texture,
}

struct FrequencyTextureViews {
    freq_r: wgpu::TextureView,
    freq_g: wgpu::TextureView,
    freq_b: wgpu::TextureView,
    freq_temp: wgpu::TextureView,
    psf_freq: wgpu::TextureView,
}

impl FrequencyTextures {
    fn create_views(&self) -> FrequencyTextureViews {
        FrequencyTextureViews {
            freq_r: self.freq_r.create_view(&Default::default()),
            freq_g: self.freq_g.create_view(&Default::default()),
            freq_b: self.freq_b.create_view(&Default::default()),
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
            "RAPID GPU check passed: {} ({:?}), max texture: {}",
            info.name,
            info.backend,
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

    /// Check if RAPID can process an image of the given size
    pub fn can_process_size(&self, width: u32, height: u32, available_vram_mb: u64) -> bool {
        let padded_w = width.next_power_of_two();
        let padded_h = height.next_power_of_two();

        // Check against max texture size
        if padded_w > self.max_texture_size || padded_h > self.max_texture_size {
            log::warn!(
                "RAPID: Padded size {}x{} exceeds max texture size {}",
                padded_w,
                padded_h,
                self.max_texture_size
            );
            return false;
        }

        // 5 frequency textures × Rg32Float (8 bytes/pixel)
        let freq_memory_mb = (padded_w as u64 * padded_h as u64 * 8 * 5) / (1024 * 1024);

        // Input/output RGBA16F textures (8 bytes/pixel)
        let rgba_memory_mb = (width as u64 * height as u64 * 8 * 2) / (1024 * 1024);

        // Total with 20% safety margin
        let total_required_mb = ((freq_memory_mb + rgba_memory_mb) as f64 * 1.2) as u64;

        let can_process = total_required_mb < available_vram_mb;

        if !can_process {
            log::warn!(
                "RAPID: Image {}x{} (padded {}x{}) requires ~{}MB, available ~{}MB",
                width,
                height,
                padded_w,
                padded_h,
                total_required_mb,
                available_vram_mb
            );
        }

        can_process
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
            freq_r: create_freq_texture("RAPID freq_r"),
            freq_g: create_freq_texture("RAPID freq_g"),
            freq_b: create_freq_texture("RAPID freq_b"),
            freq_temp: create_freq_texture("RAPID freq_temp"),
            psf_freq: create_freq_texture("RAPID psf_freq"),
        };

        let views = textures.create_views();

        self.freq_textures = Some(textures);
        self.freq_views = Some(views);
        self.allocated_width = padded_width;
        self.allocated_height = padded_height;

        let memory_mb = (padded_width as u64 * padded_height as u64 * 8 * 5) / (1024 * 1024);
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
            blur_type: params.blur_type as u32,
            motion_length: params.motion_length,
            motion_angle: params.motion_angle,
            defocus_radius: params.defocus_radius,
            gaussian_sigma: params.gaussian_sigma,
            _pad: 0.0,
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
            _pad: [0.0; 2],
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
            params.blur_type,
            params.motion_length,
            params.motion_angle,
            params.defocus_radius,
            params.gaussian_sigma
        );

        Err("RAPID deconvolution not yet implemented (Phase 2-4)".to_string())
    }

    /// High-level deconvolution that takes a DynamicImage and returns a processed DynamicImage.
    /// This handles all the texture creation, upload, processing, and readback.
    pub fn deconvolve_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &image::DynamicImage,
        params: &RapidParams,
    ) -> Result<image::DynamicImage, String> {
        use image::{GenericImageView, Rgba, RgbaImage};

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
            params.blur_type,
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

        // Step 2: Convert each RGB channel to complex and forward FFT
        log::debug!("RAPID: Converting channels to frequency domain...");

        // Process each channel with separate command submission to avoid buffer race
        // This is needed because utility_params_buffer is shared and would be overwritten

        // Red channel
        self.encode_real_to_complex(
            &mut encoder, device, queue,
            &input_view, &freq_views.freq_r,
            padded_w, padded_h, padded_w, padded_h,
            0, // channel R
        );
        // Submit and create new encoder to ensure params buffer is read correctly
        queue.submit(std::iter::once(encoder.finish()));
        encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RAPID Deconvolution G"),
        });

        // Green channel
        self.encode_real_to_complex(
            &mut encoder, device, queue,
            &input_view, &freq_views.freq_g,
            padded_w, padded_h, padded_w, padded_h,
            1, // channel G
        );
        queue.submit(std::iter::once(encoder.finish()));
        encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RAPID Deconvolution B"),
        });

        // Blue channel
        self.encode_real_to_complex(
            &mut encoder, device, queue,
            &input_view, &freq_views.freq_b,
            padded_w, padded_h, padded_w, padded_h,
            2, // channel B
        );

        // DEBUG_LEVEL 2: Skip FFT, PSF, Wiener - just test real_to_complex + readback
        let mut encoder = if DEBUG_LEVEL == 2 {
            log::info!("RAPID DEBUG: Testing real_to_complex only (no FFT)");
            // Skip directly to readback - freq_r/g/b contain the padded image
            encoder
        } else {
            // Forward FFT on each channel (each call takes ownership and returns new encoder)
            let encoder = self.forward_fft_2d(
                encoder, device, queue,
                &freq_textures.freq_r, &freq_views.freq_r,
                &freq_textures.freq_temp, &freq_views.freq_temp,
                padded_w, padded_h,
            );
            let encoder = self.forward_fft_2d(
                encoder, device, queue,
                &freq_textures.freq_g, &freq_views.freq_g,
                &freq_textures.freq_temp, &freq_views.freq_temp,
                padded_w, padded_h,
            );
            let mut encoder = self.forward_fft_2d(
                encoder, device, queue,
                &freq_textures.freq_b, &freq_views.freq_b,
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

            // Step 4: Apply Wiener filter to each channel
            log::debug!("RAPID: Applying Wiener deconvolution filter...");

            // For Wiener filter, we need to use freq_temp as output since input and output
            // can't be the same texture in storage binding. Then copy back.

            // Red channel
            self.encode_wiener_filter(
                &mut encoder, device, queue,
                &freq_views.freq_r, &freq_views.psf_freq, &freq_views.freq_temp,
                padded_w, padded_h, params,
            );
            encoder.copy_texture_to_texture(
                freq_textures.freq_temp.as_image_copy(),
                freq_textures.freq_r.as_image_copy(),
                wgpu::Extent3d { width: padded_w, height: padded_h, depth_or_array_layers: 1 },
            );

            // Green channel
            self.encode_wiener_filter(
                &mut encoder, device, queue,
                &freq_views.freq_g, &freq_views.psf_freq, &freq_views.freq_temp,
                padded_w, padded_h, params,
            );
            encoder.copy_texture_to_texture(
                freq_textures.freq_temp.as_image_copy(),
                freq_textures.freq_g.as_image_copy(),
                wgpu::Extent3d { width: padded_w, height: padded_h, depth_or_array_layers: 1 },
            );

            // Blue channel
            self.encode_wiener_filter(
                &mut encoder, device, queue,
                &freq_views.freq_b, &freq_views.psf_freq, &freq_views.freq_temp,
                padded_w, padded_h, params,
            );
            encoder.copy_texture_to_texture(
                freq_textures.freq_temp.as_image_copy(),
                freq_textures.freq_b.as_image_copy(),
                wgpu::Extent3d { width: padded_w, height: padded_h, depth_or_array_layers: 1 },
            );

            // Step 5: Inverse FFT each channel (includes normalization)
            log::debug!("RAPID: Transforming back to spatial domain...");

            let encoder = self.inverse_fft_2d(
                encoder, device, queue,
                &freq_textures.freq_r, &freq_views.freq_r,
                &freq_textures.freq_temp, &freq_views.freq_temp,
                padded_w, padded_h,
            );
            let encoder = self.inverse_fft_2d(
                encoder, device, queue,
                &freq_textures.freq_g, &freq_views.freq_g,
                &freq_textures.freq_temp, &freq_views.freq_temp,
                padded_w, padded_h,
            );
            self.inverse_fft_2d(
                encoder, device, queue,
                &freq_textures.freq_b, &freq_views.freq_b,
                &freq_textures.freq_temp, &freq_views.freq_temp,
                padded_w, padded_h,
            )
        }; // end else (DEBUG_LEVEL != 2)

        // Step 6: Read back the result from GPU
        // Create staging buffers for readback
        let bytes_per_row = padded_w * 8; // 2 channels (RG) * 4 bytes per f32
        let aligned_bytes_per_row = (bytes_per_row + 255) & !255; // Align to 256 bytes
        let buffer_size = (aligned_bytes_per_row * padded_h) as u64;

        let staging_r = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RAPID Staging R"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let staging_g = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RAPID Staging G"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let staging_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RAPID Staging B"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy textures to staging buffers
        encoder.copy_texture_to_buffer(
            freq_textures.freq_r.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_r,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(padded_h),
                },
            },
            wgpu::Extent3d { width: padded_w, height: padded_h, depth_or_array_layers: 1 },
        );
        encoder.copy_texture_to_buffer(
            freq_textures.freq_g.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_g,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(padded_h),
                },
            },
            wgpu::Extent3d { width: padded_w, height: padded_h, depth_or_array_layers: 1 },
        );
        encoder.copy_texture_to_buffer(
            freq_textures.freq_b.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &staging_b,
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

        // Map and read back the buffers
        let (tx, rx) = std::sync::mpsc::channel();
        let tx_r = tx.clone();
        let tx_g = tx.clone();
        let tx_b = tx;

        staging_r.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx_r.send(("r", result)).unwrap();
        });
        staging_g.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx_g.send(("g", result)).unwrap();
        });
        staging_b.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            tx_b.send(("b", result)).unwrap();
        });

        // Wait for all maps to complete
        device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(60)),
        }).unwrap();

        // Check for errors
        for _ in 0..3 {
            let (channel, result) = rx.recv().map_err(|e| format!("Channel receive error: {}", e))?;
            result.map_err(|e| format!("Buffer map error for {}: {:?}", channel, e))?;
        }

        // Read the data
        let r_data: Vec<f32> = {
            let view = staging_r.slice(..).get_mapped_range().map_err(|e| format!("Failed to map staging buffer: {e:?}"))?;
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
        staging_r.unmap();

        let g_data: Vec<f32> = {
            let view = staging_g.slice(..).get_mapped_range().map_err(|e| format!("Failed to map staging buffer: {e:?}"))?;
            let data: &[f32] = bytemuck::cast_slice(&view);
            let mut result = Vec::with_capacity((padded_w * padded_h) as usize);
            let f32_per_aligned_row = aligned_bytes_per_row as usize / 4;
            for y in 0..padded_h as usize {
                for x in 0..padded_w as usize {
                    let idx = y * f32_per_aligned_row + x * 2;
                    result.push(data[idx]);
                }
            }
            result
        };
        staging_g.unmap();

        let b_data: Vec<f32> = {
            let view = staging_b.slice(..).get_mapped_range().map_err(|e| format!("Failed to map staging buffer: {e:?}"))?;
            let data: &[f32] = bytemuck::cast_slice(&view);
            let mut result = Vec::with_capacity((padded_w * padded_h) as usize);
            let f32_per_aligned_row = aligned_bytes_per_row as usize / 4;
            for y in 0..padded_h as usize {
                for x in 0..padded_w as usize {
                    let idx = y * f32_per_aligned_row + x * 2;
                    result.push(data[idx]);
                }
            }
            result
        };
        staging_b.unmap();

        // Combine channels into output image (crop to original size)
        let mut output = RgbaImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let idx = (y * padded_w + x) as usize;
                let r = (r_data[idx].clamp(0.0, 1.0) * 255.0) as u8;
                let g = (g_data[idx].clamp(0.0, 1.0) * 255.0) as u8;
                let b = (b_data[idx].clamp(0.0, 1.0) * 255.0) as u8;
                output.put_pixel(x, y, Rgba([r, g, b, 255]));
            }
        }

        let elapsed = start_time.elapsed();
        log::info!("RAPID deconvolution completed in {:.2?}", elapsed);

        Ok(image::DynamicImage::ImageRgba8(output))
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
}

static RAPID_GPU: std::sync::OnceLock<Option<std::sync::Mutex<RapidGpu>>> = std::sync::OnceLock::new();

/// Lazily creates a dedicated wgpu device for blur recovery. Kept separate
/// from the main GpuContext so the pre-pass needs no plumbing through the
/// tiled pipeline; returns None (and logs) when the GPU is unsupported.
fn get_rapid_gpu() -> Option<&'static std::sync::Mutex<RapidGpu>> {
    RAPID_GPU
        .get_or_init(|| {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
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
            match RapidDeconvolver::new(&adapter, &device) {
                Ok(deconvolver) => Some(std::sync::Mutex::new(RapidGpu {
                    device,
                    queue,
                    deconvolver,
                })),
                Err(e) => {
                    log::warn!("RAPID: unsupported GPU ({e}); blur recovery disabled");
                    None
                }
            }
        })
        .as_ref()
}

/// Parses blur-recovery params from the frontend adjustment JSON.
/// Returns None when the feature is off or its section is hidden.
pub fn parse_rapid_params(adjustments: &serde_json::Value) -> Option<RapidParams> {
    let visible = adjustments
        .get("sectionVisibility")
        .and_then(|v| v.get("blurRecovery"))
        .and_then(|s| s.as_bool())
        .unwrap_or(true);
    if !visible || !adjustments["rapidEnabled"].as_bool().unwrap_or(false) {
        return None;
    }
    let blur_type = match adjustments["rapidBlurType"].as_str().unwrap_or("motion") {
        "defocus" => BlurType::Defocus,
        "gaussian" => BlurType::Gaussian,
        _ => BlurType::Motion,
    };
    Some(RapidParams {
        enabled: true,
        blur_type,
        motion_length: adjustments["rapidLength"].as_f64().unwrap_or(10.0) as f32,
        motion_angle: adjustments["rapidAngle"].as_f64().unwrap_or(0.0) as f32,
        defocus_radius: adjustments["rapidRadius"].as_f64().unwrap_or(5.0) as f32,
        gaussian_sigma: adjustments["rapidSigma"].as_f64().unwrap_or(2.0) as f32,
        lambda: adjustments["rapidLambda"].as_f64().unwrap_or(0.01) as f32,
        strength: (adjustments["rapidStrength"].as_f64().unwrap_or(100.0) as f32 / 100.0)
            .clamp(0.0, 1.0),
        adaptive: adjustments["rapidAdaptive"].as_bool().unwrap_or(false),
        ..Default::default()
    })
}

/// True when blur recovery would actually run for these adjustments
/// (enabled and its section not hidden).
pub fn is_rapid_active(adjustments: &serde_json::Value) -> bool {
    parse_rapid_params(adjustments).is_some()
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
pub fn apply_blur_recovery_scaled<'a>(
    image: std::borrow::Cow<'a, image::DynamicImage>,
    adjustments: &serde_json::Value,
    rapid_scale: f32,
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
    } = &mut *gpu;
    let start = std::time::Instant::now();

    if rapid_scale < 0.999 {
        let (w, h) = (image.width(), image.height());
        let small_w = ((w as f32 * rapid_scale).round() as u32).max(1);
        let small_h = ((h as f32 * rapid_scale).round() as u32).max(1);
        let small = image.resize_exact(small_w, small_h, image::imageops::FilterType::Triangle);
        return match deconvolver.deconvolve_image(device, queue, &small, &params.scaled(rapid_scale)) {
            Ok(out) => {
                let restored = out.resize_exact(w, h, image::imageops::FilterType::Triangle);
                log::info!(
                    "RAPID: preview-scale blur recovery pre-pass ({}x{} @ {:.3}) took {:?}",
                    small_w,
                    small_h,
                    rapid_scale,
                    start.elapsed()
                );
                std::borrow::Cow::Owned(restored)
            }
            Err(e) => {
                log::warn!("RAPID: preview-scale blur recovery failed ({e}); using original image");
                image
            }
        };
    }

    match deconvolver.deconvolve_image(device, queue, image.as_ref(), &params) {
        Ok(out) => {
            log::info!("RAPID: blur recovery pre-pass took {:?}", start.elapsed());
            std::borrow::Cow::Owned(out)
        }
        Err(e) => {
            log::warn!("RAPID: blur recovery failed ({e}); using original image");
            image
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    // ========================================================================
    // CPU Reference FFT Implementation for Validation
    // ========================================================================

    /// Complex number for CPU testing
    #[derive(Clone, Copy, Debug)]
    struct Complex {
        re: f32,
        im: f32,
    }

    impl Complex {
        fn new(re: f32, im: f32) -> Self {
            Self { re, im }
        }

        fn from_polar(r: f32, theta: f32) -> Self {
            Self {
                re: r * theta.cos(),
                im: r * theta.sin(),
            }
        }

        fn add(self, other: Self) -> Self {
            Self {
                re: self.re + other.re,
                im: self.im + other.im,
            }
        }

        fn sub(self, other: Self) -> Self {
            Self {
                re: self.re - other.re,
                im: self.im - other.im,
            }
        }

        fn mul(self, other: Self) -> Self {
            Self {
                re: self.re * other.re - self.im * other.im,
                im: self.re * other.im + self.im * other.re,
            }
        }

        fn scale(self, s: f32) -> Self {
            Self {
                re: self.re * s,
                im: self.im * s,
            }
        }

        fn magnitude(self) -> f32 {
            (self.re * self.re + self.im * self.im).sqrt()
        }
    }

    /// CPU reference implementation of 1D FFT (Cooley-Tukey radix-2)
    fn fft_1d_reference(data: &mut [Complex], forward: bool) {
        let n = data.len();
        assert!(n.is_power_of_two(), "FFT size must be power of 2");

        // Bit-reversal permutation
        let mut j = 0;
        for i in 0..n {
            if i < j {
                data.swap(i, j);
            }
            let mut m = n / 2;
            while m > 0 && j >= m {
                j -= m;
                m /= 2;
            }
            j += m;
        }

        // Cooley-Tukey FFT
        let sign = if forward { -1.0 } else { 1.0 };
        let mut len = 2;
        while len <= n {
            let half_len = len / 2;
            let angle_step = sign * 2.0 * PI / len as f32;

            for start in (0..n).step_by(len) {
                let mut angle = 0.0;
                for k in 0..half_len {
                    let twiddle = Complex::from_polar(1.0, angle);
                    let even = data[start + k];
                    let odd = data[start + k + half_len].mul(twiddle);

                    data[start + k] = even.add(odd);
                    data[start + k + half_len] = even.sub(odd);

                    angle += angle_step;
                }
            }
            len *= 2;
        }

        // Normalize for inverse FFT
        if !forward {
            let scale = 1.0 / n as f32;
            for x in data.iter_mut() {
                *x = x.scale(scale);
            }
        }
    }

    /// CPU reference implementation of 2D FFT
    fn fft_2d_reference(data: &mut [Complex], width: usize, height: usize, forward: bool) {
        // Transform rows
        for row in 0..height {
            let start = row * width;
            let mut row_data: Vec<Complex> = data[start..start + width].to_vec();
            fft_1d_reference(&mut row_data, forward);
            data[start..start + width].copy_from_slice(&row_data);
        }

        // Transform columns
        for col in 0..width {
            let mut col_data: Vec<Complex> = (0..height).map(|row| data[row * width + col]).collect();
            fft_1d_reference(&mut col_data, forward);
            for (row, &val) in col_data.iter().enumerate() {
                data[row * width + col] = val;
            }
        }
    }

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
    fn test_rapid_params_default() {
        let params = RapidParams::default();
        assert!(!params.enabled);
        assert_eq!(params.blur_type, BlurType::Motion);
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

    // ========================================================================
    // FFT Reference Tests
    // ========================================================================

    #[test]
    fn test_fft_1d_reference_impulse() {
        // FFT of impulse [1, 0, 0, 0] should be [1, 1, 1, 1]
        let mut data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];

        fft_1d_reference(&mut data, true);

        for (i, x) in data.iter().enumerate() {
            assert!(
                (x.re - 1.0).abs() < 1e-5 && x.im.abs() < 1e-5,
                "FFT of impulse failed at index {}: got ({}, {})",
                i, x.re, x.im
            );
        }
    }

    #[test]
    fn test_fft_1d_reference_dc() {
        // FFT of constant [1, 1, 1, 1] should be [4, 0, 0, 0]
        let mut data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

        fft_1d_reference(&mut data, true);

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
    fn test_fft_1d_reference_roundtrip() {
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
        fft_1d_reference(&mut data, true);
        fft_1d_reference(&mut data, false);

        for (i, (orig, result)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (orig.re - result.re).abs() < 1e-4 && (orig.im - result.im).abs() < 1e-4,
                "Roundtrip failed at index {}: expected ({}, {}), got ({}, {})",
                i, orig.re, orig.im, result.re, result.im
            );
        }
    }

    #[test]
    fn test_fft_1d_reference_sine() {
        // FFT of a single sine wave should have two peaks
        let n = 8;
        let freq = 1; // One cycle
        let mut data: Vec<Complex> = (0..n)
            .map(|i| {
                let angle = 2.0 * PI * freq as f32 * i as f32 / n as f32;
                Complex::new(angle.sin(), 0.0)
            })
            .collect();

        fft_1d_reference(&mut data, true);

        // For a sine wave, energy should be at indices 1 and n-1
        let peak1 = data[freq].magnitude();
        let peak2 = data[n - freq].magnitude();

        assert!(peak1 > 3.0, "Peak at index {} should be significant", freq);
        assert!(peak2 > 3.0, "Peak at index {} should be significant", n - freq);

        // Other frequencies should be near zero
        assert!(data[0].magnitude() < 1e-5, "DC should be zero for sine");
    }

    #[test]
    fn test_fft_2d_reference_roundtrip() {
        // 2D FFT roundtrip test
        let width = 4;
        let height = 4;
        let original: Vec<Complex> = (0..width * height)
            .map(|i| Complex::new((i + 1) as f32, 0.0))
            .collect();

        let mut data = original.clone();
        fft_2d_reference(&mut data, width, height, true);
        fft_2d_reference(&mut data, width, height, false);

        for (i, (orig, result)) in original.iter().zip(data.iter()).enumerate() {
            assert!(
                (orig.re - result.re).abs() < 1e-3 && (orig.im - result.im).abs() < 1e-3,
                "2D roundtrip failed at index {}: expected ({}, {}), got ({}, {})",
                i, orig.re, orig.im, result.re, result.im
            );
        }
    }

    #[test]
    fn test_fft_2d_reference_separable() {
        // Test that 2D FFT is separable (row FFT then col FFT = 2D FFT)
        let width = 4;
        let height = 4;
        let original: Vec<Complex> = (0..width * height)
            .map(|i| Complex::new(((i * 7) % 13) as f32, 0.0))
            .collect();

        // Method 1: Direct 2D FFT
        let mut data1 = original.clone();
        fft_2d_reference(&mut data1, width, height, true);

        // Method 2: Row FFT then column FFT (which is what fft_2d_reference does)
        // This test verifies the implementation is correct by checking consistency
        let mut data2 = original.clone();

        // Row transforms
        for row in 0..height {
            let start = row * width;
            let mut row_data: Vec<Complex> = data2[start..start + width].to_vec();
            fft_1d_reference(&mut row_data, true);
            data2[start..start + width].copy_from_slice(&row_data);
        }

        // Column transforms
        for col in 0..width {
            let mut col_data: Vec<Complex> = (0..height).map(|row| data2[row * width + col]).collect();
            fft_1d_reference(&mut col_data, true);
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
        fft_1d_reference(&mut freq, true);
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
            blur_type: BlurType::Gaussian,
            gaussian_sigma: 1.5,
            lambda: 0.01,
            strength: 1.0,
            ..Default::default()
        };

        let out = deconv
            .deconvolve_image(&device, &queue, &input, &params)
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
            fft_1d_reference(&mut row, true);
            for v in &row[k_lo..=k_hi] {
                energy += (v.re as f64).powi(2) + (v.im as f64).powi(2);
            }
        }
        energy / h as f64
    }

    /// Exit test: the edge taper must collapse FFT wrap-seam banding by
    /// an order of magnitude versus raw zero-pad. Width is an exact power of
    /// two (no pad headroom -> PSF-consistent border taper) while height pads
    /// 200 -> 256 (reflect-101 mirror margins), so both mechanisms run.
    #[test]
    fn test_gpu_edge_taper_reduces_banding() {
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
            blur_type: BlurType::Motion,
            motion_length: blur_len as f32,
            motion_angle: 0.0,
            lambda: 0.002,
            strength: 1.0,
            edge_taper: false,
            ..Default::default()
        };
        let tapered_params = RapidParams { edge_taper: true, ..base_params };

        let base = deconv
            .deconvolve_image(&device, &queue, &input, &base_params)
            .expect("baseline deconvolve failed");
        let tapered = deconv
            .deconvolve_image(&device, &queue, &input, &tapered_params)
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
        for x in 64..192 {
            let d = (out_rgb.get_pixel(x, y)[0] - in_rgb.get_pixel(x, y)[0]).abs();
            max_dev = max_dev.max(d);
        }
        assert!(max_dev < 0.05, "interior drifted by {max_dev} after tapered deconvolution");
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

        // Degenerate scales hit the kernel floors instead of collapsing.
        let tiny = p.scaled(0.001);
        assert!(tiny.motion_length >= 1.0);
        assert!(tiny.defocus_radius >= 0.5);
        assert!(tiny.gaussian_sigma >= 0.3);

        let unit = p.scaled(1.0);
        assert!((unit.motion_length - p.motion_length).abs() < 1e-6);
        assert!((unit.defocus_radius - p.defocus_radius).abs() < 1e-6);
    }

    #[test]
    fn test_psf_axis_projection() {
        // Horizontal motion: box along x, identity along y.
        let p = RapidParams {
            blur_type: BlurType::Motion,
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
            RapidParams { blur_type: BlurType::Defocus, defocus_radius: 5.0, ..Default::default() },
            RapidParams { blur_type: BlurType::Gaussian, gaussian_sigma: 2.0, ..Default::default() },
        ] {
            for axis in 0..2 {
                let k = psf_axis_projection(&p, axis);
                assert!(k.len() > 1);
                assert!((k.iter().sum::<f32>() - 1.0).abs() < 1e-4);
            }
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
        let out = apply_blur_recovery_scaled(std::borrow::Cow::Owned(img), &adjustments, 0.5);
        assert_eq!((out.width(), out.height()), (200, 150));
    }
}
