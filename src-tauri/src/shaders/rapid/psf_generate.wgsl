// ============================================================================
// RAPID PSF Generation - Analytical Frequency Domain
// ============================================================================
//
// Generate Point Spread Function (PSF) spectrum directly in frequency domain.
// This is more accurate and efficient than spatial-domain FFT for known blur models.
//
// Supported blur types:
// - Motion blur: sinc function along motion direction
// - Defocus blur: jinc function (Bessel J1)
// - Gaussian blur: Gaussian (self-similar under Fourier transform)
//
// Author: RapidRAW Mod1 Team
// Date: January 2026
// ============================================================================

// Mathematical constants
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;

// Numerical stability constants
const EPSILON: f32 = 1e-10;
// Higher floor prevents over-amplification at sinc zeros (which cause stripe artifacts)
const MAGNITUDE_FLOOR: f32 = 0.15;

// ============================================================================
// PSF Parameters
// ============================================================================

struct PSFParams {
    width: u32,
    height: u32,
    blur_type: u32,      // 0 = motion, 1 = defocus, 2 = gaussian
    motion_length: f32,  // In pixels
    motion_angle: f32,   // Degrees
    defocus_radius: f32, // In pixels
    gaussian_sigma: f32, // In pixels
    _pad: f32,
}

// Note on frequency scaling:
// The normalized frequency u,v ∈ [-0.5, 0.5] represents cycles/pixel.
// To convert pixel-based blur parameters to frequency domain:
// - Motion blur: sinc(L * f) where f is in cycles/pixel, so we use L directly
//   but f = u (since u is already in cycles/pixel for padded size)
// - Actually, u represents fraction of Nyquist, so real frequency = u * (width/2) cycles
// - For a blur of L pixels, the first zero of sinc is at f = 1/L cycles/pixel
// - In normalized coords, this zero should be at u = 1/L / (1/2) = 2/L
//
// CORRECTED: The frequency in cycles per pixel is u (when u ∈ [-0.5, 0.5])
// So for motion blur: H(u) = sinc(L * u) is CORRECT for normalized freq
// The issue is we need to scale by actual pixel frequency: u_pixels = u * width

@group(0) @binding(0) var output_tex: texture_storage_2d<rg32float, write>;
@group(0) @binding(1) var<uniform> params: PSFParams;

// ============================================================================
// Special Functions
// ============================================================================

/// Sinc function: sin(pi*x) / (pi*x)
/// Has removable singularity at x=0 where sinc(0) = 1
fn sinc(x: f32) -> f32 {
    if (abs(x) < EPSILON) {
        return 1.0;
    }
    let px = PI * x;
    return sin(px) / px;
}

/// Safe sinc with magnitude floor to prevent Wiener instability
fn sinc_safe(x: f32) -> f32 {
    let s = sinc(x);
    if (abs(s) < MAGNITUDE_FLOOR) {
        return select(-MAGNITUDE_FLOOR, MAGNITUDE_FLOOR, s >= 0.0);
    }
    return s;
}

/// Bessel function J1 approximation
/// Uses rational approximation for |x| < 8 and asymptotic expansion for |x| >= 8
fn bessel_j1(x: f32) -> f32 {
    let ax = abs(x);

    if (ax < 8.0) {
        // Rational approximation for small arguments
        let y = x * x;
        let ans1 = x * (72362614232.0 + y * (-7895059235.0 + y * (242396853.1
            + y * (-2972611.439 + y * (15704.48260 + y * (-30.16036606))))));
        let ans2 = 144725228442.0 + y * (2300535178.0 + y * (18583304.74
            + y * (99447.43394 + y * (376.9991397 + y * 1.0))));
        return ans1 / ans2;
    } else {
        // Asymptotic expansion for large arguments
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356194491;  // ax - 3*pi/4
        let ans1 = 1.0 + y * (0.183105e-2 + y * (-0.3516396496e-4
            + y * (0.2457520174e-5 + y * (-0.240337019e-6))));
        let ans2 = 0.04687499995 + y * (-0.2002690873e-3
            + y * (0.8449199096e-5 + y * (-0.88228987e-6 + y * 0.105787412e-6)));
        var ans = sqrt(0.636619772 / ax) * (cos(xx) * ans1 - z * sin(xx) * ans2);
        if (x < 0.0) {
            ans = -ans;
        }
        return ans;
    }
}

/// Jinc function: 2*J1(x)/x
/// Used for defocus (pillbox) PSF in frequency domain
fn jinc(x: f32) -> f32 {
    if (abs(x) < EPSILON) {
        return 1.0;
    }
    return 2.0 * bessel_j1(x) / x;
}

/// Safe jinc with magnitude floor
fn jinc_safe(x: f32) -> f32 {
    let j = jinc(x);
    if (abs(j) < MAGNITUDE_FLOOR) {
        return select(-MAGNITUDE_FLOOR, MAGNITUDE_FLOOR, j >= 0.0);
    }
    return j;
}

// ============================================================================
// PSF Spectrum Functions
// ============================================================================

/// Motion blur PSF in frequency domain
/// Uses Gaussian-windowed approach for stability instead of pure sinc
///
/// Pure sinc has many zeros which cause severe ringing artifacts.
/// Instead, we use a Gaussian envelope that smoothly attenuates high frequencies
/// along the motion direction, which is more stable for deconvolution.
fn motion_blur_spectrum(u: f32, v: f32, length: f32, angle_deg: f32) -> vec2<f32> {
    let angle_rad = angle_deg * PI / 180.0;
    let cos_a = cos(angle_rad);
    let sin_a = sin(angle_rad);

    // Frequency component along motion direction
    let freq_along = u * cos_a + v * sin_a;

    // Use Gaussian envelope instead of sinc to avoid zeros
    // The sigma is proportional to blur length - longer blur = narrower frequency response
    // For blur length L, effective sigma in frequency domain is ~1/L
    let sigma_freq = 1.0 / max(length, 1.0);

    // Gaussian: exp(-0.5 * (f/sigma)^2)
    let magnitude = exp(-0.5 * (freq_along * freq_along) / (sigma_freq * sigma_freq));

    // Ensure minimum magnitude for stability
    let safe_magnitude = max(magnitude, MAGNITUDE_FLOOR);

    return vec2<f32>(safe_magnitude, 0.0);
}

/// Defocus (pillbox/disk) PSF in frequency domain
/// H(u,v) = 2*J1(2*pi*r*rho) / (2*pi*r*rho) = jinc(2*pi*r*rho)
/// where rho = sqrt(u^2 + v^2)
///
/// The pillbox PSF has circular symmetry, producing ring-shaped
/// zeros in the frequency domain at the roots of J1.
///
/// For stable deconvolution, we limit the effective radius to avoid
/// too many ring-shaped zeros which cause severe artifacts.
fn defocus_blur_spectrum(u: f32, v: f32, radius: f32) -> vec2<f32> {
    let rho = sqrt(u * u + v * v);

    // Use the full blur radius for proper deconvolution
    // The safe jinc function prevents exact zeros that cause instability
    let effective_radius = radius;
    let arg = TWO_PI * effective_radius * rho;

    // H(u,v) = jinc(2*pi*r*rho)
    let magnitude = jinc_safe(arg);

    return vec2<f32>(magnitude, 0.0);
}

/// Gaussian blur PSF in frequency domain
/// H(u,v) = exp(-2*pi^2*sigma^2*(u^2 + v^2))
///
/// The Gaussian is self-similar under Fourier transform.
/// A Gaussian in spatial domain transforms to a Gaussian in frequency domain.
/// This PSF has no zeros, making it the most stable for deconvolution.
///
/// Gaussian is the most stable blur type for deconvolution since it has
/// no zeros - the transfer function decays smoothly to zero.
fn gaussian_blur_spectrum(u: f32, v: f32, sigma: f32) -> vec2<f32> {
    let freq_sq = u * u + v * v;

    // Limit effective sigma - very large sigma attenuates high frequencies too much
    let effective_sigma = min(sigma, 8.0);

    // H(u,v) = exp(-2*pi^2*sigma^2*(u^2 + v^2))
    let magnitude = exp(-2.0 * PI * PI * effective_sigma * effective_sigma * freq_sq);

    // Gaussian is always positive, no need for magnitude floor
    // But clamp to reasonable minimum for numerical stability
    let safe_magnitude = max(magnitude, MAGNITUDE_FLOOR);

    return vec2<f32>(safe_magnitude, 0.0);
}

// ============================================================================
// Main PSF Generation Kernel
// ============================================================================

@compute @workgroup_size(16, 16, 1)
fn generate_psf_spectrum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<u32>(gid.xy);

    // Bounds check
    if (coord.x >= params.width || coord.y >= params.height) {
        return;
    }

    // Convert to centered frequency coordinates [-0.5, 0.5]
    // This is the standard FFT frequency layout before fftshift
    var u = f32(coord.x) / f32(params.width);
    var v = f32(coord.y) / f32(params.height);

    // Wrap to [-0.5, 0.5] range (centered frequencies)
    if (u > 0.5) { u -= 1.0; }
    if (v > 0.5) { v -= 1.0; }

    // DEBUG: Use identity PSF to test FFT pipeline
    // If this produces the original image, FFT works and issue is in PSF/Wiener
    // If this produces garbage, FFT implementation is broken
    let DEBUG_IDENTITY_PSF = false;

    var H: vec2<f32>;

    if (DEBUG_IDENTITY_PSF) {
        // Identity PSF: H = 1.0 everywhere
        // Wiener filter becomes: F_hat = G * 1 / (1 + λ) ≈ G (scaled)
        H = vec2<f32>(1.0, 0.0);
    } else {
        // Generate PSF spectrum based on blur type
        switch (params.blur_type) {
            case 0u: {
                // Motion blur
                H = motion_blur_spectrum(u, v, params.motion_length, params.motion_angle);
            }
            case 1u: {
                // Defocus blur
                H = defocus_blur_spectrum(u, v, params.defocus_radius);
            }
            case 2u: {
                // Gaussian blur
                H = gaussian_blur_spectrum(u, v, params.gaussian_sigma);
            }
            default: {
                // Identity (no blur)
                H = vec2<f32>(1.0, 0.0);
            }
        }
    }

    textureStore(output_tex, coord, vec4<f32>(H, 0.0, 1.0));
}

// ============================================================================
// Alternative: Generate PSF in spatial domain (for debugging/comparison)
// ============================================================================

struct SpatialPSFParams {
    width: u32,
    height: u32,
    blur_type: u32,
    motion_length: f32,
    motion_angle: f32,
    defocus_radius: f32,
    gaussian_sigma: f32,
    _pad: f32,
}

@group(0) @binding(0) var spatial_output_tex: texture_storage_2d<rg32float, write>;
@group(0) @binding(1) var<uniform> spatial_params: SpatialPSFParams;

/// Generate spatial domain PSF (for debugging/visualization)
@compute @workgroup_size(16, 16, 1)
fn generate_psf_spatial(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<u32>(gid.xy);

    if (coord.x >= spatial_params.width || coord.y >= spatial_params.height) {
        return;
    }

    // Center coordinates
    let cx = f32(spatial_params.width) / 2.0;
    let cy = f32(spatial_params.height) / 2.0;
    let x = f32(coord.x) - cx;
    let y = f32(coord.y) - cy;

    var value: f32 = 0.0;

    switch (spatial_params.blur_type) {
        case 0u: {
            // Motion blur - line segment
            let angle_rad = spatial_params.motion_angle * PI / 180.0;
            let cos_a = cos(angle_rad);
            let sin_a = sin(angle_rad);
            let half_len = spatial_params.motion_length / 2.0;

            // Distance from point to line through origin at angle
            let perp_dist = abs(-x * sin_a + y * cos_a);
            let along_dist = x * cos_a + y * sin_a;

            // Point is on the blur line if perpendicular distance is small
            // and along distance is within half length
            if (perp_dist < 0.5 && abs(along_dist) <= half_len) {
                value = 1.0 / spatial_params.motion_length;
            }
        }
        case 1u: {
            // Defocus - disk/pillbox
            let r = sqrt(x * x + y * y);
            if (r <= spatial_params.defocus_radius) {
                let area = PI * spatial_params.defocus_radius * spatial_params.defocus_radius;
                value = 1.0 / area;
            }
        }
        case 2u: {
            // Gaussian
            let sigma = spatial_params.gaussian_sigma;
            let r_sq = x * x + y * y;
            value = exp(-r_sq / (2.0 * sigma * sigma)) / (TWO_PI * sigma * sigma);
        }
        default: {
            // Delta function (identity)
            if (coord.x == u32(cx) && coord.y == u32(cy)) {
                value = 1.0;
            }
        }
    }

    // Store as complex (real, 0)
    textureStore(spatial_output_tex, coord, vec4<f32>(value, 0.0, 0.0, 1.0));
}
