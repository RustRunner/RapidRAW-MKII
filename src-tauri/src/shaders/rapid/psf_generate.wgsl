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
    active_modes: u32,   // Bitmask: bit0 = motion, bit1 = defocus, bit2 = gaussian
    motion_length: f32,  // In pixels
    motion_angle: f32,   // Degrees
    defocus_radius: f32, // In pixels
    gaussian_sigma: f32, // In pixels
    hardness: f32,       // OTF shape. Motion: 0 = Gaussian envelope, 1 = hard line.
                         // Defocus: 0 = floored jinc, 1 = raw jinc (true zeros).
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
/// Blend between a Gaussian envelope and the physical line-blur OTF.
///
/// A constant-velocity motion blur is a hard line segment whose OTF is a
/// signed sinc with true zeros at k/L; deconvolving with anything else
/// reconstructs each feature as a ghost pair at +/-L. At hardness 1 this is
/// that sinc, unfloored: the Wiener form |H|/(|H|^2+lambda) is self-limiting,
/// so the zeros yield zero gain instead of floor-fabricated amplification.
/// At hardness 0 it is the legacy zero-free Gaussian envelope (floored),
/// kept for A/B compatibility; intermediate values partially fill the
/// notches for motion with acceleration or shake-smoothed endpoints.
fn motion_blur_spectrum(u: f32, v: f32, length: f32, angle_deg: f32, hardness: f32) -> vec2<f32> {
    let angle_rad = angle_deg * PI / 180.0;
    let cos_a = cos(angle_rad);
    let sin_a = sin(angle_rad);

    // Frequency component along motion direction
    let freq_along = u * cos_a + v * sin_a;

    // Gaussian envelope with sigma ~1/L: narrower response for longer blur
    let sigma_freq = 1.0 / max(length, 1.0);
    let gauss = max(
        exp(-0.5 * (freq_along * freq_along) / (sigma_freq * sigma_freq)),
        MAGNITUDE_FLOOR
    );

    // Line-blur OTF: signed sinc, zeros at k/L, no floor
    let line = sinc(length * freq_along);

    let magnitude = mix(gauss, line, hardness);

    return vec2<f32>(magnitude, 0.0);
}

/// Defocus (pillbox/disk) PSF in frequency domain
/// H(u,v) = 2*J1(2*pi*r*rho) / (2*pi*r*rho) = jinc(2*pi*r*rho)
/// where rho = sqrt(u^2 + v^2)
///
/// The pillbox PSF has circular symmetry, producing ring-shaped
/// zeros in the frequency domain at the roots of J1.
///
/// Hardness picks the treatment of those zeros. At 0 the legacy floored
/// jinc clamps |H| to MAGNITUDE_FLOOR, which makes the Wiener filter
/// amplify bands the lens destroyed (fabricated content) and puts annular
/// discontinuities into the OTF — both transform to concentric spatial
/// rings around every feature. At 1 the raw signed jinc keeps true zeros:
/// the Wiener form |H|/(|H|^2+lambda) is self-limiting there, mirroring
/// the motion hard-line OTF at hardness 1.
fn defocus_blur_spectrum(u: f32, v: f32, radius: f32, hardness: f32) -> vec2<f32> {
    let rho = sqrt(u * u + v * v);
    let arg = TWO_PI * radius * rho;

    // H(u,v) = jinc(2*pi*r*rho)
    let magnitude = mix(jinc_safe(arg), jinc(arg), hardness);

    return vec2<f32>(magnitude, 0.0);
}

/// Gaussian blur PSF in frequency domain
/// H(u,v) = exp(-2*pi^2*sigma^2*(u^2 + v^2))
///
/// A Gaussian is self-similar under Fourier transform and has no zeros, so
/// its transfer function decays smoothly without a fabricated magnitude
/// floor. Wiener regularization bounds the inverse; Gaussian-active adaptive
/// modes use the smooth confidence policy in wiener_filter.wgsl.
fn gaussian_blur_spectrum(u: f32, v: f32, sigma: f32) -> vec2<f32> {
    let freq_sq = u * u + v * v;

    // Limit effective sigma - very large sigma attenuates high frequencies too much
    let effective_sigma = min(sigma, 8.0);

    // H(u,v) = exp(-2*pi^2*sigma^2*(u^2 + v^2))
    let magnitude = exp(-2.0 * PI * PI * effective_sigma * effective_sigma * freq_sq);

    return vec2<f32>(magnitude, 0.0);
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
        // Compound OTF: blurs that occur together convolve in image space,
        // so their transfer functions MULTIPLY here — any subset of the
        // three models forms one compound kernel inverted by the single
        // Wiener pass. An empty set falls through to the identity (the old
        // default arm). Per-component floors/hardness treatments are kept:
        // small compound magnitudes reduce the Wiener gain toward zero
        // rather than spiking it (|H|/(|H|^2+λ) is bounded by 1/(2√λ)).
        var mag = 1.0;
        if ((params.active_modes & 1u) != 0u) {
            mag *= motion_blur_spectrum(u, v, params.motion_length, params.motion_angle, params.hardness).x;
        }
        if ((params.active_modes & 2u) != 0u) {
            // Hardness pinned to 1.0: the raw signed jinc with true zeros.
            // The hardness slider and estimator only exist in the motion UI,
            // and a motion-fitted value must not half-floor the defocus OTF
            // — the floored jinc's rings are a defect here, not a look.
            // Pinned in the shader rather than parse so the one hardness
            // uniform can serve motion's slider and this pin simultaneously
            // when both modes are active.
            mag *= defocus_blur_spectrum(u, v, params.defocus_radius, 1.0).x;
        }
        if ((params.active_modes & 4u) != 0u) {
            mag *= gaussian_blur_spectrum(u, v, params.gaussian_sigma).x;
        }
        H = vec2<f32>(mag, 0.0);
    }

    textureStore(output_tex, coord, vec4<f32>(H, 0.0, 1.0));
}
