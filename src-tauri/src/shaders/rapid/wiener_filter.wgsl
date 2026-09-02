// ============================================================================
// RAPID Wiener Filter - Regularized Pseudoinverse Deconvolution
// ============================================================================
//
// Implements the Wiener deconvolution filter:
//   F̂(u,v) = G(u,v) · H*(u,v) / (|H(u,v)|² + λ)
//
// Where:
//   G = degraded image spectrum (input)
//   H = PSF spectrum
//   H* = complex conjugate of H
//   λ = regularization parameter (noise-to-signal ratio)
//   F̂ = estimated original image spectrum (output)
//
// The regularization parameter λ controls the trade-off between:
//   - Small λ: More sharpening, but amplifies noise
//   - Large λ: Less noise amplification, but less sharpening
//
// Author: RapidRAW Mod1 Team
// Date: January 2026
// ============================================================================

// Numerical stability constant
const EPSILON: f32 = 1e-10;
// Dead-band threshold: MAGNITUDE_FLOOR² from psf_generate.wgsl.
// Bins below it may only raise lambda, never drop it below base. The floored
// hardness-0 motion envelope squares to this exact f32 value, so the retained
// >= comparison keeps its shipped live-side behavior bit-for-bit.
const DEAD_BAND_POWER: f32 = 0.0225;

// ============================================================================
// Complex Number Operations
// ============================================================================

/// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

/// Complex conjugate: (a + bi)* = a - bi
fn c_conj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}

/// Complex magnitude squared: |a + bi|² = a² + b²
fn c_mag_sq(a: vec2<f32>) -> f32 {
    return dot(a, a);
}

/// Complex division by real: (a + bi) / r
fn c_div_real(a: vec2<f32>, r: f32) -> vec2<f32> {
    return a / r;
}

/// Complex addition
fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a + b;
}

/// Complex scalar multiplication
fn c_scale(a: vec2<f32>, s: f32) -> vec2<f32> {
    return a * s;
}

// ============================================================================
// Wiener Filter Parameters
// ============================================================================

struct WienerParams {
    width: u32,
    height: u32,
    lambda: f32,           // Base regularization parameter
    strength: f32,         // Blend factor (0 = original, 1 = full deconvolution)
    noise_floor: f32,      // Minimum denominator value
    adaptive: u32,         // 0 = fixed lambda, 1 = adaptive
    _pad0: u32,            // Keeps the struct at 32 bytes, matching the
    _pad1: u32,            // Rust WienerParams' _pad: [u32; 2].
}

@group(0) @binding(0) var image_freq: texture_2d<f32>;    // G(u,v) - degraded image spectrum
@group(0) @binding(1) var psf_freq: texture_2d<f32>;      // H(u,v) - PSF spectrum
@group(0) @binding(2) var output_freq: texture_storage_2d<rg32float, write>;  // F̂(u,v) - result
@group(0) @binding(3) var<uniform> params: WienerParams;

// ============================================================================
// Standard Wiener Filter
// ============================================================================

/// Standard Wiener deconvolution filter
/// F̂ = G · H* / (|H|² + λ)
@compute @workgroup_size(16, 16, 1)
fn wiener_deconvolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<u32>(gid.xy);

    // Bounds check
    if (coord.x >= params.width || coord.y >= params.height) {
        return;
    }

    // Load frequency domain data
    let G = textureLoad(image_freq, coord, 0).rg;   // Image spectrum
    let H = textureLoad(psf_freq, coord, 0).rg;     // PSF spectrum

    // Compute H* (complex conjugate)
    let H_conj = c_conj(H);

    // Compute |H|² = H · H*
    let H_power = c_mag_sq(H);

    // Regularized denominator: |H|² + λ
    // Lower lambda = more aggressive deconvolution (but more noise/ringing)
    // Higher lambda = smoother result (but less sharpening)
    let effective_lambda = max(params.lambda, 0.0001);
    let denominator = max(H_power + effective_lambda, params.noise_floor);

    // Wiener filter: W = H* / (|H|² + λ)
    let W = c_div_real(H_conj, denominator);

    // Apply filter: F̂ = G · W
    let F_hat = c_mul(G, W);

    // Blend with original based on strength parameter
    // At strength=0, output equals input (G)
    // At strength=1, output equals full deconvolution (F_hat)
    let result = c_add(
        c_scale(G, 1.0 - params.strength),
        c_scale(F_hat, params.strength)
    );

    textureStore(output_freq, coord, vec4<f32>(result, 0.0, 1.0));
}

// ============================================================================
// Adaptive Wiener Filter
// ============================================================================

/// Adaptive Wiener filter with local variance-based regularization
/// Estimates local noise/signal ratio and adjusts λ accordingly
///
/// Higher λ in low-signal (noisy) regions, lower λ in high-signal regions
@compute @workgroup_size(16, 16, 1)
fn wiener_adaptive(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<u32>(gid.xy);

    if (coord.x >= params.width || coord.y >= params.height) {
        return;
    }

    let G = textureLoad(image_freq, coord, 0).rg;
    let H = textureLoad(psf_freq, coord, 0).rg;

    // Estimate local signal power from neighbors
    var local_power: f32 = 0.0;
    let window: i32 = 2;
    var count: f32 = 0.0;

    for (var dy: i32 = -window; dy <= window; dy++) {
        for (var dx: i32 = -window; dx <= window; dx++) {
            let nc = vec2<i32>(i32(coord.x) + dx, i32(coord.y) + dy);
            if (nc.x >= 0 && nc.x < i32(params.width) &&
                nc.y >= 0 && nc.y < i32(params.height)) {
                let neighbor = textureLoad(image_freq, vec2<u32>(nc), 0).rg;
                local_power += c_mag_sq(neighbor);
                count += 1.0;
            }
        }
    }
    local_power /= max(count, 1.0);

    // Estimate local SNR
    let signal_power = c_mag_sq(G);
    let noise_estimate = max(local_power - signal_power, EPSILON);
    let snr_estimate = signal_power / noise_estimate;

    // Adaptive λ: higher in low-SNR regions. Both surviving modes have a dead
    // band, so the binary rule always applies: destroyed bins may only raise λ,
    // never drop it below base.
    let base_lambda = max(params.lambda, 0.0001);
    let snr_clamped = clamp(snr_estimate, 0.1, 10.0);
    let h2 = c_mag_sq(H);
    let lambda_divisor = select(min(snr_clamped, 1.0), snr_clamped, h2 >= DEAD_BAND_POWER);
    let adaptive_lambda = base_lambda / lambda_divisor;

    // Standard Wiener filter with adaptive λ
    let H_conj = c_conj(H);
    let H_power = h2;
    let denominator = max(H_power + adaptive_lambda, params.noise_floor);
    let W = c_div_real(H_conj, denominator);
    let F_hat = c_mul(G, W);

    // Blend with original
    let result = c_add(
        c_scale(G, 1.0 - params.strength),
        c_scale(F_hat, params.strength)
    );

    textureStore(output_freq, coord, vec4<f32>(result, 0.0, 1.0));
}
