// ============================================================================
// RAPID Common Utilities - Complex Number Operations
// ============================================================================
//
// This module provides complex number arithmetic for FFT-based deconvolution.
// Complex numbers are represented as vec2<f32>(real, imaginary).
//
// Author: RapidRAW Mod1 Team
// Date: January 2026
// ============================================================================

// Complex number type alias for clarity
// Note: WGSL doesn't support type aliases in all contexts, so we use vec2<f32>
// directly in function signatures, but document the intent here.
// Complex = vec2<f32> where .x = real, .y = imaginary

// Mathematical constants
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;
const HALF_PI: f32 = 1.57079632679489661923;
const INV_PI: f32 = 0.31830988618379067154;
const INV_TWO_PI: f32 = 0.15915494309189533577;

// Numerical stability constants
const EPSILON: f32 = 1e-10;
const MAGNITUDE_FLOOR: f32 = 1e-4;

// ============================================================================
// Basic Complex Arithmetic
// ============================================================================

/// Complex addition: (a + bi) + (c + di) = (a+c) + (b+d)i
fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a + b;
}

/// Complex subtraction: (a + bi) - (c + di) = (a-c) + (b-d)i
fn c_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a - b;
}

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
/// More efficient than c_mag when you only need the squared magnitude.
fn c_mag_sq(a: vec2<f32>) -> f32 {
    return dot(a, a);
}

/// Complex magnitude: |a + bi| = sqrt(a² + b²)
fn c_mag(a: vec2<f32>) -> f32 {
    return length(a);
}

/// Complex division by real: (a + bi) / r = (a/r) + (b/r)i
fn c_div_real(a: vec2<f32>, r: f32) -> vec2<f32> {
    return a / r;
}

/// Complex multiplication by real: (a + bi) * r = (a*r) + (b*r)i
fn c_mul_real(a: vec2<f32>, r: f32) -> vec2<f32> {
    return a * r;
}

/// Complex division: (a + bi) / (c + di)
/// Uses the formula: multiply by conjugate of denominator
fn c_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let denom = c_mag_sq(b);
    // Avoid division by zero
    let safe_denom = max(denom, EPSILON);
    return vec2<f32>(
        (a.x * b.x + a.y * b.y) / safe_denom,
        (a.y * b.x - a.x * b.y) / safe_denom
    );
}

/// Safe complex division with explicit epsilon
fn c_div_safe(a: vec2<f32>, b: vec2<f32>, epsilon: f32) -> vec2<f32> {
    let denom = c_mag_sq(b);
    let safe_denom = max(denom, epsilon);
    return vec2<f32>(
        (a.x * b.x + a.y * b.y) / safe_denom,
        (a.y * b.x - a.x * b.y) / safe_denom
    );
}

// ============================================================================
// Complex Exponential and Trigonometric Functions
// ============================================================================

/// Complex exponential of pure imaginary: e^(ix) = cos(x) + i*sin(x)
/// This is Euler's formula, fundamental to FFT.
fn c_exp_i(x: f32) -> vec2<f32> {
    return vec2<f32>(cos(x), sin(x));
}

/// General complex exponential: e^(a + bi) = e^a * (cos(b) + i*sin(b))
fn c_exp(z: vec2<f32>) -> vec2<f32> {
    let r = exp(z.x);
    return vec2<f32>(r * cos(z.y), r * sin(z.y));
}

/// Complex from polar form: r * e^(i*theta) = r*cos(theta) + i*r*sin(theta)
fn c_from_polar(r: f32, theta: f32) -> vec2<f32> {
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

/// Convert complex to polar form: returns (magnitude, phase)
fn c_to_polar(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(c_mag(z), atan2(z.y, z.x));
}

// ============================================================================
// FFT Twiddle Factors
// ============================================================================

/// Forward FFT twiddle factor: W_N^k = e^(-2*pi*i*k/N)
/// Used in forward (analysis) FFT.
fn twiddle(k: u32, N: u32) -> vec2<f32> {
    let angle = -TWO_PI * f32(k) / f32(N);
    return c_exp_i(angle);
}

/// Inverse FFT twiddle factor: W_N^(-k) = e^(2*pi*i*k/N)
/// Used in inverse (synthesis) FFT.
fn twiddle_inv(k: u32, N: u32) -> vec2<f32> {
    let angle = TWO_PI * f32(k) / f32(N);
    return c_exp_i(angle);
}

/// Precomputed twiddle factor with direction parameter
/// direction: 1 for forward FFT, -1 for inverse FFT
fn twiddle_dir(k: u32, N: u32, direction: i32) -> vec2<f32> {
    let sign = f32(direction);
    let angle = -sign * TWO_PI * f32(k) / f32(N);
    return c_exp_i(angle);
}

// ============================================================================
// Special Functions for PSF Generation
// ============================================================================

/// Sinc function: sin(pi*x) / (pi*x)
/// Handles the removable singularity at x=0.
fn sinc(x: f32) -> f32 {
    if (abs(x) < EPSILON) {
        // Taylor series: sinc(x) ≈ 1 - (pi*x)^2/6 for small x
        return 1.0;
    }
    let px = PI * x;
    return sin(px) / px;
}

/// Safe sinc function with magnitude floor
/// Prevents exact zeros which cause instability in Wiener deconvolution.
fn sinc_safe(x: f32) -> f32 {
    let s = sinc(x);
    // Preserve sign but enforce minimum magnitude
    if (abs(s) < MAGNITUDE_FLOOR) {
        return select(-MAGNITUDE_FLOOR, MAGNITUDE_FLOOR, s >= 0.0);
    }
    return s;
}

/// Bessel function J1 approximation
/// Uses rational approximation for |x| < 8 and asymptotic expansion for |x| >= 8.
/// Accuracy: ~1e-7 relative error.
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

/// Jinc function: 2*J1(x)/x (used for defocus PSF)
/// Has removable singularity at x=0 where jinc(0) = 1.
fn jinc(x: f32) -> f32 {
    if (abs(x) < EPSILON) {
        return 1.0;
    }
    return 2.0 * bessel_j1(x) / x;
}

/// Safe jinc function with magnitude floor
fn jinc_safe(x: f32) -> f32 {
    let j = jinc(x);
    if (abs(j) < MAGNITUDE_FLOOR) {
        return select(-MAGNITUDE_FLOOR, MAGNITUDE_FLOOR, j >= 0.0);
    }
    return j;
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Bit reversal for FFT (used in Cooley-Tukey, not needed for Stockham)
/// Kept for reference/debugging.
fn bit_reverse(x: u32, log2n: u32) -> u32 {
    var result: u32 = 0u;
    var val = x;
    for (var i: u32 = 0u; i < log2n; i++) {
        result = (result << 1u) | (val & 1u);
        val = val >> 1u;
    }
    return result;
}

/// Check if a number is a power of 2
fn is_power_of_2(x: u32) -> bool {
    return (x != 0u) && ((x & (x - 1u)) == 0u);
}

/// Compute log2 of a power of 2
/// Assumes input is a power of 2; undefined behavior otherwise.
fn log2_u32(x: u32) -> u32 {
    var result: u32 = 0u;
    var val = x;
    while (val > 1u) {
        val = val >> 1u;
        result++;
    }
    return result;
}

/// Linear interpolation for complex numbers
fn c_lerp(a: vec2<f32>, b: vec2<f32>, t: f32) -> vec2<f32> {
    return a + (b - a) * t;
}

/// Clamp complex magnitude while preserving phase
fn c_clamp_mag(z: vec2<f32>, max_mag: f32) -> vec2<f32> {
    let mag = c_mag(z);
    if (mag > max_mag && mag > EPSILON) {
        return z * (max_mag / mag);
    }
    return z;
}

// ============================================================================
// Debug/Visualization Helpers
// ============================================================================

/// Convert complex to RGB for visualization (magnitude as brightness, phase as hue)
fn c_to_rgb(z: vec2<f32>, max_mag: f32) -> vec3<f32> {
    let mag = c_mag(z) / max_mag;
    let phase = atan2(z.y, z.x);

    // HSV to RGB with hue from phase, saturation = 1, value = magnitude
    let h = (phase + PI) * INV_TWO_PI;  // 0 to 1
    let s = 1.0;
    let v = clamp(mag, 0.0, 1.0);

    // HSV to RGB conversion
    let c = v * s;
    let hp = h * 6.0;
    let x = c * (1.0 - abs(fract(hp / 2.0) * 2.0 - 1.0));
    let m = v - c;

    var rgb: vec3<f32>;
    let hi = u32(hp) % 6u;
    switch (hi) {
        case 0u: { rgb = vec3<f32>(c, x, 0.0); }
        case 1u: { rgb = vec3<f32>(x, c, 0.0); }
        case 2u: { rgb = vec3<f32>(0.0, c, x); }
        case 3u: { rgb = vec3<f32>(0.0, x, c); }
        case 4u: { rgb = vec3<f32>(x, 0.0, c); }
        default: { rgb = vec3<f32>(c, 0.0, x); }
    }

    return rgb + m;
}
