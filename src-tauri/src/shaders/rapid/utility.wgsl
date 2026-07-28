// ============================================================================
// RAPID Utility Operations - Windowing, Padding, Conversion
// ============================================================================
//
// This module provides utility operations for the RAPID pipeline:
// - Tukey (cosine-tapered) window for edge artifact reduction
// - Real to complex conversion with zero padding
// - Complex to real conversion with cropping
// - Channel extraction and combination
//
// Author: RapidRAW Mod1 Team
// Date: January 2026
// ============================================================================

// Mathematical constants
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;

// ============================================================================
// Utility Parameters
// ============================================================================

struct UtilityParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    window_alpha: f32,    // Tukey window parameter (0 = rectangular, 1 = Hann)
    normalize_factor: f32,
    channel: u32,         // 0=R, 1=G, 2=B for channel extraction
    _pad: u32,
}

// ============================================================================
// Window Functions
// ============================================================================

/// 1D Tukey (cosine-tapered) window
/// alpha = 0: rectangular window
/// alpha = 1: Hann window
/// alpha in between: cosine-tapered edges with flat center
fn tukey_window_1d(x: f32, N: f32, alpha: f32) -> f32 {
    if (alpha <= 0.0) {
        return 1.0;  // Rectangular window
    }

    if (alpha >= 1.0) {
        // Hann window
        return 0.5 * (1.0 - cos(TWO_PI * x / N));
    }

    let width = alpha * N / 2.0;

    if (x < width) {
        // Left taper
        return 0.5 * (1.0 - cos(PI * x / width));
    } else if (x > N - width) {
        // Right taper
        return 0.5 * (1.0 - cos(PI * (N - x) / width));
    }

    // Flat center
    return 1.0;
}

/// 2D Tukey window (separable)
fn tukey_window_2d(x: f32, y: f32, W: f32, H: f32, alpha: f32) -> f32 {
    return tukey_window_1d(x, W, alpha) * tukey_window_1d(y, H, alpha);
}

/// Hann window (special case of Tukey with alpha=1)
fn hann_window_1d(x: f32, N: f32) -> f32 {
    return 0.5 * (1.0 - cos(TWO_PI * x / N));
}

/// 2D Hann window
fn hann_window_2d(x: f32, y: f32, W: f32, H: f32) -> f32 {
    return hann_window_1d(x, W) * hann_window_1d(y, H);
}

// ============================================================================
// Real to Complex Conversion with Windowing and Zero Padding
// ============================================================================

@group(0) @binding(0) var input_rgba: texture_2d<f32>;
@group(0) @binding(1) var output_complex: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> params: UtilityParams;

/// Convert single channel from RGBA to complex with windowing and zero padding
/// Extracts one channel (R, G, or B) and applies Tukey window
@compute @workgroup_size(16, 16, 1)
fn real_to_complex_windowed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<u32>(gid.xy);

    // Bounds check against destination size
    if (coord.x >= params.dst_width || coord.y >= params.dst_height) {
        return;
    }

    var value: f32 = 0.0;

    // Check if within source image bounds
    if (coord.x < params.src_width && coord.y < params.src_height) {
        // Load RGBA value
        let rgba = textureLoad(input_rgba, coord, 0);

        // Extract requested channel
        switch (params.channel) {
            case 0u: { value = rgba.r; }
            case 1u: { value = rgba.g; }
            case 2u: { value = rgba.b; }
            default: { value = rgba.r; }
        }

        // Apply Tukey window
        let window = tukey_window_2d(
            f32(coord.x), f32(coord.y),
            f32(params.src_width), f32(params.src_height),
            params.window_alpha
        );
        value *= window;
    }
    // Else: zero padding (value stays 0.0)

    // Output as complex (real, 0)
    textureStore(output_complex, coord, vec4<f32>(value, 0.0, 0.0, 1.0));
}

/// Convert all RGB channels to complex in a single pass
/// Uses channel index from z coordinate
@compute @workgroup_size(16, 16, 1)
fn real_to_complex_all_channels(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<u32>(gid.xy);

    if (coord.x >= params.dst_width || coord.y >= params.dst_height) {
        return;
    }

    var value: f32 = 0.0;

    if (coord.x < params.src_width && coord.y < params.src_height) {
        let rgba = textureLoad(input_rgba, coord, 0);

        // Channel is passed via params for single-channel operation
        switch (params.channel) {
            case 0u: { value = rgba.r; }
            case 1u: { value = rgba.g; }
            case 2u: { value = rgba.b; }
            default: { value = rgba.r; }
        }

        let window = tukey_window_2d(
            f32(coord.x), f32(coord.y),
            f32(params.src_width), f32(params.src_height),
            params.window_alpha
        );
        value *= window;
    }

    textureStore(output_complex, coord, vec4<f32>(value, 0.0, 0.0, 1.0));
}

// ============================================================================
// Complex to Real Conversion with Cropping
// ============================================================================

@group(0) @binding(0) var input_complex: texture_2d<f32>;
@group(0) @binding(1) var output_real: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> crop_params: UtilityParams;

/// Convert complex to real, crop padding, and normalize
/// Takes real part of complex number, applies normalization, clips to [0,1]
@compute @workgroup_size(16, 16, 1)
fn complex_to_real_crop(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<u32>(gid.xy);

    // Only process within original (cropped) dimensions
    if (coord.x >= crop_params.src_width || coord.y >= crop_params.src_height) {
        return;
    }

    // Load complex value (may be from larger padded texture)
    let complex_val = textureLoad(input_complex, coord, 0).rg;

    // Take real part and apply normalization
    var real_val = complex_val.x * crop_params.normalize_factor;

    // Clip to valid range [0, 1]
    real_val = clamp(real_val, 0.0, 1.0);

    // Output as single channel (stored in R, G is 0)
    textureStore(output_real, coord, vec4<f32>(real_val, 0.0, 0.0, 1.0));
}

// ============================================================================
// Channel Combination
// ============================================================================

struct CombineParams {
    width: u32,
    height: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var channel_r: texture_2d<f32>;
@group(0) @binding(1) var channel_g: texture_2d<f32>;
@group(0) @binding(2) var channel_b: texture_2d<f32>;
@group(0) @binding(3) var output_rgba: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> combine_params: CombineParams;

/// Combine R, G, B channels back into RGBA
@compute @workgroup_size(16, 16, 1)
fn combine_rgb_channels(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<u32>(gid.xy);

    if (coord.x >= combine_params.width || coord.y >= combine_params.height) {
        return;
    }

    // Load individual channels (stored as complex, take real part)
    let r = textureLoad(channel_r, coord, 0).r;
    let g = textureLoad(channel_g, coord, 0).r;
    let b = textureLoad(channel_b, coord, 0).r;

    // Combine into RGBA (alpha = 1)
    textureStore(output_rgba, coord, vec4<f32>(r, g, b, 1.0));
}

// ============================================================================
// Edge Extension (for boundary handling)
// ============================================================================

struct ExtendParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    mode: u32,  // 0 = zero, 1 = reflect, 2 = replicate
    _pad: vec3<u32>,
}

@group(0) @binding(0) var extend_input: texture_2d<f32>;
@group(0) @binding(1) var extend_output: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> extend_params: ExtendParams;

/// Extend image boundaries with various modes
@compute @workgroup_size(16, 16, 1)
fn extend_boundaries(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<u32>(gid.xy);

    if (coord.x >= extend_params.dst_width || coord.y >= extend_params.dst_height) {
        return;
    }

    var src_coord = vec2<i32>(coord);
    let src_w = i32(extend_params.src_width);
    let src_h = i32(extend_params.src_height);

    var value: vec2<f32> = vec2<f32>(0.0, 0.0);

    switch (extend_params.mode) {
        case 0u: {
            // Zero padding
            if (src_coord.x >= 0 && src_coord.x < src_w &&
                src_coord.y >= 0 && src_coord.y < src_h) {
                value = textureLoad(extend_input, vec2<u32>(src_coord), 0).rg;
            }
        }
        case 1u: {
            // Reflect
            if (src_coord.x < 0) { src_coord.x = -src_coord.x - 1; }
            if (src_coord.y < 0) { src_coord.y = -src_coord.y - 1; }
            if (src_coord.x >= src_w) { src_coord.x = 2 * src_w - src_coord.x - 1; }
            if (src_coord.y >= src_h) { src_coord.y = 2 * src_h - src_coord.y - 1; }

            src_coord.x = clamp(src_coord.x, 0, src_w - 1);
            src_coord.y = clamp(src_coord.y, 0, src_h - 1);
            value = textureLoad(extend_input, vec2<u32>(src_coord), 0).rg;
        }
        case 2u: {
            // Replicate (clamp to edge)
            src_coord.x = clamp(src_coord.x, 0, src_w - 1);
            src_coord.y = clamp(src_coord.y, 0, src_h - 1);
            value = textureLoad(extend_input, vec2<u32>(src_coord), 0).rg;
        }
        default: {
            // Default to zero padding
            if (src_coord.x >= 0 && src_coord.x < src_w &&
                src_coord.y >= 0 && src_coord.y < src_h) {
                value = textureLoad(extend_input, vec2<u32>(src_coord), 0).rg;
            }
        }
    }

    textureStore(extend_output, coord, vec4<f32>(value, 0.0, 1.0));
}

// ============================================================================
// Texture Copy Utilities
// ============================================================================

struct CopyParams {
    width: u32,
    height: u32,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var copy_input: texture_2d<f32>;
@group(0) @binding(1) var copy_output: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> copy_params: CopyParams;

/// Simple texture copy (Rg32Float)
@compute @workgroup_size(16, 16, 1)
fn texture_copy_rg(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<u32>(gid.xy);

    if (coord.x >= copy_params.width || coord.y >= copy_params.height) {
        return;
    }

    let val = textureLoad(copy_input, coord, 0).rg;
    textureStore(copy_output, coord, vec4<f32>(val, 0.0, 1.0));
}
