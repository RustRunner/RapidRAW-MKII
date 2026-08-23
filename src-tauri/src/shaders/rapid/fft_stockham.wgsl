// ============================================================================
// RAPID FFT - Stockham Auto-Sort Algorithm
// ============================================================================
//
// The Stockham FFT algorithm performs implicit bit-reversal during computation,
// eliminating the need for a separate permutation step. This makes it ideal
// for GPU implementation where sequential memory access is important.
//
// Key properties:
// - Auto-sorting: output is in natural order
// - Ping-pong: alternates between two buffers
// - Cache-friendly: sequential memory access pattern
// - No bit-reversal: implicit reordering during butterflies
//
// For N-point FFT: log₂(N) passes required
// Each pass: N/2 butterfly operations
//
// Author: RapidRAW Mod1 Team
// Date: January 2026
// ============================================================================

// Include common complex math functions
// Note: In WGSL, we can't actually #include, so these are duplicated or
// the shader sources are concatenated at load time in Rust.

// Mathematical constants
const PI: f32 = 3.14159265358979323846;
const TWO_PI: f32 = 6.28318530717958647692;

// ============================================================================
// Local complex-number operations used by this standalone shader module
// ============================================================================

/// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

/// Complex addition
fn c_add(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a + b;
}

/// Complex subtraction
fn c_sub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return a - b;
}

/// Complex exponential of pure imaginary: e^(ix) = cos(x) + i*sin(x)
fn c_exp_i(x: f32) -> vec2<f32> {
    return vec2<f32>(cos(x), sin(x));
}

// ============================================================================
// FFT Parameters
// ============================================================================

struct FFTParams {
    size: u32,           // FFT size (must be power of 2)
    pass_num: u32,       // Current pass number (0 to log2(size)-1)
    direction: i32,      // 1 = forward FFT, -1 = inverse FFT
    is_horizontal: u32,  // 1 = row-wise, 0 = column-wise
    width: u32,          // Texture width
    height: u32,         // Texture height
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var output_tex: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> params: FFTParams;

// ============================================================================
// Stockham FFT Butterfly Pass
// ============================================================================
//
// The Stockham algorithm works by performing butterflies with a specific
// addressing pattern that results in naturally-ordered output.
//
// For each pass p (0 to log2(N)-1):
//   - Block size = 2^(p+1)
//   - Half block = 2^p
//   - Each butterfly reads from positions separated by half_block
//   - Output is written to sequential positions
//
// Example for N=8:
//   Pass 0: blocks of 2, butterflies at (0,1), (2,3), (4,5), (6,7)
//   Pass 1: blocks of 4, butterflies at (0,2), (1,3), (4,6), (5,7)
//   Pass 2: blocks of 8, butterflies at (0,4), (1,5), (2,6), (3,7)
// ============================================================================

@compute @workgroup_size(256, 1, 1)
fn fft_horizontal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.size;
    let half_N = N / 2u;

    // Each thread handles one butterfly operation
    let butterfly_idx = gid.x;
    let row = gid.y;

    // Bounds check
    if (butterfly_idx >= half_N || row >= params.height) {
        return;
    }

    // Cooley-Tukey radix-2 DIT (Decimation In Time)
    // After bit-reversal input, we do standard butterfly passes
    let stage = params.pass_num;
    let stride = 1u << stage;                  // 1, 2, 4, 8, ... (distance between butterfly pairs)
    let half_stride = stride;
    let block_size = stride << 1u;             // 2, 4, 8, 16, ... (full butterfly block)

    // Which block are we in, and which butterfly within the block?
    let block_idx = butterfly_idx / stride;
    let pos_in_block = butterfly_idx % stride;

    // Input/output indices - same positions for in-place style FFT
    let idx_even = block_idx * block_size + pos_in_block;
    let idx_odd = idx_even + stride;

    // Load input values (complex numbers stored as RG)
    let even_val = textureLoad(input_tex, vec2<u32>(idx_even, row), 0).rg;
    let odd_val = textureLoad(input_tex, vec2<u32>(idx_odd, row), 0).rg;

    // Compute twiddle factor: W_N^k = e^(-2πik/N) for forward, e^(2πik/N) for inverse
    // k is position within the half-block, twiddle repeats with period block_size
    let k = pos_in_block;
    let sign = f32(-params.direction);  // -1 for forward, +1 for inverse
    let angle = sign * TWO_PI * f32(k) / f32(block_size);
    let twiddle = c_exp_i(angle);

    // Butterfly operation: even' = even + W*odd, odd' = even - W*odd
    let odd_twisted = c_mul(twiddle, odd_val);
    let out_even = c_add(even_val, odd_twisted);
    let out_odd = c_sub(even_val, odd_twisted);

    // Write output to same positions (ping-pong between textures handles the "in-place")
    textureStore(output_tex, vec2<u32>(idx_even, row), vec4<f32>(out_even, 0.0, 1.0));
    textureStore(output_tex, vec2<u32>(idx_odd, row), vec4<f32>(out_odd, 0.0, 1.0));
}

@compute @workgroup_size(1, 256, 1)
fn fft_vertical(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.size;
    let half_N = N / 2u;

    // Each thread handles one butterfly operation
    let col = gid.x;
    let butterfly_idx = gid.y;

    // Bounds check
    if (col >= params.width || butterfly_idx >= half_N) {
        return;
    }

    // Cooley-Tukey radix-2 DIT (Decimation In Time)
    let stage = params.pass_num;
    let stride = 1u << stage;                  // 1, 2, 4, 8, ...
    let block_size = stride << 1u;             // 2, 4, 8, 16, ...

    // Which block are we in, and which butterfly within the block?
    let block_idx = butterfly_idx / stride;
    let pos_in_block = butterfly_idx % stride;

    // Input/output indices
    let idx_even = block_idx * block_size + pos_in_block;
    let idx_odd = idx_even + stride;

    // Load input values
    let even_val = textureLoad(input_tex, vec2<u32>(col, idx_even), 0).rg;
    let odd_val = textureLoad(input_tex, vec2<u32>(col, idx_odd), 0).rg;

    // Compute twiddle factor
    let k = pos_in_block;
    let sign = f32(-params.direction);
    let angle = sign * TWO_PI * f32(k) / f32(block_size);
    let twiddle = c_exp_i(angle);

    // Butterfly operation
    let odd_twisted = c_mul(twiddle, odd_val);
    let out_even = c_add(even_val, odd_twisted);
    let out_odd = c_sub(even_val, odd_twisted);

    // Write output to same positions
    textureStore(output_tex, vec2<u32>(col, idx_even), vec4<f32>(out_even, 0.0, 1.0));
    textureStore(output_tex, vec2<u32>(col, idx_odd), vec4<f32>(out_odd, 0.0, 1.0));
}

// ============================================================================
// Bit-Reversal Permutation
// ============================================================================
//
// Cooley-Tukey FFT requires bit-reversed input order. This shader performs
// the permutation either on input (before FFT) or output (after FFT).
// ============================================================================

struct BitRevParams {
    width: u32,
    height: u32,
    log2_width: u32,
    log2_height: u32,
}

@group(0) @binding(0) var bitrev_input_tex: texture_2d<f32>;
@group(0) @binding(1) var bitrev_output_tex: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> bitrev_params: BitRevParams;

// Reverse bits of a number with given bit width
fn reverse_bits(x: u32, bits: u32) -> u32 {
    var result = 0u;
    var val = x;
    for (var i = 0u; i < bits; i = i + 1u) {
        result = (result << 1u) | (val & 1u);
        val = val >> 1u;
    }
    return result;
}

@compute @workgroup_size(16, 16, 1)
fn bit_reverse_2d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= bitrev_params.width || y >= bitrev_params.height) {
        return;
    }

    // Bit-reverse both coordinates
    let rev_x = reverse_bits(x, bitrev_params.log2_width);
    let rev_y = reverse_bits(y, bitrev_params.log2_height);

    // Read from natural order, write to bit-reversed order
    // This prepares data for Cooley-Tukey FFT
    let val = textureLoad(bitrev_input_tex, vec2<u32>(x, y), 0).rg;
    textureStore(bitrev_output_tex, vec2<u32>(rev_x, rev_y), vec4<f32>(val, 0.0, 1.0));
}

// Inverse bit-reversal: read from bit-reversed, write to natural
@compute @workgroup_size(16, 16, 1)
fn bit_reverse_2d_inverse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if (x >= bitrev_params.width || y >= bitrev_params.height) {
        return;
    }

    // Bit-reverse coordinates to find source position
    let rev_x = reverse_bits(x, bitrev_params.log2_width);
    let rev_y = reverse_bits(y, bitrev_params.log2_height);

    // Read from bit-reversed position, write to natural order
    let val = textureLoad(bitrev_input_tex, vec2<u32>(rev_x, rev_y), 0).rg;
    textureStore(bitrev_output_tex, vec2<u32>(x, y), vec4<f32>(val, 0.0, 1.0));
}

// ============================================================================
// IFFT Normalization
// ============================================================================
//
// After inverse FFT, we need to divide by N to get the correct scaling.
// This is done as a separate pass for efficiency.
// ============================================================================

struct NormalizeParams {
    width: u32,
    height: u32,
    scale: f32,  // 1/N for IFFT normalization
    _pad: u32,
}

@group(0) @binding(0) var norm_input_tex: texture_2d<f32>;
@group(0) @binding(1) var norm_output_tex: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> norm_params: NormalizeParams;

@compute @workgroup_size(16, 16, 1)
fn fft_normalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<u32>(gid.xy);

    if (coord.x >= norm_params.width || coord.y >= norm_params.height) {
        return;
    }

    let val = textureLoad(norm_input_tex, coord, 0).rg;
    let normalized = val * norm_params.scale;

    textureStore(norm_output_tex, coord, vec4<f32>(normalized, 0.0, 1.0));
}

// ============================================================================
// Texture Copy (for when we need to copy between textures)
// ============================================================================

@compute @workgroup_size(16, 16, 1)
fn texture_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<u32>(gid.xy);

    if (coord.x >= norm_params.width || coord.y >= norm_params.height) {
        return;
    }

    let val = textureLoad(norm_input_tex, coord, 0).rg;
    textureStore(norm_output_tex, coord, vec4<f32>(val, 0.0, 1.0));
}
