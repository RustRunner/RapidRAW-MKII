// ============================================================================
// CPU FFT - iterative radix-2 Cooley-Tukey
// ============================================================================
//
// Small self-contained FFT used where the GPU pipeline is overkill: the
// cepstral blur estimator runs it on a downscaled working copy, and the
// rapid_processing tests use it as the reference implementation to validate
// the GPU FFT against. O(n log n) with bit-reversal, power-of-two sizes only.

use std::f32::consts::PI;

/// Complex number for CPU-side FFT work
#[derive(Clone, Copy, Debug)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    pub fn from_polar(r: f32, theta: f32) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    pub fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    pub fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    pub fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    pub fn scale(self, s: f32) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }

    pub fn magnitude(self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }
}

/// 1D FFT (iterative Cooley-Tukey radix-2), in place. Inverse includes the
/// 1/n normalization.
pub fn fft_1d(data: &mut [Complex], forward: bool) {
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

/// 2D FFT: rows then columns, in place on a row-major buffer.
pub fn fft_2d(data: &mut [Complex], width: usize, height: usize, forward: bool) {
    // Transform rows
    for row in 0..height {
        let start = row * width;
        let mut row_data: Vec<Complex> = data[start..start + width].to_vec();
        fft_1d(&mut row_data, forward);
        data[start..start + width].copy_from_slice(&row_data);
    }

    // Transform columns
    for col in 0..width {
        let mut col_data: Vec<Complex> = (0..height).map(|row| data[row * width + col]).collect();
        fft_1d(&mut col_data, forward);
        for (row, &val) in col_data.iter().enumerate() {
            data[row * width + col] = val;
        }
    }
}
