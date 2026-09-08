export interface ImageIdentity {
  path: string;
  generation: string;
}

export interface OwnedEstimate<T> {
  identity: ImageIdentity;
  estimate: T;
}

export interface DomainNoise {
  sigma_y: number;
  sigma_cb: number;
  sigma_cr: number;
}

export interface NoiseMeasurement {
  version: number;
  bins: Array<{
    mean_encoded_y: number;
    linear: DomainNoise;
    encoded: DomainNoise;
    linear_mad_scale: DomainNoise;
    encoded_mad_scale: DomainNoise;
    sampled_patches: number;
    accepted_patches: number;
    accepted_unclipped_patches: number;
    accepted_black_clipped_patches: number;
    black_clipped_fraction: number;
    accepted_pixels: number;
    represented_pixels: number;
    quantization_limited: [boolean, boolean, boolean];
    structure_ratio: number;
    lag1: number;
    lag8: number;
    highpass_to_marginal: number;
    increment_disagreement: number;
  }>;
  quality: {
    sampled_patches: number;
    resampled_patches: number;
    sampled_origins: Array<[number, number]>;
    sampled_by_bin: number[];
    qualified_by_bin: number[];
    rejected_nonfinite: number;
    rejected_clipped: number;
    qualified_unclipped_patches: number;
    qualified_black_clipped_patches: number;
    clipped_pixel_fraction: number;
    min_patch_clipped_fraction: number;
    max_patch_clipped_fraction: number;
    rejected_structure: number;
    rejected_long_correlation: number;
    source_code_step: number;
    insufficient_bins: number;
  };
}

export interface NoiseEstimate {
  calibration_version: number;
  linear_bin_median: DomainNoise;
  encoded_bin_median: DomainNoise;
  native_chroma_spacing: number;
  detail: number;
  strength_range: [number, number];
  chroma_range: [number, number];
  measurement: NoiseMeasurement;
  strength: number;
  chroma: number;
}

export interface GlareEstimate {
  amount: number;
  veilSize: number;
  maxBoost: number;
  glareRatio: number;
  confident: boolean;
}

export function sameImage(a: ImageIdentity | null | undefined, b: ImageIdentity | null | undefined): boolean {
  return !!a && !!b && a.path === b.path && a.generation === b.generation;
}

export function readyImageIdentity(image: { isReady: boolean; path: string; identity?: ImageIdentity } | null) {
  return image?.isReady && image.identity?.path === image.path ? image.identity : undefined;
}

export function isStaleEstimateError(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'stale';
}

export function estimateErrorMessage(error: unknown): string {
  return typeof error === 'object' && error !== null && 'message' in error ? String(error.message) : String(error);
}
