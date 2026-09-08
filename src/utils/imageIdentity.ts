export interface ImageIdentity {
  path: string;
  generation: string;
}

export interface OwnedEstimate<T> {
  identity: ImageIdentity;
  estimate: T;
}

export interface NoiseEstimate {
  sigma_luma: number;
  sigma_chroma: number;
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
