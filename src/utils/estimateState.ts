import type { Adjustments } from './adjustments';
import { ImageIdentity, NoiseEstimate, GlareEstimate, sameImage } from './imageIdentity';

export type EstimateTool = 'denoise' | 'glare';
export interface EstimateRequest {
  token: symbol;
  identity: ImageIdentity;
  status: 'pending' | 'success' | 'error';
  result?: NoiseEstimate | GlareEstimate;
  error?: string;
}
export type EstimateRequests = Partial<Record<EstimateTool, EstimateRequest>>;
export interface VeilFlash {
  token: symbol;
  panel: symbol;
  identity: ImageIdentity;
}
export const estimateKeys: Record<EstimateTool, (keyof Adjustments)[]> = {
  denoise: ['denoiseEnabled', 'denoiseStrength', 'denoiseDetail', 'denoiseChroma'],
  glare: ['glareEnabled', 'glareAmount', 'glareVeilSize', 'glareMaxBoost', 'glareShowVeil'],
};

// Only preview calls receive this overlay. History, export, presets, and
// sidecar persistence continue to consume the committed adjustments object.
export function withVeilFlash(
  adjustments: Adjustments,
  flash: VeilFlash | null,
  identity?: ImageIdentity,
): Adjustments {
  return flash && sameImage(flash.identity, identity) ? { ...adjustments, glareShowVeil: true } : adjustments;
}
