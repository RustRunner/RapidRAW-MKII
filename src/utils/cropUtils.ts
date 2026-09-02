import { Crop, PercentCrop } from 'react-image-crop';

export function getOrientedDimensions(
  imageWidth: number,
  imageHeight: number,
  orientationSteps: number,
): { width: number; height: number } {
  const isSwapped = orientationSteps === 1 || orientationSteps === 3;
  return {
    width: isSwapped ? imageHeight : imageWidth,
    height: isSwapped ? imageWidth : imageHeight,
  };
}

/**
 * Default half-width, in oriented pixels, of the tolerance band used when
 * asking whether a rectangle "is" another rectangle. Matches the +/-2px the
 * geometry effect already uses for its maximized-crop test.
 */
export const CROP_EDGE_TOLERANCE_PX = 2;

/**
 * Convert a react-image-crop percent rectangle into the oriented pixel space
 * that `adjustments.crop` is stored in. Keeps the ceil-origin / floor-size
 * rounding the editor has always used, so committed crops stay byte-identical
 * to the pre-draft behaviour.
 */
export function percentToPixelCrop(
  percentCrop: { x: number; y: number; width: number; height: number } | null | undefined,
  imageWidth: number,
  imageHeight: number,
  orientationSteps: number,
): Crop | null {
  if (!percentCrop) return null;

  const { width: W, height: H } = getOrientedDimensions(imageWidth, imageHeight, orientationSteps);

  return {
    unit: 'px',
    x: Math.ceil((percentCrop.x / 100) * W),
    y: Math.ceil((percentCrop.y / 100) * H),
    width: Math.floor((percentCrop.width / 100) * W),
    height: Math.floor((percentCrop.height / 100) * H),
  };
}

/** Inverse of `percentToPixelCrop`, in the same oriented frame. */
export function pixelToPercentCrop(
  pixelCrop: Crop | null | undefined,
  imageWidth: number,
  imageHeight: number,
  orientationSteps: number,
): PercentCrop | null {
  if (!pixelCrop) return null;

  const { width: W, height: H } = getOrientedDimensions(imageWidth, imageHeight, orientationSteps);
  if (W <= 0 || H <= 0) return null;

  return {
    unit: '%',
    x: (pixelCrop.x / W) * 100,
    y: (pixelCrop.y / H) * 100,
    width: (pixelCrop.width / W) * 100,
    height: (pixelCrop.height / H) * 100,
  };
}

/**
 * True when every edge of `crop` sits within `tolerance` of the oriented frame.
 * A full-frame crop and `crop: null` are the same picture -- the backend
 * short-circuits `rect == full` in `apply_crop` -- so this is what lets the
 * editor keep `null` canonical instead of committing a redundant rectangle.
 */
export function isFullFrameCrop(
  crop: Crop | null | undefined,
  orientedWidth: number,
  orientedHeight: number,
  tolerance: number = CROP_EDGE_TOLERANCE_PX,
): boolean {
  if (!crop) return false;

  return (
    Math.abs(crop.x) <= tolerance &&
    Math.abs(crop.y) <= tolerance &&
    Math.abs(crop.x + crop.width - orientedWidth) <= tolerance &&
    Math.abs(crop.y + crop.height - orientedHeight) <= tolerance
  );
}

export function calculateCenteredCrop(
  imageWidth: number,
  imageHeight: number,
  orientationSteps: number,
  aspectRatio: number | null,
  rotation: number = 0,
): Crop | null {
  if (!aspectRatio || aspectRatio <= 0) return null;

  const { width: W, height: H } = getOrientedDimensions(imageWidth, imageHeight, orientationSteps);

  const angle = Math.abs(rotation);
  const rad = ((angle % 180) * Math.PI) / 180;
  const sin = Math.sin(rad);
  const cos = Math.cos(rad);

  const h_c = Math.min(H / (aspectRatio * sin + cos), W / (aspectRatio * cos + sin));
  const w_c = aspectRatio * h_c;

  return {
    unit: 'px',
    x: Math.round((W - w_c) / 2),
    y: Math.round((H - h_c) / 2),
    width: Math.round(w_c),
    height: Math.round(h_c),
  };
}

function isCropWithinBounds(crop: Crop, imageW: number, imageH: number, rotation: number): boolean {
  const cx = imageW / 2;
  const cy = imageH / 2;
  const rad = (-rotation * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const pts = [
    { x: crop.x, y: crop.y },
    { x: crop.x + crop.width, y: crop.y },
    { x: crop.x, y: crop.y + crop.height },
    { x: crop.x + crop.width, y: crop.y + crop.height },
  ];
  for (let i = 0; i < 4; i++) {
    const nx = cos * (pts[i].x - cx) - sin * (pts[i].y - cy) + cx;
    const ny = sin * (pts[i].x - cx) + cos * (pts[i].y - cy) + cy;
    if (nx < -1 || nx > imageW + 1 || ny < -1 || ny > imageH + 1) return false;
  }
  return true;
}

export function calculateAreaPreservingCrop(
  imageWidth: number,
  imageHeight: number,
  orientationSteps: number,
  aspectRatio: number | null,
  rotation: number,
  currentCrop: Crop | null | undefined,
): Crop | null {
  if (!aspectRatio || aspectRatio <= 0 || !currentCrop || !currentCrop.width || !currentCrop.height) return null;

  const { width: W, height: H } = getOrientedDimensions(imageWidth, imageHeight, orientationSteps);

  const area = currentCrop.width * currentCrop.height;
  const newH = Math.sqrt(area / aspectRatio);
  const newW = aspectRatio * newH;
  const centerX = currentCrop.x + currentCrop.width / 2;
  const centerY = currentCrop.y + currentCrop.height / 2;

  const candidate: Crop = {
    unit: 'px',
    x: Math.round(centerX - newW / 2),
    y: Math.round(centerY - newH / 2),
    width: Math.round(newW),
    height: Math.round(newH),
  };

  return isCropWithinBounds(candidate, W, H, rotation) ? candidate : null;
}

export function rotateCropCenter(
  crop: Crop,
  orientedWidth: number,
  orientedHeight: number,
  deltaDegrees: number,
): Crop {
  const rad = (deltaDegrees * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const cx = orientedWidth / 2;
  const cy = orientedHeight / 2;
  const px = crop.x + crop.width / 2 - cx;
  const py = crop.y + crop.height / 2 - cy;
  const rx = px * cos - py * sin;
  const ry = px * sin + py * cos;
  return {
    unit: 'px',
    x: Math.round(cx + rx - crop.width / 2),
    y: Math.round(cy + ry - crop.height / 2),
    width: crop.width,
    height: crop.height,
  };
}
