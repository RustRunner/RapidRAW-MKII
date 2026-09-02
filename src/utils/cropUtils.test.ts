import { describe, expect, it } from 'vitest';
import type { Crop } from 'react-image-crop';
import {
  CROP_EDGE_TOLERANCE_PX,
  getOrientedDimensions,
  isFullFrameCrop,
  percentToPixelCrop,
  pixelToPercentCrop,
  rotatePixelCrop90,
} from './cropUtils';

const W = 400;
const H = 300;

/**
 * Where a pixel lands after image::rotate90 (clockwise) / rotate270, used as an
 * independent oracle for rotatePixelCrop90 rather than restating its formula.
 */
const rotatePointCw = (x: number, y: number, _w: number, h: number) => ({ x: h - 1 - y, y: x });
const rotatePointCcw = (x: number, y: number, w: number, _h: number) => ({ x: y, y: w - 1 - x });

function boundingBoxAfterRotation(
  crop: { x: number; y: number; width: number; height: number },
  w: number,
  h: number,
  rotatePoint: (x: number, y: number, w: number, h: number) => { x: number; y: number },
) {
  const corners = [
    [crop.x, crop.y],
    [crop.x + crop.width - 1, crop.y],
    [crop.x, crop.y + crop.height - 1],
    [crop.x + crop.width - 1, crop.y + crop.height - 1],
  ].map(([x, y]) => rotatePoint(x, y, w, h));

  const xs = corners.map((c) => c.x);
  const ys = corners.map((c) => c.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);

  return {
    x: minX,
    y: minY,
    width: Math.max(...xs) - minX + 1,
    height: Math.max(...ys) - minY + 1,
  };
}

describe('rotatePixelCrop90', () => {
  const crop = { unit: 'px' as const, x: 30, y: 20, width: 100, height: 60 };

  it('matches a pixel-level simulation of a clockwise quarter turn', () => {
    const expected = boundingBoxAfterRotation(crop, W, H, rotatePointCw);
    expect(rotatePixelCrop90(crop, W, H, 'cw')).toMatchObject(expected);
  });

  it('matches a pixel-level simulation of a counter-clockwise quarter turn', () => {
    const expected = boundingBoxAfterRotation(crop, W, H, rotatePointCcw);
    expect(rotatePixelCrop90(crop, W, H, 'ccw')).toMatchObject(expected);
  });

  it('transposes the frame, so the result fits the rotated image', () => {
    const rotated = rotatePixelCrop90(crop, W, H, 'cw');
    expect(rotated.width).toBe(crop.height);
    expect(rotated.height).toBe(crop.width);
    expect(rotated.x + rotated.width).toBeLessThanOrEqual(H);
    expect(rotated.y + rotated.height).toBeLessThanOrEqual(W);
  });

  it('returns to the original rectangle after four clockwise steps', () => {
    let current: Crop = crop;
    let [w, h] = [W, H];
    for (let i = 0; i < 4; i++) {
      current = rotatePixelCrop90(current, w, h, 'cw');
      [w, h] = [h, w];
    }
    expect(current).toMatchObject({ x: crop.x, y: crop.y, width: crop.width, height: crop.height });
  });

  it('is reversible: clockwise then counter-clockwise is the identity', () => {
    const there = rotatePixelCrop90(crop, W, H, 'cw');
    expect(rotatePixelCrop90(there, H, W, 'ccw')).toMatchObject({
      x: crop.x,
      y: crop.y,
      width: crop.width,
      height: crop.height,
    });
  });

  it('keeps a full-frame rectangle full frame in the rotated frame', () => {
    const full = { unit: 'px' as const, x: 0, y: 0, width: W, height: H };
    const rotated = rotatePixelCrop90(full, W, H, 'cw');
    expect(isFullFrameCrop(rotated, H, W)).toBe(true);
  });
});

describe('isFullFrameCrop', () => {
  const full = { unit: 'px' as const, x: 0, y: 0, width: W, height: H };

  it('accepts the exact frame and rejects a null crop', () => {
    expect(isFullFrameCrop(full, W, H)).toBe(true);
    expect(isFullFrameCrop(null, W, H)).toBe(false);
  });

  it('accepts each edge displaced within tolerance', () => {
    const t = CROP_EDGE_TOLERANCE_PX;
    expect(isFullFrameCrop({ ...full, x: t, width: W - t }, W, H)).toBe(true);
    expect(isFullFrameCrop({ ...full, y: t, height: H - t }, W, H)).toBe(true);
    expect(isFullFrameCrop({ ...full, width: W - t }, W, H)).toBe(true);
    expect(isFullFrameCrop({ ...full, height: H - t }, W, H)).toBe(true);
  });

  it('rejects each edge displaced just past tolerance', () => {
    const over = CROP_EDGE_TOLERANCE_PX + 1;
    expect(isFullFrameCrop({ ...full, x: over, width: W - over }, W, H)).toBe(false);
    expect(isFullFrameCrop({ ...full, y: over, height: H - over }, W, H)).toBe(false);
    expect(isFullFrameCrop({ ...full, width: W - over }, W, H)).toBe(false);
    expect(isFullFrameCrop({ ...full, height: H - over }, W, H)).toBe(false);
  });

  it('rejects a genuine crop', () => {
    expect(isFullFrameCrop({ unit: 'px', x: 30, y: 20, width: 100, height: 60 }, W, H)).toBe(false);
  });
});

describe('percent/pixel conversion', () => {
  it.each([0, 1, 2, 3])('round-trips a rectangle at orientationSteps %i', (orientationSteps) => {
    const { width: ow, height: oh } = getOrientedDimensions(W, H, orientationSteps);
    const pixel = { unit: 'px' as const, x: 40, y: 25, width: 120, height: 80 };

    const percent = pixelToPercentCrop(pixel, W, H, orientationSteps);
    const back = percentToPixelCrop(percent, W, H, orientationSteps);

    // ceil-origin / floor-size rounding can move an edge by at most 1px.
    expect(back!.x).toBeCloseTo(pixel.x, 0);
    expect(back!.y).toBeCloseTo(pixel.y, 0);
    expect(Math.abs(back!.width - pixel.width)).toBeLessThanOrEqual(1);
    expect(Math.abs(back!.height - pixel.height)).toBeLessThanOrEqual(1);
    expect(pixel.x + pixel.width).toBeLessThanOrEqual(ow);
    expect(pixel.y + pixel.height).toBeLessThanOrEqual(oh);
  });

  it('uses the swapped frame when orientation transposes the image', () => {
    const percent = { unit: '%' as const, x: 0, y: 0, width: 50, height: 100 };
    expect(percentToPixelCrop(percent, W, H, 0)).toMatchObject({ width: W / 2, height: H });
    // At one step the oriented frame is H x W, so the same percentages
    // describe a different pixel rectangle.
    expect(percentToPixelCrop(percent, W, H, 1)).toMatchObject({ width: H / 2, height: W });
  });

  it('maps a full-frame percentage onto a crop the frame test accepts', () => {
    const percent = { unit: '%' as const, x: 0, y: 0, width: 100, height: 100 };
    expect(isFullFrameCrop(percentToPixelCrop(percent, W, H, 0), W, H)).toBe(true);
  });

  it('passes null through', () => {
    expect(percentToPixelCrop(null, W, H, 0)).toBeNull();
    expect(pixelToPercentCrop(null, W, H, 0)).toBeNull();
  });
});
