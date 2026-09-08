import { describe, expect, it } from 'vitest';
import { isStaleEstimateError, readyImageIdentity, sameImage } from './imageIdentity';

describe('committed image identity', () => {
  const identity = { path: 'A', generation: '9007199254740993' };
  it('preserves generations beyond JavaScript integer precision', () => {
    expect(sameImage(identity, { ...identity })).toBe(true);
    expect(sameImage(identity, { ...identity, generation: '9007199254740992' })).toBe(false);
    expect(sameImage(identity, { ...identity, path: 'B' })).toBe(false);
    expect(sameImage(undefined, undefined)).toBe(false);
  });
  it('cannot use an old identity while a same-path reload is pending', () => {
    expect(readyImageIdentity({ path: 'A', isReady: false, identity })).toBeUndefined();
    expect(readyImageIdentity({ path: 'B', isReady: true, identity })).toBeUndefined();
    expect(readyImageIdentity({ path: 'A', isReady: true, identity })).toEqual(identity);
  });
  it('distinguishes normal staleness from analysis failure', () => {
    expect(isStaleEstimateError({ code: 'stale' })).toBe(true);
    expect(isStaleEstimateError({ code: 'failed', message: 'failure' })).toBe(false);
  });
});
