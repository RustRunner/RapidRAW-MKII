import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { invoke } from '@tauri-apps/api/core';
import { useEditorStore } from './useEditorStore';
import { debouncedSave, debouncedSetHistory } from './editorPersistence';
import {
  cancelEstimate,
  supersedeEstimateEdits,
  dismissVeilFlash,
  mountEstimatePanel,
  requestEstimate,
  VEIL_FLASH_MS,
} from './estimateRequests';
import { INITIAL_ADJUSTMENTS } from '../utils/adjustments';
import { EstimateTool, withVeilFlash } from '../utils/estimateState';
import { ImageIdentity } from '../utils/imageIdentity';
import { Invokes, SelectedImage } from '../components/ui/AppProperties';

vi.mock('@tauri-apps/api/core', () => ({ invoke: vi.fn() }));
const identity = (path = 'A', generation = '1'): ImageIdentity => ({ path, generation });
const selected = (id = identity(), isReady = true) => ({ path: id.path, identity: id, isReady }) as SelectedImage;
const noise = { linear_bin_median: { sigma_y: 0.01, sigma_cb: 0.02, sigma_cr: 0.01 }, strength: 45.4, chroma: 70.6 };
const glare = { amount: 62.4, veilSize: 35.6, maxBoost: 47.1, glareRatio: 0.2, confident: true };
const state = () => useEditorStore.getState();
let panel: symbol;
let close: () => void;
let onError: ReturnType<typeof vi.fn<(message: string) => void>>;
function pending(tool: EstimateTool, owner = panel) {
  let resolve!: (value: unknown) => void;
  let reject!: (reason: unknown) => void;
  vi.mocked(invoke).mockImplementationOnce(
    () =>
      new Promise((yes, no) => {
        resolve = yes;
        reject = no;
      }),
  );
  const done = requestEstimate(tool, owner, onError);
  return {
    done,
    reject,
    resolve: (id = identity(), estimate = tool === 'denoise' ? noise : glare) => resolve({ identity: id, estimate }),
  };
}
function edit(patch: Partial<typeof INITIAL_ADJUSTMENTS>) {
  state().setEditor({ adjustments: { ...state().adjustments, ...patch } });
}
function saved() {
  return vi.mocked(invoke).mock.calls.filter(([command]) => command === Invokes.SaveMetadataAndUpdateThumbnail);
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(invoke).mockReset().mockResolvedValue(undefined);
  useEditorStore.setState(useEditorStore.getInitialState(), true);
  state().setEditor({ selectedImage: selected() });
  panel = Symbol('panel');
  close = mountEstimatePanel(panel);
  onError = vi.fn();
});
afterEach(() => {
  close();
  debouncedSave.cancel();
  debouncedSetHistory.cancel();
  vi.clearAllTimers();
  vi.useRealTimers();
});

describe.each<EstimateTool>(['denoise', 'glare'])('%s editor ownership', (tool) => {
  it('applies once after panel close, preserves unrelated edits, records history and saves committed settings', async () => {
    const job = pending(tool);
    expect(invoke).toHaveBeenLastCalledWith(
      tool === 'denoise' ? Invokes.EstimateNoiseLevel : Invokes.EstimateGlareVeil,
      { expectedIdentity: identity(), ...(tool === 'denoise' ? { detail: INITIAL_ADJUSTMENTS.denoiseDetail } : {}) },
    );
    close();
    edit({ exposure: 1.25 });
    debouncedSetHistory(state().adjustments, state().adjustmentsSnapshotVersion);
    job.resolve();
    await job.done;
    expect(state().adjustments.exposure).toBe(1.25);
    expect(state().history).toHaveLength(3); // initial, earlier manual edit, estimate
    expect(state().history[1].exposure).toBe(1.25);
    expect(state().estimateRequests[tool]?.status).toBe('success');
    expect(state().veilFlash).toBeNull();
    expect(state().adjustments.glareShowVeil).toBe(false);
    expect(tool === 'denoise' ? state().adjustments.denoiseStrength : state().adjustments.glareAmount).toBe(
      tool === 'denoise' ? 45 : 62,
    );
    await vi.advanceTimersByTimeAsync(1500);
    expect(state().history).toHaveLength(3);
    expect(saved()).toHaveLength(1);
    expect(saved()[0][1]).toEqual({ path: 'A', adjustments: state().adjustments });
    expect(JSON.stringify(saved()[0][1])).not.toMatch(/generation|estimateRequests|veilFlash/);
    state().undo();
    expect(tool === 'denoise' ? state().adjustments.denoiseStrength : state().adjustments.glareAmount).toBe(0);
    state().redo();
    expect(tool === 'denoise' ? state().adjustments.denoiseStrength : state().adjustments.glareAmount).toBe(
      tool === 'denoise' ? 45 : 62,
    );
  });

  it.each(['switch', 'reload', 'roundtrip', 'loading'] as const)('discards completion after %s', async (change) => {
    const job = pending(tool);
    if (change === 'switch' || change === 'roundtrip')
      state().setEditor({ selectedImage: selected(identity('B', '2')) });
    if (change === 'reload' || change === 'roundtrip')
      state().setEditor({ selectedImage: selected(identity('A', '3')) });
    if (change === 'loading') state().setEditor({ selectedImage: selected(identity(), false) });
    job.resolve();
    await job.done;
    expect(state().history).toHaveLength(1);
    expect(state().estimateRequests[tool]).toBeUndefined();
    expect(state().veilFlash).toBeNull();
    expect(onError).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(2000);
    expect(saved()).toHaveLength(0);
  });

  it('does not start while the image is loading', async () => {
    state().setEditor({ selectedImage: selected(identity(), false) });
    await requestEstimate(tool, panel, onError);
    expect(invoke).not.toHaveBeenCalled();
    expect(state().selectedImage?.identity).toBeUndefined();
  });

  it.each(['resolve', 'reject'] as const)(
    'old %s cannot end or report an error for a newer request',
    async (outcome) => {
      const old = pending(tool);
      const latest = pending(tool);
      if (outcome === 'resolve') old.resolve();
      else old.reject(new Error('old failure'));
      await old.done;
      expect(state().estimateRequests[tool]?.status).toBe('pending');
      expect(onError).not.toHaveBeenCalled();
      latest.resolve();
      await latest.done;
      expect(state().history).toHaveLength(2);
      expect(state().estimateRequests[tool]?.status).toBe('success');
    },
  );

  it('a late old success cannot replace an already committed newer success', async () => {
    const old = pending(tool);
    const latest = pending(tool);
    latest.resolve();
    await latest.done;
    const accepted = state().adjustments;
    old.resolve();
    await old.done;
    expect(state().adjustments).toBe(accepted);
    expect(state().history).toHaveLength(2);
  });

  it('manual tool edits invalidate the token even when edited back to the initial value', async () => {
    const job = pending(tool);
    const key = tool === 'denoise' ? 'denoiseStrength' : 'glareAmount';
    edit({ [key]: 25 });
    edit({ [key]: 0 });
    job.resolve();
    await job.done;
    expect(state().adjustments[key]).toBe(0);
    expect(state().estimateRequests[tool]).toBeUndefined();
    expect(state().history).toHaveLength(1);
  });

  it.each(['undo', 'redo', 'resetHistory', 'goToHistoryIndex', 'cancel'] as const)('respects %s', async (action) => {
    edit({ exposure: 1 });
    state().pushHistory(state().adjustments);
    if (action === 'redo') state().undo();
    const job = pending(tool);
    if (action === 'resetHistory') state().resetHistory({ ...INITIAL_ADJUSTMENTS });
    else if (action === 'goToHistoryIndex') state().goToHistoryIndex(0);
    else if (action === 'cancel') cancelEstimate(tool);
    else state()[action]();
    const intended = state().adjustments;
    const history = state().history;
    job.resolve();
    await job.done;
    expect(state().adjustments).toBe(intended);
    expect(state().history).toBe(history);
    expect(state().estimateRequests[tool]).toBeUndefined();
  });

  it('rejects a mismatched response identity and silently handles backend cancellation', async () => {
    const wrong = pending(tool);
    wrong.resolve(identity('B', '2'));
    await wrong.done;
    expect(state().estimateRequests[tool]).toBeUndefined();
    const stale = pending(tool);
    stale.reject({ code: 'stale' });
    await stale.done;
    expect(state().estimateRequests[tool]).toBeUndefined();
    expect(onError).not.toHaveBeenCalled();
    expect(state().history).toHaveLength(1);
  });

  it('reports only a current failure and does not call closed panel callbacks', async () => {
    const current = pending(tool);
    current.reject({ code: 'failed', message: 'analysis failed' });
    await current.done;
    expect(onError).toHaveBeenCalledExactlyOnceWith('analysis failed');
    expect(state().estimateRequests[tool]?.status).toBe('error');
    onError.mockClear();
    const closed = pending(tool);
    close();
    closed.reject(new Error('closed'));
    await closed.done;
    expect(onError).not.toHaveBeenCalled();
    expect(state().history).toHaveLength(1);
  });
});

it('the two tools can complete independently and preserve each other’s edits', async () => {
  const denoise = pending('denoise');
  const glareJob = pending('glare');
  denoise.resolve();
  await denoise.done;
  expect(state().estimateRequests.glare?.status).toBe('pending');
  glareJob.resolve();
  await glareJob.done;
  expect(state().adjustments).toMatchObject({
    denoiseStrength: 45,
    denoiseChroma: 71,
    glareEnabled: true,
    glareAmount: 62,
  });
  expect(state().history).toHaveLength(3);
});

it('flash affects preview only and expires without history or persistence mutations', async () => {
  const job = pending('glare');
  job.resolve();
  await job.done;
  const committed = state().adjustments;
  const flash = state().veilFlash;
  expect(flash).not.toBeNull();
  expect(withVeilFlash(committed, flash, identity()).glareShowVeil).toBe(true);
  expect(withVeilFlash(committed, flash, identity('B', '2'))).toBe(committed);
  expect(committed.glareShowVeil).toBe(false);
  await vi.advanceTimersByTimeAsync(VEIL_FLASH_MS);
  expect(state().veilFlash).toBeNull();
  expect(state().adjustments).toBe(committed);
  expect(state().history).toHaveLength(2);
  expect(saved()[0][1]).toEqual({ path: 'A', adjustments: committed });
});

it.each([true, false])('timeout respects subsequent manual Show Veil = %s', async (show) => {
  const job = pending('glare');
  job.resolve();
  await job.done;
  dismissVeilFlash();
  edit({ glareShowVeil: show });
  await vi.advanceTimersByTimeAsync(VEIL_FLASH_MS);
  expect(state().adjustments.glareShowVeil).toBe(show);
  expect(state().veilFlash).toBeNull();
});

it('an old timeout or old panel cleanup cannot clear a newer panel’s flash', async () => {
  const old = pending('glare');
  old.resolve();
  await old.done;
  await vi.advanceTimersByTimeAsync(600);
  const newPanel = Symbol('new panel');
  const closeNew = mountEstimatePanel(newPanel);
  const latest = pending('glare', newPanel);
  latest.resolve();
  await latest.done;
  const flash = state().veilFlash;
  close();
  expect(state().veilFlash).toBe(flash);
  await vi.advanceTimersByTimeAsync(600);
  expect(state().veilFlash).toBe(flash);
  closeNew();
  expect(state().veilFlash).toBeNull();
});

it('remounting a panel does not give an old request permission to flash', async () => {
  const job = pending('glare');
  close();
  const closeNew = mountEstimatePanel(Symbol('remounted'));
  job.resolve();
  await job.done;
  expect(state().adjustments.glareEnabled).toBe(true);
  expect(state().veilFlash).toBeNull();
  closeNew();
});

it('an old timeout cannot change the next image’s manual veil state', async () => {
  const job = pending('glare');
  job.resolve();
  await job.done;
  state().setEditor({ selectedImage: selected(identity('B', '2')) });
  edit({ glareShowVeil: true });
  await vi.advanceTimersByTimeAsync(VEIL_FLASH_MS);
  expect(state().adjustments.glareShowVeil).toBe(true);
  expect(state().veilFlash).toBeNull();
});

it('unconfident Glare never force-enables the stage', async () => {
  const job = pending('glare');
  job.resolve(identity(), { ...glare, confident: false });
  await job.done;
  expect(state().adjustments.glareEnabled).toBe(false);
  expect(state().history).toHaveLength(1);
  expect(state().veilFlash).toBeNull();
  expect(onError).toHaveBeenCalledOnce();
});

// Explicit intent must win even when reset/paste/preset values equal defaults.
it.each([
  'denoiseStrength',
  'denoiseEnabled',
  'denoiseDetail',
  'denoiseChroma',
  'glareAmount',
  'glareEnabled',
  'glareVeilSize',
  'glareMaxBoost',
  'glareShowVeil',
] as const)('an explicit patch containing %s supersedes analysis', async (key) => {
  const tool = key.startsWith('denoise') ? 'denoise' : 'glare';
  const job = pending(tool);
  supersedeEstimateEdits({ [key]: INITIAL_ADJUSTMENTS[key] });
  job.resolve();
  await job.done;
  expect(state().history).toHaveLength(1);
  expect(state().estimateRequests[tool]).toBeUndefined();
});

it.each(['denoise', 'glare'] as const)(
  'undo before %s save debounce cannot persist the undone estimate',
  async (tool) => {
    const job = pending(tool);
    job.resolve();
    await job.done;
    state().undo();
    await vi.advanceTimersByTimeAsync(350);
    expect(saved()).toHaveLength(0);
    // The normal editor render schedules the current settings after undo.
    debouncedSave('A', state().adjustments, identity());
    await vi.advanceTimersByTimeAsync(350);
    expect(saved()).toHaveLength(1);
    expect(saved()[0][1]).toEqual({ path: 'A', adjustments: INITIAL_ADJUSTMENTS });
  },
);

it('a delayed save cannot write an earlier generation after same-path reload', async () => {
  const job = pending('denoise');
  job.resolve();
  await job.done;
  state().setEditor({ selectedImage: selected(identity('A', '2')) });
  await vi.advanceTimersByTimeAsync(350);
  expect(saved()).toHaveLength(0);
});

it('captures current Detail in the native request and discards it after a Detail edit', async () => {
  edit({ denoiseDetail: 75 });
  const job = pending('denoise');
  expect(invoke).toHaveBeenLastCalledWith(Invokes.EstimateNoiseLevel, { expectedIdentity: identity(), detail: 75 });
  edit({ denoiseDetail: 40 });
  job.resolve();
  await job.done;
  expect(state().adjustments.denoiseDetail).toBe(40);
  expect(state().adjustments.denoiseStrength).toBe(0);
  expect(state().history).toHaveLength(1);
});
