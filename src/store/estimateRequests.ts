import { invoke } from '@tauri-apps/api/core';
import { Invokes } from '../components/ui/AppProperties';
import type { Adjustments } from '../utils/adjustments';
import { estimateKeys, type EstimateTool, type EstimateRequest } from '../utils/estimateState';
import {
  NoiseEstimate,
  GlareEstimate,
  OwnedEstimate,
  readyImageIdentity,
  sameImage,
  isStaleEstimateError,
  estimateErrorMessage,
} from '../utils/imageIdentity';
import { useEditorStore } from './useEditorStore';
import { debouncedSetHistory, debouncedSave } from './editorPersistence';

const panels = new Set<symbol>();
export const VEIL_FLASH_MS = 1200;

export function mountEstimatePanel(panel: symbol) {
  panels.add(panel);
  return () => {
    panels.delete(panel);
    dismissVeilFlash(panel);
  };
}

// With an owner, cleanup only ends that panel's flash. A manual Show Veil
// choice omits the owner and takes precedence over any temporary preview.
export function dismissVeilFlash(panel?: symbol) {
  const state = useEditorStore.getState();
  if (state.veilFlash && (!panel || state.veilFlash.panel === panel)) {
    state.setEditor({ veilFlash: null });
  }
}

export function cancelEstimate(tool: EstimateTool) {
  const state = useEditorStore.getState();
  const requests = { ...state.estimateRequests };
  delete requests[tool];
  state.setEditor({ estimateRequests: requests, ...(tool === 'glare' ? { veilFlash: null } : {}) });
}

// Explicit patches (paste, preset, section reset) supersede included tool
// settings even if their values happen to equal the current values.
export function supersedeEstimateEdits(patch: Partial<Adjustments>) {
  for (const tool of ['denoise', 'glare'] as const) {
    if (estimateKeys[tool].some((key) => Object.prototype.hasOwnProperty.call(patch, key))) cancelEstimate(tool);
  }
}

function isCurrent(tool: EstimateTool, request: EstimateRequest) {
  const state = useEditorStore.getState();
  return (
    state.estimateRequests[tool]?.token === request.token &&
    sameImage(readyImageIdentity(state.selectedImage), request.identity)
  );
}

export async function requestEstimate(tool: EstimateTool, panel: symbol, onError: (message: string) => void) {
  const state = useEditorStore.getState();
  const identity = readyImageIdentity(state.selectedImage);
  if (!identity) return;
  const request: EstimateRequest = { token: Symbol(tool), identity, status: 'pending' };
  state.setEditor({
    estimateRequests: { ...state.estimateRequests, [tool]: request },
    ...(tool === 'glare' ? { veilFlash: null } : {}),
  });
  try {
    const result = await invoke<OwnedEstimate<NoiseEstimate | GlareEstimate>>(
      tool === 'denoise' ? Invokes.EstimateNoiseLevel : Invokes.EstimateGlareVeil,
      { expectedIdentity: identity },
    );
    if (!isCurrent(tool, request)) return;
    if (!sameImage(result.identity, identity)) {
      cancelEstimate(tool);
      return;
    }
    let patch: Partial<Adjustments>;
    if (tool === 'denoise') {
      const estimate = result.estimate as NoiseEstimate;
      patch = { denoiseStrength: Math.round(estimate.strength), denoiseChroma: Math.round(estimate.chroma) };
    } else {
      const estimate = result.estimate as GlareEstimate;
      if (!estimate.confident) throw new Error('Insufficient usable image data');
      const clamp = (v: number) => Math.min(100, Math.max(0, Math.round(v)));
      patch = {
        glareEnabled: true,
        glareAmount: clamp(estimate.amount),
        glareVeilSize: clamp(estimate.veilSize),
        glareMaxBoost: clamp(estimate.maxBoost),
      };
    }
    // One discrete history action, preserving any unrelated edits made while
    // analysis was running. Closing the initiating panel does not cancel it.
    debouncedSetHistory.flush();
    const current = useEditorStore.getState();
    const adjustments = { ...current.adjustments, ...patch };
    const flash = tool === 'glare' && panels.has(panel) ? { token: request.token, panel, identity } : null;
    current.setEditor({
      adjustments,
      estimateRequests: {
        ...current.estimateRequests,
        [tool]: { ...request, status: 'success', result: result.estimate },
      },
      ...(tool === 'glare' ? { veilFlash: flash } : {}),
    });
    useEditorStore.getState().pushHistory(adjustments);
    debouncedSave(identity.path, adjustments, identity);
    if (flash) {
      setTimeout(() => {
        const latest = useEditorStore.getState();
        if (latest.veilFlash?.token === flash.token && sameImage(readyImageIdentity(latest.selectedImage), identity)) {
          latest.setEditor({ veilFlash: null });
        }
      }, VEIL_FLASH_MS);
    }
  } catch (error) {
    if (!isCurrent(tool, request)) return;
    if (isStaleEstimateError(error)) {
      cancelEstimate(tool);
      return;
    }
    const message = estimateErrorMessage(error);
    const current = useEditorStore.getState();
    current.setEditor({
      estimateRequests: { ...current.estimateRequests, [tool]: { ...request, status: 'error', error: message } },
    });
    if (panels.has(panel)) onError(message);
  }
}
