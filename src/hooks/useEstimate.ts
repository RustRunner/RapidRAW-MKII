import { useLayoutEffect, useRef } from 'react';
import { useEditorStore } from '../store/useEditorStore';
import { mountEstimatePanel, requestEstimate, dismissVeilFlash } from '../store/estimateRequests';
import type { EstimateTool } from '../utils/estimateState';

export function useEstimate(tool: EstimateTool, onError: (message: string) => void, isVisible = true) {
  const panel = useRef(Symbol(tool));
  useLayoutEffect(() => {
    if (!isVisible) return;
    // A reopen is a new visibility lifetime, even if the component stayed mounted.
    panel.current = Symbol(tool);
    return mountEstimatePanel(panel.current);
  }, [isVisible, tool]);
  const isFlashing = useEditorStore((state) => !!state.veilFlash);
  const request = useEditorStore((state) => state.estimateRequests[tool]);
  return {
    isFlashing,
    isEstimating: request?.status === 'pending',
    result: request?.result,
    estimate: () => requestEstimate(tool, panel.current, onError),
    dismissFlash: () => dismissVeilFlash(),
  };
}
