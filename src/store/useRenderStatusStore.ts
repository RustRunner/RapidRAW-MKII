import { create } from 'zustand';

const EMA_ALPHA = 0.3;

interface RenderStatusState {
  issuedJobId: number;
  settledJobId: number;
  newestIsInteractive: boolean;
  emaInteractiveMs: number | null;
  emaFinalMs: number | null;

  // Actions
  jobStarted: (id: number, interactive: boolean) => void;
  jobSettled: (id: number, durationMs: number, counted: boolean, interactive: boolean) => void;
  reset: () => void;
}

export const useRenderStatusStore = create<RenderStatusState>((set) => ({
  issuedJobId: 0,
  settledJobId: 0,
  newestIsInteractive: false,
  emaInteractiveMs: null,
  emaFinalMs: null,

  jobStarted: (id, interactive) => set({ issuedJobId: id, newestIsInteractive: interactive }),

  jobSettled: (id, durationMs, counted, interactive) =>
    set((state) => {
      const next: Partial<RenderStatusState> = {
        settledJobId: Math.max(state.settledJobId, id),
      };
      // A job that settles after reset() (image switched mid-flight) has an id
      // above the fresh issuedJobId — its duration must not seed the new
      // image's estimate.
      if (counted && id <= state.issuedJobId) {
        if (interactive) {
          next.emaInteractiveMs =
            state.emaInteractiveMs === null
              ? durationMs
              : state.emaInteractiveMs * (1 - EMA_ALPHA) + durationMs * EMA_ALPHA;
        } else {
          next.emaFinalMs =
            state.emaFinalMs === null
              ? durationMs
              : state.emaFinalMs * (1 - EMA_ALPHA) + durationMs * EMA_ALPHA;
        }
      }
      return next;
    }),

  reset: () =>
    set({
      issuedJobId: 0,
      settledJobId: 0,
      newestIsInteractive: false,
      emaInteractiveMs: null,
      emaFinalMs: null,
    }),
}));

export const useIsRenderPending = () => useRenderStatusStore((s) => s.issuedJobId > s.settledJobId);
