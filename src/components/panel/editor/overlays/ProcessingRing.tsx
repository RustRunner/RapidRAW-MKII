import { memo, useCallback, useEffect, useLayoutEffect, useRef } from 'react';
import clsx from 'clsx';
import { useIsRenderPending, useRenderStatusStore } from '../../../../store/useRenderStatusStore';

export const APPEAR_DELAY = 180;
export const MIN_VISIBLE = 350;
export const SNAP_MS = 130;
export const FADE_MS = 300;
export const DEFAULT_FINAL_MS = 400;
export const DEFAULT_INTERACTIVE_MS = 150;
const APPEAR_POP_MS = 120;
const RADIUS = 18;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;
const GLOW_NORMAL = 'drop-shadow(0 0 5px rgba(255,255,255,0.75)) drop-shadow(0 0 14px rgba(255,255,255,0.28))';
const GLOW_SNAP = 'drop-shadow(0 0 9px rgba(255,255,255,0.95)) drop-shadow(0 0 22px rgba(255,255,255,0.45))';

export type RingPhase = 'hidden' | 'arming' | 'visible' | 'snapping' | 'fading';

export interface RingState {
  phase: RingPhase;
  phaseStart: number;
  shownAt: number;
  fillStart: number;
  est: number;
  progress: number;
}

export const RING_HIDDEN: RingState = {
  phase: 'hidden',
  phaseStart: 0,
  shownAt: 0,
  fillStart: 0,
  est: DEFAULT_FINAL_MS,
  progress: 0,
};

function fillProgress(elapsed: number, est: number, prevProgress: number): number {
  const p =
    elapsed <= est ? Math.min(0.9, (elapsed / est) * 0.9) : 0.9 + 0.07 * (1 - Math.exp(-(elapsed - est) / est));
  return Math.max(prevProgress, p);
}

export function stepRing(prev: RingState, now: number, outstanding: boolean, currentEst: number): RingState {
  switch (prev.phase) {
    case 'hidden':
      return outstanding ? { ...RING_HIDDEN, phase: 'arming', phaseStart: now, fillStart: now } : prev;
    case 'arming':
      if (!outstanding) return RING_HIDDEN;
      if (now - prev.phaseStart >= APPEAR_DELAY) {
        return {
          phase: 'visible',
          phaseStart: now,
          shownAt: now,
          fillStart: prev.fillStart,
          est: currentEst,
          progress: fillProgress(now - prev.fillStart, currentEst, 0),
        };
      }
      return prev;
    case 'visible':
      if (!outstanding && now - prev.shownAt >= MIN_VISIBLE) {
        return { ...prev, phase: 'snapping', phaseStart: now, progress: 1 };
      }
      return { ...prev, progress: fillProgress(now - prev.fillStart, prev.est, prev.progress) };
    case 'snapping':
      if (outstanding) {
        return {
          phase: 'visible',
          phaseStart: now,
          shownAt: prev.shownAt,
          fillStart: now,
          est: currentEst,
          progress: 0,
        };
      }
      if (now - prev.phaseStart >= SNAP_MS) return { ...prev, phase: 'fading', phaseStart: now };
      return prev;
    case 'fading':
      if (now - prev.phaseStart >= FADE_MS) {
        return outstanding ? { ...RING_HIDDEN, phase: 'arming', phaseStart: now, fillStart: now } : RING_HIDDEN;
      }
      return prev;
  }
}

interface ProcessingRingProps {
  suppressed: boolean;
}

function ProcessingRing({ suppressed }: ProcessingRingProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const arcRef = useRef<SVGCircleElement>(null);
  const ringStateRef = useRef<RingState>(RING_HIDDEN);
  const rafRef = useRef<number>(0);
  const runningRef = useRef(false);
  const lastPhaseRef = useRef<RingPhase>('hidden');
  const reducedMotionRef = useRef(false);

  const isPending = useIsRenderPending();

  const applyRingFrame = useCallback((state: RingState, now: number) => {
    const root = rootRef.current;
    const arc = arcRef.current;
    if (!root || !arc) return;

    arc.style.strokeDashoffset = String(CIRCUMFERENCE * (1 - state.progress));

    let opacity = 0;
    let popProgress = 1;
    if (state.phase === 'visible' || state.phase === 'snapping') {
      popProgress = Math.min(1, (now - state.shownAt) / APPEAR_POP_MS);
      opacity = popProgress;
    } else if (state.phase === 'fading') {
      opacity = Math.max(0, 1 - (now - state.phaseStart) / FADE_MS);
    }
    root.style.opacity = String(opacity);
    root.style.transform = reducedMotionRef.current ? 'scale(1)' : `scale(${0.9 + 0.1 * popProgress})`;

    if (lastPhaseRef.current !== state.phase) {
      lastPhaseRef.current = state.phase;
      arc.style.filter = state.phase === 'snapping' ? GLOW_SNAP : GLOW_NORMAL;
    }
  }, []);

  const startLoop = useCallback(() => {
    if (runningRef.current) return;
    runningRef.current = true;
    const loop = () => {
      const s = useRenderStatusStore.getState();
      const outstanding = s.issuedJobId > s.settledJobId;
      const est = s.newestIsInteractive
        ? (s.emaInteractiveMs ?? DEFAULT_INTERACTIVE_MS)
        : (s.emaFinalMs ?? DEFAULT_FINAL_MS);
      const now = performance.now();
      const next = stepRing(ringStateRef.current, now, outstanding, est);
      ringStateRef.current = next;
      applyRingFrame(next, now);
      if (next.phase === 'hidden') {
        runningRef.current = false;
        return;
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
  }, [applyRingFrame]);

  useLayoutEffect(() => {
    reducedMotionRef.current = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (rootRef.current) rootRef.current.style.opacity = '0';
    if (arcRef.current) {
      arcRef.current.style.strokeDashoffset = String(CIRCUMFERENCE);
      arcRef.current.style.filter = GLOW_NORMAL;
    }
  }, []);

  useEffect(() => {
    if (isPending) startLoop();
  }, [isPending, startLoop]);

  useEffect(() => {
    const unsubscribe = useRenderStatusStore.subscribe((s) => {
      if (s.issuedJobId > s.settledJobId && !runningRef.current) startLoop();
    });
    return () => {
      unsubscribe();
      cancelAnimationFrame(rafRef.current);
      runningRef.current = false;
    };
  }, [startLoop]);

  return (
    <div
      ref={rootRef}
      aria-hidden="true"
      className={clsx('absolute bottom-5 right-5 z-40 pointer-events-none', suppressed && 'hidden')}
    >
      <div
        style={{
          position: 'absolute',
          left: -20,
          top: -20,
          width: 84,
          height: 84,
          background: 'radial-gradient(circle, rgba(8,9,12,0.55) 0%, transparent 72%)',
          pointerEvents: 'none',
        }}
      />
      <svg width={44} height={44} viewBox="0 0 44 44" style={{ position: 'relative', display: 'block' }}>
        <circle cx={22} cy={22} r={RADIUS} fill="none" stroke="#3a3e46" strokeOpacity={0.75} strokeWidth={3} />
        <circle
          ref={arcRef}
          cx={22}
          cy={22}
          r={RADIUS}
          fill="none"
          stroke="#ffffff"
          strokeWidth={3}
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          transform="rotate(-90 22 22)"
        />
      </svg>
    </div>
  );
}

export default memo(ProcessingRing);
