import React, { useCallback, useRef } from 'react';

interface SplitViewLineProps {
  pos: number;
  onPosChange: (pos: number) => void;
}

export default function SplitViewLine({ pos, onPosChange }: SplitViewLineProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const isDraggingRef = useRef(false);

  const handlePointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    e.stopPropagation();
    e.preventDefault();
    isDraggingRef.current = true;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, []);

  const handlePointerMove = useCallback(
    (e: React.PointerEvent<HTMLDivElement>) => {
      if (!isDraggingRef.current) {
        return;
      }
      const rect = rootRef.current?.getBoundingClientRect();
      if (!rect || rect.width <= 0) {
        return;
      }
      e.stopPropagation();
      onPosChange(Math.min(0.95, Math.max(0.05, (e.clientX - rect.left) / rect.width)));
    },
    [onPosChange],
  );

  const handlePointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    isDraggingRef.current = false;
    (e.target as HTMLElement).releasePointerCapture?.(e.pointerId);
  }, []);

  return (
    <div ref={rootRef} className="absolute inset-0 z-40 pointer-events-none">
      <div
        style={{
          position: 'absolute',
          left: `${pos * 100}%`,
          top: 0,
          height: '100%',
          width: '0px',
        }}
      >
        <div
          style={{
            position: 'absolute',
            left: '-1px',
            top: 0,
            width: '2px',
            height: '100%',
            background: 'rgba(255, 255, 255, 0.85)',
            boxShadow: '0 0 4px rgba(0, 0, 0, 0.6)',
          }}
        />
        <div
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          style={{
            position: 'absolute',
            left: '-9px',
            top: 0,
            width: '18px',
            height: '100%',
            cursor: 'ew-resize',
            touchAction: 'none',
            pointerEvents: 'auto',
          }}
        />
        <div
          style={{
            position: 'absolute',
            left: '-12px',
            top: 'calc(50% - 12px)',
            width: '24px',
            height: '24px',
            borderRadius: '50%',
            background: 'rgba(255, 255, 255, 0.9)',
            boxShadow: '0 0 4px rgba(0, 0, 0, 0.6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: 'rgba(0, 0, 0, 0.75)',
            fontSize: '11px',
            lineHeight: 1,
            userSelect: 'none',
          }}
        >
          {'◀▶'}
        </div>
      </div>
    </div>
  );
}
