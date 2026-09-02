import { useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import debounce from 'lodash.debounce';
import { toast } from 'react-toastify';
import { useEditorStore } from '../store/useEditorStore';
import { useLibraryStore } from '../store/useLibraryStore';
import { useSettingsStore } from '../store/useSettingsStore';
import { useProcessStore } from '../store/useProcessStore';
import { useUIStore } from '../store/useUIStore';
import {
  Adjustments,
  INITIAL_ADJUSTMENTS,
  COPYABLE_ADJUSTMENT_KEYS,
  PasteMode,
  LensAdjustment,
  completeRecoveryGroups,
  normalizeLoadedAdjustments,
} from '../utils/adjustments';
import {
  calculateCenteredCrop,
  getOrientedDimensions,
  isFullFrameCrop,
  percentToPixelCrop,
  rotatePixelCrop90,
} from '../utils/cropUtils';
import { Crop, PercentCrop } from 'react-image-crop';
import { Invokes, Panel, SelectedImage } from '../components/ui/AppProperties';
import { globalImageCache } from '../utils/ImageLRUCache';

// Scheduled with the snapshot version that was current when the edit was made.
// A later image load, reset, undo/redo or metadata replacement invalidates the
// queued push even if the caller forgot an eager .cancel().
export const debouncedSetHistory = debounce((newAdj: Adjustments, snapshotVersion: number) => {
  const state = useEditorStore.getState();
  if (state.adjustmentsSnapshotVersion !== snapshotVersion) return;
  state.pushHistory(newAdj);
}, 500);

/** Geometric equality, ignoring the `unit` tag. */
function cropsAreEqual(a: Crop | null | undefined, b: Crop | null | undefined): boolean {
  if (!a || !b) return !a && !b;
  return a.x === b.x && a.y === b.y && a.width === b.width && a.height === b.height;
}

/**
 * Fold a pending crop draft into `prev`, canonicalising a full-frame rectangle
 * to `null` (D7). Returns `prev` unchanged when there is no draft, so callers
 * can fold unconditionally.
 */
function foldDraftCrop(prev: Adjustments, draftCrop: PercentCrop | null, selectedImage: SelectedImage | null) {
  if (!draftCrop || !selectedImage?.width || !selectedImage?.height) return prev;

  const orientationSteps = prev.orientationSteps || 0;
  const pixelCrop = percentToPixelCrop(draftCrop, selectedImage.width, selectedImage.height, orientationSteps);
  const { width: W, height: H } = getOrientedDimensions(selectedImage.width, selectedImage.height, orientationSteps);

  return { ...prev, crop: pixelCrop && isFullFrameCrop(pixelCrop, W, H) ? null : pixelCrop };
}

export const debouncedSave = debounce((path: string, adjustmentsToSave: Adjustments) => {
  invoke(Invokes.SaveMetadataAndUpdateThumbnail, { path, adjustments: adjustmentsToSave }).catch((err) => {
    console.error('Auto-save failed:', err);
    toast.error(`Failed to save changes: ${err}`);
  });
}, 300);

export function useEditorActions() {
  const setEditor = useEditorStore((s) => s.setEditor);

  const setAdjustments = useCallback(
    (value: Partial<Adjustments> | ((prev: Adjustments) => Adjustments)) => {
      setEditor((state) => {
        const prev = state.adjustments;
        const newAdjustments = typeof value === 'function' ? value(prev) : { ...prev, ...value };
        // An updater returning `prev` means "nothing changed" -- honour it
        // rather than pushing a duplicate history snapshot.
        if (newAdjustments === prev) return {};
        debouncedSetHistory(newAdjustments, state.adjustmentsSnapshotVersion);
        return { adjustments: newAdjustments };
      });
    },
    [setEditor],
  );

  /**
   * The only way a geometry control may write adjustments while the crop panel
   * is open. Folds any pending drag into the committed crop first, then applies
   * `value` over the folded state, then clears the draft -- all in one store
   * transition, so the geometry effect never observes a stale draft beside new
   * geometry. A `value` carrying its own `crop` overrides the fold, which is
   * what step rotation and an explicit pasted crop rely on.
   */
  const setAdjustmentsFoldingDraft = useCallback(
    (value: Partial<Adjustments> | ((prev: Adjustments) => Adjustments)) => {
      setEditor((state) => {
        const prev = state.adjustments;
        const folded = foldDraftCrop(prev, state.draftCrop, state.selectedImage);
        const next = typeof value === 'function' ? value(folded) : { ...folded, ...value };

        if (next === prev) {
          return state.draftCrop === null ? {} : { draftCrop: null };
        }

        debouncedSetHistory(next, state.adjustmentsSnapshotVersion);
        return { adjustments: next, draftCrop: null };
      });
    },
    [setEditor],
  );

  /**
   * Discrete-action write: no debounce, its own synchronous history entry, and
   * a flush of any older pending edit first so the two land as ordered entries.
   * Returns whether anything was actually written.
   */
  const commitAdjustmentsImmediately = useCallback(
    (value: Partial<Adjustments> | ((prev: Adjustments) => Adjustments)): boolean => {
      const store = useEditorStore.getState();
      const prev = store.adjustments;
      const next = typeof value === 'function' ? value(prev) : { ...prev, ...value };
      if (next === prev) return false;

      debouncedSetHistory.flush();
      store.setEditor({ adjustments: next });
      useEditorStore.getState().pushHistory(next);
      return true;
    },
    [],
  );

  /** Pending draft as an oriented pixel rectangle, or null when there is none. */
  const draftToPixelCrop = useCallback((): Crop | null => {
    const { draftCrop, selectedImage, adjustments } = useEditorStore.getState();
    if (!draftCrop || !selectedImage?.width || !selectedImage?.height) return null;
    return percentToPixelCrop(draftCrop, selectedImage.width, selectedImage.height, adjustments.orientationSteps || 0);
  }, []);

  /**
   * Apply: commit the drafted rectangle and leave the panel.
   *
   * With no draft this is a pure, write-free exit -- which is what makes Enter
   * on an untouched panel free. A full-frame draft commits as `crop: null`
   * (D7), and a draft equal to what is already committed writes nothing and
   * adds no history entry.
   */
  const commitDraftCrop = useCallback(() => {
    const { selectedImage, adjustments, draftCrop } = useEditorStore.getState();
    const draftPx = draftToPixelCrop();

    if (draftPx && selectedImage?.width && selectedImage?.height) {
      const { width: W, height: H } = getOrientedDimensions(
        selectedImage.width,
        selectedImage.height,
        adjustments.orientationSteps || 0,
      );
      const canonicalDraft = isFullFrameCrop(draftPx, W, H) ? null : draftPx;

      commitAdjustmentsImmediately((prev) =>
        cropsAreEqual(prev.crop, canonicalDraft) ? prev : { ...prev, crop: canonicalDraft },
      );
    }

    if (draftCrop !== null) {
      useEditorStore.getState().setEditor({ draftCrop: null });
    }

    // setRightPanel toggles the panel closed when handed the active one, so
    // only switch while Crop is still open. That keeps a second activation
    // (button plus Enter, say) idempotent.
    const { activeRightPanel, setRightPanel } = useUIStore.getState();
    if (activeRightPanel === Panel.Crop) {
      setRightPanel(Panel.Adjustments);
    }
  }, [commitAdjustmentsImmediately, draftToPixelCrop]);

  // History navigation flushes the pending edit first, so it becomes its own
  // entry and the move lands where the user expects.
  const undoAdjustments = useCallback(() => {
    debouncedSetHistory.flush();
    useEditorStore.getState().undo();
  }, []);

  const redoAdjustments = useCallback(() => {
    debouncedSetHistory.flush();
    useEditorStore.getState().redo();
  }, []);

  const goToAdjustmentsHistoryIndex = useCallback((index: number) => {
    debouncedSetHistory.flush();
    useEditorStore.getState().goToHistoryIndex(index);
  }, []);

  /**
   * 90-degree step rotation, shared by the crop panel's buttons and the
   * keyboard action so both take the same branch.
   *
   * With a pending drag the drawn rectangle is folded in and then mapped into
   * the new oriented frame, so an off-centre crop stays where the user put it.
   * With no draft this keeps the historical centred-crop behaviour.
   */
  const handleRotate = useCallback(
    (degrees: number) => {
      const { selectedImage, draftCrop } = useEditorStore.getState();
      const increment = degrees > 0 ? 1 : 3;
      const direction = increment === 1 ? 'cw' : 'ccw';
      const hadDraft = draftCrop !== null;

      setAdjustmentsFoldingDraft((prev) => {
        const newAspectRatio = prev.aspectRatio && prev.aspectRatio !== 0 ? 1 / prev.aspectRatio : null;
        const newOrientationSteps = ((prev.orientationSteps || 0) + increment) % 4;

        let newCrop: Crop | null = null;
        if (selectedImage?.width && selectedImage?.height) {
          // `prev` is the folded state, so prev.crop already carries the drag.
          if (hadDraft && prev.crop) {
            const from = getOrientedDimensions(selectedImage.width, selectedImage.height, prev.orientationSteps || 0);
            const to = getOrientedDimensions(selectedImage.width, selectedImage.height, newOrientationSteps);
            const mapped = rotatePixelCrop90(prev.crop, from.width, from.height, direction);
            newCrop = isFullFrameCrop(mapped, to.width, to.height) ? null : mapped;
          } else {
            newCrop = calculateCenteredCrop(
              selectedImage.width,
              selectedImage.height,
              newOrientationSteps,
              newAspectRatio,
            );
          }
        }

        return {
          ...prev,
          aspectRatio: newAspectRatio,
          orientationSteps: newOrientationSteps,
          rotation: 0,
          crop: newCrop,
        };
      });
    },
    [setAdjustmentsFoldingDraft],
  );

  const handleAutoAdjustments = useCallback(async () => {
    const selectedImage = useEditorStore.getState().selectedImage;
    if (!selectedImage?.isReady) return;
    try {
      const autoAdjustments: Adjustments = await invoke(Invokes.CalculateAutoAdjustments);
      setAdjustments((prev: Adjustments) => ({
        ...prev,
        ...autoAdjustments,
      }));
    } catch (err) {
      toast.error(`Failed to apply auto adjustments: ${err}`);
    }
  }, [setAdjustments]);

  const handleLutSelect = useCallback(
    async (path: string) => {
      const isAndroid = useSettingsStore.getState().osPlatform === 'android';
      try {
        const result: { size: number } = await invoke('load_and_parse_lut', { path });
        let name = isAndroid && path.startsWith('content://')
          ? await invoke<string>('resolve_android_content_uri_name', { uriStr: path })
          : path.split(/[\\/]/).pop() || 'LUT';
        setAdjustments((prev: Adjustments) => ({
          ...prev,
          lutPath: path,
          lutName: name,
          lutSize: result.size,
          lutIntensity: 100,
        }));
      } catch (err) {
        toast.error(`Failed to load LUT: ${err}`);
      }
    },
    [setAdjustments],
  );

  const setLutPreviewOverride = useCallback(
    (path: string | null) => {
      setEditor((state) => {
        if (!path) return { previewOverride: null };
        const name = path.split(/[\\/]/).pop() || 'LUT';
        return {
          previewOverride: {
            ...state.adjustments,
            lutPath: path,
            lutName: name,
            lutIntensity: state.adjustments.lutIntensity,
          },
        };
      });
    },
    [setEditor],
  );

  const handleResetAdjustments = useCallback((paths?: string[]) => {
    const { multiSelectedPaths, libraryActivePath, setLibrary } = useLibraryStore.getState();
    const { selectedImage, resetHistory } = useEditorStore.getState();
    const pathsToReset = paths || multiSelectedPaths;
    if (pathsToReset.length === 0) return;

    pathsToReset.forEach((p) => globalImageCache.delete(p));
    debouncedSetHistory.cancel();

    invoke(Invokes.ResetAdjustmentsForPaths, { paths: pathsToReset })
      .then(() => {
        if (libraryActivePath && pathsToReset.includes(libraryActivePath))
          setLibrary({ libraryActiveAdjustments: { ...INITIAL_ADJUSTMENTS } });
        if (selectedImage && pathsToReset.includes(selectedImage.path)) {
          const aspect =
            selectedImage.width && selectedImage.height ? selectedImage.width / selectedImage.height : null;
          const resetData = { ...INITIAL_ADJUSTMENTS, aspectRatio: aspect, aiPatches: [] };
          resetHistory(resetData);
        }
      })
      .catch((err) => toast.error(`Failed to reset adjustments: ${err}`));
  }, []);

  const handleCopyAdjustments = useCallback(async (pathOrEvent?: string | any) => {
    const pathOverride = typeof pathOrEvent === 'string' ? pathOrEvent : undefined;
    const { selectedImage, adjustments } = useEditorStore.getState();
    const { libraryActivePath, multiSelectedPaths } = useLibraryStore.getState();
    let sourceAdjustments: any = null;

    const pathToCopyFrom =
      pathOverride || (selectedImage ? selectedImage.path : libraryActivePath || multiSelectedPaths[0]);

    if (selectedImage && pathToCopyFrom === selectedImage.path) {
      sourceAdjustments = adjustments;
    } else if (pathToCopyFrom) {
      try {
        const meta: any = await invoke(Invokes.LoadMetadata, { path: pathToCopyFrom });
        if (meta?.adjustments && !meta.adjustments.is_null) {
          sourceAdjustments = normalizeLoadedAdjustments(meta.adjustments);
        } else {
          sourceAdjustments = INITIAL_ADJUSTMENTS;
        }
      } catch (err) {
        toast.error(`Failed to load metadata for copying: ${err}`);
        return;
      }
    }

    if (!sourceAdjustments) return;

    const adjustmentsToCopy: any = {};

    for (const key of COPYABLE_ADJUSTMENT_KEYS) {
      if (Object.prototype.hasOwnProperty.call(sourceAdjustments, key)) {
        adjustmentsToCopy[key] = structuredClone(sourceAdjustments[key]);
      }
    }
    useEditorStore.getState().setEditor({ copiedAdjustments: adjustmentsToCopy });
    useProcessStore.getState().setProcess({ isCopied: true });
  }, []);

  const handlePasteAdjustments = useCallback(
    (paths?: string[]) => {
      const { copiedAdjustments, selectedImage } = useEditorStore.getState();
      const { multiSelectedPaths } = useLibraryStore.getState();
      const { appSettings } = useSettingsStore.getState();
      const { setProcess } = useProcessStore.getState();

      if (!copiedAdjustments || !appSettings) return;

      const { mode, includedAdjustments } = appSettings.copyPasteSettings;
      const adjustmentsToApply: Partial<Adjustments> = {};

      for (const key of includedAdjustments) {
        if (Object.prototype.hasOwnProperty.call(copiedAdjustments, key)) {
          const value = copiedAdjustments[key as keyof Adjustments];
          if (mode === PasteMode.Merge) {
            const defaultValue = INITIAL_ADJUSTMENTS[key as keyof Adjustments];
            if (JSON.stringify(value) !== JSON.stringify(defaultValue))
              adjustmentsToApply[key as keyof Adjustments] = value;
          } else {
            adjustmentsToApply[key as keyof Adjustments] = value;
          }
        }
      }
      // Merge mode strips INITIAL-equal keys, which would leave recovery
      // subsystems half-carried (e.g. a toggle without its strength) and
      // break the absent-key-means-legacy rule on the receiving side.
      completeRecoveryGroups(adjustmentsToApply, copiedAdjustments);

      if (includedAdjustments.includes(LensAdjustment.LensMaker)) {
        if (!adjustmentsToApply.lensMaker) {
          adjustmentsToApply.lensDistortionParams = null;
        }
      }

      if (Object.keys(adjustmentsToApply).length === 0) {
        setProcess({ isPasted: true });
        return;
      }

      const pathsToUpdate =
        paths || (multiSelectedPaths.length > 0 ? multiSelectedPaths : selectedImage ? [selectedImage.path] : []);
      if (pathsToUpdate.length === 0) return;

      pathsToUpdate.forEach((p) => globalImageCache.delete(p));

      if (selectedImage && pathsToUpdate.includes(selectedImage.path)) {
        // Only the patch. Passing the whole stale adjustments object would
        // carry its old crop over a pending draft and undo the fold; if the
        // patch itself includes crop, that pasted crop intentionally wins.
        setAdjustmentsFoldingDraft(adjustmentsToApply);
      }

      invoke(Invokes.ApplyAdjustmentsToPaths, { paths: pathsToUpdate, adjustments: adjustmentsToApply })
        .then(() => {
          if (selectedImage && pathsToUpdate.includes(selectedImage.path)) {
            invoke('load_metadata', { path: selectedImage.path }).then((meta: any) => {
              if (meta.adjustments) {
                setAdjustments((prev: any) => ({
                  ...prev,
                  lensMaker: meta.adjustments.lensMaker,
                  lensModel: meta.adjustments.lensModel,
                  lensDistortionParams: meta.adjustments.lensDistortionParams,
                }));
              }
            });
          }
        })
        .catch((err) => toast.error(`Failed to paste adjustments: ${err}`));

      setProcess({ isPasted: true });
    },
    [setAdjustments],
  );

  const handleZoomChange = useCallback((zoomValue: number, fitToWindow: boolean = false) => {
    const { originalSize, baseRenderSize, adjustments } = useEditorStore.getState();
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    let targetZoomPercent: number;

    const orientationSteps = adjustments.orientationSteps || 0;
    const isSwapped = orientationSteps === 1 || orientationSteps === 3;
    const effectiveOriginalWidth = isSwapped ? originalSize.height : originalSize.width;
    const effectiveOriginalHeight = isSwapped ? originalSize.width : originalSize.height;

    if (fitToWindow) {
      if (
        effectiveOriginalWidth > 0 &&
        effectiveOriginalHeight > 0 &&
        baseRenderSize.width > 0 &&
        baseRenderSize.height > 0
      ) {
        const originalAspect = effectiveOriginalWidth / effectiveOriginalHeight;
        const baseAspect = baseRenderSize.width / baseRenderSize.height;
        targetZoomPercent =
          originalAspect > baseAspect
            ? baseRenderSize.width / effectiveOriginalWidth
            : baseRenderSize.height / effectiveOriginalHeight;
      } else {
        targetZoomPercent = 1.0;
      }
    } else {
      targetZoomPercent = zoomValue / dpr;
    }

    targetZoomPercent = Math.max(0.1 / dpr, Math.min(2.0, targetZoomPercent));

    let transformZoom = 1.0;
    if (
      effectiveOriginalWidth > 0 &&
      effectiveOriginalHeight > 0 &&
      baseRenderSize.width > 0 &&
      baseRenderSize.height > 0
    ) {
      const originalAspect = effectiveOriginalWidth / effectiveOriginalHeight;
      const baseAspect = baseRenderSize.width / baseRenderSize.height;
      if (originalAspect > baseAspect) {
        transformZoom = (targetZoomPercent * effectiveOriginalWidth) / baseRenderSize.width;
      } else {
        transformZoom = (targetZoomPercent * effectiveOriginalHeight) / baseRenderSize.height;
      }
    }
    useEditorStore.getState().setEditor({ zoom: transformZoom });
  }, []);

  return {
    setAdjustments,
    setAdjustmentsFoldingDraft,
    commitAdjustmentsImmediately,
    draftToPixelCrop,
    commitDraftCrop,
    undoAdjustments,
    redoAdjustments,
    goToAdjustmentsHistoryIndex,
    handleRotate,
    handleAutoAdjustments,
    handleLutSelect,
    setLutPreviewOverride,
    handleResetAdjustments,
    handleCopyAdjustments,
    handlePasteAdjustments,
    handleZoomChange,
  };
}
