import debounce from 'lodash.debounce';
import { invoke } from '@tauri-apps/api/core';
import { toast } from 'react-toastify';
import { useEditorStore } from './useEditorStore';
import type { Adjustments } from '../utils/adjustments';
import { ImageIdentity, readyImageIdentity, sameImage } from '../utils/imageIdentity';
import { Invokes } from '../components/ui/AppProperties';

// Scheduled with the snapshot version that was current when the edit was made.
// A later image load, reset, undo/redo or metadata replacement invalidates the
// queued push even if the caller forgot an eager .cancel().
export const debouncedSetHistory = debounce((newAdj: Adjustments, snapshotVersion: number) => {
  const state = useEditorStore.getState();
  if (state.adjustmentsSnapshotVersion !== snapshotVersion) return;
  state.pushHistory(newAdj);
}, 500);

export const debouncedSave = debounce((path: string, adjustmentsToSave: Adjustments, identity?: ImageIdentity) => {
  const state = useEditorStore.getState();
  if (
    identity &&
    (!sameImage(identity, readyImageIdentity(state.selectedImage)) || state.adjustments !== adjustmentsToSave)
  )
    return;
  invoke(Invokes.SaveMetadataAndUpdateThumbnail, { path, adjustments: adjustmentsToSave }).catch((err) => {
    if (identity && !sameImage(identity, readyImageIdentity(useEditorStore.getState().selectedImage))) return;
    console.error('Auto-save failed:', err);
    toast.error(`Failed to save changes: ${err}`);
  });
}, 300);
