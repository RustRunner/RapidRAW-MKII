import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { invoke } from '@tauri-apps/api/core';
import { Info } from 'lucide-react';
import { toast } from 'react-toastify';

import Slider from '../ui/Slider';
import Switch from '../ui/Switch';
import { Adjustments, GlareRecoveryAdjustment } from '../../utils/adjustments';
import { Invokes } from '../ui/AppProperties';
import { useEditorStore } from '../../store/useEditorStore';
import {
  GlareEstimate,
  OwnedEstimate,
  readyImageIdentity,
  sameImage,
  isStaleEstimateError,
  estimateErrorMessage,
} from '../../utils/imageIdentity';

interface GlareRecoveryPanelProps {
  adjustments: Adjustments;
  setAdjustments(adjustments: Partial<Adjustments> | ((prev: Adjustments) => Partial<Adjustments>)): any;
  onDragStateChange?(dragging: boolean): void;
}

const VEIL_FLASH_MS = 1200;

export default function GlareRecoveryPanel({
  adjustments,
  setAdjustments,
  onDragStateChange,
}: GlareRecoveryPanelProps) {
  const { t } = useTranslation();
  const [isEstimating, setIsEstimating] = useState(false);
  const selectedImage = useEditorStore((s) => s.selectedImage);
  useEffect(() => setIsEstimating(false), [selectedImage?.path, selectedImage?.identity?.generation]);
  const flashTimeoutRef = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (flashTimeoutRef.current !== null) {
        clearTimeout(flashTimeoutRef.current);
      }
    },
    [],
  );

  const handleValueChange = (key: GlareRecoveryAdjustment, e: any) => {
    const numericValue = parseFloat(e.target.value);
    setAdjustments((prev: Adjustments) => ({ ...prev, [key]: numericValue }));
  };

  const handleEstimateGlare = async () => {
    const expectedIdentity = readyImageIdentity(useEditorStore.getState().selectedImage);
    if (isEstimating || !expectedIdentity) {
      return;
    }
    setIsEstimating(true);
    try {
      const result = await invoke<OwnedEstimate<GlareEstimate>>(Invokes.EstimateGlareVeil, { expectedIdentity });
      if (
        !sameImage(result.identity, expectedIdentity) ||
        !sameImage(readyImageIdentity(useEditorStore.getState().selectedImage), expectedIdentity)
      )
        return;
      const estimate = result.estimate;
      if (!estimate?.confident) {
        toast.error(t('editor.adjustments.glareRecovery.estimateFailed'));
        return;
      }
      const clamp = (v: number) => Math.min(100, Math.max(0, Math.round(v)));
      // Apply the suggestion and flash the veil so the user sees the gray
      // field about to be subtracted. A landed estimate must never leave
      // the stage off.
      setAdjustments((prev: Adjustments) => ({
        ...prev,
        [GlareRecoveryAdjustment.GlareEnabled]: true,
        [GlareRecoveryAdjustment.GlareAmount]: clamp(estimate.amount),
        [GlareRecoveryAdjustment.GlareVeilSize]: clamp(estimate.veilSize),
        [GlareRecoveryAdjustment.GlareMaxBoost]: clamp(estimate.maxBoost),
        [GlareRecoveryAdjustment.GlareShowVeil]: true,
      }));
      if (flashTimeoutRef.current !== null) {
        clearTimeout(flashTimeoutRef.current);
      }
      flashTimeoutRef.current = window.setTimeout(() => {
        flashTimeoutRef.current = null;
        if (!sameImage(readyImageIdentity(useEditorStore.getState().selectedImage), expectedIdentity)) return;
        setAdjustments((prev: Adjustments) => ({
          ...prev,
          [GlareRecoveryAdjustment.GlareShowVeil]: false,
        }));
      }, VEIL_FLASH_MS);
    } catch (err) {
      if (
        sameImage(readyImageIdentity(useEditorStore.getState().selectedImage), expectedIdentity) &&
        !isStaleEstimateError(err)
      ) {
        toast.error(`${t('editor.adjustments.glareRecovery.estimateFailed')} (${estimateErrorMessage(err)})`);
      }
    } finally {
      if (sameImage(readyImageIdentity(useEditorStore.getState().selectedImage), expectedIdentity))
        setIsEstimating(false);
    }
  };

  return (
    <div>
      <div className="mb-4 p-2 bg-bg-secondary rounded-md flex items-start gap-2">
        <Info size={14} className="text-text-secondary mt-0.5 flex-shrink-0" />
        <p className="text-xs text-text-secondary">{t('editor.adjustments.glareRecovery.description')}</p>
      </div>

      <div className="mb-4 p-2 bg-bg-tertiary rounded-md">
        <div className="flex items-center justify-between">
          <p className="text-sm font-medium text-text-primary">{t('editor.adjustments.glareRecovery.enable')}</p>
          <Switch
            id="glare-enable-toggle"
            label=""
            checked={!!adjustments.glareEnabled}
            onChange={(checked: boolean) =>
              setAdjustments((prev: Adjustments) => ({
                ...prev,
                [GlareRecoveryAdjustment.GlareEnabled]: checked,
              }))
            }
          />
        </div>
        {adjustments.glareEnabled && (
          <div className="space-y-2 pt-2 mt-2 border-t border-bg-secondary">
            <button
              className={`w-full py-2 px-4 rounded font-medium text-sm transition-colors border-2 ${
                isEstimating
                  ? 'bg-gray-500/20 text-gray-300 border-gray-500 cursor-wait'
                  : 'bg-transparent text-primary border-primary hover:bg-primary hover:text-white'
              }`}
              onClick={handleEstimateGlare}
              disabled={isEstimating}
            >
              {isEstimating
                ? t('editor.adjustments.glareRecovery.estimating')
                : t('editor.adjustments.glareRecovery.estimate')}
            </button>
            <Slider
              label={t('editor.adjustments.glareRecovery.amount')}
              max={100}
              min={0}
              onChange={(e: any) => handleValueChange(GlareRecoveryAdjustment.GlareAmount, e)}
              step={1}
              value={adjustments.glareAmount}
              onDragStateChange={onDragStateChange}
            />
            <Slider
              label={t('editor.adjustments.glareRecovery.veilSize')}
              max={100}
              min={0}
              defaultValue={50}
              onChange={(e: any) => handleValueChange(GlareRecoveryAdjustment.GlareVeilSize, e)}
              step={1}
              value={adjustments.glareVeilSize}
              onDragStateChange={onDragStateChange}
            />
            <Slider
              label={t('editor.adjustments.glareRecovery.maxBoost')}
              max={100}
              min={0}
              defaultValue={50}
              onChange={(e: any) => handleValueChange(GlareRecoveryAdjustment.GlareMaxBoost, e)}
              step={1}
              value={adjustments.glareMaxBoost}
              onDragStateChange={onDragStateChange}
            />
            <div className="flex items-center justify-between pt-2 border-t border-bg-secondary">
              <p className="text-sm font-medium text-text-primary">{t('editor.adjustments.glareRecovery.showVeil')}</p>
              <Switch
                id="glare-show-veil-toggle"
                label=""
                checked={!!adjustments.glareShowVeil}
                onChange={(checked: boolean) =>
                  setAdjustments((prev: Adjustments) => ({
                    ...prev,
                    [GlareRecoveryAdjustment.GlareShowVeil]: checked,
                  }))
                }
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
