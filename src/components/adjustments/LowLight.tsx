import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { invoke } from '@tauri-apps/api/core';
import { Info } from 'lucide-react';
import { toast } from 'react-toastify';

import Slider from '../ui/Slider';
import Switch from '../ui/Switch';
import { Adjustments, LowLightAdjustment } from '../../utils/adjustments';
import { Invokes } from '../ui/AppProperties';
import { useEditorStore } from '../../store/useEditorStore';

interface NoiseEstimate {
  sigma_luma: number;
  sigma_chroma: number;
  strength: number;
  chroma: number;
}

interface LowLightPanelProps {
  adjustments: Adjustments;
  setAdjustments(adjustments: Partial<Adjustments> | ((prev: Adjustments) => Partial<Adjustments>)): any;
  onDragStateChange?(dragging: boolean): void;
}

export default function LowLightPanel({ adjustments, setAdjustments, onDragStateChange }: LowLightPanelProps) {
  const { t } = useTranslation();
  const selectedImage = useEditorStore((state: any) => state.selectedImage);
  const [isEstimating, setIsEstimating] = useState(false);
  const [noiseEstimate, setNoiseEstimate] = useState<NoiseEstimate | null>(null);

  const path: string | null = selectedImage?.path ?? null;

  useEffect(() => {
    setNoiseEstimate(null);
  }, [path]);

  const handleValueChange = (key: LowLightAdjustment, e: any) => {
    const numericValue = parseFloat(e.target.value);
    setAdjustments((prev: Adjustments) => ({ ...prev, [key]: numericValue }));
  };

  const handleCheckedChange = (key: LowLightAdjustment, checked: boolean) => {
    setAdjustments((prev: Adjustments) => ({ ...prev, [key]: checked }));
  };

  const handleEstimateNoise = async () => {
    if (isEstimating) {
      return;
    }
    setIsEstimating(true);
    try {
      const estimate = await invoke<NoiseEstimate>(Invokes.EstimateNoiseLevel);
      const strength = Math.round(estimate.strength);
      const chroma = Math.round(estimate.chroma);
      setNoiseEstimate(estimate);
      setAdjustments((prev: Adjustments) => ({
        ...prev,
        [LowLightAdjustment.DenoiseStrength]: strength,
        [LowLightAdjustment.DenoiseChroma]: chroma,
      }));
    } catch (err) {
      toast.error(`${t('editor.adjustments.lowlight.estimateFailed')} (${err})`);
    } finally {
      setIsEstimating(false);
    }
  };

  return (
    <div>
      <div className="mb-4 p-2 bg-bg-secondary rounded-md flex items-start gap-2">
        <Info size={14} className="text-text-secondary mt-0.5 flex-shrink-0" />
        <p className="text-xs text-text-secondary">{t('editor.adjustments.lowlight.description')}</p>
      </div>

      <div className="mb-4 p-2 bg-bg-tertiary rounded-md">
        <div className="flex items-center justify-between mb-2">
          <p className="text-sm font-medium text-text-primary">{t('editor.adjustments.lowlight.hotPixels')}</p>
          <Switch
            id="hot-pixel-toggle"
            label=""
            checked={!!adjustments.hotPixelEnabled}
            onChange={(checked: boolean) => handleCheckedChange(LowLightAdjustment.HotPixelEnabled, checked)}
          />
        </div>
        {adjustments.hotPixelEnabled && (
          <div className="space-y-2 pt-2 border-t border-bg-secondary">
            <Slider
              label={t('editor.adjustments.lowlight.threshold')}
              max={100}
              min={0}
              onChange={(e: any) => handleValueChange(LowLightAdjustment.HotPixelThreshold, e)}
              step={1}
              value={adjustments.hotPixelThreshold}
              onDragStateChange={onDragStateChange}
            />
          </div>
        )}
      </div>

      <div className="mb-4 p-2 bg-bg-tertiary rounded-md">
        <div className="flex items-center justify-between mb-2">
          <p className="text-sm font-medium text-text-primary">{t('editor.adjustments.lowlight.denoise')}</p>
          <Switch
            id="denoise-toggle"
            label=""
            checked={!!adjustments.denoiseEnabled}
            onChange={(checked: boolean) => handleCheckedChange(LowLightAdjustment.DenoiseEnabled, checked)}
          />
        </div>
        {adjustments.denoiseEnabled && (
          <div className="space-y-2 pt-2 border-t border-bg-secondary">
            <button
              className={`w-full py-2 px-4 rounded font-medium text-sm transition-colors border-2 ${
                isEstimating
                  ? 'bg-gray-500/20 text-gray-300 border-gray-500 cursor-wait'
                  : 'bg-transparent text-primary border-primary hover:bg-primary hover:text-white'
              }`}
              onClick={handleEstimateNoise}
              disabled={isEstimating}
            >
              {isEstimating
                ? t('editor.adjustments.lowlight.estimating')
                : t('editor.adjustments.lowlight.estimate')}
            </button>
            {noiseEstimate && (
              <div className="p-2 bg-bg-secondary rounded text-xs text-text-secondary">
                {t('editor.adjustments.lowlight.measured', {
                  strength: Math.round(noiseEstimate.strength),
                  chroma: Math.round(noiseEstimate.chroma),
                })}
              </div>
            )}
            <Slider
              label={t('editor.adjustments.lowlight.strength')}
              max={100}
              min={0}
              onChange={(e: any) => handleValueChange(LowLightAdjustment.DenoiseStrength, e)}
              step={1}
              value={adjustments.denoiseStrength}
              onDragStateChange={onDragStateChange}
            />
            <Slider
              label={t('editor.adjustments.lowlight.detail')}
              max={100}
              min={0}
              onChange={(e: any) => handleValueChange(LowLightAdjustment.DenoiseDetail, e)}
              step={1}
              value={adjustments.denoiseDetail}
              onDragStateChange={onDragStateChange}
            />
            <Slider
              label={t('editor.adjustments.lowlight.chroma')}
              max={100}
              min={0}
              onChange={(e: any) => handleValueChange(LowLightAdjustment.DenoiseChroma, e)}
              step={1}
              value={adjustments.denoiseChroma}
              onDragStateChange={onDragStateChange}
            />
          </div>
        )}
      </div>
    </div>
  );
}
