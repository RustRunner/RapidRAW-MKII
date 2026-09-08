import { useTranslation } from 'react-i18next';
import { Info } from 'lucide-react';
import { toast } from 'react-toastify';

import Slider from '../ui/Slider';
import Switch from '../ui/Switch';
import { Adjustments, LowLightAdjustment } from '../../utils/adjustments';
import { useEstimate } from '../../hooks/useEstimate';
import type { NoiseEstimate } from '../../utils/imageIdentity';

interface LowLightPanelProps {
  adjustments: Adjustments;
  isVisible?: boolean;
  setAdjustments(adjustments: Partial<Adjustments> | ((prev: Adjustments) => Partial<Adjustments>)): any;
  onDragStateChange?(dragging: boolean): void;
}

export default function LowLightPanel({
  adjustments,
  setAdjustments,
  onDragStateChange,
  isVisible,
}: LowLightPanelProps) {
  const { t } = useTranslation();
  const { isEstimating, result, estimate } = useEstimate(
    'denoise',
    (message) => toast.error(`${t('editor.adjustments.lowlight.estimateFailed')} (${message})`),
    isVisible,
  );
  const noiseEstimate = result as NoiseEstimate | undefined;

  const handleValueChange = (key: LowLightAdjustment, e: any) => {
    const numericValue = parseFloat(e.target.value);
    setAdjustments((prev: Adjustments) => ({ ...prev, [key]: numericValue }));
  };

  const handleCheckedChange = (key: LowLightAdjustment, checked: boolean) => {
    setAdjustments((prev: Adjustments) => ({ ...prev, [key]: checked }));
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
            {/* The shader's detection threshold is inverted (0 flags every
                deviating pixel, 100 flags none), so the display shows
                Sensitivity = 100 - stored: 0 = detect nothing (inert,
                the backend gates the stage off), 100 = flag everything.
                Stored sidecar values keep their threshold meaning. */}
            <Slider
              label={t('editor.adjustments.lowlight.sensitivity')}
              max={100}
              min={0}
              defaultValue={0}
              onChange={(e: any) =>
                setAdjustments((prev: Adjustments) => ({
                  ...prev,
                  [LowLightAdjustment.HotPixelThreshold]: 100 - parseFloat(e.target.value),
                }))
              }
              step={1}
              value={100 - adjustments.hotPixelThreshold}
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
            <p className="text-xs text-text-secondary">{t('editor.adjustments.lowlight.denoiseDescription')}</p>
            <button
              className={`w-full py-2 px-4 rounded font-medium text-sm transition-colors border-2 ${
                isEstimating
                  ? 'bg-gray-500/20 text-gray-300 border-gray-500 cursor-wait'
                  : 'bg-transparent text-primary border-primary hover:bg-primary hover:text-white'
              }`}
              onClick={estimate}
              disabled={isEstimating}
            >
              {isEstimating ? t('editor.adjustments.lowlight.estimating') : t('editor.adjustments.lowlight.estimate')}
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
              defaultValue={50}
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
