import { useTranslation } from 'react-i18next';
import { Info } from 'lucide-react';
import { toast } from 'react-toastify';

import Slider from '../ui/Slider';
import Switch from '../ui/Switch';
import { Adjustments, GlareRecoveryAdjustment } from '../../utils/adjustments';
import { useEstimate } from '../../hooks/useEstimate';

interface GlareRecoveryPanelProps {
  adjustments: Adjustments;
  isVisible?: boolean;
  setAdjustments(adjustments: Partial<Adjustments> | ((prev: Adjustments) => Partial<Adjustments>)): any;
  onDragStateChange?(dragging: boolean): void;
}

export default function GlareRecoveryPanel({
  adjustments,
  setAdjustments,
  onDragStateChange,
  isVisible,
}: GlareRecoveryPanelProps) {
  const { t } = useTranslation();
  const { isEstimating, isFlashing, estimate, dismissFlash } = useEstimate(
    'glare',
    (message) => toast.error(`${t('editor.adjustments.glareRecovery.estimateFailed')} (${message})`),
    isVisible,
  );

  const handleValueChange = (key: GlareRecoveryAdjustment, e: any) => {
    const numericValue = parseFloat(e.target.value);
    setAdjustments((prev: Adjustments) => ({ ...prev, [key]: numericValue }));
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
              onClick={estimate}
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
                checked={!!adjustments.glareShowVeil || isFlashing}
                onChange={(checked: boolean) => {
                  dismissFlash();
                  setAdjustments((prev: Adjustments) => ({
                    ...prev,
                    [GlareRecoveryAdjustment.GlareShowVeil]: checked,
                  }));
                }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
