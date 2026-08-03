import { useTranslation } from 'react-i18next';
import { Info } from 'lucide-react';

import Slider from '../ui/Slider';
import Switch from '../ui/Switch';
import { Adjustments, GlareRecoveryAdjustment } from '../../utils/adjustments';

interface GlareRecoveryPanelProps {
  adjustments: Adjustments;
  setAdjustments(adjustments: Partial<Adjustments> | ((prev: Adjustments) => Partial<Adjustments>)): any;
  onDragStateChange?(dragging: boolean): void;
}

// No enable switch by design: Amount 0 disables the stage entirely, and Show
// veil previews the estimated veil even at Amount 0.
export default function GlareRecoveryPanel({
  adjustments,
  setAdjustments,
  onDragStateChange,
}: GlareRecoveryPanelProps) {
  const { t } = useTranslation();

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
        <div className="space-y-2">
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
            onChange={(e: any) => handleValueChange(GlareRecoveryAdjustment.GlareVeilSize, e)}
            step={1}
            value={adjustments.glareVeilSize}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={t('editor.adjustments.glareRecovery.maxBoost')}
            max={100}
            min={0}
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
      </div>
    </div>
  );
}
