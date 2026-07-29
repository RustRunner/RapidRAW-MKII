import { useTranslation } from 'react-i18next';
import { Info } from 'lucide-react';

import Slider from '../ui/Slider';
import Switch from '../ui/Switch';
import { Adjustments, BlurRecoveryAdjustment } from '../../utils/adjustments';
import { useEditorStore } from '../../store/useEditorStore';

interface BlurRecoveryPanelProps {
  adjustments: Adjustments;
  setAdjustments(adjustments: Partial<Adjustments> | ((prev: Adjustments) => Partial<Adjustments>)): any;
  onDragStateChange?(dragging: boolean): void;
}

const BLUR_TYPES = ['motion', 'defocus', 'gaussian'] as const;
const MOTION_LENGTH_PRESETS = [50, 100, 150, 200];
const DEFOCUS_RADIUS_PRESETS = [25, 50, 75, 100];

export default function BlurRecoveryPanel({ adjustments, setAdjustments, onDragStateChange }: BlurRecoveryPanelProps) {
  const { t } = useTranslation();
  const setEditor = useEditorStore((state: any) => state.setEditor);

  const handleValueChange = (key: BlurRecoveryAdjustment, e: any) => {
    const numericValue = parseFloat(e.target.value);
    setAdjustments((prev: Adjustments) => ({ ...prev, [key]: numericValue }));
  };

  const handleAngleChange = (e: any) => {
    const numericValue = parseFloat(e.target.value);
    setEditor({ blurOverlayAngle: numericValue });
    setAdjustments((prev: Adjustments) => ({ ...prev, [BlurRecoveryAdjustment.RapidAngle]: numericValue }));
  };

  const handleAngleDragState = (dragging: boolean) => {
    setEditor({ isBlurAngleAdjusting: dragging, blurOverlayAngle: adjustments.rapidAngle });
    onDragStateChange?.(dragging);
  };

  return (
    <div>
      <div className="mb-4 p-2 bg-bg-secondary rounded-md flex items-start gap-2">
        <Info size={14} className="text-text-secondary mt-0.5 flex-shrink-0" />
        <p className="text-xs text-text-secondary">{t('editor.adjustments.blurRecovery.description')}</p>
      </div>

      <div className="mb-4 p-2 bg-bg-tertiary rounded-md">
        <div className="flex items-center justify-between mb-2">
          <p className="text-sm font-medium text-text-primary">{t('editor.adjustments.blurRecovery.enable')}</p>
          <Switch
            id="blur-recovery-toggle"
            label=""
            checked={!!adjustments.rapidEnabled}
            onChange={(checked: boolean) =>
              setAdjustments((prev: Adjustments) => ({ ...prev, [BlurRecoveryAdjustment.RapidEnabled]: checked }))
            }
          />
        </div>

        {adjustments.rapidEnabled && (
          <div className="space-y-2 pt-2 border-t border-bg-secondary">
            <div className="flex gap-1">
              {BLUR_TYPES.map((type) => (
                <button
                  key={type}
                  className={`flex-1 py-1 px-2 rounded text-xs font-medium transition-colors ${
                    adjustments.rapidBlurType === type
                      ? 'bg-primary text-white'
                      : 'bg-bg-secondary text-text-secondary hover:text-text-primary'
                  }`}
                  onClick={() =>
                    setAdjustments((prev: Adjustments) => ({ ...prev, [BlurRecoveryAdjustment.RapidBlurType]: type }))
                  }
                >
                  {t(`editor.adjustments.blurRecovery.${type}`)}
                </button>
              ))}
            </div>

            {adjustments.rapidBlurType === 'motion' && (
              <>
                <div className="flex gap-1">
                  {MOTION_LENGTH_PRESETS.map((preset) => (
                    <button
                      key={preset}
                      className={`flex-1 py-1 px-2 rounded text-xs font-medium transition-colors ${
                        adjustments.rapidLength === preset
                          ? 'bg-primary text-white'
                          : 'bg-bg-secondary text-text-secondary hover:text-text-primary'
                      }`}
                      onClick={() =>
                        setAdjustments((prev: Adjustments) => ({
                          ...prev,
                          [BlurRecoveryAdjustment.RapidLength]: preset,
                        }))
                      }
                    >
                      {preset}
                    </button>
                  ))}
                </div>
                <Slider
                  label={t('editor.adjustments.blurRecovery.length')}
                  max={200}
                  min={1}
                  onChange={(e: any) => handleValueChange(BlurRecoveryAdjustment.RapidLength, e)}
                  step={1}
                  value={adjustments.rapidLength}
                  onDragStateChange={onDragStateChange}
                />
                <Slider
                  label={t('editor.adjustments.blurRecovery.angle')}
                  max={180}
                  min={0}
                  onChange={handleAngleChange}
                  step={1}
                  value={adjustments.rapidAngle}
                  onDragStateChange={handleAngleDragState}
                />
              </>
            )}

            {adjustments.rapidBlurType === 'defocus' && (
              <>
                <div className="flex gap-1">
                  {DEFOCUS_RADIUS_PRESETS.map((preset) => (
                    <button
                      key={preset}
                      className={`flex-1 py-1 px-2 rounded text-xs font-medium transition-colors ${
                        adjustments.rapidRadius === preset
                          ? 'bg-primary text-white'
                          : 'bg-bg-secondary text-text-secondary hover:text-text-primary'
                      }`}
                      onClick={() =>
                        setAdjustments((prev: Adjustments) => ({
                          ...prev,
                          [BlurRecoveryAdjustment.RapidRadius]: preset,
                        }))
                      }
                    >
                      {preset}
                    </button>
                  ))}
                </div>
                <Slider
                  label={t('editor.adjustments.blurRecovery.radius')}
                  max={100}
                  min={1}
                  onChange={(e: any) => handleValueChange(BlurRecoveryAdjustment.RapidRadius, e)}
                  step={0.5}
                  value={adjustments.rapidRadius}
                  onDragStateChange={onDragStateChange}
                />
              </>
            )}

            {adjustments.rapidBlurType === 'gaussian' && (
              <Slider
                label={t('editor.adjustments.blurRecovery.sigma')}
                max={10}
                min={0.5}
                onChange={(e: any) => handleValueChange(BlurRecoveryAdjustment.RapidSigma, e)}
                step={0.1}
                value={adjustments.rapidSigma}
                onDragStateChange={onDragStateChange}
              />
            )}

            <Slider
              label={t('editor.adjustments.blurRecovery.lambda')}
              max={0.1}
              min={0.001}
              onChange={(e: any) => handleValueChange(BlurRecoveryAdjustment.RapidLambda, e)}
              step={0.001}
              value={adjustments.rapidLambda}
              onDragStateChange={onDragStateChange}
            />
            <Slider
              label={t('editor.adjustments.blurRecovery.strength')}
              max={100}
              min={0}
              onChange={(e: any) => handleValueChange(BlurRecoveryAdjustment.RapidStrength, e)}
              step={1}
              value={adjustments.rapidStrength}
              onDragStateChange={onDragStateChange}
            />

            <div className="flex items-center justify-between pt-1">
              <p className="text-xs text-text-secondary">{t('editor.adjustments.blurRecovery.adaptive')}</p>
              <Switch
                id="blur-recovery-adaptive-toggle"
                label=""
                checked={!!adjustments.rapidAdaptive}
                onChange={(checked: boolean) =>
                  setAdjustments((prev: Adjustments) => ({ ...prev, [BlurRecoveryAdjustment.RapidAdaptive]: checked }))
                }
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
