import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { invoke } from '@tauri-apps/api/core';
import { Info } from 'lucide-react';
import { toast } from 'react-toastify';

import Slider from '../ui/Slider';
import Switch from '../ui/Switch';
import { Adjustments, BlurRecoveryAdjustment } from '../../utils/adjustments';
import { Invokes } from '../ui/AppProperties';
import { useEditorStore } from '../../store/useEditorStore';

interface BlurEstimate {
  length: number;
  angle: number;
  confidence: number;
  confident: boolean;
  hardness: number;
}

interface BlurRecoveryPanelProps {
  adjustments: Adjustments;
  setAdjustments(adjustments: Partial<Adjustments> | ((prev: Adjustments) => Partial<Adjustments>)): any;
  onDragStateChange?(dragging: boolean): void;
}

const BLUR_TYPES = ['motion', 'defocus', 'gaussian'] as const;
const MOTION_LENGTH_PRESETS = [50, 100, 150, 200];
const DEFOCUS_RADIUS_PRESETS = [25, 50, 75, 100];

// The "Artifact suppression" slider is a log-scale view over the stored raw
// lambda: s in [0, 100] maps to lambda = 0.001 x 100^(s/100), so the useful
// 0.001-0.01 range gets half the travel instead of the first few pixels.
// Lambda stays raw in adjustments/sidecars; s is derived for display only.
const LAMBDA_MIN = 0.001;
const LAMBDA_MAX = 0.1;
const suppressionToLambda = (s: number) => LAMBDA_MIN * Math.pow(LAMBDA_MAX / LAMBDA_MIN, s / 100);
const lambdaToSuppression = (lambda: number) => {
  const clamped = Math.min(LAMBDA_MAX, Math.max(LAMBDA_MIN, lambda));
  return (100 * Math.log10(clamped / LAMBDA_MIN)) / Math.log10(LAMBDA_MAX / LAMBDA_MIN);
};

export default function BlurRecoveryPanel({ adjustments, setAdjustments, onDragStateChange }: BlurRecoveryPanelProps) {
  const { t } = useTranslation();
  const setEditor = useEditorStore((state: any) => state.setEditor);
  const [isEstimating, setIsEstimating] = useState(false);

  const handleValueChange = (key: BlurRecoveryAdjustment, e: any) => {
    const numericValue = parseFloat(e.target.value);
    setAdjustments((prev: Adjustments) => ({ ...prev, [key]: numericValue }));
  };

  const handleEstimateBlur = async () => {
    if (isEstimating) {
      return;
    }
    setIsEstimating(true);
    try {
      const estimate = await invoke<BlurEstimate>(Invokes.EstimateBlurKernel);
      if (!estimate?.confident) {
        toast.error(t('editor.adjustments.blurRecovery.estimateFailed'));
        return;
      }
      const length = Math.min(200, Math.max(1, Math.round(estimate.length)));
      const angle = Math.min(180, Math.max(0, Math.round(estimate.angle)));
      const hardness = Math.min(100, Math.max(0, Math.round(100 * estimate.hardness)));
      setAdjustments((prev: Adjustments) => ({
        ...prev,
        [BlurRecoveryAdjustment.RapidLength]: length,
        [BlurRecoveryAdjustment.RapidAngle]: angle,
        [BlurRecoveryAdjustment.RapidHardness]: hardness,
      }));
      // Flash the angle overlay so the detected direction is visible.
      setEditor({ isBlurAngleAdjusting: true, blurOverlayAngle: angle });
      setTimeout(() => setEditor({ isBlurAngleAdjusting: false }), 1200);
    } catch (err) {
      toast.error(`${t('editor.adjustments.blurRecovery.estimateFailed')} (${err})`);
    } finally {
      setIsEstimating(false);
    }
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
                <button
                  className={`w-full py-2 px-4 rounded font-medium text-sm transition-colors border-2 ${
                    isEstimating
                      ? 'bg-gray-500/20 text-gray-300 border-gray-500 cursor-wait'
                      : 'bg-transparent text-primary border-primary hover:bg-primary hover:text-white'
                  }`}
                  onClick={handleEstimateBlur}
                  disabled={isEstimating}
                >
                  {isEstimating
                    ? t('editor.adjustments.blurRecovery.estimating')
                    : t('editor.adjustments.blurRecovery.estimate')}
                </button>
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
                <Slider
                  label={t('editor.adjustments.blurRecovery.hardness')}
                  max={100}
                  min={0}
                  onChange={(e: any) => handleValueChange(BlurRecoveryAdjustment.RapidHardness, e)}
                  step={1}
                  value={adjustments.rapidHardness}
                  onDragStateChange={onDragStateChange}
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
              max={100}
              min={0}
              onChange={(e: any) => {
                const s = parseFloat(e.target.value);
                setAdjustments((prev: Adjustments) => ({
                  ...prev,
                  [BlurRecoveryAdjustment.RapidLambda]: suppressionToLambda(s),
                }));
              }}
              step={1}
              value={Math.round(lambdaToSuppression(adjustments.rapidLambda))}
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
          </div>
        )}
      </div>
    </div>
  );
}
