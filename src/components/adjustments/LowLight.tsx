import { useEffect, useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { invoke } from '@tauri-apps/api/core';
import { emit } from '@tauri-apps/api/event';
import { Info, Zap } from 'lucide-react';
import { toast } from 'react-toastify';

import Slider from '../ui/Slider';
import Switch from '../ui/Switch';
import { Adjustments, LowLightAdjustment } from '../../utils/adjustments';
import { Invokes } from '../ui/AppProperties';
import { useEditorStore } from '../../store/useEditorStore';
import { useProcessStore } from '../../store/useProcessStore';

interface LowLightPanelProps {
  adjustments: Adjustments;
  setAdjustments(adjustments: Partial<Adjustments> | ((prev: Adjustments) => Partial<Adjustments>)): any;
  onDragStateChange?(dragging: boolean): void;
}

// ISO-to-strength curve: gentle at base ISO, aggressive at ISO 12800+.
function calculateIsoMultiplier(iso: number): number {
  if (!iso || iso <= 0) {
    return 1.0;
  }
  const logIso = Math.log2(iso / 100);
  if (iso <= 400) {
    return 0.3 + (logIso / 2) * 0.2;
  }
  if (iso <= 1600) {
    return 0.5 + ((logIso - 2) / 2) * 0.3;
  }
  if (iso <= 6400) {
    return 0.8 + ((logIso - 4) / 2) * 0.4;
  }
  return Math.min(1.5, 1.2 + ((logIso - 6) / 2) * 0.3);
}

export default function LowLightPanel({ adjustments, setAdjustments, onDragStateChange }: LowLightPanelProps) {
  const { t } = useTranslation();
  const selectedImage = useEditorStore((state: any) => state.selectedImage);
  const [isDenoising, setIsDenoising] = useState(false);

  const path: string | null = selectedImage?.path ?? null;

  const iso = useMemo(() => {
    const raw =
      selectedImage?.exif?.PhotographicSensitivity ??
      selectedImage?.exif?.ISOSpeedRatings ??
      selectedImage?.exif?.ISO ??
      '0';
    return parseInt(String(raw), 10) || 0;
  }, [selectedImage?.exif]);

  const suggestedStrength = useMemo(
    () => (iso > 0 ? Math.round(Math.min(100, 50 * calculateIsoMultiplier(iso))) : 50),
    [iso],
  );
  const [denoiseStrength, setDenoiseStrength] = useState(suggestedStrength);
  useEffect(() => {
    setDenoiseStrength(suggestedStrength);
  }, [suggestedStrength, path]);

  // Keep the sidecar-persisted multiplier in sync with this image's ISO while
  // the live denoiser is enabled; render stays reproducible from the sidecar
  // alone. Guarded so untouched images are never marked edited.
  const targetMultiplier = useMemo(() => {
    if (!adjustments.denoiseAutoIso || iso <= 0) {
      return 1.0;
    }
    return Math.round(calculateIsoMultiplier(iso) * 100) / 100;
  }, [adjustments.denoiseAutoIso, iso]);

  useEffect(() => {
    if (!adjustments.denoiseEnabled || adjustments.denoiseIsoMultiplier === targetMultiplier) {
      return;
    }
    setAdjustments((prev: Adjustments) => ({
      ...prev,
      [LowLightAdjustment.DenoiseIsoMultiplier]: targetMultiplier,
    }));
  }, [adjustments.denoiseEnabled, adjustments.denoiseIsoMultiplier, targetMultiplier, setAdjustments]);

  const handleValueChange = (key: LowLightAdjustment, e: any) => {
    const numericValue = parseFloat(e.target.value);
    setAdjustments((prev: Adjustments) => ({ ...prev, [key]: numericValue }));
  };

  const handleCheckedChange = (key: LowLightAdjustment, checked: boolean) => {
    setAdjustments((prev: Adjustments) => ({ ...prev, [key]: checked }));
  };

  const handleDeepClean = async () => {
    if (!path || isDenoising) {
      return;
    }
    setIsDenoising(true);
    try {
      await invoke(Invokes.ApplyDenoising, { path, intensity: denoiseStrength / 100, method: 'bm3d' });
      const savedPath = await invoke<string>(Invokes.SaveDenoisedImage, { originalPathStr: path });
      await emit('indexing-finished');
      useProcessStore.getState().setProcess({ initialFileToOpen: savedPath });
    } catch (err) {
      toast.error(`Failed to denoise image: ${err}`);
    } finally {
      setIsDenoising(false);
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
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Zap size={14} className={adjustments.denoiseAutoIso ? 'text-primary' : 'text-text-secondary'} />
                <label className="text-sm font-medium text-text-primary">
                  {t('editor.adjustments.lowlight.autoIso')}
                </label>
              </div>
              <Switch
                id="denoise-auto-iso-toggle"
                label=""
                checked={!!adjustments.denoiseAutoIso}
                onChange={(checked: boolean) => handleCheckedChange(LowLightAdjustment.DenoiseAutoIso, checked)}
              />
            </div>

            {adjustments.denoiseAutoIso && (
              <div className="p-2 bg-bg-secondary rounded text-xs text-text-secondary">
                {iso > 0 ? (
                  <span>
                    {t('editor.adjustments.lowlight.isoMultiplier', {
                      iso,
                      multiplier: targetMultiplier.toFixed(2),
                    })}
                  </span>
                ) : (
                  <span>{t('editor.adjustments.lowlight.noIso')}</span>
                )}
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

      <div className="p-2 bg-bg-tertiary rounded-md">
        <div className="flex items-center gap-2 mb-2">
          <Zap size={14} className={iso > 0 ? 'text-primary' : 'text-text-secondary'} />
          <p className="text-sm font-medium text-text-primary">{t('editor.adjustments.lowlight.deepClean')}</p>
        </div>
        <div className="p-2 mb-2 bg-bg-secondary rounded text-xs text-text-secondary">
          {iso > 0 ? (
            <span>{t('editor.adjustments.lowlight.isoDetected', { iso, strength: suggestedStrength })}</span>
          ) : (
            <span>{t('editor.adjustments.lowlight.noIso')}</span>
          )}
        </div>
        <Slider
          label={t('editor.adjustments.lowlight.strength')}
          max={100}
          min={0}
          onChange={(e: any) => setDenoiseStrength(parseFloat(e.target.value))}
          step={1}
          value={denoiseStrength}
          onDragStateChange={onDragStateChange}
        />
        <button
          className={`w-full mt-2 py-2 px-4 rounded font-medium text-sm transition-colors border-2 ${
            isDenoising
              ? 'bg-gray-500/20 text-gray-300 border-gray-500 cursor-wait'
              : 'bg-transparent text-primary border-primary hover:bg-primary hover:text-white'
          }`}
          onClick={handleDeepClean}
          disabled={isDenoising || !path}
        >
          {isDenoising ? t('editor.adjustments.lowlight.applying') : t('editor.adjustments.lowlight.apply')}
        </button>
      </div>
    </div>
  );
}
