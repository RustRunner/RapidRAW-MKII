import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { invoke } from '@tauri-apps/api/core';
import { Info } from 'lucide-react';
import { toast } from 'react-toastify';
import { useShallow } from 'zustand/react/shallow';

import Slider from '../ui/Slider';
import Switch from '../ui/Switch';
import { Adjustments, BlurRecoveryAdjustment } from '../../utils/adjustments';
import { Invokes } from '../ui/AppProperties';
import { useEditorStore } from '../../store/useEditorStore';
import { useSettingsStore } from '../../store/useSettingsStore';

interface BlurEstimate {
  length: number;
  angle: number;
  confidence: number;
  confident: boolean;
  hardness: number;
  lambda: number;
}

interface DefocusEstimate {
  radius: number;
  confidence: number;
  confident: boolean;
  lambda: number;
}

interface GaussianEstimate {
  sigma: number;
  confidence: number;
  confident: boolean;
  lambda: number;
}

interface BlurRecoveryPanelProps {
  adjustments: Adjustments;
  setAdjustments(adjustments: Partial<Adjustments> | ((prev: Adjustments) => Partial<Adjustments>)): any;
  onDragStateChange?(dragging: boolean): void;
}

const BLUR_TYPES = ['motion', 'defocus', 'gaussian'] as const;
type BlurType = (typeof BLUR_TYPES)[number];
const MOTION_LENGTH_PRESETS = [50, 100, 150, 200];
const DEFOCUS_RADIUS_PRESETS = [5, 10, 15, 20];

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

const finiteOr = (value: unknown, fallback: number) =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;
const booleanOr = (value: unknown, fallback: boolean) => (typeof value === 'boolean' ? value : fallback);

const canonicalGeometrySnapshot = (value: Adjustments) => {
  const cropValues = value.crop && [value.crop.x, value.crop.y, value.crop.width, value.crop.height];
  const crop =
    cropValues && cropValues.every((part) => typeof part === 'number' && Number.isFinite(part))
      ? {
          x: value.crop!.x,
          y: value.crop!.y,
          width: value.crop!.width,
          height: value.crop!.height,
        }
      : null;
  const lens = value.lensDistortionParams;
  return {
    crop,
    orientationSteps: finiteOr(value.orientationSteps, 0),
    flipHorizontal: booleanOr(value.flipHorizontal, false),
    flipVertical: booleanOr(value.flipVertical, false),
    rotation: finiteOr(value.rotation, 0),
    transformDistortion: finiteOr(value.transformDistortion, 0),
    transformVertical: finiteOr(value.transformVertical, 0),
    transformHorizontal: finiteOr(value.transformHorizontal, 0),
    transformRotate: finiteOr(value.transformRotate, 0),
    transformAspect: finiteOr(value.transformAspect, 0),
    transformScale: finiteOr(value.transformScale, 100),
    transformXOffset: finiteOr(value.transformXOffset, 0),
    transformYOffset: finiteOr(value.transformYOffset, 0),
    lensDistortionAmount: finiteOr(value.lensDistortionAmount, 100),
    lensVignetteAmount: finiteOr(value.lensVignetteAmount, 100),
    lensTcaAmount: finiteOr(value.lensTcaAmount, 100),
    lensDistortionEnabled: booleanOr(value.lensDistortionEnabled, true),
    lensTcaEnabled: booleanOr(value.lensTcaEnabled, true),
    lensVignetteEnabled: booleanOr(value.lensVignetteEnabled, true),
    lensDistortionParams: lens
      ? {
          k1: finiteOr(lens.k1, 0),
          k2: finiteOr(lens.k2, 0),
          k3: finiteOr(lens.k3, 0),
          model: finiteOr(lens.model, 0),
          tca_vr: finiteOr(lens.tca_vr, 1),
          tca_vb: finiteOr(lens.tca_vb, 1),
          vig_k1: finiteOr(lens.vig_k1, 0),
          vig_k2: finiteOr(lens.vig_k2, 0),
          vig_k3: finiteOr(lens.vig_k3, 0),
        }
      : null,
  };
};

const geometryFingerprint = (value: Adjustments) => JSON.stringify(canonicalGeometrySnapshot(value));

interface DefocusEstimateRequest {
  requestId: number;
  path: string | undefined;
  geometryFingerprint: string;
}

const MODE_TOGGLE_KEYS = {
  motion: BlurRecoveryAdjustment.RapidMotionEnabled,
  defocus: BlurRecoveryAdjustment.RapidDefocusEnabled,
  gaussian: BlurRecoveryAdjustment.RapidGaussianEnabled,
} as const;

const estimateButtonClass = (busy: boolean) =>
  `w-full py-2 px-4 rounded font-medium text-sm transition-colors border-2 ${
    busy
      ? 'bg-gray-500/20 text-gray-300 border-gray-500 cursor-wait'
      : 'bg-transparent text-primary border-primary hover:bg-primary hover:text-white'
  }`;

export default function BlurRecoveryPanel({ adjustments, setAdjustments, onDragStateChange }: BlurRecoveryPanelProps) {
  const { t } = useTranslation();
  const setEditor = useEditorStore((state: any) => state.setEditor);
  const { appSettings, handleSettingsChange } = useSettingsStore(
    useShallow((state) => ({
      appSettings: state.appSettings,
      handleSettingsChange: state.handleSettingsChange,
    })),
  );
  const [isEstimating, setIsEstimating] = useState(false);
  const flashTimeoutRef = useRef<number | null>(null);
  const defocusRequestSequenceRef = useRef(0);
  const defocusRequestRef = useRef<DefocusEstimateRequest | null>(null);

  useEffect(
    () => () => {
      if (flashTimeoutRef.current !== null) {
        clearTimeout(flashTimeoutRef.current);
      }
    },
    [],
  );

  // Modes are independent: any subset may be enabled, and the enabled
  // modes compose into one compound kernel backend-side. A mode
  // contributes iff its switch is on AND its kernel is positive
  // (zero-start sliders make toggle-on inert until dialed in). The
  // displayed tab is pure browsing state - the switches carry activation,
  // so tab clicks never write adjustments.
  const contributes: Record<BlurType, boolean> = {
    motion: adjustments.rapidMotionEnabled && adjustments.rapidLength > 0,
    defocus: adjustments.rapidDefocusEnabled && adjustments.rapidRadius > 0,
    gaussian: adjustments.rapidGaussianEnabled && adjustments.rapidSigma > 0,
  };
  // lastBlurMode is user-editable JSON on disk - whitelist it so a
  // malformed value cannot leave the panel with no tab and no body.
  const storedMode = appSettings?.lastBlurMode;
  const displayedMode: BlurType =
    storedMode && (BLUR_TYPES as readonly string[]).includes(storedMode) ? storedMode : 'motion';

  const handleTabClick = (type: BlurType) => {
    if (appSettings) {
      handleSettingsChange({ ...appSettings, lastBlurMode: type });
    }
  };

  const handleValueChange = (key: BlurRecoveryAdjustment, e: any) => {
    const numericValue = parseFloat(e.target.value);
    setAdjustments((prev: Adjustments) => ({ ...prev, [key]: numericValue }));
  };

  const handleToggle = (type: BlurType, checked: boolean) => {
    setAdjustments((prev: Adjustments) => ({ ...prev, [MODE_TOGGLE_KEYS[type]]: checked }));
  };

  // An estimate takes seconds on big frames; if the user navigates to
  // another image before it lands, the result must be discarded - the
  // store-routed setAdjustments follows navigation.
  const imagePath = () => useEditorStore.getState().selectedImage?.path;

  const defocusRequestMatches = (requestId: number, liveState: ReturnType<typeof useEditorStore.getState>) => {
    const request = defocusRequestRef.current;
    return (
      request?.requestId === requestId &&
      request.path === liveState.selectedImage?.path &&
      request.geometryFingerprint === geometryFingerprint(liveState.adjustments)
    );
  };

  const handleEstimateMotion = async () => {
    if (isEstimating) {
      return;
    }
    setIsEstimating(true);
    const pathAtStart = imagePath();
    try {
      const estimate = await invoke<BlurEstimate>(Invokes.EstimateBlurKernel);
      if (imagePath() !== pathAtStart) {
        return;
      }
      if (!estimate?.confident) {
        toast.error(t('editor.adjustments.blurRecovery.estimateFailedMotion'));
        return;
      }
      const length = Math.min(200, Math.max(1, Math.round(estimate.length)));
      const angle = Math.min(180, Math.max(0, Math.round(estimate.angle)));
      const hardness = Math.min(100, Math.max(0, Math.round(100 * estimate.hardness)));
      const lambda = Math.min(LAMBDA_MAX, Math.max(LAMBDA_MIN, estimate.lambda));
      // A landed estimate must never leave its own mode off.
      setAdjustments((prev: Adjustments) => ({
        ...prev,
        [BlurRecoveryAdjustment.RapidLength]: length,
        [BlurRecoveryAdjustment.RapidAngle]: angle,
        [BlurRecoveryAdjustment.RapidHardness]: hardness,
        [BlurRecoveryAdjustment.RapidLambda]: lambda,
        [BlurRecoveryAdjustment.RapidMotionEnabled]: true,
      }));
      // Flash the angle overlay so the detected direction is visible.
      setEditor({ isBlurAngleAdjusting: true, blurOverlayAngle: angle });
      if (flashTimeoutRef.current !== null) {
        clearTimeout(flashTimeoutRef.current);
      }
      flashTimeoutRef.current = window.setTimeout(() => setEditor({ isBlurAngleAdjusting: false }), 1200);
    } catch (err) {
      toast.error(`${t('editor.adjustments.blurRecovery.estimateFailedMotion')} (${err})`);
    } finally {
      setIsEstimating(false);
    }
  };

  const handleEstimateDefocus = async () => {
    if (isEstimating) {
      return;
    }

    const startingState = useEditorStore.getState();
    const snapshot = canonicalGeometrySnapshot(startingState.adjustments);
    const requestId = ++defocusRequestSequenceRef.current;
    defocusRequestRef.current = {
      requestId,
      path: startingState.selectedImage?.path,
      geometryFingerprint: JSON.stringify(snapshot),
    };
    setIsEstimating(true);

    try {
      const invokeArgs = snapshot.crop === null ? {} : { roi: snapshot };
      const estimate = await invoke<DefocusEstimate>(Invokes.EstimateDefocusKernel, invokeArgs);
      const liveState = useEditorStore.getState();
      if (!defocusRequestMatches(requestId, liveState)) {
        return;
      }
      if (!estimate?.confident) {
        toast.error(t('editor.adjustments.blurRecovery.estimateFailedDefocus'));
        return;
      }
      // Round to the slider's 0.1 step before checking its upper rail.
      const roundedRadius = Math.round(estimate.radius * 10) / 10;
      if (roundedRadius > 20) {
        toast.error(
          t('editor.adjustments.blurRecovery.estimateOutOfRangeDefocus', {
            radius: estimate.radius.toFixed(1),
          }),
        );
        return;
      }
      // The floor of 1 keeps a confident estimate from writing 0 and
      // leaving the mode inert. No hardness write - the shader pins the
      // raw jinc for the defocus component.
      const radius = Math.max(1, roundedRadius);
      const lambda = Math.min(LAMBDA_MAX, Math.max(LAMBDA_MIN, estimate.lambda));
      setAdjustments((prev: Adjustments) => ({
        ...prev,
        [BlurRecoveryAdjustment.RapidRadius]: radius,
        [BlurRecoveryAdjustment.RapidLambda]: lambda,
        [BlurRecoveryAdjustment.RapidDefocusEnabled]: true,
      }));
      toast.success(t('editor.adjustments.blurRecovery.estimateSuccessDefocus', { radius: radius.toFixed(1) }));
    } catch (err) {
      const liveState = useEditorStore.getState();
      if (!defocusRequestMatches(requestId, liveState)) {
        return;
      }
      toast.error(`${t('editor.adjustments.blurRecovery.estimateFailedDefocus')} (${err})`);
    } finally {
      if (defocusRequestRef.current?.requestId === requestId) {
        setIsEstimating(false);
      }
    }
  };

  const handleEstimateGaussian = async () => {
    if (isEstimating) {
      return;
    }
    setIsEstimating(true);
    const pathAtStart = imagePath();
    try {
      const estimate = await invoke<GaussianEstimate>(Invokes.EstimateGaussianKernel);
      if (imagePath() !== pathAtStart) {
        return;
      }
      if (!estimate?.confident) {
        toast.error(t('editor.adjustments.blurRecovery.estimateFailedGaussian'));
        return;
      }
      // Top clamp 8: the PSF caps effective sigma there, so applying more
      // would lie about what renders. Snap to the slider's 0.1 step.
      const sigma = Math.min(8, Math.max(0.5, Math.round(estimate.sigma * 10) / 10));
      const lambda = Math.min(LAMBDA_MAX, Math.max(LAMBDA_MIN, estimate.lambda));
      setAdjustments((prev: Adjustments) => ({
        ...prev,
        [BlurRecoveryAdjustment.RapidSigma]: sigma,
        [BlurRecoveryAdjustment.RapidLambda]: lambda,
        [BlurRecoveryAdjustment.RapidGaussianEnabled]: true,
      }));
      toast.success(t('editor.adjustments.blurRecovery.estimateSuccessGaussian', { sigma: sigma.toFixed(1) }));
    } catch (err) {
      toast.error(`${t('editor.adjustments.blurRecovery.estimateFailedGaussian')} (${err})`);
    } finally {
      setIsEstimating(false);
    }
  };

  const handleAngleChange = (e: any) => {
    const numericValue = parseFloat(e.target.value);
    setEditor({ blurOverlayAngle: numericValue });
    setAdjustments((prev: Adjustments) => ({
      ...prev,
      [BlurRecoveryAdjustment.RapidAngle]: numericValue,
    }));
  };

  const handleAngleDragState = (dragging: boolean) => {
    setEditor({ isBlurAngleAdjusting: dragging, blurOverlayAngle: adjustments.rapidAngle });
    onDragStateChange?.(dragging);
  };

  const estimateButton = (onClick: () => void) => (
    <button className={estimateButtonClass(isEstimating)} onClick={onClick} disabled={isEstimating}>
      {isEstimating ? t('editor.adjustments.blurRecovery.estimating') : t('editor.adjustments.blurRecovery.estimate')}
    </button>
  );

  const modeSwitchRow = (type: BlurType) => (
    <div className="flex items-center justify-between">
      <p className="text-sm font-medium text-text-primary">{t('editor.adjustments.blurRecovery.enable')}</p>
      <Switch
        id={`blur-${type}-toggle`}
        label=""
        checked={!!adjustments[MODE_TOGGLE_KEYS[type]]}
        onChange={(checked: boolean) => handleToggle(type, checked)}
      />
    </div>
  );

  return (
    <div>
      <div className="mb-4 p-2 bg-bg-secondary rounded-md flex items-start gap-2">
        <Info size={14} className="text-text-secondary mt-0.5 flex-shrink-0" />
        <p className="text-xs text-text-secondary">{t('editor.adjustments.blurRecovery.description')}</p>
      </div>

      <div className="mb-4 p-2 bg-bg-tertiary rounded-md">
        <div className="space-y-2">
          <div className="flex gap-1">
            {BLUR_TYPES.map((type) => (
              <button
                key={type}
                className={`relative flex-1 py-1 px-2 rounded text-xs font-medium transition-colors ${
                  displayedMode === type
                    ? 'bg-primary text-white'
                    : 'bg-bg-secondary text-text-secondary hover:text-text-primary'
                }`}
                onClick={() => handleTabClick(type)}
              >
                {t(`editor.adjustments.blurRecovery.${type}`)}
                {contributes[type] && (
                  <span
                    className={`absolute top-1 right-1 w-1.5 h-1.5 rounded-full ${
                      displayedMode === type ? 'bg-white' : 'bg-primary'
                    }`}
                  />
                )}
              </button>
            ))}
          </div>

          {displayedMode === 'motion' && (
            <>
              {modeSwitchRow('motion')}
              {adjustments.rapidMotionEnabled && (
                <div className="space-y-2 pt-2 border-t border-bg-secondary">
                  {estimateButton(handleEstimateMotion)}
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
                    min={0}
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
                    defaultValue={50}
                    onChange={(e: any) => handleValueChange(BlurRecoveryAdjustment.RapidHardness, e)}
                    step={1}
                    value={adjustments.rapidHardness}
                    onDragStateChange={onDragStateChange}
                  />
                </div>
              )}
            </>
          )}

          {displayedMode === 'defocus' && (
            <>
              {modeSwitchRow('defocus')}
              {adjustments.rapidDefocusEnabled && (
                <div className="space-y-2 pt-2 border-t border-bg-secondary">
                  {estimateButton(handleEstimateDefocus)}
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
                    max={20}
                    min={0}
                    onChange={(e: any) => handleValueChange(BlurRecoveryAdjustment.RapidRadius, e)}
                    step={0.1}
                    value={adjustments.rapidRadius}
                    onDragStateChange={onDragStateChange}
                  />
                </div>
              )}
            </>
          )}

          {displayedMode === 'gaussian' && (
            <>
              {modeSwitchRow('gaussian')}
              {adjustments.rapidGaussianEnabled && (
                <div className="space-y-2 pt-2 border-t border-bg-secondary">
                  {estimateButton(handleEstimateGaussian)}
                  <Slider
                    label={t('editor.adjustments.blurRecovery.sigma')}
                    max={8}
                    min={0}
                    onChange={(e: any) => handleValueChange(BlurRecoveryAdjustment.RapidSigma, e)}
                    step={0.1}
                    value={adjustments.rapidSigma}
                    onDragStateChange={onDragStateChange}
                  />
                </div>
              )}
            </>
          )}

          {/* The shared inversion controls (Artifact suppression, Strength)
              follow the displayed tab's switch: a tab with its mode off
              shows nothing below the Enable row. They govern the whole
              compound set - one value across all modes, hence the
              subheading - and persist across tabs. */}
          {adjustments[MODE_TOGGLE_KEYS[displayedMode]] && (
            <>
              <div className="pt-2 border-t border-bg-secondary">
                <p className="text-xs font-medium text-text-secondary">{t('editor.adjustments.blurRecovery.shared')}</p>
              </div>
              <Slider
                label={t('editor.adjustments.blurRecovery.lambda')}
                max={100}
                min={0}
                defaultValue={50}
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
                defaultValue={50}
                onChange={(e: any) => handleValueChange(BlurRecoveryAdjustment.RapidStrength, e)}
                step={1}
                value={adjustments.rapidStrength}
                onDragStateChange={onDragStateChange}
              />
            </>
          )}
        </div>
      </div>
    </div>
  );
}
