import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { invoke } from '@tauri-apps/api/core';
import { Info } from 'lucide-react';
import { toast } from 'react-toastify';

import { Invokes } from '../ui/AppProperties';
import { useEditorStore } from '../../store/useEditorStore';
import { useProcessStore } from '../../store/useProcessStore';

export default function UpscalePanel() {
  const { t } = useTranslation();
  const selectedImage = useEditorStore((state: any) => state.selectedImage);
  const [isUpscaling, setIsUpscaling] = useState(false);

  const path: string | null = selectedImage?.path ?? null;
  const isUpscaled = !!path && path.includes('_upscaled');

  const handleApply = async () => {
    if (!path || isUpscaling || isUpscaled) {
      return;
    }
    setIsUpscaling(true);
    try {
      const jsAdjustments = useEditorStore.getState().adjustments;
      const newPath = await invoke<string>(Invokes.UpscaleAndSaveImage, { path, jsAdjustments });
      useProcessStore.getState().setProcess({ initialFileToOpen: newPath });
    } catch (err) {
      toast.error(`Failed to upscale image: ${err}`);
    } finally {
      setIsUpscaling(false);
    }
  };

  const buttonLabel = isUpscaled
    ? t('editor.adjustments.upscale.applied')
    : isUpscaling
      ? t('editor.adjustments.upscale.applying')
      : t('editor.adjustments.upscale.apply');

  return (
    <div>
      <div className="mb-4 p-2 bg-bg-secondary rounded-md flex items-start gap-2">
        <Info size={14} className="text-text-secondary mt-0.5 flex-shrink-0" />
        <p className="text-xs text-text-secondary">{t('editor.adjustments.upscale.description')}</p>
      </div>

      <div className="p-2 bg-bg-tertiary rounded-md">
        <p className="text-md font-semibold mb-2 text-primary">{t('editor.adjustments.upscale.heading')}</p>
        <button
          className={`w-full py-2 px-4 rounded font-medium text-sm transition-colors border-2 ${
            isUpscaled
              ? 'bg-green-600/20 text-green-400 border-green-600 cursor-not-allowed'
              : isUpscaling
                ? 'bg-gray-500/20 text-gray-300 border-gray-500 cursor-wait'
                : 'bg-transparent text-primary border-primary hover:bg-primary hover:text-white'
          }`}
          onClick={handleApply}
          disabled={isUpscaled || isUpscaling || !path}
        >
          {buttonLabel}
        </button>
        {isUpscaled && (
          <p className="text-xs text-text-secondary mt-2">{t('editor.adjustments.upscale.alreadyUpscaled')}</p>
        )}
      </div>
    </div>
  );
}
