import { KEY_CAMERA_SETTINGS_MAP } from '../components/panel/right/MetadataPanel';
import { formatDmsCompact, toSignedDecimal } from './gpsUtils';
import { latLonToMgrs } from './mgrs';

export const DEFAULT_CALLOUT_TEMPLATE = 'Time: \nLocation: \nNotes: ';

type ExifMap = { [key: string]: string };

const LABEL_WIDTH = 10;

function formatValue(mapKey: string, value: string | undefined): string | undefined {
  if (value === undefined || value === null || value === '') return undefined;
  const entry = KEY_CAMERA_SETTINGS_MAP[mapKey];
  return entry?.format ? String(entry.format(value as unknown as number)) : String(value);
}

function locationLine(exif: ExifMap, mgrs: boolean): string | undefined {
  const lat = exif.GPSLatitude;
  const latRef = exif.GPSLatitudeRef;
  const lon = exif.GPSLongitude;
  const lonRef = exif.GPSLongitudeRef;
  if (!lat || !latRef || !lon || !lonRef) return undefined;

  if (mgrs) {
    const latDd = toSignedDecimal(lat, latRef);
    const lonDd = toSignedDecimal(lon, lonRef);
    if (latDd !== null && lonDd !== null) {
      const grid = latLonToMgrs(latDd, lonDd);
      if (grid) return grid;
    }
  }

  const latText = formatDmsCompact(lat, latRef);
  const lonText = formatDmsCompact(lon, lonRef);
  if (!latText || !lonText) return undefined;
  return `${latText}, ${lonText}`;
}

export function buildPrefillBlock(exif: ExifMap | null, options: { mgrs: boolean }): string {
  if (!exif) return '';
  const lines: string[] = [];
  const push = (label: string, value: string | undefined) => {
    if (value) lines.push(`${(label + ':').padEnd(LABEL_WIDTH)}${value}`);
  };

  push('Camera', [exif.Make, exif.Model].filter(Boolean).join(' ') || undefined);
  push('Lens', formatValue('LensModel', exif.LensModel ?? exif.LensSpecification));

  const iso = exif.ISOSpeed ?? exif.PhotographicSensitivity ?? exif.ISOSpeedRatings;
  const exposureParts = [
    formatValue('ExposureTime', exif.ExposureTime),
    formatValue('FNumber', exif.FNumber),
    iso ? `ISO ${iso}` : undefined,
    formatValue('FocalLengthIn35mmFilm', exif.FocalLength ?? exif.FocalLengthIn35mmFilm),
  ].filter(Boolean);
  push('Exposure', exposureParts.length ? exposureParts.join(' · ') : undefined);

  push('Time', exif.DateTimeOriginal);
  push('Location', locationLine(exif, options.mgrs));

  return lines.join('\n');
}

export function insertIntoNotes(existing: string, block: string): string {
  if (!block) return existing;
  if (existing.trim() === '') return block;
  return existing.replace(/\s+$/, '') + '\n\n' + block;
}
