export function parseDms(dmsString: string): number | null {
  if (!dmsString) return null;
  const parts = dmsString.match(/(\d+\.?\d*)\s+deg\s+(\d+\.?\d*)\s+min\s+(\d+\.?\d*)\s+sec/);
  if (!parts) return null;
  const degrees = parseFloat(parts[1]);
  const minutes = parseFloat(parts[2]);
  const seconds = parseFloat(parts[3]);
  return degrees + minutes / 60 + seconds / 3600;
}

export function toSignedDecimal(dmsString: string, ref: string): number | null {
  const value = parseDms(dmsString);
  if (value === null) return null;
  const hemisphere = ref.trim().charAt(0).toUpperCase();
  return hemisphere === 'S' || hemisphere === 'W' ? -value : value;
}

export function formatDmsCompact(dmsString: string, ref: string): string | null {
  const decimal = parseDms(dmsString);
  if (decimal === null) return null;
  let degrees = Math.floor(decimal);
  const minutesFloat = (decimal - degrees) * 60;
  let minutes = Math.floor(minutesFloat);
  let seconds = Math.round((minutesFloat - minutes) * 60);
  if (seconds === 60) {
    seconds = 0;
    minutes += 1;
  }
  if (minutes === 60) {
    minutes = 0;
    degrees += 1;
  }
  return `${degrees}°${minutes}'${seconds}"${ref.trim().charAt(0).toUpperCase()}`;
}
