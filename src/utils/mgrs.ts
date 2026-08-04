/**
 * WGS84 lat/long -> MGRS grid reference (zone + band, 100 km square, 5+5
 * digits = 1 m). Hand-rolled Snyder-series UTM plus the AA lettering scheme;
 * no dependency, one-way only.
 *
 * Verification vectors (no JS test runner in this repo — checked by hand):
 *   CN Tower   43.642567, -79.387139 -> "17T PJ 30084 33438"
 *              (published UTM reference 17T 630084 4833438; this
 *              implementation agrees within 1 m)
 *   Vent       46.857222,  10.926111 -> "32T PS 46821 91098"
 *   Sydney    -33.8568,   151.2153   -> "56H LH 34900 52288"
 *   Polar      85, 10                -> null (UPS territory, out of scope)
 */
export function latLonToMgrs(lat: number, lon: number): string | null {
  if (lat < -80 || lat > 84 || !Number.isFinite(lat) || !Number.isFinite(lon)) return null;

  const a = 6378137.0;
  const f = 1 / 298.257223563;
  const k0 = 0.9996;
  const e2 = f * (2 - f);
  const ep2 = e2 / (1 - e2);

  let zone = Math.floor((lon + 180) / 6) + 1;
  if (lat >= 56 && lat < 64 && lon >= 3 && lon < 12) zone = 32;
  if (lat >= 72 && lat < 84) {
    if (lon >= 0 && lon < 9) zone = 31;
    else if (lon >= 9 && lon < 21) zone = 33;
    else if (lon >= 21 && lon < 33) zone = 35;
    else if (lon >= 33 && lon < 42) zone = 37;
  }

  const lon0 = (((zone - 1) * 6 - 180 + 3) * Math.PI) / 180;
  const la = (lat * Math.PI) / 180;
  const lo = (lon * Math.PI) / 180;

  const n = a / Math.sqrt(1 - e2 * Math.sin(la) * Math.sin(la));
  const t = Math.tan(la) * Math.tan(la);
  const c = ep2 * Math.cos(la) * Math.cos(la);
  const bigA = Math.cos(la) * (lo - lon0);
  const m =
    a *
    ((1 - e2 / 4 - (3 * e2 * e2) / 64 - (5 * e2 * e2 * e2) / 256) * la -
      ((3 * e2) / 8 + (3 * e2 * e2) / 32 + (45 * e2 * e2 * e2) / 1024) * Math.sin(2 * la) +
      ((15 * e2 * e2) / 256 + (45 * e2 * e2 * e2) / 1024) * Math.sin(4 * la) -
      ((35 * e2 * e2 * e2) / 3072) * Math.sin(6 * la));

  const easting =
    k0 *
      n *
      (bigA +
        ((1 - t + c) * Math.pow(bigA, 3)) / 6 +
        ((5 - 18 * t + t * t + 72 * c - 58 * ep2) * Math.pow(bigA, 5)) / 120) +
    500000;
  let northing =
    k0 *
    (m +
      n *
        Math.tan(la) *
        ((bigA * bigA) / 2 +
          ((5 - t + 9 * c + 4 * c * c) * Math.pow(bigA, 4)) / 24 +
          ((61 - 58 * t + t * t + 600 * c - 330 * ep2) * Math.pow(bigA, 6)) / 720));
  if (lat < 0) northing += 10000000;

  const bands = 'CDEFGHJKLMNPQRSTUVWX';
  const band = bands[Math.min(Math.floor((lat + 80) / 8), 19)];

  const columnSets = ['ABCDEFGH', 'JKLMNPQR', 'STUVWXYZ'];
  const column = columnSets[(zone - 1) % 3][Math.floor(easting / 100000) - 1];
  const rows = 'ABCDEFGHJKLMNPQRSTUV';
  const row = rows[(Math.floor(northing / 100000) + (zone % 2 === 1 ? 0 : 5)) % 20];

  const eastDigits = String(Math.floor(easting % 100000)).padStart(5, '0');
  const northDigits = String(Math.floor(northing % 100000)).padStart(5, '0');
  return `${zone}${band} ${column}${row} ${eastDigits} ${northDigits}`;
}
