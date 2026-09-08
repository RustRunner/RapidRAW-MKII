# TypeScript cleanup — 8 September 2026

The strict TypeScript check now passes, down from 54 diagnostics at `94ced8f4`. Compiler settings and translation-key checking remain unchanged; no suppression directives or dependency changes were added.

The cleanup types finite translation choices, mask controls/factory results, culling progress, folder-tree responses, and icons at their declarations. Virtual lists pass function rows to react-window 2, which already memoizes them internally. The library color filter retains its separate “none” choice.

Compiler findings also exposed three behavior fixes: a dropped mask now receives an additive blend mode and the intended insertion index in their respective arguments; delayed preview cleanup captures the URL to revoke before state changes; paste uses the existing merge/all-keys default when copy/paste settings are absent.

Validation:

- `npm run typecheck`: passed, zero diagnostics.
- `npm test`: 74 tests passed.
- `npm run build`: passed; existing bundle-size advisory remains.
- `npm run i18n:runtime-check`: 952 plural resolutions passed across 12 locales.
- `git diff --check`: passed.

This is a strict-check cleanup, not removal of every legacy `any` or a whole-repository lint/format overhaul. No estimator calculations, persisted adjustment fields, or RAW fixtures changed.
