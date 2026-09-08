# Estimate ownership validation

## Backend identity (08 September 2026)

Load start invalidates the committed image and noise cache under the same mutex used for final publication. Ready metadata and both estimate commands carry a path and opaque decimal generation string. Both analyses retain the decoded image Arc and encoding from one committed snapshot; Glare uses that same snapshot for its noise budget. Stale results use a typed `code: stale` error. Numerical estimation and slider mappings are unchanged.

Decoded-cache hits verify a BLAKE3 content fingerprint, including same-size replacements with preserved timestamps. This adds a sequential file read on cache hits, while avoiding a repeat decode. The fingerprint is captured from the bytes actually decoded. An old load cannot publish a decoded-cache entry after a newer load starts.

Validation with Rust 1.96.1:

- `cargo +1.96.1 test --lib image_identity`: 4 passed (concurrent completion, overlapping publication, cache/reload ownership, wire format).
- `cargo +1.96.1 test --lib decoded_freshness_tests`: 1 passed; equal-size BMP replacement with preserved mtime invalidates the cache and decodes new pixels.
- `cargo +1.96.1 test --lib denoising::tests`: 4 passed.
- `cargo +1.96.1 test --lib glare_recovery::tests`: 5 passed, 1 optional test ignored.
- `npm test -- src/utils/imageIdentity.test.ts`: 3 passed.
- `npm run build`: passed.
- `npm run typecheck`: 66 diagnostics, identical to an archived HEAD baseline before this slice. No new diagnostics.
- `git diff --check`: passed.

The frontend callers are migrated atomically with the command contract. Panel lifetime, manual edits, and transient flash ownership are addressed in the following editor slice.
