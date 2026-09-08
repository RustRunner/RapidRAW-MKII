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

## Editor request ownership (08 September 2026)

Denoise and Glare requests now belong to the editor store. Each tool has an opaque request token, the requested image identity, and runtime status/result. Removing the token on a relevant edit serves as the edit revision: editing back to an earlier value cannot revive old work. Explicit reset, paste, and preset patches supersede included tools even when their values equal the current settings. Undo/redo/history navigation and image replacement clear requests. Unrelated edits and the other tool's successful estimate remain intact.

A valid completion flushes preceding debounced history, merges its settings into the latest editor adjustments, adds one discrete history entry, and schedules autosave. Save callbacks verify image identity and the current adjustments object, so an undone or reloaded estimate cannot be saved later. Reset cancels a pending save. Cached navigation retains its preview while using the common loader to publish a fresh ready identity; a cached identity never authorizes analysis during loading.

Glare force-enables only on an accepted result. Its 1200 ms veil flash is runtime preview state with an image, request token, and initiating panel visibility lifetime. Collapsible sections keep their components mounted, so visibility is passed explicitly. Collapse, unmount, or manual Show Veil ends the appropriate flash; reopening cannot give an old request a new flash. Timeout cleanup only clears its own token. Preview rendering overlays the veil flag; history, sidecar saves, presets, export inputs, and original-image rendering use the committed settings. Flash-only renders do not schedule extra saves or synchronization.

### Final validation

- `npm test`: 74 tests passed across 3 files. The request suite contains 54 deferred-promise/fake-timer tests covering both successful consumers, panel-close completion, one history action and persisted settings, image switches, same-path reload, A/B/A, overlapping requests, mismatched/stale responses, relevant edits, explicit patches, cancellation, reset/undo/redo/history navigation, independent tool completion, and flash ownership.
- `REQUIRE_GPU_TESTS=1 cargo +1.96.1 test --manifest-path src-tauri/Cargo.toml --lib -- --test-threads=1`: 133 passed, 6 optional tests ignored. GPU execution was required, not adapter-skipped.
- `npm run build`: passed (existing bundle-size advisory).
- `npm run typecheck`: 54 existing diagnostics, down from 66 in the archived pre-ownership baseline. Comparing file/error messages while ignoring shifted line numbers found no new diagnostics.
- ESLint and Prettier passed for the new request, state-type, persistence, hook, and test modules. `git diff --check` passed. Whole-repository formatting/strict lint limitations remain as previously documented.

A local headless Firefox harness ran 15 checks against the actual React panels, `useEditorActions`, `useImageLoader`, and cached navigation. Only Tauri transport was mocked. It verified real unmount completion/save, collapse while mounted, reopen, the visible Show Veil checkbox with unchanged saved settings, manual-off/timeout behavior, identity clearing during reload, old completion rejection, new-generation success, overlapping same-path loads, cached-preview retention with fresh readiness, and equal-value explicit action supersession. Session harness and transcript: `/tmp/estimate-browser/` (`results.log`). This exercises browser/component integration; it does not claim a native desktop IPC or export-dialog smoke test. Backend command ownership and full-resolution RAW rendering were validated separately by Rust/GPU tests.

The two user-supplied CR3 files and local planning documents remain ignored. Estimator algorithms, the 4500 sigma-to-slider mapping, and Glare's 0.025 budget are unchanged. Numerical calibration remains separate future work.
