use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use image::{DynamicImage, GenericImageView, ImageFormat};

use tauri::{Emitter, Manager};

use crate::app_state::AppState;
use crate::file_management::parse_virtual_path;
use crate::image_processing::get_or_init_gpu_context;

/// Cap on either output dimension. Bounds the worst-case RGBA8 output buffer
/// at 1 GiB and matches wgpu's typical max_texture_dimension_2d ceiling.
const MAX_OUTPUT_DIMENSION: u32 = 16384;

/// Validates the rendered dimensions before the 2x resize allocates the 4x
/// buffer. Rejects (1) renders past the device texture limit - the GPU pass
/// silently returns the *unprocessed* image past that limit, so the edits
/// were never applied - and (2) outputs past MAX_OUTPUT_DIMENSION. Returns
/// the output (2x) dimensions on success. No clamping: scale stays exactly 2x.
fn validate_upscale_dimensions(
    width: u32,
    height: u32,
    max_texture_dimension: u32,
) -> Result<(u32, u32), String> {
    if width > max_texture_dimension || height > max_texture_dimension {
        return Err(format!(
            "Rendered image is {}×{} px, exceeding this GPU's {} px texture limit, so edits could not be applied",
            width, height, max_texture_dimension
        ));
    }
    let (out_w, out_h) = (width * 2, height * 2);
    if out_w > MAX_OUTPUT_DIMENSION || out_h > MAX_OUTPUT_DIMENSION {
        return Err(format!(
            "Upscaled output would be {}×{} px, exceeding the {} px limit",
            out_w, out_h, MAX_OUTPUT_DIMENSION
        ));
    }
    Ok((out_w, out_h))
}

#[tauri::command]
pub async fn upscale_and_save_image(
    path: String,
    js_adjustments: serde_json::Value,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let (original_image, is_raw) = {
        let lock = state.original_image.lock().unwrap();
        lock.as_ref()
            .map(|img| (Arc::clone(&img.image), img.is_raw))
            .ok_or_else(|| "No image loaded".to_string())?
    };

    // GpuContext is Clone with all-Arc fields; obtain it here where
    // tauri::State is still available and move the owned copy into the
    // blocking task.
    let context = get_or_init_gpu_context(&state, &app_handle)?;

    let (source_real_path, _) = parse_virtual_path(&path);
    let stem = source_real_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image")
        .to_string();
    let parent = source_real_path
        .parent()
        .unwrap_or(Path::new("."))
        .to_path_buf();
    let extension = source_real_path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase());

    // RAW sources (and unknown formats) cannot be re-encoded to their original
    // container; save them as PNG with a matching extension.
    let (format, out_ext) = if is_raw {
        (ImageFormat::Png, "png".to_string())
    } else {
        match extension.as_deref() {
            Some(ext @ ("jpg" | "jpeg")) => (ImageFormat::Jpeg, ext.to_string()),
            Some("png") => (ImageFormat::Png, "png".to_string()),
            Some("webp") => (ImageFormat::WebP, "webp".to_string()),
            Some(ext @ ("tif" | "tiff")) => (ImageFormat::Tiff, ext.to_string()),
            _ => (ImageFormat::Png, "png".to_string()),
        }
    };

    let output_path = parent.join(format!("{}_upscaled.{}", stem, out_ext));
    let source_path_str = source_real_path.to_string_lossy().to_string();

    // tauri::State can't cross into spawn_blocking; move a cloned AppHandle in
    // and re-derive State inside the closure, as batch export does.
    let app_handle_clone = app_handle.clone();
    let output_path_str = tokio::task::spawn_blocking(move || -> Result<String, String> {
        let state = app_handle_clone.state::<AppState>();

        let mut js_adjustments = js_adjustments;
        crate::hydrate_adjustments(&state, &mut js_adjustments);

        // Full export render path: rotate/flip/crop, masks with crop offsets,
        // GPU tone/color/LUT pass. Blocks on GPU readback - hence
        // spawn_blocking for the whole hydrate -> render -> resize -> save
        // sequence.
        let rendered = crate::export_processing::process_image_for_export_pipeline(
            &path,
            original_image.as_ref(),
            &js_adjustments,
            &context,
            &state,
            is_raw,
            "upscale_and_save_image",
            &app_handle_clone,
        )?;

        let (w, h) = rendered.dimensions();
        let (out_w, out_h) =
            validate_upscale_dimensions(w, h, context.limits.max_texture_dimension_2d)?;

        log::info!("Upscaling image 2x: {}x{} -> {}x{}", w, h, out_w, out_h);
        let upscaled = DynamicImage::ImageRgba8(image::imageops::resize(
            &rendered,
            out_w,
            out_h,
            image::imageops::FilterType::Lanczos3,
        ));
        drop(rendered);

        // Encode to bytes so EXIF can be embedded before the write.
        // The JPEG encoder rejects RGBA input; flatten to RGB for it.
        let mut image_bytes = Vec::new();
        let mut cursor = Cursor::new(&mut image_bytes);
        let encode_result = if format == ImageFormat::Jpeg {
            upscaled.to_rgb8().write_to(&mut cursor, format)
        } else {
            upscaled.write_to(&mut cursor, format)
        };
        encode_result.map_err(|e| format!("Failed to encode upscaled image: {}", e))?;

        // Embed EXIF into the output bytes (PNG zTXt / JPEG APP1) for
        // external tools; a silent no-op for unsupported containers.
        crate::exif_processing::write_image_with_metadata(
            &mut image_bytes,
            &source_path_str,
            &out_ext,
            true,
            false,
        )?;

        std::fs::write(&output_path, image_bytes)
            .map_err(|e| format!("Failed to save upscaled image: {}", e))?;

        // EXIF sidecar - the in-app metadata display reads only sidecars.
        // Deliberately no copy of the source adjustments sidecar: edits are
        // baked, the output starts with fresh defaults.
        let _ = crate::exif_processing::write_rrexif_sidecar(&source_path_str, &output_path);

        Ok(output_path.to_string_lossy().to_string())
    })
    .await
    .map_err(|e| format!("Upscale task failed: {}", e))??;

    log::info!("Upscaled image saved to: {}", output_path_str);

    // Refresh the library so the new file shows up in the filmstrip.
    let _ = app_handle.emit("indexing-finished", ());

    Ok(output_path_str)
}

#[cfg(test)]
mod upscale_dimension_tests {
    use super::validate_upscale_dimensions;

    #[test]
    fn test_validate_upscale_dimensions() {
        // Under both limits: returns doubled output dimensions.
        assert_eq!(
            validate_upscale_dimensions(4000, 3000, 16384),
            Ok((8000, 6000))
        );
        // Exact boundary: 8192 doubles to the 16384 cap and passes.
        assert_eq!(
            validate_upscale_dimensions(8192, 8192, 16384),
            Ok((16384, 16384))
        );
        // Device texture limit exceeded: the GPU pass would have silently
        // skipped the edits, so this must hard-error.
        assert!(validate_upscale_dimensions(9000, 3000, 8192).is_err());
        // 2x output exceeds the cap; error names actual vs. limit.
        let err = validate_upscale_dimensions(9216, 6144, 16384).unwrap_err();
        assert!(err.contains("18432"));
        assert!(err.contains("16384"));
    }
}
