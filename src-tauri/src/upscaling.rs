use std::path::Path;
use std::sync::Arc;

use image::{DynamicImage, GenericImageView, ImageFormat};

use tauri::Emitter;

use crate::app_state::AppState;
use crate::file_management::parse_virtual_path;

#[tauri::command]
pub async fn upscale_and_save_image(
    path: String,
    app_handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<String, String> {
    let (original_image, is_raw) = {
        let lock = state.original_image.lock().unwrap();
        lock.as_ref()
            .map(|img| (Arc::clone(&img.image), img.is_raw))
            .ok_or_else(|| "No image loaded".to_string())?
    };

    let (w, h) = original_image.dimensions();
    log::info!("Upscaling image 2x: {}x{} -> {}x{}", w, h, w * 2, h * 2);

    let upscaled = tokio::task::spawn_blocking(move || {
        DynamicImage::ImageRgba8(image::imageops::resize(
            original_image.as_ref(),
            w * 2,
            h * 2,
            image::imageops::FilterType::Lanczos3,
        ))
    })
    .await
    .map_err(|e| format!("Upscale task failed: {}", e))?;

    let (source_path, _) = parse_virtual_path(&path);
    let stem = source_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");
    let parent = source_path.parent().unwrap_or(Path::new("."));

    let extension = source_path
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

    // The JPEG encoder rejects RGBA input; flatten to RGB for it.
    let save_result = if format == ImageFormat::Jpeg {
        upscaled.to_rgb8().save_with_format(&output_path, format)
    } else {
        upscaled.save_with_format(&output_path, format)
    };
    save_result.map_err(|e| format!("Failed to save upscaled image: {}", e))?;

    let output_path_str = output_path.to_string_lossy().to_string();
    log::info!("Upscaled image saved to: {}", output_path_str);

    // Refresh the library so the new file shows up in the filmstrip.
    let _ = app_handle.emit("indexing-finished", ());

    Ok(output_path_str)
}
