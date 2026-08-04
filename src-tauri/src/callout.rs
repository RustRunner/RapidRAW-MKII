use std::sync::LazyLock;

use ab_glyph::{FontRef, PxScale};
use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};
use imageproc::drawing::{draw_text_mut, text_size};

use crate::export_processing::{CalloutSettings, anchored_position};

static FONT: LazyLock<FontRef<'static>> = LazyLock::new(|| {
    FontRef::try_from_slice(include_bytes!("../fonts/CourierPrime-Regular.ttf"))
        .expect("embedded Courier Prime parses")
});

const BOX_GREY: [u8; 3] = [38, 38, 38];
const MIN_FONT_PX: f32 = 8.0;
const FIT_LIMIT: f32 = 0.92;
const PAD_FACTOR: f32 = 0.6;
const LINE_ADVANCE_FACTOR: f32 = 1.35;

fn max_line_width(lines: &[&str], px: f32) -> u32 {
    let scale = PxScale::from(px);
    lines
        .iter()
        .filter(|line| !line.trim().is_empty())
        .map(|line| text_size(scale, &*FONT, line).0)
        .max()
        .unwrap_or(0)
}

fn box_dims(lines: &[&str], px: f32) -> (u32, u32) {
    let pad = PAD_FACTOR * px;
    let advance = LINE_ADVANCE_FACTOR * px;
    let max_w = max_line_width(lines, px);
    let box_w = (max_w as f32 + 2.0 * pad).ceil() as u32;
    let box_h = (lines.len() as f32 * advance + 2.0 * pad).ceil() as u32;
    (box_w, box_h)
}

pub fn render_callout(
    text: &str,
    base_w: u32,
    base_h: u32,
    size: f32,
    opacity: f32,
) -> Option<RgbaImage> {
    let mut lines: Vec<&str> = text.split('\n').collect();
    while lines.last().is_some_and(|line| line.trim().is_empty()) {
        lines.pop();
    }
    if lines.iter().all(|line| line.trim().is_empty()) {
        return None;
    }

    let base_min_dim = base_w.min(base_h) as f32;
    let mut px = (base_min_dim * size / 100.0).max(MIN_FONT_PX);

    let (mut box_w, mut box_h) = box_dims(&lines, px);
    let limit_w = FIT_LIMIT * base_w as f32;
    let limit_h = FIT_LIMIT * base_h as f32;
    if box_w as f32 > limit_w || box_h as f32 > limit_h {
        // Monospace metrics scale linearly, so one pass lands on the target;
        // below the font floor the box may still overflow and overlay() clips.
        let ratio = (limit_w / box_w as f32).min(limit_h / box_h as f32);
        px = (px * ratio).max(MIN_FONT_PX);
        (box_w, box_h) = box_dims(&lines, px);
    }

    let alpha = ((opacity / 100.0).clamp(0.0, 1.0) * 255.0) as u8;
    let mut img = RgbaImage::from_pixel(
        box_w.max(1),
        box_h.max(1),
        Rgba([BOX_GREY[0], BOX_GREY[1], BOX_GREY[2], alpha]),
    );

    let pad = PAD_FACTOR * px;
    let advance = LINE_ADVANCE_FACTOR * px;
    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        draw_text_mut(
            &mut img,
            Rgba([255, 255, 255, 255]),
            pad.round() as i32,
            (pad + i as f32 * advance).round() as i32,
            PxScale::from(px),
            &*FONT,
            line,
        );
    }

    Some(img)
}

pub fn apply_callout(image: &mut DynamicImage, settings: &CalloutSettings) -> Result<(), String> {
    let (base_w, base_h) = image.dimensions();
    let Some(overlay) = render_callout(
        &settings.text,
        base_w,
        base_h,
        settings.size,
        settings.opacity,
    ) else {
        return Ok(());
    };

    let base_min_dim = base_w.min(base_h) as f32;
    let spacing_px = (base_min_dim * (settings.spacing / 100.0)) as i64;
    let (x, y) = anchored_position(
        &settings.anchor,
        base_w,
        base_h,
        overlay.width(),
        overlay.height(),
        spacing_px,
    );

    image::imageops::overlay(image, &overlay, x, y);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export_processing::WatermarkAnchor;

    #[test]
    fn box_dimensions_match_formula() {
        let img = render_callout("Time: 12:00\nNotes: dusk", 4000, 3000, 2.5, 50.0).unwrap();
        let px = 3000.0 * 2.5 / 100.0;
        let lines = ["Time: 12:00", "Notes: dusk"];
        let max_w = max_line_width(&lines, px);
        let expected_w = (max_w as f32 + 2.0 * PAD_FACTOR * px).ceil() as u32;
        let expected_h = (2.0 * LINE_ADVANCE_FACTOR * px + 2.0 * PAD_FACTOR * px).ceil() as u32;
        assert_eq!(img.width(), expected_w);
        assert_eq!(img.height(), expected_h);
    }

    #[test]
    fn glyph_interiors_are_opaque_white() {
        let img = render_callout("HHHH", 4000, 3000, 5.0, 50.0).unwrap();
        let white = img
            .pixels()
            .filter(|p| p.0 == [255, 255, 255, 255])
            .count();
        assert!(white > 0, "expected fully opaque white glyph interiors");
    }

    #[test]
    fn background_carries_box_alpha() {
        let img = render_callout("X", 4000, 3000, 5.0, 50.0).unwrap();
        assert_eq!(img.get_pixel(0, 0).0, [38, 38, 38, 127]);
    }

    #[test]
    fn long_line_trips_fit_guard() {
        let long = "x".repeat(300);
        let img = render_callout(&long, 4000, 3000, 2.5, 50.0).unwrap();
        // One linear rescale plus per-line ceil rounding; allow a few px of slack.
        assert!(
            (img.width() as f32) <= FIT_LIMIT * 4000.0 + 4.0,
            "fit guard should cap width, got {}",
            img.width()
        );
    }

    #[test]
    fn whitespace_text_renders_nothing() {
        assert!(render_callout("", 4000, 3000, 2.5, 50.0).is_none());
        assert!(render_callout("   \n\t\n  ", 4000, 3000, 2.5, 50.0).is_none());
    }

    #[test]
    fn trailing_blank_lines_are_trimmed() {
        let with = render_callout("line\n\n\n", 4000, 3000, 2.5, 50.0).unwrap();
        let without = render_callout("line", 4000, 3000, 2.5, 50.0).unwrap();
        assert_eq!(with.height(), without.height());
    }

    #[test]
    fn composite_blends_box_over_base() {
        let mut base = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            400,
            300,
            Rgba([255, 255, 255, 255]),
        ));
        let settings = CalloutSettings {
            text: "X".to_string(),
            anchor: WatermarkAnchor::TopLeft,
            size: 5.0,
            spacing: 0.0,
            opacity: 50.0,
        };
        apply_callout(&mut base, &settings).unwrap();
        // 50% #262626 over white ~= (38 + 255) / 2 per channel.
        let px = base.get_pixel(0, 0).0;
        for c in &px[0..3] {
            assert!(
                (*c as i32 - 146).abs() <= 3,
                "expected ~146 blend, got {:?}",
                px
            );
        }
    }

    #[test]
    fn empty_text_leaves_image_untouched() {
        let mut base = DynamicImage::ImageRgba8(RgbaImage::from_pixel(
            400,
            300,
            Rgba([10, 20, 30, 255]),
        ));
        let settings = CalloutSettings {
            text: "  ".to_string(),
            anchor: WatermarkAnchor::Center,
            size: 2.5,
            spacing: 5.0,
            opacity: 50.0,
        };
        apply_callout(&mut base, &settings).unwrap();
        assert_eq!(base.get_pixel(200, 150).0, [10, 20, 30, 255]);
    }
}
