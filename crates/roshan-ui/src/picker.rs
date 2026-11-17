//! Choosing the area for a terrain fetch on an OpenStreetMap view.
//!
//! The rectangle to fetch stays in the middle of the view and has the size set in the side
//! panel; the user drags the map underneath it. That needs no drawing gestures and keeps the
//! latitude/longitude fields and the map showing the same place.

use eframe::egui;
use walkers::sources::OpenStreetMap;
use walkers::{HttpTiles, Map, MapMemory, lon_lat};

use crate::Area;

pub struct Picker {
    tiles: HttpTiles,
    memory: MapMemory,
}

impl Picker {
    pub fn new(ctx: egui::Context) -> Self {
        let mut memory = MapMemory::default();
        let _ = memory.set_zoom(14.0); // a few kilometres across
        Self {
            tiles: HttpTiles::new(OpenStreetMap, ctx),
            memory,
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, area: &mut Area) {
        // Dragging detaches the map from the given position. Take the new centre as the area
        // and attach again, so the fields in the side panel can move the map too.
        if let Some(centre) = self.memory.detached() {
            (area.lon, area.lat) = (centre.x(), centre.y());
            self.memory.follow_my_position();
        }
        let response = ui.add(Map::new(
            Some(&mut self.tiles),
            &mut self.memory,
            lon_lat(area.lon, area.lat),
        ));

        // Web-Mercator ground resolution: metres per point at this latitude and zoom.
        let metres_per_point =
            156_543.03 * area.lat.to_radians().cos() / 2f64.powf(self.memory.zoom());
        let size = egui::vec2(
            (area.width_km * 1000.0 / metres_per_point) as f32,
            (area.height_km * 1000.0 / metres_per_point) as f32,
        );
        let rect = egui::Rect::from_center_size(response.rect.center(), size);
        let painter = ui.painter_at(response.rect);
        painter.rect_filled(
            rect,
            0.0,
            egui::Color32::from_rgba_unmultiplied(255, 90, 30, 40),
        );
        painter.rect_stroke(
            rect,
            0.0,
            egui::Stroke::new(2.0, egui::Color32::from_rgb(220, 60, 20)),
            egui::StrokeKind::Outside,
        );
        painter.text(
            rect.left_top() - egui::vec2(0.0, 4.0),
            egui::Align2::LEFT_BOTTOM,
            format!(
                "{:.2} km x {:.2} km at {:.4}, {:.4}",
                area.width_km, area.height_km, area.lat, area.lon
            ),
            egui::FontId::proportional(13.0),
            egui::Color32::BLACK,
        );
    }
}
