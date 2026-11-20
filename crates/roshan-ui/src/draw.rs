//! Everything painted on the map: the cell picture, the things on top of it, and the clicks.
//!
//! The cell picture is one texture. Its unchanging part (land cover, relief, contours) is
//! computed once per restart; fire, ash and water are laid over it once per fire step. Nothing
//! is repainted or uploaded while nothing changes.

use crate::{Mode, Shared, Sim, Snapshot, Tool, Viewer};
use eframe::egui::{self, Color32, Pos2, Rect, Stroke, vec2};
use roshan_core::allocate::{TASK_FIRE, TASK_STATION};
use roshan_core::belief::{Belief, NEVER};
use roshan_core::fire::{BURNED, BURNING};
use roshan_core::fuel::SB40;
use roshan_core::rothermel::FuelBed;
use roshan_core::scenario::Scenario;
use roshan_core::world::World;
use std::collections::VecDeque;
use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};
use std::time::Instant;

/// CLC+ Backbone land-cover classes (index = class code): name and map colour.
const LANDCOVER: [(&str, [u8; 3]); 12] = [
    ("no data", [60, 60, 66]),
    ("sealed", [128, 128, 134]),
    ("needle-leaved trees", [30, 78, 56]),
    ("broadleaved deciduous trees", [86, 140, 62]),
    ("broadleaved evergreen trees", [34, 116, 98]),
    ("low-growing woody plants", [150, 132, 70]),
    ("permanent herbaceous", [176, 204, 112]),
    ("periodically herbaceous", [222, 204, 124]),
    ("lichens and mosses", [164, 186, 170]),
    ("sparsely vegetated", [196, 174, 146]),
    ("water", [62, 104, 176]),
    ("snow and ice", [236, 240, 246]),
];

fn landcover_entry(class: u8) -> (&'static str, [u8; 3]) {
    LANDCOVER[if (class as usize) < LANDCOVER.len() {
        class as usize
    } else {
        0
    }]
}

/// One colour per fuel model: a hue per group, darker with the model number.
fn fuel_color(id: u8) -> [u8; 3] {
    if id == 0 {
        return [84, 88, 98];
    }
    let name = SB40[id as usize - 1].name;
    let (light, dark, count) = match &name[..2] {
        "GR" => ([226, 230, 150], [140, 172, 72], 9),
        "GS" => ([204, 196, 116], [146, 132, 60], 4),
        "SH" => ([190, 160, 110], [110, 78, 44], 9),
        "TU" => ([128, 184, 150], [40, 110, 92], 5),
        "TL" => ([110, 168, 96], [20, 66, 40], 9),
        _ => ([190, 150, 150], [120, 84, 96], 4),
    };
    let number: f32 = name[2..].parse().unwrap_or(1.0);
    mix(light, dark, (number - 1.0) / (count - 1) as f32)
}

/// What a cell is called in the legend and the tooltip, and its colour when not on fire.
/// With land cover the class decides, otherwise the fuel model.
fn describe(landcover: Option<u8>, fuel: u8) -> (String, [u8; 3]) {
    let model = if fuel == 0 {
        "does not burn"
    } else {
        SB40[fuel as usize - 1].name
    };
    match landcover {
        Some(class) => {
            let (name, color) = landcover_entry(class);
            (format!("{name} ({model})"), color)
        }
        None => (model.to_string(), fuel_color(fuel)),
    }
}

fn mix(a: [u8; 3], b: [u8; 3], t: f32) -> [u8; 3] {
    [0, 1, 2].map(|i| (a[i] as f32 + (b[i] as f32 - a[i] as f32) * t.clamp(0.0, 1.0)) as u8)
}

/// What the colour of an unburned cell stands for.
#[derive(Clone, Copy, PartialEq, Default, Debug, serde::Serialize, serde::Deserialize)]
pub enum ColorBy {
    #[default]
    LandCover,
    Elevation,
    SpreadRate,
    FlameLength,
}

/// What is drawn. Remembered between runs.
#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Layers {
    pub color_by: ColorBy,
    pub hillshade: bool,
    pub contours: bool,
    pub isochrones: bool,
    pub grid: bool,
    pub trails: bool,
    pub goals: bool,
    pub labels: bool,
    pub wind: bool,
    /// What the drones have seen of the fire, where they carry sensors.
    pub belief: bool,
}

impl Default for Layers {
    fn default() -> Self {
        Self {
            color_by: ColorBy::LandCover,
            hillshade: true,
            contours: false,
            isochrones: false,
            grid: true,
            trails: true,
            goals: true,
            labels: true,
            wind: true,
            belief: true,
        }
    }
}

/// Explorers and the lines to their waypoints.
const EXPLORER: Color32 = Color32::from_rgb(200, 150, 255);

/// Minutes between two arrival-time lines.
pub const ISOCHRONE_S: f32 = 300.0;

/// A repeatable pseudo-random number in 0..1 for a cell.
fn hash(i: u32) -> f32 {
    let mut h = i.wrapping_mul(0x9E37_79B9);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    (h & 0xFFFF) as f32 / 65535.0
}

fn scale(c: [u8; 3], factor: f32) -> [u8; 3] {
    c.map(|v| (v as f32 * factor).clamp(0.0, 255.0) as u8)
}

/// How bright a slope looks with the light in the upper left: 1 on flat ground.
/// `slope` is the terrain gradient (rise per metre towards +x and +y; +y is down the screen).
fn shade(slope: [f32; 2]) -> f32 {
    const EXAGGERATE: f32 = 3.0; // real terrain is gentle; this makes relief readable
    const LIGHT: [f32; 3] = [-0.5, -0.5, 0.707]; // towards the light, unit length
    let (sx, sy) = (slope[0] * EXAGGERATE, slope[1] * EXAGGERATE);
    let lit = (-sx * LIGHT[0] - sy * LIGHT[1] + LIGHT[2]) / (sx * sx + sy * sy + 1.0).sqrt();
    (lit / LIGHT[2]).clamp(0.6, 1.3)
}

/// Dark blue to yellow, for layers that show a number.
fn ramp(t: f32) -> [u8; 3] {
    const STOPS: [[u8; 3]; 5] = [
        [68, 1, 84],
        [59, 82, 139],
        [33, 145, 140],
        [94, 201, 98],
        [253, 231, 37],
    ];
    let x = t.clamp(0.0, 1.0) * 4.0;
    let k = (x as usize).min(3);
    mix(STOPS[k], STOPS[k + 1], x - k as f32)
}

/// A round step that splits `range` into roughly ten bands.
fn nice_step(range: f32) -> f32 {
    let raw = (range / 10.0).max(1e-3);
    let power = 10f32.powf(raw.log10().floor());
    [1.0, 2.0, 5.0, 10.0]
        .map(|m| m * power)
        .into_iter()
        .find(|&s| s >= raw)
        .unwrap_or(raw)
}

/// Label and colour of everything that occurs on the map.
pub type Legend = Vec<(String, [u8; 3])>;

/// The part of the picture that only changes with the map: one colour per cell, and the legend.
fn base(world: &World, scenario: &Scenario, layers: Layers) -> (Vec<[u8; 3]>, Legend) {
    let grid = world.terrain.grid;
    let n = grid.len();
    let landcover = scenario.landcover.as_deref();
    // A number per cell (None where it means nothing) and its unit, for the numeric layers.
    let values: Option<(Vec<Option<f32>>, &str)> = match layers.color_by {
        ColorBy::LandCover => None,
        ColorBy::Elevation => {
            (scenario.elevation.as_ref()).map(|e| (e.iter().map(|&v| Some(v)).collect(), "m"))
        }
        // ponytail: both are computed for the wind at restart and not updated when the wind
        // schedule moves on; recompute on a wind change if that matters.
        ColorBy::SpreadRate | ColorBy::FlameLength => {
            let flame = layers.color_by == ColorBy::FlameLength;
            let beds: Vec<Option<FuelBed>> = (0..=SB40.len())
                .map(|id| (id > 0).then(|| FuelBed::new(&SB40[id - 1], &scenario.moisture)))
                .collect();
            let cells = (0..n).map(|i| {
                let bed = beds[world.terrain.fuel[i] as usize].as_ref()?;
                let (x, y) = grid.xy(i);
                let head = world.spread(x, y).head;
                Some(if flame {
                    // m/s to ft/min, feet to metres
                    bed.flame_length_ft(head as f64 * 196.85) as f32 * 0.3048
                } else {
                    head
                })
            });
            Some((cells.collect(), if flame { "m flame" } else { "m/s" }))
        }
    };

    let (mut colours, legend): (Vec<[u8; 3]>, Legend) = match &values {
        Some((values, unit)) => {
            let known = || values.iter().flatten().copied();
            let (lo, hi) = (
                known().fold(f32::INFINITY, f32::min),
                known().fold(f32::NEG_INFINITY, f32::max),
            );
            let span = (hi - lo).max(1e-6);
            let colours = (values.iter())
                .map(|v| v.map_or([84, 88, 98], |v| ramp((v - lo) / span)))
                .collect();
            let legend = (0..5)
                .map(|k| k as f32 / 4.0)
                .map(|t| (format!("{:.2} {unit}", lo + t * span), ramp(t)))
                .collect();
            (colours, legend)
        }
        None => {
            let colour = |i: usize| match landcover {
                Some(classes) => landcover_entry(classes[i]).1,
                None => fuel_color(world.terrain.fuel[i]),
            };
            // The legend lists what occurs on this map, in class (or fuel) order.
            let seen: std::collections::BTreeSet<(u8, u8)> = (0..n)
                .map(|i| (landcover.map_or(0, |l| l[i]), world.terrain.fuel[i]))
                .collect();
            let mut legend: Vec<_> = seen
                .iter()
                .map(|&(class, fuel)| describe(landcover.map(|_| class), fuel))
                .collect();
            if layers.color_by == ColorBy::Elevation {
                legend.insert(0, ("this map has no elevation".to_string(), [84, 88, 98]));
            }
            ((0..n).map(colour).collect(), legend)
        }
    };

    let numeric = values.is_some();
    for (i, c) in colours.iter_mut().enumerate() {
        let mut factor = 1.0;
        if layers.hillshade {
            factor *= shade(world.terrain.slope[i]);
        }
        if !numeric {
            factor *= 0.96 + 0.08 * hash(i as u32); // so a forest is not one flat slab
        }
        *c = scale(*c, factor);
    }
    if let (true, Some(elevation)) = (layers.contours, &scenario.elevation) {
        let (lo, hi) = elevation
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        let step = nice_step(hi - lo);
        let band = |i: usize| (elevation[i] / step).floor();
        for (i, colour) in colours.iter_mut().enumerate() {
            let crosses = [(1, 0), (0, 1)]
                .into_iter()
                .filter_map(|(dx, dy)| grid.offset(i, dx, dy))
                .any(|j| band(j) != band(i));
            if crosses {
                *colour = scale(*colour, 0.72);
            }
        }
    }
    (colours, legend)
}

/// The picture of the map right now: the base with fire, ash and water laid over it. With a
/// `belief` (and its layer on) the picture also shows where the drones' knowledge is wrong:
/// cells nobody has looked at are dark, a fire the drones do not know of is grey, and a fire
/// they still believe in although it is gone is violet.
fn paint(world: &World, belief: Option<&Belief>, base: &[[u8; 3]], layers: Layers) -> Vec<Color32> {
    let belief = belief.filter(|_| layers.belief);
    let (fire, grid) = (&world.fire, world.terrain.grid);
    let now = world.time_s();
    let step = (world.tick / world.cfg.fire_every as u64) as u32;
    let rgb = |c: [u8; 3]| Color32::from_rgb(c[0], c[1], c[2]);
    let mut pixels: Vec<Color32> = (0..base.len())
        .map(|i| {
            let age = now - fire.t_ignite[i];
            let mut c = match fire.state[i] {
                BURNING => {
                    // White-hot, then yellow, then a deep red; every cell flickers on its own.
                    let flame = if age < 20.0 {
                        mix([255, 250, 205], [255, 205, 70], age / 20.0)
                    } else {
                        mix([255, 205, 70], [205, 45, 20], (age - 20.0) / 90.0)
                    };
                    scale(
                        flame,
                        0.86 + 0.28 * hash(i as u32 ^ step.wrapping_mul(0x632B_E5AB)),
                    )
                }
                // Embers that cool to ash, which keeps a trace of what grew there.
                BURNED => mix(
                    [150, 52, 22],
                    mix([36, 33, 32], base[i], 0.15),
                    (age - 60.0) / 400.0,
                ),
                _ => base[i],
            };
            let water = world.water.load[i];
            if water > 0.0 {
                // Cyan (lakes are blue), full strength at 2 L/m².
                c = mix(c, [90, 220, 245], (0.3 + 0.25 * water).min(0.8));
            }
            if let Some(b) = belief {
                match (fire.state[i] == BURNING, b.state[i] == BURNING) {
                    (true, false) => {
                        let grey = ((c[0] as u16 + c[1] as u16 + c[2] as u16) / 3) as u8;
                        c = mix(c, [grey; 3], 0.85);
                    }
                    (false, true) => c = mix(c, [190, 70, 230], 0.7),
                    _ => {}
                }
                if b.seen[i] == NEVER {
                    c = scale(c, 0.5);
                }
            }
            rgb(c)
        })
        .collect();

    // The fire lights up what is next to it.
    let glow = Color32::from_rgb(255, 150, 50);
    for cell in fire.burning() {
        for (dx, dy) in (-1..=1).flat_map(|dy| (-1..=1).map(move |dx| (dx, dy))) {
            if let Some(j) = grid.offset(cell as usize, dx, dy)
                && fire.state[j] != BURNING
            {
                pixels[j] = pixels[j].lerp_to_gamma(glow, 0.2);
            }
        }
    }
    if layers.isochrones {
        let band = |i: usize| (fire.t_ignite[i] / ISOCHRONE_S).floor();
        for i in (0..base.len()).filter(|&i| fire.state[i] == BURNED) {
            let crosses = [(1, 0), (0, 1)]
                .into_iter()
                .filter_map(|(dx, dy)| grid.offset(i, dx, dy))
                .any(|j| fire.state[j] == BURNED && band(j) != band(i));
            if crosses {
                pixels[i] = pixels[i].lerp_to_gamma(Color32::WHITE, 0.45);
            }
        }
    }
    pixels
}

/// The cell picture and what it was painted for.
#[derive(Default)]
pub struct Canvas {
    base: Vec<[u8; 3]>,
    base_for: Option<(u64, Layers)>,
    /// Fire step, restart and hand-edit counters of the last picture.
    painted_for: Option<(u64, u64, u64)>,
    fresh: Option<Vec<Color32>>,
    pub legend: Legend,
}

impl Canvas {
    /// Repaints if the simulation moved on. Called under the sim lock, so it does nothing
    /// when there is nothing to do.
    // ponytail: a repaint redoes and uploads the whole map once per fire step. On maps of
    // millions of cells, upload only the burning cells' bounding box (TextureHandle::set_partial).
    pub(crate) fn update(&mut self, s: &Shared, layers: Layers) {
        let world = s.sim.world();
        if self.base_for != Some((s.restarts, layers)) {
            (self.base, self.legend) = base(world, &s.scenario, layers);
            (self.base_for, self.painted_for) = (Some((s.restarts, layers)), None);
        }
        let now = (
            world.tick / world.cfg.fire_every as u64,
            s.restarts,
            s.edits,
        );
        if self.painted_for != Some(now) {
            self.painted_for = Some(now);
            self.fresh = Some(paint(world, s.sim.belief(), &self.base, layers));
        }
    }

    /// A new picture to upload, if there is one.
    pub fn take(&mut self) -> Option<Vec<Color32>> {
        self.fresh.take()
    }
}

/// Short-lived decoration that lives in the UI only: flight trails and water splashes.
#[derive(Default)]
pub struct Effects {
    trails: Vec<VecDeque<[f32; 2]>>,
    water: Vec<f32>,
    /// Where and when water was dropped.
    splashes: Vec<([f32; 2], Instant)>,
    restarts: u64,
}

const TRAIL_POINTS: usize = 40;
const SPLASH_S: f32 = 0.7;

impl Effects {
    pub(crate) fn update(&mut self, snap: &Snapshot) {
        if snap.restarts != self.restarts || self.trails.len() != snap.drones.len() {
            self.trails = vec![VecDeque::new(); snap.drones.len()];
            self.water = snap
                .drones
                .iter()
                .map(|d| d.water.map_or(0.0, |w| w.0))
                .collect();
            self.splashes.clear();
            self.restarts = snap.restarts;
        }
        for (i, drone) in snap.drones.iter().enumerate() {
            let trail = &mut self.trails[i];
            let moved = trail.back().map_or(f32::INFINITY, |last| {
                (last[0] - drone.pos[0]).hypot(last[1] - drone.pos[1])
            });
            if moved > 8.0 {
                trail.clear(); // a jump: a new episode started
            }
            if moved > 0.5 {
                trail.push_back(drone.pos);
                if trail.len() > TRAIL_POINTS {
                    trail.pop_front();
                }
            }
            let water = drone.water.map_or(0.0, |w| w.0);
            if water < self.water[i] - 1.0 {
                self.splashes.push((drone.pos, Instant::now()));
            }
            self.water[i] = water;
        }
        self.splashes
            .retain(|(_, at)| at.elapsed().as_secs_f32() < SPLASH_S);
    }

    /// Something is animating, so frames must keep coming even while paused.
    pub fn busy(&self) -> bool {
        !self.splashes.is_empty()
    }
}

/// A quadcopter seen from above, `r` from centre to rotor, its nose towards where it flies.
fn quadcopter(painter: &egui::Painter, at: Pos2, r: f32, vel: [f32; 2], fill: Color32) {
    let heading = if vel[0].hypot(vel[1]) > 0.05 {
        vel[1].atan2(vel[0])
    } else {
        -FRAC_PI_2
    };
    let arm = |k: usize| {
        let angle = heading + FRAC_PI_4 + k as f32 * FRAC_PI_2;
        vec2(angle.cos(), angle.sin()) * r
    };
    let dark = Color32::from_rgb(18, 20, 24);
    for (width, colour) in [(0.36, dark), (0.16, fill)] {
        for k in 0..2 {
            painter.line_segment([at - arm(k), at + arm(k)], Stroke::new(width * r, colour));
        }
    }
    for k in 0..4 {
        let rotor = Stroke::new(0.13 * r, fill);
        painter.circle(at + arm(k), 0.42 * r, Color32::from_black_alpha(110), rotor);
    }
    let nose = vec2(heading.cos(), heading.sin()) * r;
    let red = Color32::from_rgb(255, 80, 40);
    painter.line_segment(
        [at + nose * 0.3, at + nose * 0.95],
        Stroke::new(0.16 * r, red),
    );
    painter.circle(at, 0.36 * r, fill, Stroke::new(0.1 * r, dark));
}

impl Viewer<'_> {
    /// Applies the selected tool to a cell.
    fn click(&mut self, x: u32, y: u32, tool: Tool, snap: &Snapshot) {
        fn toggle(cells: &mut Vec<[u32; 2]>, cell: [u32; 2]) {
            match cells.iter().position(|c| *c == cell) {
                Some(i) => drop(cells.remove(i)),
                None => cells.push(cell),
            }
        }
        match tool {
            Tool::Select => {
                // The nearest drone within two cells of the click. Drones in the air come
                // first; at a station it is the first one ready, else the first in service.
                let d = |i: usize| {
                    let p = snap.drones[i].pos;
                    (p[0] - x as f32 - 0.5).hypot(p[1] - y as f32 - 0.5)
                };
                let rank = |i: usize| {
                    let drone = &snap.drones[i];
                    match (drone.docked, drone.ready) {
                        (false, _) => (0, d(i)),
                        (true, true) => (1, 0.0),
                        (true, false) => (2, 0.0),
                    }
                };
                let nearest = (0..snap.drones.len())
                    .filter(|&i| d(i) <= 2.0)
                    .min_by(|&a, &b| {
                        rank(a)
                            .0
                            .cmp(&rank(b).0)
                            .then(rank(a).1.total_cmp(&rank(b).1))
                    });
                self.shared.lock().unwrap().selected = nearest;
            }
            Tool::Send => {
                let mut s = self.shared.lock().unwrap();
                let selected = s.selected;
                let sent = match (&mut s.sim, selected) {
                    (Sim::Allocate(env, _), Some(i)) => env.send(i, [x, y]),
                    _ => Err("select a drone first"),
                };
                drop(s);
                if let Err(why) = sent {
                    self.message = why.to_string();
                }
            }
            Tool::Ignite | Tool::Water => {
                let mut s = self.shared.lock().unwrap();
                s.edits += 1;
                let world = s.sim.world_mut();
                let grid = world.terrain.grid;
                // Enough to stop a front of 4 MW/m, beyond which fires are out of control anyway.
                let cfg = world.cfg;
                let litres = (2.0 * cfg.water_crit_l_m2).max(4.0 * cfg.water_per_mw_l_m2);
                let r = self.prefs.brush as i32;
                for (dx, dy) in (-r..=r).flat_map(|dy| (-r..=r).map(move |dx| (dx, dy))) {
                    let Some(i) = grid.offset(grid.idx(x, y), dx, dy) else {
                        continue;
                    };
                    let (cx, cy) = grid.xy(i);
                    match tool {
                        Tool::Ignite => drop(world.ignite(cx, cy)),
                        _ => world.add_water(cx, cy, litres),
                    }
                }
            }
            Tool::StartFire => self.edit(|s| toggle(&mut s.ignitions, [x, y])),
            Tool::Station => self.edit(|s| toggle(&mut s.stations, [x, y])),
            Tool::Erase => self.edit(|s| {
                for cells in [&mut s.ignitions, &mut s.stations] {
                    cells.retain(|c| *c != [x, y]);
                }
            }),
        }
    }

    pub(crate) fn map(&mut self, ui: &mut egui::Ui, snap: &Snapshot) {
        let Some(texture) = &self.texture else { return };
        let texture = texture.id();
        let size = vec2(snap.size[0] as f32, snap.size[1] as f32); // one scene unit per cell
        let selected = snap.inspect.as_ref().map(|i| i.index);
        if let (true, Some(drone)) = (self.follow, selected.and_then(|i| snap.drones.get(i))) {
            let centre = Pos2::new(drone.pos[0], drone.pos[1]);
            self.view = Rect::from_center_size(centre, self.view.size());
        }
        let scene = egui::Scene::new()
            .zoom_range(0.2..=60.0)
            .max_inner_size(size)
            .scroll_zooms(true)
            .drag_pan_buttons(egui::DragPanButtons::PRIMARY | egui::DragPanButtons::MIDDLE);
        let (mut hovered, mut clicked) = (None, None);
        let (tool, layers, visible) = (self.tool, self.prefs.layers, self.view);
        let effects = &self.effects;
        let area = ui.available_rect_before_wrap();
        // Text is drawn afterwards, straight onto the screen, so it stays sharp at any zoom.
        let mut labels: Vec<(Pos2, String)> = Vec::new();
        let mut to_screen = egui::emath::TSTransform::IDENTITY;
        scene.show(ui, &mut self.view, |ui| {
            let response =
                ui.add(egui::Image::from_texture((texture, size)).sense(egui::Sense::click()));
            let origin = response.rect.min;
            to_screen = (ui.ctx())
                .layer_transform_to_global(ui.layer_id())
                .unwrap_or_default();
            let zoom = to_screen.scaling; // screen points per cell
            // A size in cells that never gets smaller than so many points on screen.
            let at_least = |cells: f32, points: f32| cells.max(points / zoom);
            let at = |p: [f32; 2]| origin + vec2(p[0], p[1]);
            let centre = |c: &[u32; 2]| at([c[0] as f32 + 0.5, c[1] as f32 + 0.5]);
            let painter = ui.painter();
            let hair = 1.0 / zoom; // one screen point

            if layers.grid && zoom >= 12.0 {
                let stroke = Stroke::new(hair, Color32::from_black_alpha(45));
                let (x0, x1) = (visible.min.x.max(0.0), visible.max.x.min(size.x));
                let (y0, y1) = (visible.min.y.max(0.0), visible.max.y.min(size.y));
                for x in x0.ceil() as u32..=x1.floor() as u32 {
                    painter.line_segment([at([x as f32, y0]), at([x as f32, y1])], stroke);
                }
                for y in y0.ceil() as u32..=y1.floor() as u32 {
                    painter.line_segment([at([x0, y as f32]), at([x1, y as f32])], stroke);
                }
            }

            // All particles as one mesh: thousands of them are in flight in a big fire.
            let mut sparks = egui::Mesh::default();
            for &(p, radiation) in &snap.particles {
                let colour = if radiation {
                    Color32::from_rgb(255, 120, 40)
                } else {
                    Color32::from_rgb(255, 240, 150)
                };
                let side = at_least(0.22, 1.5);
                sparks.add_colored_rect(Rect::from_center_size(at(p), vec2(side, side)), colour);
            }
            painter.add(egui::Shape::mesh(sparks));

            for c in &snap.start_fires {
                let r = at_least(0.9, 6.0);
                let red = Color32::from_rgb(255, 60, 30);
                painter.circle_stroke(centre(c), r, Stroke::new(0.2 * r, red));
            }
            // A ground station is a helipad; beside it, how many of its drones are ready to
            // be sent and how many are docked.
            for (k, c) in snap.stations.iter().enumerate() {
                let (p, r) = (centre(c), at_least(1.1, 10.0));
                let white = Stroke::new(0.16 * r, Color32::WHITE);
                painter.circle(p, r, Color32::from_rgb(40, 90, 190), white);
                let (w, h) = (0.32 * r, 0.45 * r);
                for x in [-w, w] {
                    painter.line_segment([p + vec2(x, -h), p + vec2(x, h)], white);
                }
                painter.line_segment([p + vec2(-w, 0.0), p + vec2(w, 0.0)], white);
                if let Some((ready, docked)) = snap.fleet.get(k) {
                    labels.push((p + vec2(1.2 * r, 0.6 * r), format!("{ready}/{docked}")));
                }
            }
            // Fire targets on offer to the planner, numbered in slot order after the stations.
            for (k, p) in snap.targets.iter().enumerate() {
                let (p, r) = (at(*p), at_least(0.8, 6.0));
                let white = Stroke::new(0.14 * r, Color32::WHITE);
                painter.circle_stroke(p, r, white);
                for tick in [
                    vec2(1.0, 0.0),
                    vec2(0.0, 1.0),
                    vec2(-1.0, 0.0),
                    vec2(0.0, -1.0),
                ] {
                    painter.line_segment([p + tick * r * 0.6, p + tick * r * 1.5], white);
                }
                if layers.labels {
                    let slot = (snap.stations.len() + k).to_string();
                    labels.push((p + vec2(r, -r), slot));
                }
            }

            if layers.trails {
                for (trail, drone) in effects.trails.iter().zip(&snap.drones) {
                    if drone.docked {
                        continue;
                    }
                    let points: Vec<Pos2> = trail.iter().map(|&p| at(p)).collect();
                    for (k, pair) in points.windows(2).enumerate() {
                        let alpha = 20 + (140 * k / TRAIL_POINTS) as u8;
                        let stroke =
                            Stroke::new(at_least(0.12, 1.5), Color32::from_white_alpha(alpha));
                        painter.line_segment([pair[0], pair[1]], stroke);
                    }
                }
            }
            for (p, when) in &effects.splashes {
                let t = when.elapsed().as_secs_f32() / SPLASH_S;
                let ring =
                    Color32::from_rgba_unmultiplied(120, 230, 255, (255.0 * (1.0 - t)) as u8);
                let r = at_least(0.6, 5.0) * (1.0 + 4.0 * t);
                painter.circle_stroke(at(*p), r, Stroke::new(at_least(0.2, 2.0), ring));
            }

            for (i, drone) in snap.drones.iter().enumerate() {
                let chosen = selected == Some(i);
                if drone.docked && !chosen {
                    continue; // inside the station
                }
                let (p, navigate) = (at(drone.pos), snap.rings.is_some());
                // Smaller on the fly task's small map, where distances between drones matter.
                let r = if navigate {
                    at_least(0.32, 7.0)
                } else {
                    at_least(0.7, 10.0)
                };
                if let (Some(goal), true) = (drone.goal, layers.goals || chosen) {
                    let colour = if drone.kind == TASK_FIRE {
                        Color32::from_rgb(255, 150, 60)
                    } else if drone.kind == TASK_STATION {
                        Color32::from_rgb(110, 170, 255)
                    } else if drone.explorer {
                        EXPLORER
                    } else {
                        Color32::WHITE
                    };
                    let line = Stroke::new(
                        hair * if chosen { 2.2 } else { 1.2 },
                        colour.gamma_multiply(if chosen { 0.95 } else { 0.5 }),
                    );
                    painter.line_segment([p, at(goal)], line);
                    painter.circle_stroke(at(goal), at_least(0.35, 3.0), line);
                }
                if let (true, Some((view, collide))) = (chosen, snap.rings) {
                    let faint = Stroke::new(at_least(0.12, 1.0), Color32::from_white_alpha(150));
                    let square = Rect::from_center_size(p, vec2(2.0 * view, 2.0 * view));
                    painter.rect_stroke(square, 0.0, faint, egui::StrokeKind::Outside);
                    let red = Stroke::new(at_least(0.1, 1.0), Color32::RED);
                    painter.circle_stroke(p, collide, red);
                }
                if chosen && drone.sense > 0.0 && !drone.docked {
                    // What its fire sensor sees.
                    let faint = Stroke::new(at_least(0.12, 1.0), Color32::from_white_alpha(150));
                    painter.circle_stroke(p, drone.sense, faint);
                }
                if chosen {
                    let halo = Stroke::new(0.18 * r, Color32::YELLOW);
                    painter.circle_stroke(p, 1.75 * r, halo);
                }
                if drone.docked {
                    continue; // only the halo shows where the selected drone is
                }
                let fill = match (chosen, drone.deciding) {
                    (true, _) => Color32::YELLOW,
                    (_, true) => Color32::from_rgb(255, 170, 60), // waiting for a decision
                    _ if drone.explorer => EXPLORER,
                    _ => Color32::from_rgb(235, 238, 242),
                };
                quadcopter(painter, p, r, drone.vel, fill);
                if let Some((litres, tank)) = drone.water.filter(|w| w.1 > 0.0) {
                    // How full the tank is, as a bar under the drone.
                    let bar =
                        Rect::from_center_size(p + vec2(0.0, 1.75 * r), vec2(2.2 * r, 0.3 * r));
                    painter.rect_filled(bar, 0.1 * r, Color32::from_black_alpha(170));
                    let full = (litres / tank).clamp(0.0, 1.0);
                    let level =
                        Rect::from_min_size(bar.min, vec2(bar.width() * full, bar.height()));
                    painter.rect_filled(level, 0.1 * r, Color32::from_rgb(90, 210, 245));
                }
                if layers.labels {
                    labels.push((p + vec2(1.2 * r, -1.2 * r), i.to_string()));
                }
            }

            // hover_pos is already in scene coordinates, i.e. in cells.
            let Some(pos) = response.hover_pos() else {
                return;
            };
            let cell = pos - origin;
            if cell.x < 0.0 || cell.y < 0.0 || cell.x >= size.x || cell.y >= size.y {
                return;
            }
            let xy = (cell.x as u32, cell.y as u32);
            hovered = Some(xy);
            if response.clicked() {
                clicked = Some((xy, tool));
            } else if response.secondary_clicked() && snap.mode != Mode::Navigate {
                clicked = Some((xy, Tool::Water));
            }
        });

        // The scene is a layer of its own above this panel's; text goes on one above that
        // (and still below the floating windows).
        let overlay = egui::LayerId::new(ui.layer_id().order, egui::Id::unique("map overlay"));
        ui.ctx().set_sublayer(ui.layer_id(), overlay);
        let painter = ui.ctx().layer_painter(overlay).with_clip_rect(area);
        let font = egui::FontId::proportional(12.0);
        for (p, text) in labels {
            let p = to_screen * p;
            for (offset, colour) in [(1.0, Color32::BLACK), (0.0, Color32::WHITE)] {
                let anchor = egui::Align2::LEFT_BOTTOM;
                painter.text(
                    p + vec2(offset, offset),
                    anchor,
                    &text,
                    font.clone(),
                    colour,
                );
            }
        }
        if layers.wind && snap.mode != Mode::Navigate {
            // Where the wind blows to, longer when it blows harder.
            let c = area.right_top() + vec2(-44.0, 44.0);
            painter.circle_filled(c, 32.0, Color32::from_black_alpha(130));
            let blow =
                vec2(snap.wind.to[0], snap.wind.to[1]) * (10.0 + snap.wind.speed_ms.min(20.0));
            if snap.wind.speed_ms > 0.0 {
                painter.arrow(c - blow * 0.5, blow, Stroke::new(2.5, Color32::WHITE));
            }
            let speed = format!("{:.0} m/s", snap.wind.speed_ms);
            painter.text(
                c + vec2(0.0, 34.0),
                egui::Align2::CENTER_TOP,
                speed,
                font,
                Color32::WHITE,
            );
        }

        if let Some(((x, y), tool)) = clicked {
            self.click(x, y, tool, snap);
        }
        self.hover.clear();
        if let (Some((x, y)), true) = (hovered, snap.mode != Mode::Navigate) {
            let s = self.shared.lock().unwrap();
            let world = s.sim.world();
            let grid = world.terrain.grid;
            if x >= grid.nx || y >= grid.ny {
                return; // the map was just replaced
            }
            let i = grid.idx(x, y);
            let class = s.scenario.landcover.as_ref().map(|l| l[i]);
            let (fuel, _) = describe(class, world.terrain.fuel[i]);
            let rate = world.spread(x, y).head;
            let water = world.water.load[i];
            let slope = world.terrain.slope[i];
            let height = (s.scenario.elevation.as_ref())
                .map_or(String::new(), |e| format!("   {:.0} m high", e[i]));
            let believed = s.sim.belief().map(|b| match (b.seen[i], b.state[i]) {
                (NEVER, _) => "no drone has looked here yet".to_string(),
                (seen, state) => format!(
                    "the drones saw it {:.0} s ago: {}",
                    (world.tick - seen as u64) as f32 * world.cfg.dt_s,
                    ["not burning", "burning", "burned"][state.min(2) as usize]
                ),
            });
            drop(s);
            self.hover = format!("cell ({x}, {y})   {fuel}");
            let anchor = egui::PopupAnchor::Pointer;
            egui::Tooltip::always_open(
                ui.ctx().clone(),
                ui.layer_id(),
                egui::Id::unique("cell"),
                anchor,
            )
            .show(|ui| {
                ui.label(format!("cell ({x}, {y})   {fuel}"));
                ui.label(format!(
                    "head rate {rate:.3} m/s   slope {:.0} %   water {water:.2} L/m²{height}",
                    slope[0].hypot(slope[1]) * 100.0
                ));
                if let Some(believed) = &believed {
                    ui.label(believed);
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relief_is_lit_from_the_upper_left() {
        assert_eq!(shade([0.0, 0.0]), 1.0);
        // Ground rising towards the lower right faces the light; falling ground is in shadow.
        assert!(shade([0.05, 0.05]) > 1.0);
        assert!(shade([-0.05, -0.05]) < 1.0);
    }

    #[test]
    fn picture_follows_the_fire() {
        let scenario = crate::demo_scenario();
        let mut world = scenario.world(Default::default(), 1).unwrap();
        let (base, legend) = base(&world, &scenario, Layers::default());
        assert_eq!(base.len(), scenario.grid.len());
        assert!(!legend.is_empty());
        // Every layer paints every cell.
        for color_by in [
            ColorBy::Elevation,
            ColorBy::SpreadRate,
            ColorBy::FlameLength,
        ] {
            let layers = Layers {
                color_by,
                contours: true,
                ..Layers::default()
            };
            assert_eq!(super::base(&world, &scenario, layers).0.len(), base.len());
        }
        let burning = world
            .fire
            .burning()
            .next()
            .expect("the demo starts with a fire") as usize;
        let before = paint(&world, None, &base, Layers::default());
        assert_ne!(before[burning].to_array()[..3], base[burning]);

        // The belief layer: one look at the fire, then a fire the drones only believe in.
        let grid = scenario.grid;
        let centre = |cell: usize| {
            let (x, y) = grid.xy(cell);
            [
                (x as f32 + 0.5) * grid.cell_m,
                (y as f32 + 0.5) * grid.cell_m,
            ]
        };
        let (far, gone) = (0, grid.idx(grid.nx - 1, grid.ny - 1));
        assert!(far != burning && gone != burning);
        let mut seen = Belief::new(grid.len());
        seen.look(grid, &world.fire.state, |_| None, centre(burning), 15.0, 1);
        let mut fake = world.fire.state.clone();
        fake[gone] = BURNING;
        seen.look(grid, &fake, |_| None, centre(gone), 1.0, 1);
        let with = paint(&world, Some(&seen), &base, Layers::default());
        assert_eq!(
            with[burning], before[burning],
            "a fire the drones know looks as it is"
        );
        assert_ne!(
            with[gone], before[gone],
            "a fire that is gone but believed is marked"
        );
        let dark = |c: Color32| c.r() as u32 + c.g() as u32 + c.b() as u32;
        assert!(dark(with[far]) < dark(before[far]), "nobody looked there");
        let unknown = paint(
            &world,
            Some(&Belief::new(grid.len())),
            &base,
            Layers::default(),
        );
        assert_ne!(
            unknown[burning], before[burning],
            "a fire the drones do not know of"
        );
        let off = Layers {
            belief: false,
            ..Layers::default()
        };
        assert_eq!(paint(&world, Some(&seen), &base, off), before);

        world.run(600.0);
        let after = paint(
            &world,
            None,
            &base,
            Layers {
                isochrones: true,
                ..Layers::default()
            },
        );
        assert_ne!(before, after);
    }

    #[test]
    fn contour_steps_are_round() {
        assert_eq!(nice_step(95.0), 10.0);
        assert_eq!(nice_step(12.0), 2.0);
        assert_eq!(nice_step(0.0), 0.001);
    }
}
