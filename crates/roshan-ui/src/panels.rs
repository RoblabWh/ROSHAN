//! Menus, toolbar, side panel, status bar, inspector, keyboard shortcuts and the small windows.
//! Everything a click or a key can ask for goes through `Viewer::act`.

use crate::draw::{ColorBy, ISOCHRONE_S};
use crate::{Flight, Mode, Outcome, Planner, Snapshot, Tool, Viewer, demo_scenario};
use eframe::egui::{self, Key, Modifiers};
use roshan_core::fire::FireModel;
use roshan_core::world::Wind;
use std::path::PathBuf;
use std::sync::mpsc;

/// The window's colours: egui's dark theme, a touch cooler, with fire orange for selections.
pub fn visuals() -> egui::Visuals {
    let mut v = egui::Visuals::dark();
    v.panel_fill = egui::Color32::from_rgb(25, 27, 32);
    v.window_fill = egui::Color32::from_rgb(30, 32, 38);
    v.extreme_bg_color = egui::Color32::from_rgb(14, 15, 18);
    v.selection.bg_fill = egui::Color32::from_rgb(196, 86, 28);
    v.selection.stroke.color = egui::Color32::WHITE;
    v
}

const SCENARIO_FILES: &[&str] = &["npz", "tif", "tiff"];
const GEOTIFF_FILES: &[&str] = &["tif", "tiff"];

pub(crate) enum Action {
    Pause,
    Step,
    Restart,
    Fit,
    /// Ask for a file with one of these endings and show it.
    Open(&'static [&'static str]),
    OpenPath(PathBuf),
    Save,
    SaveAs,
    Demo,
    Screenshot,
    Quit,
    LoadPolicy,
    /// Take the planner (false) or the pilot (true) away again.
    Detach(bool),
}

/// Tool, toolbar label, and what it does.
const TOOLS: [(Tool, &str, &str); 7] = [
    (
        Tool::Select,
        "👆 select",
        "select a drone to inspect; on a station: its next ready drone",
    ),
    (
        Tool::Send,
        "➡ send",
        "send the selected drone here: to a fire, a place, or home to a station",
    ),
    (Tool::Ignite, "🔥 ignite", "set fire now"),
    (
        Tool::Water,
        "💧 water",
        "drop water now (also: right click)",
    ),
    (
        Tool::StartFire,
        "✴ start fire",
        "toggle a fire the scenario starts with (restarts)",
    ),
    (
        Tool::Station,
        "🏠 station",
        "toggle a ground station with its drones (restarts); Settings says how many",
    ),
    (
        Tool::Erase,
        "🗑 erase",
        "erase what is placed here (restarts)",
    ),
];

fn tools(mode: Mode) -> impl Iterator<Item = (Tool, &'static str, &'static str)> {
    TOOLS.into_iter().filter(move |(tool, ..)| match mode {
        Mode::Navigate => *tool == Tool::Select,
        Mode::Fire => !matches!(tool, Tool::Select | Tool::Send),
        Mode::Allocate => true,
    })
}

const SHORTCUTS: [(&str, &str); 14] = [
    ("Space", "pause or resume"),
    (".", "one fire step (while paused)"),
    ("R", "restart"),
    ("F", "show the whole map"),
    ("1 - 7", "choose a tool"),
    ("[   ]", "half or double the speed"),
    ("Esc", "close the inspector"),
    ("Ctrl+O", "open a scenario"),
    ("Ctrl+S", "save"),
    ("Ctrl+Shift+S", "save as"),
    ("F12", "screenshot"),
    ("Right click", "drop water"),
    ("Drag, scroll", "pan, zoom"),
    ("W A S D", "steer the selected drone (fly task)"),
];

fn degrees(to: [f32; 2]) -> f32 {
    to[1].atan2(to[0]).to_degrees().rem_euclid(360.0)
}

fn towards(degrees: f32) -> [f32; 2] {
    [degrees.to_radians().cos(), degrees.to_radians().sin()]
}

/// Writes what is on screen to a PNG and says where.
fn save_png(image: &egui::ColorImage) -> Result<String, String> {
    let dir = if std::path::Path::new("docs").is_dir() {
        "docs/img"
    } else {
        "."
    };
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let path = format!("{dir}/roshan-{stamp}.png");
    let [w, h] = image.size;
    image::save_buffer(
        &path,
        image.as_raw(),
        w as u32,
        h as u32,
        image::ColorType::Rgba8,
    )
    .map_err(|e| e.to_string())?;
    Ok(format!("screenshot: {path}"))
}

impl Viewer<'_> {
    pub(crate) fn hotkeys(&mut self, ctx: &egui::Context, snap: &Snapshot) {
        if ctx.egui_wants_keyboard_input() {
            return; // typing into a field
        }
        let pressed = |modifiers, key| ctx.input_mut(|i| i.consume_key(modifiers, key));
        let (plain, ctrl) = (Modifiers::NONE, Modifiers::COMMAND);
        // Ctrl+Shift+S before Ctrl+S: the shorter one also matches with Shift held.
        let action = if pressed(ctrl | Modifiers::SHIFT, Key::S) {
            Some(Action::SaveAs)
        } else if pressed(ctrl, Key::S) {
            Some(Action::Save)
        } else if pressed(ctrl, Key::O) {
            Some(Action::Open(SCENARIO_FILES))
        } else if pressed(plain, Key::Space) {
            Some(Action::Pause)
        } else if pressed(plain, Key::Period) {
            Some(Action::Step)
        } else if pressed(plain, Key::R) {
            Some(Action::Restart)
        } else if pressed(plain, Key::F) {
            Some(Action::Fit)
        } else if pressed(plain, Key::F12) {
            Some(Action::Screenshot)
        } else {
            None
        };
        if let Some(action) = action {
            self.act(ctx, action);
        }
        let digits = [
            Key::Num1,
            Key::Num2,
            Key::Num3,
            Key::Num4,
            Key::Num5,
            Key::Num6,
            Key::Num7,
        ];
        for ((tool, ..), key) in tools(snap.mode).zip(digits) {
            if pressed(plain, key) {
                self.tool = tool;
            }
        }
        let faster =
            pressed(plain, Key::CloseBracket) as i32 - pressed(plain, Key::OpenBracket) as i32;
        if faster != 0 {
            let mut s = self.shared.lock().unwrap();
            s.speed = (s.speed * 2f32.powi(faster)).clamp(0.25, 600.0);
        }
        if pressed(plain, Key::Escape) {
            self.shared.lock().unwrap().selected = None;
        }
    }

    pub(crate) fn act(&mut self, ctx: &egui::Context, action: Action) {
        match action {
            Action::Pause => {
                let mut s = self.shared.lock().unwrap();
                s.paused = !s.paused;
                s.error = None;
            }
            Action::Step => self.shared.lock().unwrap().step_once = true,
            Action::Restart => {
                let mut s = self.shared.lock().unwrap();
                if s.advance_seed {
                    s.seed += 1;
                }
                if let Err(why) = s.restart() {
                    self.message = format!("restart failed: {why}");
                }
            }
            Action::Fit => self.fit(),
            Action::Open(endings) => {
                let Some(open) = self.open.clone() else {
                    return;
                };
                let dir = self.prefs.dir.clone();
                self.spawn("choose a file ...", move || {
                    let mut dialog = rfd::FileDialog::new().add_filter("maps", endings);
                    if let Some(dir) = dir {
                        dialog = dialog.set_directory(dir);
                    }
                    let Some(path) = dialog.pick_file() else {
                        return Ok(Outcome::Cancelled);
                    };
                    let line = format!("opened {}", path.display());
                    Ok(Outcome::Scenario(open(&path)?, Some(path), line))
                });
            }
            Action::OpenPath(path) => self.open_path(path),
            Action::Save => {
                let Some(save) = self.save.clone() else {
                    return;
                };
                let scenario = self.shared.lock().unwrap().scenario.clone();
                match save(&scenario, self.path.as_deref()) {
                    Ok(path) => self.saved(path),
                    Err(why) => self.message = format!("save failed: {why}"),
                }
            }
            Action::SaveAs => {
                let Some(save) = self.save.clone() else {
                    return;
                };
                let scenario = self.shared.lock().unwrap().scenario.clone();
                let dir = self.prefs.dir.clone();
                self.spawn("choose where to save ...", move || {
                    let mut dialog = rfd::FileDialog::new()
                        .add_filter("scenarios", &["npz"])
                        .set_file_name("scenario.npz");
                    if let Some(dir) = dir {
                        dialog = dialog.set_directory(dir);
                    }
                    let Some(path) = dialog.save_file() else {
                        return Ok(Outcome::Cancelled);
                    };
                    save(&scenario, Some(&path.with_extension("npz"))).map(Outcome::Saved)
                });
            }
            Action::Demo => {
                if self.show_scenario(demo_scenario()) {
                    (self.path, self.message) = (None, "the built-in demo".to_string());
                    self.dirty = false;
                }
            }
            Action::Screenshot => {
                if self.job.is_some() {
                    self.message = "still busy with the previous request".to_string();
                    return;
                }
                let (tx, rx) = mpsc::channel();
                self.job = Some(rx);
                ctx.request_screenshot(move |image| {
                    drop(tx.send(save_png(&image).map(Outcome::Message)));
                });
            }
            Action::Quit => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
            Action::LoadPolicy => {
                let Some(load) = self.load_policy.clone() else {
                    return;
                };
                let dir = self.prefs.dir.clone();
                self.spawn("choose a checkpoint ...", move || {
                    let mut dialog = rfd::FileDialog::new().add_filter("checkpoints", &["pt"]);
                    if let Some(dir) = dir {
                        dialog = dialog.set_directory(dir);
                    }
                    let Some(path) = dialog.pick_file() else {
                        return Ok(Outcome::Cancelled);
                    };
                    Ok(Outcome::Policy(load(&path)?, path))
                });
            }
            Action::Detach(pilot) => {
                let mut s = self.shared.lock().unwrap();
                s.incoming.push((pilot, None));
                if pilot {
                    s.has_fly = false;
                    if s.flight == Flight::Policy {
                        s.flight = Flight::Scripted;
                    }
                } else {
                    s.has_planner = false;
                    if s.planner == Planner::Policy {
                        s.planner = Planner::Greedy;
                    }
                }
                self.policy_names[pilot as usize] = None;
            }
        }
    }

    /// Opens a file without asking (a recent one, or the default at start).
    pub(crate) fn open_path(&mut self, path: PathBuf) {
        let Some(open) = self.open.clone() else {
            self.message = "opening files needs the viewer started from Python".to_string();
            return;
        };
        self.spawn(&format!("opening {} ...", path.display()), move || {
            let line = format!("opened {}", path.display());
            Ok(Outcome::Scenario(open(&path)?, Some(path), line))
        });
    }

    pub(crate) fn menu_bar(&mut self, ui: &mut egui::Ui, snap: &Snapshot) {
        let ctx = ui.ctx().clone();
        let mut action = None;
        let scenario = snap.mode != Mode::Navigate;
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("File", |ui| {
                let (can_open, can_save) = (self.open.is_some(), self.save.is_some() && scenario);
                if ui.add_enabled(can_open, egui::Button::new("Open ...   Ctrl+O")).on_hover_text("open a saved scenario (.npz)").clicked() {
                    action = Some(Action::Open(SCENARIO_FILES));
                }
                ui.add_enabled_ui(can_open && !self.prefs.recent.is_empty(), |ui| {
                    ui.menu_button("Open recent", |ui| {
                        for path in &self.prefs.recent {
                            if ui.button(path.display().to_string()).on_hover_text("open this scenario again").clicked() {
                                action = Some(Action::OpenPath(path.clone()));
                            }
                        }
                    });
                });
                let import = egui::Button::new("Import land-cover GeoTIFF ...");
                if ui.add_enabled(can_open, import).on_hover_text("make a scenario from a land-cover GeoTIFF, such as a map of the old ROSHAN").clicked() {
                    action = Some(Action::Open(GEOTIFF_FILES));
                }
                let fetch = egui::Button::new("Fetch terrain of a real place ...");
                if ui.add_enabled(self.fetch.is_some(), fetch).on_hover_text("download land cover and elevation of an area in Europe as a new scenario").clicked() {
                    self.show_fetch = true;
                }
                if ui.button("Built-in demo").on_hover_text("the made-up landscape that needs no map data").clicked() {
                    action = Some(Action::Demo);
                }
                ui.separator();
                if ui.add_enabled(can_save, egui::Button::new("Save   Ctrl+S")).on_hover_text("save the scenario with everything placed on it to its file (a new file if it has none); refused for a map that experiment files use").clicked() {
                    action = Some(Action::Save);
                }
                let save_as = egui::Button::new("Save as ...   Ctrl+Shift+S");
                if ui.add_enabled(can_save, save_as).on_hover_text("save the scenario to a new file").clicked() {
                    action = Some(Action::SaveAs);
                }
                ui.separator();
                let set = egui::Button::new("Open this scenario at every start");
                let set = ui.add_enabled(self.path.is_some(), set);
                if set.on_hover_text("start the viewer with this scenario instead of the demo")
                    .on_disabled_hover_text("save or open it first")
                    .clicked() {
                    self.prefs.default.clone_from(&self.path);
                    self.message = "set as the default scenario".to_string();
                }
                if let Some(default) = self.prefs.default.clone() {
                    let name = default.file_name().unwrap_or_default().to_string_lossy();
                    if ui.button(format!("Forget the default ({name})")).on_hover_text("start the viewer with the demo again").clicked() {
                        self.prefs.default = None;
                    }
                }
                ui.separator();
                if ui.button("Screenshot   F12").on_hover_text("save a picture of the window as a PNG").clicked() {
                    action = Some(Action::Screenshot);
                }
                if ui.button("Quit").on_hover_text("close the viewer").clicked() {
                    action = Some(Action::Quit);
                }
                if !can_open {
                    ui.small("Files need the viewer started from Python.");
                }
            });

            ui.menu_button("Simulation", |ui| {
                let mut navigating = !scenario;
                ui.radio_value(&mut navigating, false, "Scenario: fire, and drones at stations")
                    .on_hover_text("the fire spreads on the map; drones from the ground stations fight it");
                ui.radio_value(&mut navigating, true, "Fly task: waypoints on an empty map")
                    .on_hover_text("drones fly to waypoints without colliding; no fire");
                if navigating == scenario {
                    let mut s = self.shared.lock().unwrap();
                    if let Err(why) = s.set_navigating(navigating) {
                        self.message = format!("not switched: {why}");
                    }
                }
                ui.separator();
                let mut model = snap.fire_model;
                ui.radio_value(&mut model, FireModel::Rothermel, "Fire model: Rothermel")
                    .on_hover_text("spread rate from fuel, moisture, wind and slope; restarts the run");
                ui.radio_value(&mut model, FireModel::Firespin, "Fire model: FireSPIN (particles)")
                    .on_hover_text("the particle model of the old ROSHAN; restarts the run");
                if model != snap.fire_model {
                    let mut s = self.shared.lock().unwrap();
                    s.cfg.fire_model = model;
                    if let Err(why) = s.restart() {
                        self.message = format!("restart failed: {why}");
                    }
                }
                ui.separator();
                let mut advance = snap.advance_seed;
                let next = ui.checkbox(&mut advance, "Restart with the next seed");
                if next.on_hover_text("every restart is a new episode; off repeats the same one").changed() {
                    self.shared.lock().unwrap().advance_seed = advance;
                }
                for (name, first) in [("validation", 1_000_000), ("test", 2_000_000)] {
                    if ui.button(format!("Go to the first {name} seed ({first})")).on_hover_text("restart on the first seed of this evaluation split").clicked() {
                        let mut s = self.shared.lock().unwrap();
                        s.seed = first;
                        let _ = s.restart();
                    }
                }
                ui.separator();
                if ui.button("Settings ...").on_hover_text("every parameter of the simulation and the two tasks").clicked() {
                    self.forms.open = true;
                }
            });

            ui.menu_button("Policy", |ui| {
                let load = egui::Button::new("Load a checkpoint ...");
                if ui.add_enabled(self.load_policy.is_some(), load).on_hover_text("attach a trained planner or pilot from a .pt file").clicked() {
                    action = Some(Action::LoadPolicy);
                }
                ui.small("A planner hands out targets, a pilot flies the drones.\nBoth can be attached at once.");
                ui.separator();
                for (pilot, role) in [(false, "planner"), (true, "pilot")] {
                    match &self.policy_names[pilot as usize] {
                        Some(name) => {
                            if ui.button(format!("Detach the {role} ({name})")).on_hover_text("stop using this policy").clicked() {
                                action = Some(Action::Detach(pilot));
                            }
                        }
                        None => drop(ui.add_enabled(false, egui::Button::new(format!("no {role} attached")))),
                    }
                }
            });

            ui.menu_button("View", |ui| {
                let layers = &mut self.prefs.layers;
                ui.label("Colour the ground by");
                ui.radio_value(&mut layers.color_by, ColorBy::LandCover, "land cover / fuel")
                    .on_hover_text("what grows on each cell");
                ui.radio_value(&mut layers.color_by, ColorBy::Elevation, "elevation")
                    .on_hover_text("height of the ground, low to high");
                ui.radio_value(&mut layers.color_by, ColorBy::SpreadRate, "spread rate at the start")
                    .on_hover_text("how fast fire would run through each cell in the starting weather");
                ui.radio_value(&mut layers.color_by, ColorBy::FlameLength, "flame length at the start")
                    .on_hover_text("how long the flames would be on each cell in the starting weather");
                ui.separator();
                ui.checkbox(&mut layers.hillshade, "Relief shading")
                    .on_hover_text("light and shadow from the slopes");
                ui.checkbox(&mut layers.contours, "Elevation contours")
                    .on_hover_text("lines of equal height");
                let minutes = ISOCHRONE_S / 60.0;
                ui.checkbox(&mut layers.isochrones, format!("Fire arrival lines ({minutes:.0} min)"))
                    .on_hover_text("where the fire front stood at each interval");
                ui.checkbox(&mut layers.grid, "Cell grid when zoomed in")
                    .on_hover_text("outline every cell once they are large enough");
                ui.checkbox(&mut layers.wind, "Wind arrow")
                    .on_hover_text("direction and speed of the wind");
                ui.separator();
                ui.checkbox(&mut layers.trails, "Flight trails")
                    .on_hover_text("where each drone has just been");
                ui.checkbox(&mut layers.goals, "Lines to targets")
                    .on_hover_text("a line from each drone to where it is heading");
                ui.checkbox(&mut layers.labels, "Numbers on drones and targets")
                    .on_hover_text("the indices the planner and the inspector use");
                ui.separator();
                if ui.button("Whole map   F").on_hover_text("zoom out until the whole map fits").clicked() {
                    action = Some(Action::Fit);
                }
                ui.checkbox(&mut self.follow, "Follow the selected drone")
                    .on_hover_text("keep the selected drone in the middle of the view");
                ui.checkbox(&mut self.prefs.side_panel, "Side panel")
                    .on_hover_text("planner, weather and legend at the side");
            });

            ui.menu_button("Help", |ui| {
                if ui.button("Keyboard and mouse").on_hover_text("the list of shortcuts").clicked() {
                    self.show_help = true;
                }
            });
        });
        if let Some(action) = action {
            self.act(&ctx, action);
        }
    }

    pub(crate) fn toolbar(&mut self, ui: &mut egui::Ui, snap: &Snapshot) {
        let ctx = ui.ctx().clone();
        let mut action = None;
        ui.horizontal(|ui| {
            let play = ui.button(if snap.paused { "▶" } else { "⏸" });
            if play.on_hover_text("pause or resume (Space)").clicked() {
                action = Some(Action::Pause);
            }
            let step = ui.add_enabled(snap.paused, egui::Button::new("⏭"));
            if step.on_hover_text("one fire step (.)").clicked() {
                action = Some(Action::Step);
            }
            if ui.button("⟲").on_hover_text("restart (R)").clicked() {
                action = Some(Action::Restart);
            }
            let mut speed = snap.speed;
            let slider = egui::Slider::new(&mut speed, 0.25..=600.0)
                .logarithmic(true)
                .suffix("x");
            if ui
                .add(slider)
                .on_hover_text("simulated seconds per second ([ and ])")
                .changed()
            {
                self.shared.lock().unwrap().speed = speed;
            }
            ui.separator();
            if !tools(snap.mode).any(|(tool, ..)| tool == self.tool) {
                self.tool = tools(snap.mode).next().expect("every mode has a tool").0;
            }
            for (k, (tool, label, hint)) in tools(snap.mode).enumerate() {
                let button = ui.selectable_value(&mut self.tool, tool, label);
                button.on_hover_text(format!("{hint} ({})", k + 1));
            }
            if matches!(self.tool, Tool::Ignite | Tool::Water) {
                ui.separator();
                let brush = egui::DragValue::new(&mut self.prefs.brush)
                    .range(0..=10)
                    .prefix("brush ");
                ui.add(brush)
                    .on_hover_text("cells around the click that are reached");
            }
        });
        if let Some(action) = action {
            self.act(&ctx, action);
        }
    }

    pub(crate) fn status_bar(&mut self, ui: &mut egui::Ui, snap: &Snapshot) {
        ui.horizontal(|ui| {
            ui.label(clock(snap.time_s))
                .on_hover_text("simulated time since the start");
            ui.separator();
            let (at, time_s, _) = self.pace;
            let wall = at.elapsed().as_secs_f32();
            if wall >= 0.5 {
                let pace = ((snap.time_s - time_s) / wall).max(0.0); // a restart is not a pace
                self.pace = (std::time::Instant::now(), snap.time_s, pace);
            }
            let pace = self.pace.2;
            let pace = if pace >= 60.0 {
                format!("{:.1} min/s", pace / 60.0)
            } else {
                format!("{pace:.1} s/s")
            };
            ui.label(pace)
                .on_hover_text("simulated time per real second, as measured");
            ui.separator();
            if snap.mode != Mode::Navigate {
                ui.label(format!("{} burning", snap.sources))
                    .on_hover_text("cells on fire right now");
                ui.separator();
                ui.label(format!("{:.1} % burned", snap.burned * 100.0))
                    .on_hover_text("share of the burnable cells that burned or are burning");
                ui.separator();
                if snap.fire_model == FireModel::Rothermel {
                    ui.label(format!("fronts up to {:.0} kW/m", snap.front_kw))
                        .on_hover_text(
                            "fireline intensity of the most intense front burning now. Small \
                             drones plausibly stop up to about 500 kW/m; above 4000 kW/m \
                             (water_limit_kw_m) water does nothing",
                        );
                    ui.separator();
                }
                let wind = format!(
                    "wind {:.1} m/s to {:.0}°",
                    snap.wind.speed_ms,
                    degrees(snap.wind.to)
                );
                ui.label(wind)
                    .on_hover_text("0° is right (+x), 90° is down (+y)");
                ui.separator();
            }
            if snap.mode == Mode::Allocate {
                let docked = snap.drones.iter().filter(|d| d.docked).count();
                let ready = snap.drones.iter().filter(|d| d.ready).count();
                ui.label(format!(
                    "{} flying, {ready} ready, {} in service",
                    snap.drones.len() - docked,
                    docked - ready
                ))
                .on_hover_text(
                    "drones in the air, docked and free to be sent, docked and waiting for \
                     or on a pad",
                );
                ui.separator();
            }
            if !snap.particles.is_empty() {
                ui.label(format!("{} particles", snap.particles.len()))
                    .on_hover_text("FireSPIN particles in flight");
                ui.separator();
            }
            ui.label(format!(
                "{} x {} cells of {} m",
                snap.size[0], snap.size[1], snap.cell_m
            ))
            .on_hover_text("size of the map");
            ui.separator();
            ui.label(format!("seed {}", snap.seed))
                .on_hover_text("the random seed of this episode");
            if let Some(fingerprint) = snap.fingerprint {
                ui.label(format!("fire #{fingerprint}")).on_hover_text(
                    "identifies the fire this episode started with: equal to Fire_Fingerprint \
                     in an evaluation CSV when seed, scenario and settings are the evaluation's",
                );
            }
            if snap.has_planner || snap.has_fly {
                ui.separator();
                ui.label(format!("{} policy calls", snap.policy_calls))
                    .on_hover_text("how often an attached policy has been asked");
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let line = if self.hover.is_empty() {
                    &self.message
                } else {
                    &self.hover
                };
                ui.add(egui::Label::new(line).truncate());
            });
        });
    }

    /// The side panel: who plans, the weather, and what the colours mean.
    pub(crate) fn side(&mut self, ui: &mut egui::Ui, snap: &Snapshot) {
        ui.strong("Run");
        let mut seed = snap.seed;
        ui.horizontal(|ui| {
            ui.label("seed");
            let field = ui.add(egui::DragValue::new(&mut seed));
            if field
                .on_hover_text("random seed of the episode; changing it restarts")
                .changed()
            {
                let mut s = self.shared.lock().unwrap();
                s.seed = seed;
                let _ = s.restart();
            }
        });
        if !snap.note.is_empty() {
            ui.label(&snap.note);
        }
        if !self.message.is_empty() {
            ui.label(&self.message);
        }

        if snap.mode == Mode::Allocate {
            ui.separator();
            ui.strong("Planner");
            let mut planner = snap.planner;
            ui.horizontal_wrapped(|ui| {
                ui.radio_value(&mut planner, Planner::Off, "off")
                    .on_hover_text("nobody plans: drones move only when sent by hand");
                ui.radio_value(&mut planner, Planner::Greedy, "greedy")
                    .on_hover_text("each ready drone to the nearest fire that still needs water");
                ui.radio_value(&mut planner, Planner::Head, "head")
                    .on_hover_text(
                        "each ready drone to the most intense fire that still needs water",
                    );
                ui.radio_value(&mut planner, Planner::Flank, "flank")
                    .on_hover_text(
                        "each ready drone to the least intense fire that still needs water",
                    );
                ui.radio_value(&mut planner, Planner::Rollout, "rollout")
                    .on_hover_text("greedy improved by simulating the alternatives ahead");
                if snap.has_planner {
                    ui.radio_value(&mut planner, Planner::Policy, "policy")
                        .on_hover_text("the attached planner checkpoint decides");
                }
            });
            if planner != snap.planner {
                self.shared.lock().unwrap().planner = planner;
            }
            if planner == Planner::Rollout {
                let (mut horizon, mut candidates) = snap.rollout;
                ui.horizontal(|ui| {
                    let ahead = egui::DragValue::new(&mut horizon)
                        .range(10.0..=1800.0)
                        .suffix(" s ahead");
                    let tries = egui::DragValue::new(&mut candidates)
                        .range(1..=16)
                        .suffix(" tries");
                    let ahead = ui
                        .add(ahead)
                        .on_hover_text("how far each alternative is simulated");
                    let tries = ui
                        .add(tries)
                        .on_hover_text("fires tried per drone: those lacking the most water");
                    if ahead.changed() | tries.changed() {
                        self.shared.lock().unwrap().rollout = (horizon, candidates);
                    }
                });
            }
            if let Some(name) = &self.policy_names[0] {
                ui.small(format!("planner policy: {name}"));
            }
        }
        if let (Some(name), true) = (&self.policy_names[1], snap.has_fly) {
            ui.small(format!("pilot: {name}"));
        }
        if snap.mode == Mode::Navigate {
            ui.separator();
            ui.strong("Pilot");
            let mut flight = snap.flight;
            ui.horizontal_wrapped(|ui| {
                ui.radio_value(&mut flight, Flight::Off, "off")
                    .on_hover_text("nobody flies: drones stand still unless you steer one");
                ui.radio_value(&mut flight, Flight::Straight, "straight")
                    .on_hover_text("straight at the waypoint, without avoiding other drones");
                ui.radio_value(&mut flight, Flight::Scripted, "scripted")
                    .on_hover_text("sidesteps other drones; centralised: it sees every drone and moves them in turn");
                ui.radio_value(&mut flight, Flight::Decentral, "decentral")
                    .on_hover_text("each drone avoids on its own, seeing only what a fly policy sees");
                if snap.has_fly {
                    ui.radio_value(&mut flight, Flight::Policy, "policy")
                        .on_hover_text("the attached fly checkpoint steers");
                }
            });
            if flight != snap.flight {
                self.shared.lock().unwrap().flight = flight;
            }
            return self.footer(ui, snap);
        }

        ui.separator();
        ui.strong("Weather at the start");
        let start = snap.winds[0];
        let (mut wind_speed, mut heading) = (start.speed_ms, degrees(start.to));
        let (mut d1, mut herb, mut woody) = (
            snap.moisture.d1 * 100.0,
            snap.moisture.herb * 100.0,
            snap.moisture.woody * 100.0,
        );
        let mut changed = false;
        let mut slider = |value: &mut f32, range, suffix: &str, text: &str, hint: &str| {
            changed |= ui
                .add(egui::Slider::new(value, range).suffix(suffix).text(text))
                .on_hover_text(hint)
                .changed();
        };
        slider(
            &mut wind_speed,
            0.0..=25.0,
            " m/s",
            "wind",
            "wind speed at the start",
        );
        slider(
            &mut heading,
            0.0..=359.0,
            "°",
            "towards",
            "where the wind blows to: 0° is right (+x), 90° is down (+y)",
        );
        slider(
            &mut d1,
            2.0..=40.0,
            " %",
            "dead fuel",
            "moisture of fine dead fuel; coarser dead fuel is one and two points wetter",
        );
        slider(
            &mut herb,
            30.0..=300.0,
            " %",
            "live herbs",
            "moisture of live grasses and herbs",
        );
        slider(
            &mut woody,
            30.0..=300.0,
            " %",
            "live woody",
            "moisture of live shrubs and twigs",
        );
        ui.small("Moisture in % of dry weight. 0° is right (+x), 90° is down (+y).");
        if changed {
            self.edit(|s| {
                s.wind[0].speed_ms = wind_speed;
                s.wind[0].to = towards(heading);
                // The coarser dead fuels follow the fine ones at the usual one-point steps.
                (s.moisture.d1, s.moisture.d10, s.moisture.d100) =
                    (d1 / 100.0, d1 / 100.0 + 0.01, d1 / 100.0 + 0.02);
                (s.moisture.herb, s.moisture.woody) = (herb / 100.0, woody / 100.0);
            });
        }
        self.wind_changes(ui, snap);
        self.footer(ui, snap);
    }

    /// The rest of the wind schedule: when the wind turns, and to what.
    fn wind_changes(&mut self, ui: &mut egui::Ui, snap: &Snapshot) {
        let title = format!("Wind changes later ({})", snap.winds.len() - 1);
        ui.collapsing(title, |ui| {
            let mut winds = snap.winds.clone();
            let (mut changed, mut remove) = (false, None);
            for (k, wind) in winds.iter_mut().enumerate().skip(1) {
                ui.horizontal(|ui| {
                    let (mut minutes, mut heading) = (wind.t_s / 60.0, degrees(wind.to));
                    let after = egui::DragValue::new(&mut minutes)
                        .range(0.1..=10_000.0)
                        .speed(0.5)
                        .prefix("after ")
                        .suffix(" min");
                    if ui
                        .add(after)
                        .on_hover_text("when this wind sets in")
                        .changed()
                    {
                        (wind.t_s, changed) = (minutes * 60.0, true);
                    }
                    let speed = egui::DragValue::new(&mut wind.speed_ms)
                        .range(0.0..=25.0)
                        .speed(0.1)
                        .suffix(" m/s");
                    changed |= ui
                        .add(speed)
                        .on_hover_text("wind speed from then on")
                        .changed();
                    let to = egui::DragValue::new(&mut heading)
                        .range(0.0..=359.0)
                        .suffix("°");
                    if ui
                        .add(to)
                        .on_hover_text("where it blows to: 0° is right, 90° is down")
                        .changed()
                    {
                        (wind.to, changed) = (towards(heading), true);
                    }
                    if ui
                        .small_button("✖")
                        .on_hover_text("remove this change")
                        .clicked()
                    {
                        remove = Some(k);
                    }
                });
            }
            if ui
                .button("add a change")
                .on_hover_text("let the wind turn once more, ten minutes after the last change")
                .clicked()
            {
                let last = winds[winds.len() - 1];
                winds.push(Wind {
                    t_s: last.t_s + 600.0,
                    ..last
                });
                changed = true;
            }
            if let Some(k) = remove {
                winds.remove(k);
                changed = true;
            }
            if changed {
                winds[1..].sort_by(|a, b| a.t_s.total_cmp(&b.t_s));
                self.edit(|s| s.wind = winds);
            }
        });
    }

    /// Legend and error display, at the bottom of the side panel.
    fn footer(&mut self, ui: &mut egui::Ui, snap: &Snapshot) {
        if snap.mode != Mode::Navigate {
            ui.separator();
            ui.strong("Legend");
            for (label, c) in &self.canvas.legend {
                ui.horizontal(|ui| {
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                    ui.painter()
                        .rect_filled(rect, 2.0, egui::Color32::from_rgb(c[0], c[1], c[2]));
                    ui.small(label);
                });
            }
            ui.small("burning: white to red   burned: embers, then ash\nwater dropped: cyan");
        }
        if let Some(trace) = &snap.error {
            ui.separator();
            ui.colored_label(
                egui::Color32::LIGHT_RED,
                "Policy failed, simulation paused:",
            );
            ui.monospace(trace);
        }
    }

    /// Everything about the selected drone: what it observes and what it earned.
    pub(crate) fn inspector(&mut self, ui: &mut egui::Ui, snap: &Snapshot) {
        let Some(inspect) = &snap.inspect else { return };
        ui.horizontal(|ui| {
            ui.heading(format!("Drone {}", inspect.index));
            if ui
                .small_button("close")
                .on_hover_text("deselect the drone")
                .clicked()
            {
                self.shared.lock().unwrap().selected = None;
            }
        });
        if snap.mode == Mode::Navigate {
            ui.checkbox(&mut self.steering, "steer it myself (W A S D)")
                .on_hover_text("fly this drone with the keyboard instead of the pilot");
        }
        ui.checkbox(&mut self.follow, "keep it in view")
            .on_hover_text("keep this drone in the middle of the view");
        ui.separator();
        ui.strong("Observation");
        egui::Grid::new("observation")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                for (name, value) in &inspect.rows {
                    ui.small(name);
                    ui.monospace(format!("{value:+.3}"));
                    ui.end_row();
                }
            });
        for line in &inspect.slots {
            ui.small(line);
        }
        ui.separator();
        ui.strong("Reward terms");
        egui::Grid::new("terms")
            .num_columns(3)
            .striped(true)
            .show(ui, |ui| {
                ui.small("");
                ui.small("last tick");
                ui.small("episode");
                ui.end_row();
                for (name, last, sum) in &inspect.terms {
                    ui.small(*name);
                    ui.monospace(format!("{last:+.4}"));
                    ui.monospace(format!("{sum:+.3}"));
                    ui.end_row();
                }
            });
        ui.small("Terms are unweighted; the trainer multiplies them by its reward weights.");
    }

    /// The windows that float over the map: settings, terrain fetch, help.
    pub(crate) fn windows(&mut self, ctx: &egui::Context, _snap: &Snapshot) {
        self.settings_window(ctx);

        let mut open = self.show_fetch;
        egui::Window::new("Terrain of a real place")
            .open(&mut open)
            .resizable(false)
            .show(ctx, |ui| {
                let area = &mut self.prefs.area;
                ui.horizontal(|ui| {
                    ui.label("lat");
                    ui.add(
                        egui::DragValue::new(&mut area.lat)
                            .speed(0.001)
                            .max_decimals(5),
                    )
                    .on_hover_text("latitude of the centre, degrees north");
                    ui.label("lon");
                    ui.add(
                        egui::DragValue::new(&mut area.lon)
                            .speed(0.001)
                            .max_decimals(5),
                    )
                    .on_hover_text("longitude of the centre, degrees east");
                });
                ui.horizontal(|ui| {
                    ui.label("km");
                    ui.add(
                        egui::DragValue::new(&mut area.width_km)
                            .speed(0.05)
                            .range(0.2..=20.0),
                    )
                    .on_hover_text("width of the area, west to east");
                    ui.label("x");
                    ui.add(
                        egui::DragValue::new(&mut area.height_km)
                            .speed(0.05)
                            .range(0.2..=20.0),
                    )
                    .on_hover_text("height of the area, north to south");
                });
                if cfg!(feature = "map") {
                    ui.checkbox(&mut self.picking, "choose on the street map")
                        .on_hover_text("show a street map and drag it to choose the area");
                    if self.picking {
                        ui.small("Drag the map until the rectangle covers the area, then fetch.");
                    }
                }
                let ready = self.fetch.is_some() && self.job.is_none();
                if ui
                    .add_enabled(ready, egui::Button::new("Fetch"))
                    .on_hover_text("download the area and open it as a new scenario")
                    .clicked()
                {
                    let (fetch, area) = (self.fetch.clone().expect("checked above"), *area);
                    self.spawn("fetching land cover and elevation ...", move || {
                        let scenario = fetch(area)?;
                        let line =
                            format!("fetched a {} x {} map", scenario.grid.nx, scenario.grid.ny);
                        Ok(Outcome::Scenario(scenario, None, line))
                    });
                }
                ui.small("Land cover and elevation of Europe, 10 m cells.");
                if !self.message.is_empty() {
                    ui.label(&self.message);
                }
            });
        if !open {
            self.picking = false;
        }
        self.show_fetch = open;

        egui::Window::new("Keyboard and mouse")
            .open(&mut self.show_help)
            .resizable(false)
            .show(ctx, |ui| {
                egui::Grid::new("shortcuts")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        for (keys, what) in SHORTCUTS {
                            ui.monospace(keys);
                            ui.label(what);
                            ui.end_row();
                        }
                    });
            });
    }
}

/// Simulated time for the status bar: hours appear once there are any.
fn clock(time_s: f32) -> String {
    let t = time_s as u32;
    let (h, m, s) = (t / 3600, t / 60 % 60, t % 60);
    if h > 0 {
        format!("{h} h {m:02} min {s:02} s")
    } else {
        format!("{m} min {s:02} s")
    }
}

#[test]
fn clock_switches_to_hours() {
    assert_eq!(clock(59.9), "0 min 59 s");
    assert_eq!(clock(3599.0), "59 min 59 s");
    assert_eq!(clock(60_000.0), "16 h 40 min 00 s");
}
