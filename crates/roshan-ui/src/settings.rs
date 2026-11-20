//! What the viewer remembers between runs, and the Settings window.
//!
//! The Settings window is one form for every configuration struct: a struct is turned into
//! JSON, each number and switch gets a field named after it, and the result is read back. A new
//! parameter in the core therefore shows up here without anybody writing a widget for it.
//!
//! The window loads and saves experiment files (the `[sim]`, `[allocate]` and `[navigate]`
//! sections that training and evaluation read), so a setup tried here can be trained on and
//! the world of a finished run can be looked at.

use crate::draw::Layers;
use crate::{Area, Outcome, Viewer};
use eframe::egui;
use roshan_core::allocate::AllocConfig;
use roshan_core::navigate::NavConfig;
use roshan_core::world::SimConfig;
use serde_json::{Value, json as json_value};
use std::path::{Path, PathBuf};

/// Remembered between runs (stored by eframe in the user's data directory).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Prefs {
    /// The scenario to open when the viewer is started without one.
    pub default: Option<PathBuf>,
    /// Files opened or saved lately, newest first.
    pub recent: Vec<PathBuf>,
    /// The area of the last terrain fetch.
    pub area: Area,
    pub layers: Layers,
    /// Simulation speed for scenarios.
    pub speed: Option<f32>,
    pub side_panel: bool,
    /// Ignite and water tools reach this many cells around the click.
    pub brush: u32,
    /// Where the last file dialog ended up.
    pub dir: Option<PathBuf>,
}

impl Default for Prefs {
    fn default() -> Self {
        Self {
            default: None,
            recent: vec![],
            // Die Haard near Haltern am See, the area shown in the thesis.
            area: Area {
                lat: 51.72,
                lon: 7.19,
                width_km: 1.5,
                height_km: 1.0,
            },
            layers: Layers::default(),
            speed: None,
            side_panel: true,
            brush: 1,
            dir: None,
        }
    }
}

impl Prefs {
    /// Puts a file at the top of the recent list.
    pub fn remember(&mut self, path: &Path) {
        self.recent.retain(|p| p != path);
        self.recent.insert(0, path.to_path_buf());
        self.recent.truncate(8);
        self.dir = path.parent().map(Path::to_path_buf);
    }
}

/// State of the Settings window.
#[derive(Default)]
pub struct Forms {
    pub open: bool,
    tab: usize,
    /// The values being edited: simulation, allocate task, fly task.
    draft: Option<[Value; 3]>,
}

impl Forms {
    /// Shows the settings of an experiment file (JSON of its sections) in the window. Sections
    /// and values the file does not have are the defaults, as they are for training.
    pub(crate) fn load(&mut self, text: &str) -> Result<(), String> {
        let file: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        let section = |i: usize| {
            file.get(SECTIONS[i])
                .cloned()
                .unwrap_or_else(|| json_value!({}))
        };
        let read = || -> Result<[Value; 3], serde_json::Error> {
            Ok([
                json(&serde_json::from_value::<SimConfig>(section(0))?),
                json(&serde_json::from_value::<AllocConfig>(section(1))?),
                json(&serde_json::from_value::<NavConfig>(section(2))?),
            ])
        };
        self.draft = Some(read().map_err(|e| e.to_string())?);
        self.open = true;
        Ok(())
    }
}

/// A file dialog for experiment files, starting in `experiments/` when there is one.
fn dialog() -> rfd::FileDialog {
    let dialog = rfd::FileDialog::new().add_filter("experiment files", &["toml"]);
    match std::path::absolute("experiments") {
        Ok(dir) if dir.is_dir() => dialog.set_directory(dir),
        _ => dialog,
    }
}

/// The tabs' sections in an experiment file.
const SECTIONS: [&str; 3] = ["sim", "allocate", "navigate"];
const TABS: [&str; 3] = ["Simulation", "Drones and planner", "Fly task"];
const TAB_HINTS: [&str; 3] = [
    "time step, water and the fire models",
    "the task where drones are sent to fires",
    "the task where drones fly to waypoints",
];

/// What a parameter is, for its tooltip. `salt` tells the FireSPIN fields (in "….spin") from
/// the rest. A parameter without an entry fails the test below.
fn hint(name: &str, salt: &str) -> &'static str {
    match (name, salt.ends_with(".spin")) {
        ("dt_s", false) => "length of one tick, seconds",
        ("fire_every", _) => "the fire advances once every this many ticks",
        ("water_crit_l_m2", _) => {
            "lowest water load (L/m²) that stops fire: what a weak front needs"
        }
        ("water_per_mw_grass_l_m2", _) => {
            "water (L/m²) per MW/m of fire intensity that stops a front in grass"
        }
        ("water_per_mw_l_m2", _) => {
            "water (L/m²) per MW/m of fire intensity that stops a front in shrub and slash"
        }
        ("water_limit_kw_m", _) => {
            "fire intensity (kW/m) above which no amount of water stops a front"
        }
        ("water_per_mw_timber_l_m2", _) => {
            "water (L/m²) per MW/m of fire intensity that stops a front in litter under trees"
        }
        ("sheltered_waf", _) => {
            "share of the wind that reaches fuel under trees: 0.1 dense stands, 0.2 open"
        }
        ("water_tau_s", _) => {
            "time constant with which water on the ground disappears, seconds (an assumption)"
        }
        ("ember_s", _) => "how long a fully blocked burning cell can still resume, seconds",
        ("ros_noise", _) => "spread of the per-cell spread-rate noise; 0 is deterministic",
        ("dt_s", true) => "time step of the particle model, seconds",
        ("emit_convective", _) => "burning cells send out convection particles",
        ("emit_radiation", _) => "burning cells send out radiation particles",
        ("conv_y_st", _) => "heat a convection particle starts with",
        ("conv_y_lim", _) => "heat below which a convection particle vanishes",
        ("conv_tau_mem_s", _) => {
            "time constant with which a convection particle loses heat, seconds"
        }
        ("conv_fl", _) => "scale between a convection particle's velocity and its movement",
        ("conv_c0", _) => "Kolmogorov constant of the Langevin equation",
        ("lt_m", _) => "turbulent length scale, metres",
        ("wind_a", _) => "turbulence intensity: velocity fluctuation = this x wind speed",
        ("rad_y_st", _) => "heat a radiation particle starts with",
        ("rad_y_lim", _) => "heat below which a radiation particle vanishes",
        ("generic_burn_s", _) => "burning duration of cells without a land-cover class, seconds",
        ("generic_ignite_s", _) => "ignition delay of cells without a land-cover class, seconds",
        ("max_speed_ms", _) => "top speed of a drone, m/s",
        ("tank_l", _) => "tank size, litres",
        ("drop_l", _) => "water released per drop, litres; a drop lands on one cell",
        ("drop_efficiency", _) => "share of a drop that reaches the fuel (an assumption)",
        ("service_s", _) => {
            "time on a pad to swap the battery and fill the tank at a ground station, seconds"
        }
        ("pads", _) => "drones a ground station services at once; the others queue",
        ("endurance_s", _) => {
            "longest time a drone may be in the air between two services, seconds"
        }
        ("drones_per_station", _) => "drones based at every ground station",
        ("arrive_m", _) => "a target counts as reached within this distance, metres",
        ("station_m", _) => "a ground station counts as reached within this distance, metres",
        ("safe_m", _) => {
            "the controller keeps drones in the air this far apart, metres; 0: straight lines, \
             no separation (height lanes assumed)"
        }
        ("sense_m", _) => {
            "every drone in the air sees the fire within this distance, metres, once per fire \
             step; all drones share what they saw. 0: no sensor. The planners read the true \
             fire unless plan_on_belief is set"
        }
        ("explorers_per_station", _) => {
            "explorer drones based at every ground station besides its attack drones: no \
             tank, a camera. They take off by themselves and sweep the map for fire"
        }
        ("explorer_sense_m", _) => {
            "an explorer sees the fire within this distance, metres; its sweep lanes are spaced \
             to match"
        }
        ("explorers_follow", _) => {
            "between two legs of its sweep an explorer visits the fire it believes to burn in \
             its stripe, where the last look is oldest"
        }
        ("plan_on_belief", _) => {
            "the planners work from what the drones have seen, not from the true fire: no \
             target until the fire is found. Needs sense_m or explorers"
        }
        ("block_cells", _) => {
            "cells with a fire front are grouped into square blocks this many cells wide; each \
             is one target"
        }
        ("max_targets", _) => {
            "at most this many fire targets are offered, those needing the most water first"
        }
        ("max_interval_s", _) => "docked drones told to wait are asked again this often, seconds",
        ("time_limit_s", _) => "an episode ends after this long, seconds",
        ("random_ignitions", _) => "above zero: this many random fires replace the scenario's",
        ("random_wind", _) => {
            "turn the scenario's wind by a random angle per episode; its wind changes turn along"
        }
        ("wind_scale", _) => "factor on every wind speed of the scenario",
        ("preburn_s", _) => {
            "the fire burns unattended this long before the episode starts, seconds"
        }
        ("preburn_cells", _) => {
            "... or until this many cells have caught fire, whichever comes first; 0 = no cap"
        }
        ("drones", _) => "number of drones",
        ("view_m", _) => "how far a drone sees other drones and the map edge, metres",
        ("collide_m", _) => "two drones closer than this have collided, metres",
        ("goals", _) => "waypoints each drone has to reach per episode",
        ("station_start", _) => "probability that all drones start at a ground station in a corner",
        ("time_factor", _) => "time limit = goals x map diagonal / top speed x this",
        ("near_drone", _) => {
            "band (fraction of view_m) in which closing in on a drone gives the approach reward term"
        }
        ("near_edge", _) => {
            "band (fraction of view_m) in which closing in on the map edge gives the approach reward term"
        }
        _ => "",
    }
}

fn json<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).expect("a configuration is plain data")
}

/// One row per number and switch of a JSON object; nested objects fold out below. Fields that
/// differ from `default` are bold and can be put back. Text fields are skipped: the one there
/// is, the fire model, has its own menu entry.
fn form(ui: &mut egui::Ui, value: &mut Value, default: &Value, salt: &str) {
    let Value::Object(fields) = value else { return };
    egui::Grid::new(salt)
        .num_columns(3)
        .striped(true)
        .show(ui, |ui| {
            for (name, v) in fields.iter_mut() {
                let standard = &default[name.as_str()];
                if !(v.is_number() || v.is_boolean()) {
                    continue;
                }
                let differs = v != standard;
                let label = egui::RichText::new(name);
                ui.label(if differs { label.strong() } else { label })
                    .on_hover_text(hint(name, salt));
                if let Value::Bool(on) = v {
                    ui.checkbox(on, "");
                } else if let Some(mut whole) = v.as_u64() {
                    if ui.add(egui::DragValue::new(&mut whole)).changed() {
                        *v = whole.into();
                    }
                } else {
                    let mut x = v.as_f64().unwrap_or(0.0);
                    let lowest = if standard.as_f64() < Some(0.0) {
                        f64::MIN
                    } else {
                        0.0
                    };
                    let speed = (x.abs() * 0.01).max(0.001);
                    let field = egui::DragValue::new(&mut x)
                        .speed(speed)
                        .range(lowest..=f64::MAX)
                        .max_decimals(4);
                    if ui.add(field).changed() {
                        *v = x.into();
                    }
                }
                if differs {
                    let back = ui
                        .small_button("⟲")
                        .on_hover_text(format!("back to {standard}"));
                    if back.clicked() {
                        *v = standard.clone();
                    }
                } else {
                    ui.label("");
                }
                ui.end_row();
            }
        });
    for (name, v) in fields.iter_mut().filter(|(_, v)| v.is_object()) {
        ui.collapsing(name.clone(), |ui| {
            form(ui, v, &default[name.as_str()], &format!("{salt}.{name}"));
        });
    }
}

impl Viewer<'_> {
    pub(crate) fn settings_window(&mut self, ctx: &egui::Context) {
        if !self.forms.open {
            self.forms.draft = None;
            return;
        }
        let defaults = [
            json(&SimConfig::default()),
            json(&AllocConfig::default()),
            json(&NavConfig::default()),
        ];
        let shared = &self.shared;
        let draft = self.forms.draft.get_or_insert_with(|| {
            let s = shared.lock().unwrap();
            [json(&s.cfg), json(&s.alloc), json(&s.nav)]
        });
        let tab = &mut self.forms.tab;
        let (mut open, mut apply) = (true, false);
        let (can_load, can_save) = (self.load_settings.is_some(), self.save_settings.is_some());
        // Set by the buttons: read a file; write these sections (JSON).
        let (mut load, mut save) = (false, None);
        egui::Window::new("Settings")
            .open(&mut open)
            .default_width(340.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    for (i, name) in TABS.iter().enumerate() {
                        ui.selectable_value(tab, i, *name)
                            .on_hover_text(TAB_HINTS[i]);
                    }
                });
                ui.separator();
                egui::ScrollArea::vertical()
                    .max_height(520.0)
                    .show(ui, |ui| {
                        form(ui, &mut draft[*tab], &defaults[*tab], TABS[*tab]);
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    apply = ui
                        .button("Apply and restart")
                        .on_hover_text("use these values; the run starts over, paused")
                        .clicked();
                    if ui
                        .button("All to default")
                        .on_hover_text("put every value on this tab back")
                        .clicked()
                    {
                        draft[*tab] = defaults[*tab].clone();
                    }
                    load = ui
                        .add_enabled(can_load, egui::Button::new("Load ..."))
                        .on_hover_text(
                            "show the [sim], [allocate] and [navigate] of an experiment file \
                             (one written by hand, by this window, or the experiment.toml of a run)",
                        )
                        .clicked();
                    if ui
                        .add_enabled(can_save, egui::Button::new("Save ..."))
                        .on_hover_text(
                            "write every value of all three tabs as the [sim], [allocate] and \
                             [navigate] of an experiment file",
                        )
                        .clicked()
                    {
                        let sections = SECTIONS.iter().zip(draft.iter());
                        let all = sections.map(|(name, values)| (name.to_string(), values.clone()));
                        save = Some(Value::Object(all.collect()).to_string());
                    }
                });
                ui.small(
                    "Names are the parameters' own: _s seconds, _m metres, _l litres, _ms m/s.",
                );
            });
        self.forms.open = open;
        if apply {
            self.apply_settings();
        }
        if let (true, Some(read)) = (load, self.load_settings.clone()) {
            self.spawn("choose an experiment file ...", move || {
                let Some(path) = dialog().pick_file() else {
                    return Ok(Outcome::Cancelled);
                };
                Ok(Outcome::Settings(read(&path)?, path))
            });
        }
        if let (Some(text), Some(write)) = (save, self.save_settings.clone()) {
            self.spawn("choose where to save ...", move || {
                let Some(path) = dialog().set_file_name("settings.toml").save_file() else {
                    return Ok(Outcome::Cancelled);
                };
                write(&path, &text).map(Outcome::Message)
            });
        }
    }

    /// Puts the edited values to use; anything the simulation rejects changes nothing.
    fn apply_settings(&mut self) {
        let Some([sim, alloc, nav]) = self.forms.draft.clone() else {
            return;
        };
        let parsed = || -> Result<(SimConfig, AllocConfig, NavConfig), serde_json::Error> {
            Ok((
                serde_json::from_value(sim)?,
                serde_json::from_value(alloc)?,
                serde_json::from_value(nav)?,
            ))
        };
        let mut s = self.shared.lock().unwrap();
        let before = (s.cfg, s.alloc, s.nav);
        let applied = parsed().map_err(|e| e.to_string()).and_then(|new| {
            (s.cfg, s.alloc, s.nav) = new;
            s.restart()
        });
        match applied {
            Ok(()) => {
                s.paused = true;
                self.message = "settings applied".to_string();
            }
            Err(why) => {
                (s.cfg, s.alloc, s.nav) = before;
                self.message = format!("not applied: {why}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The form relies on this: every configuration survives the trip through JSON, and every
    /// field is something the form can show (a number, a switch, text or a nested object).
    #[test]
    fn configurations_survive_the_form() {
        fn check<T>(config: T)
        where
            T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
        {
            let value = json(&config);
            fn showable(v: &Value) -> bool {
                match v {
                    Value::Object(fields) => fields.values().all(showable),
                    other => other.is_number() || other.is_boolean() || other.is_string(),
                }
            }
            assert!(showable(&value), "{value}");
            assert_eq!(serde_json::from_value::<T>(value).unwrap(), config);
        }
        check(SimConfig::default());
        check(AllocConfig::default());
        check(NavConfig::default());
    }

    /// Every parameter the form shows explains itself on hover.
    #[test]
    fn every_parameter_has_a_tooltip() {
        fn check(value: &Value, salt: &str) {
            for (name, v) in value.as_object().unwrap() {
                if v.is_object() {
                    check(v, &format!("{salt}.{name}"));
                } else if !v.is_string() {
                    assert!(!hint(name, salt).is_empty(), "no tooltip for {salt}.{name}");
                }
            }
        }
        check(&json(&SimConfig::default()), TABS[0]);
        check(&json(&AllocConfig::default()), TABS[1]);
        check(&json(&NavConfig::default()), TABS[2]);
    }

    /// A loaded file shows as defaults plus its values, and a name the simulation does not
    /// know is refused instead of dropped.
    #[test]
    fn experiment_files_survive_the_window() {
        let file = json_value!({"allocate": {"tank_l": 250.0}, "sim": {"spin": {"lt_m": 90.0}}});
        let mut forms = Forms::default();
        forms.load(&file.to_string()).unwrap();
        let draft = forms.draft.clone().unwrap();
        assert_eq!(draft[1]["drop_l"], json(&AllocConfig::default())["drop_l"]);
        assert_eq!(draft[1]["tank_l"], 250.0);
        assert_eq!(draft[0]["spin"]["lt_m"], 90.0);
        assert_eq!(draft[2], json(&NavConfig::default()));
        assert!(forms.load(r#"{"allocate": {"tank": 1}}"#).is_err());
        assert_eq!(forms.draft, Some(draft));
    }

    #[test]
    fn preferences_survive_storage() {
        let mut prefs = Prefs::default();
        for k in 0..10 {
            prefs.remember(Path::new(&format!("maps/{k}.npz")));
        }
        prefs.remember(Path::new("maps/5.npz"));
        assert_eq!(prefs.recent.len(), 8);
        assert_eq!(prefs.recent[0], Path::new("maps/5.npz"));
        assert_eq!(prefs.dir.as_deref(), Some(Path::new("maps")));
        let stored = serde_json::to_string(&prefs).unwrap();
        assert_eq!(serde_json::from_str::<Prefs>(&stored).unwrap(), prefs);
        // Settings written by an older version, with fields missing, still load.
        assert_eq!(
            serde_json::from_str::<Prefs>("{}").unwrap(),
            Prefs::default()
        );
    }
}
