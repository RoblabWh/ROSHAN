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
use crate::{Area, Mode, Outcome, Viewer};
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
    /// The values being edited: simulation, allocate task, fly task. Kept while the window
    /// is closed, so closing it loses nothing.
    draft: Option<[Value; 3]>,
    /// What was running when the draft was last compared with it (`rebase`).
    base: [Value; 3],
    filter: String,
    changed_only: bool,
    /// The experiment file the draft was loaded from.
    from: Option<PathBuf>,
}

impl Forms {
    /// Open on a tab from the start (`Options::show_settings`), or closed.
    pub(crate) fn opened_on(tab: Option<usize>) -> Self {
        Self {
            open: tab.is_some(),
            tab: tab.unwrap_or(0).min(TABS.len() - 1),
            ..Self::default()
        }
    }

    /// Shows the settings of an experiment file (JSON of its sections) in the window. Sections
    /// and values the file does not have are the defaults, as they are for training.
    pub(crate) fn load(&mut self, text: &str, from: &Path) -> Result<(), String> {
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
        self.from = Some(from.to_path_buf());
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

/// What a parameter is, for its tooltip. `salt` is its section; it tells the FireSPIN fields
/// (in "sim.spin") from the rest. A parameter without an entry fails the test below.
fn hint(name: &str, salt: &str) -> &'static str {
    match (name, salt.ends_with(".spin")) {
        ("dt_s", false) => "length of one tick, seconds",
        ("fire_model", _) => "which fire model runs: Rothermel spread rates or FireSPIN particles",
        ("spin", _) => "parameters of the FireSPIN particle model; used only while it is chosen",
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
        ("explorer_speed_ms", _) => "top speed of an explorer, m/s: no tank, so the faster drone",
        ("report_m", _) => {
            "above zero: every fire is reported at the start, off by up to this many metres; \
             explorers fly to the reports first and search around them. 0: nobody is told"
        }
        ("explorers_follow", _) => {
            "between two legs of its sweep an explorer visits the fire it believes to burn in \
             its stripe, where the last look is oldest"
        }
        ("explorers_track", _) => {
            "up to this many explorers stay with a fire once one is known, flying from one \
             believed front cell to the next, the longest unseen first; the others keep sweeping"
        }
        ("forecast", _) => {
            "between looks the belief is a forecast: the fire model runs on from what was seen \
             (known wind and fuel, no spread-rate noise) and every look corrects it. Off: the \
             belief is the last look. Needs a sensor; Rothermel only"
        }
        ("plan_on_belief", _) => {
            "the planners work from what the drones have seen, not from the true fire: no \
             target until the fire is found. Needs sense_m or explorers"
        }
        ("triage", _) => {
            "the numbers of the triage planner: which blocks get no drone, and how the rest \
             are ranked"
        }
        ("cap_kw_m", _) => {
            "triage: blocks whose strongest front is above this intensity (kW/m) get no drone"
        }
        ("per_mw_m", _) => "triage: metres added to a block's distance per MW/m of its front",
        ("per_drop_m", _) => {
            "triage: metres added per drop the block still lacks; nearly stopped blocks first"
        }
        ("per_neighbour_m", _) => {
            "triage: metres added per neighbouring block that still lacks water"
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

/// A configuration as the form holds it: fields in the order they are declared (serde_json's
/// `preserve_order`), numbers as they are written (0.1, not the f32's 0.10000000149).
fn json<T: serde::Serialize>(value: &T) -> Value {
    let text = serde_json::to_string(value).expect("a configuration is plain data");
    serde_json::from_str(&text).expect("it was just written")
}

/// Group headings: (section, the key the group starts with, title). Keys follow in the order
/// the struct declares them, so a new parameter lands in the group it is declared in. For a
/// nested section the key is its name and the title that of its fold.
const HEADINGS: [(&str, &str, &str); 20] = [
    ("sim", "dt_s", "Clock"),
    ("sim", "water_crit_l_m2", "Water"),
    ("sim", "sheltered_waf", "Fire"),
    ("sim", "spin", "FireSPIN particle model"),
    ("sim.spin", "dt_s", "Particles"),
    ("sim.spin", "conv_y_st", "Convection"),
    ("sim.spin", "rad_y_st", "Radiation"),
    ("sim.spin", "generic_burn_s", "Cells without land cover"),
    ("allocate", "max_speed_ms", "Drone"),
    ("allocate", "service_s", "Ground station"),
    ("allocate", "arrive_m", "Flight"),
    ("allocate", "sense_m", "Finding the fire"),
    ("allocate", "block_cells", "Targets"),
    ("allocate", "time_limit_s", "Episode"),
    ("allocate", "triage", "Triage planner"),
    (
        "allocate.triage",
        "cap_kw_m",
        "Score of a block, in metres of flight",
    ),
    ("navigate", "drones", "Drones"),
    ("navigate", "view_m", "Distances"),
    ("navigate", "goals", "Episode"),
    ("navigate", "near_drone", "Reward bands"),
];

fn heading(salt: &str, name: &str) -> Option<&'static str> {
    let found = HEADINGS.iter().find(|(s, n, _)| (*s, *n) == (salt, name));
    found.map(|h| h.2)
}

/// The unit shown after a number, from the ending of the parameter's name.
fn unit(name: &str) -> &'static str {
    // Longer endings first: `_kw_m` is not metres.
    const UNITS: [(&str, &str); 7] = [
        ("_l_m2", " L/m²"),
        ("_kw_m", " kW/m"),
        ("_cells", " cells"),
        ("_ms", " m/s"),
        ("_s", " s"),
        ("_m", " m"),
        ("_l", " L"),
    ];
    let found = UNITS.iter().find(|(ending, _)| name.ends_with(ending));
    found.map_or("", |u| u.1)
}

/// What a text parameter can be.
/// ponytail: the one text parameter there is. A new one needs its choices here; a test fails
/// without them.
fn choices(name: &str) -> &'static [&'static str] {
    match name {
        "fire_model" => &["rothermel", "firespin"],
        _ => &[],
    }
}

/// Why a parameter changes nothing right now, given the tab's values (`root`); None when it
/// does. Such rows are dimmed but stay editable.
fn idle(salt: &str, name: &str, root: &Value) -> Option<&'static str> {
    let firespin = root["fire_model"] == "firespin";
    let rothermel_only = matches!(name, "ros_noise" | "sheltered_waf" | "water_limit_kw_m")
        || name.starts_with("water_per_mw");
    match (salt, name) {
        ("sim.spin", _) if !firespin => Some("the FireSPIN model is not chosen"),
        ("sim", _) if firespin && rothermel_only => Some("the FireSPIN model does not use it"),
        (
            "allocate",
            "explorer_sense_m" | "explorer_speed_ms" | "explorers_follow" | "explorers_track",
        ) if root["explorers_per_station"] == 0 => Some("explorers_per_station is 0"),
        ("allocate", "preburn_cells") if root["preburn_s"] == 0.0 => Some("preburn_s is 0"),
        _ => None,
    }
}

/// Why a whole tab changes nothing right now.
fn idle_tab(tab: usize, mode: Mode) -> Option<&'static str> {
    match (tab, mode) {
        (1, Mode::Navigate) => Some("The fly task is running: these apply to a scenario."),
        (1, Mode::Fire) => Some(
            "This scenario has no ground station, so no drones: place one with the station tool.",
        ),
        (2, Mode::Fire | Mode::Allocate) => {
            Some("The fly task is not running: switch to it in the Simulation menu.")
        }
        _ => None,
    }
}

/// The filter box: `filter` is found in the parameter's name or in its description.
fn matches(name: &str, hint: &str, filter: &str) -> bool {
    let filter = filter.trim().to_lowercase();
    name.contains(&filter) || hint.to_lowercase().contains(&filter)
}

/// Takes over what changed in the running world (`was` to `now`) for every value the user
/// has not edited, so that Apply never undoes a change made elsewhere (the fire model in the
/// menu, a loaded checkpoint's settings).
fn rebase(draft: &mut Value, was: &Value, now: &Value) {
    if let Value::Object(fields) = draft {
        for (name, v) in fields.iter_mut() {
            rebase(v, &was[name.as_str()], &now[name.as_str()]);
        }
    } else if draft == was {
        *draft = now.clone();
    }
}

/// What Apply would change, one line per value: "pads 8 → 16".
fn changes(draft: &Value, now: &Value, prefix: &str) -> Vec<String> {
    let Value::Object(fields) = draft else {
        return vec![];
    };
    let mut out = vec![];
    for (name, v) in fields {
        let old = &now[name.as_str()];
        if v.is_object() {
            out.extend(changes(v, old, &format!("{prefix}{name}.")));
        } else if v != old {
            out.push(format!("{prefix}{name} {old} → {v}"));
        }
    }
    out
}

/// The part of a configuration that differs from `default`, as an object of its own.
fn differing(value: &Value, default: &Value) -> Value {
    let mut out = serde_json::Map::new();
    for (name, v) in value.as_object().into_iter().flatten() {
        let standard = &default[name.as_str()];
        let part = if v.is_object() {
            differing(v, standard)
        } else {
            v.clone()
        };
        if part != *standard && part != json_value!({}) {
            out.insert(name.clone(), part);
        }
    }
    Value::Object(out)
}

/// The draft as flags for `python -m roshan.eval`: what differs from the defaults in the
/// simulation and in the allocate task. Empty when nothing does.
fn overrides(draft: &[Value; 3], defaults: &[Value; 3]) -> String {
    let flag = |(name, i): (&str, usize)| {
        let part = differing(&draft[i], &defaults[i]);
        (part != json_value!({})).then(|| format!("{name} '{part}'"))
    };
    let flags: Vec<String> = [("--sim", 0), ("--config", 1)]
        .into_iter()
        .filter_map(flag)
        .collect();
    flags.join(" ")
}

/// Puts every number and switch back to its default; text (the fire model) stays.
fn to_default(value: &mut Value, default: &Value) {
    if let Value::Object(fields) = value {
        for (name, v) in fields.iter_mut() {
            to_default(v, &default[name.as_str()]);
        }
    } else if !value.is_string() {
        *value = default.clone();
    }
}

/// The draft as configurations, or why the simulation would refuse it.
fn parse(draft: &[Value; 3]) -> Result<(SimConfig, AllocConfig, NavConfig), String> {
    fn read<T: serde::de::DeserializeOwned>(draft: &[Value; 3], i: usize) -> Result<T, String> {
        serde_json::from_value(draft[i].clone()).map_err(|e| format!("[{}] {e}", SECTIONS[i]))
    }
    let (sim, alloc, nav): (SimConfig, AllocConfig, NavConfig) =
        (read(draft, 0)?, read(draft, 1)?, read(draft, 2)?);
    sim.check()?;
    alloc.check()?;
    nav.check()?;
    Ok((sim, alloc, nav))
}

/// Which rows the form shows.
struct View<'a> {
    filter: &'a str,
    changed_only: bool,
    /// The tab's values as they were at the start of the frame, for `idle`.
    root: &'a Value,
}

const LABEL_W: f32 = 190.0;
const VALUE_W: f32 = 96.0;

/// The rows of a JSON object under their group headings; nested objects fold out below.
fn form(ui: &mut egui::Ui, value: &mut Value, default: &Value, salt: &str, view: &View) {
    let Value::Object(fields) = value else { return };
    let mut groups: Vec<(&str, Vec<String>)> = vec![("", vec![])];
    for (name, v) in fields.iter().filter(|(_, v)| !v.is_object()) {
        if let Some(title) = heading(salt, name) {
            groups.push((title, vec![]));
        }
        let changed = *v != default[name.as_str()];
        if matches(name, hint(name, salt), view.filter) && (changed || !view.changed_only) {
            groups.last_mut().unwrap().1.push(name.clone());
        }
    }
    for (title, names) in groups.iter().filter(|g| !g.1.is_empty()) {
        ui.add_space(6.0);
        if !title.is_empty() {
            ui.strong(*title);
        }
        egui::Grid::new((salt, *title))
            .num_columns(3)
            .striped(true)
            .show(ui, |ui| {
                for name in names {
                    let (v, standard) = (&mut fields[name.as_str()], &default[name.as_str()]);
                    row(ui, name, v, standard, salt, view.root);
                    ui.end_row();
                }
            });
    }
    for (name, v) in fields.iter_mut().filter(|(_, v)| v.is_object()) {
        let inner = format!("{salt}.{name}");
        let running = idle(&inner, "", view.root).is_none();
        // While rows are being searched for, a fold must not hide them.
        let searching = !view.filter.trim().is_empty() || view.changed_only;
        ui.add_space(6.0);
        egui::CollapsingHeader::new(heading(salt, name).unwrap_or(name))
            .id_salt(&inner)
            .default_open(running)
            .open(searching.then_some(true))
            .show_unindented(ui, |ui| {
                form(ui, v, &default[name.as_str()], &inner, view);
            })
            .header_response
            .on_hover_text(hint(name, salt));
    }
}

/// One parameter: its name, its value with the unit, and a way back to the default.
fn row(ui: &mut egui::Ui, name: &str, v: &mut Value, standard: &Value, salt: &str, root: &Value) {
    let differs = v != standard;
    let idle = idle(salt, name, root);
    let mut tip = format!("{}\ndefault: {standard}", hint(name, salt));
    if let Some(why) = idle {
        tip += &format!("\nno effect right now: {why}");
    }
    let label = egui::RichText::new(name);
    let label = match (idle, differs) {
        (Some(_), _) => label.weak(),
        (None, true) => label.strong(),
        (None, false) => label,
    };
    ui.horizontal(|ui| {
        ui.set_min_width(LABEL_W);
        ui.label(label).on_hover_text(&tip);
    });
    let size = egui::vec2(VALUE_W, ui.spacing().interact_size.y);
    if let Value::Bool(on) = v {
        ui.checkbox(on, "").on_hover_text(&tip);
    } else if let Value::String(chosen) = v {
        ui.horizontal(|ui| {
            for choice in choices(name) {
                ui.radio_value(chosen, choice.to_string(), *choice)
                    .on_hover_text(&tip);
            }
        });
    } else if let Some(mut whole) = v.as_u64() {
        let field = egui::DragValue::new(&mut whole)
            .min_size(size)
            .suffix(unit(name));
        if ui.add(field).on_hover_text(&tip).changed() {
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
            .min_size(size)
            .suffix(unit(name))
            .speed(speed)
            .range(lowest..=f64::MAX)
            .max_decimals(4)
            // As short as the number allows: 0.7, not 0.700.
            .custom_formatter(|x, _| (x as f32).to_string());
        if ui.add(field).on_hover_text(&tip).changed() {
            *v = x.into();
        }
    }
    if differs {
        // Not reachable with Tab: Enter on it would undo the value just typed.
        let back = egui::Button::new("⟲")
            .small()
            .frame(false)
            .sense(egui::Sense::CLICK);
        if ui
            .add(back)
            .on_hover_text(format!("back to the default, {standard}"))
            .clicked()
        {
            *v = standard.clone();
        }
    } else {
        ui.label("");
    }
}

impl Viewer<'_> {
    pub(crate) fn settings_window(&mut self, ctx: &egui::Context, mode: Mode) {
        if !self.forms.open {
            return;
        }
        let defaults = [
            json(&SimConfig::default()),
            json(&AllocConfig::default()),
            json(&NavConfig::default()),
        ];
        let now = {
            let s = self.shared.lock().unwrap();
            [json(&s.cfg), json(&s.alloc), json(&s.nav)]
        };
        let Forms {
            tab,
            draft,
            base,
            filter,
            changed_only,
            from,
            ..
        } = &mut self.forms;
        match draft {
            Some(draft) => {
                for i in 0..3 {
                    rebase(&mut draft[i], &base[i], &now[i]);
                }
            }
            None => *draft = Some(now.clone()),
        }
        *base = now.clone();
        let draft = draft.as_mut().expect("just made");
        let problem = parse(draft).err();
        let pending: Vec<String> = (0..3)
            .flat_map(|i| changes(&draft[i], &now[i], ""))
            .collect();
        let can_apply = problem.is_none() && !pending.is_empty();
        let (can_load, can_save) = (self.load_settings.is_some(), self.save_settings.is_some());
        // Set by the buttons: apply; read a file; write these sections (JSON); copied text.
        let mut revert = false;
        let (mut open, mut apply, mut load, mut save, mut copied) =
            (true, false, false, None, None);
        egui::Window::new("Settings")
            .open(&mut open)
            .default_size([450.0, 700.0])
            .min_width(440.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    for (i, name) in TABS.iter().enumerate() {
                        ui.selectable_value(tab, i, *name)
                            .on_hover_text(TAB_HINTS[i]);
                    }
                });
                ui.horizontal(|ui| {
                    let find = egui::TextEdit::singleline(filter)
                        .hint_text("filter")
                        .desired_width(150.0);
                    ui.add(find).on_hover_text(
                        "show only the parameters whose name or description holds this text, \
                         from all three tabs",
                    );
                    ui.checkbox(changed_only, "changed only")
                        .on_hover_text("show only the values that differ from the defaults");
                    if ui
                        .button("All to default")
                        .on_hover_text("put every number and switch on this tab back")
                        .clicked()
                    {
                        to_default(&mut draft[*tab], &defaults[*tab]);
                    }
                });
                ui.separator();
                egui::Panel::bottom("settings_actions")
                    .frame(egui::Frame::NONE)
                    .show(ui, |ui| {
                        ui.add_space(4.0);
                        if let Some(why) = &problem {
                            let why = egui::RichText::new(why).color(egui::Color32::LIGHT_RED);
                            ui.add(egui::Label::new(why).wrap());
                        }
                        if let Some(path) = from.as_ref() {
                            let text = format!("loaded from {}", path.display());
                            let text = egui::RichText::new(text).small();
                            ui.add(egui::Label::new(text).truncate());
                        }
                        egui::Sides::new().show(
                            ui,
                            |ui| {
                                load = ui
                                    .add_enabled(can_load, egui::Button::new("Load ..."))
                                    .on_hover_text(
                                        "show the [sim], [allocate] and [navigate] of an \
                                         experiment file (one written by hand, by this window, \
                                         or the experiment.toml of a run)",
                                    )
                                    .clicked();
                                if ui
                                    .add_enabled(can_save, egui::Button::new("Save ..."))
                                    .on_hover_text(
                                        "write every value of all three tabs as the [sim], \
                                         [allocate] and [navigate] of an experiment file",
                                    )
                                    .clicked()
                                {
                                    let sections = SECTIONS.iter().zip(draft.iter());
                                    let all = sections.map(|(n, v)| (n.to_string(), v.clone()));
                                    save = Some(Value::Object(all.collect()).to_string());
                                }
                                if ui
                                    .button("Copy")
                                    .on_hover_text(
                                        "copy what differs from the defaults as --sim and \
                                         --config flags for python -m roshan.eval",
                                    )
                                    .clicked()
                                {
                                    copied = Some(overrides(draft, &defaults));
                                }
                            },
                            |ui| {
                                let count = match pending.len() {
                                    0 => String::new(),
                                    n => format!(" ({n})"),
                                };
                                let button = egui::Button::new(format!("Apply and restart{count}"))
                                    .selected(can_apply);
                                let tip = format!(
                                    "use these values; the run starts over (Ctrl+Enter)\n{}",
                                    pending.join("\n")
                                );
                                apply = ui
                                    .add_enabled(can_apply, button)
                                    .on_hover_text(&tip)
                                    .on_disabled_hover_text(match &problem {
                                        Some(why) => why.as_str(),
                                        None => "nothing differs from what is running",
                                    })
                                    .clicked();
                                let back = egui::Button::new("Revert");
                                if ui
                                    .add_enabled(!pending.is_empty(), back)
                                    .on_hover_text("drop the edits and show what is running now")
                                    .clicked()
                                {
                                    revert = true;
                                }
                            },
                        );
                    });
                egui::ScrollArea::vertical()
                    .auto_shrink(false)
                    .show(ui, |ui| {
                        // A filter looks through all three tabs.
                        let all = !filter.trim().is_empty();
                        for i in (0..3).filter(|&i| all || i == *tab) {
                            if all {
                                ui.add_space(6.0);
                                ui.label(egui::RichText::new(TABS[i]).heading());
                            }
                            if let Some(note) = idle_tab(i, mode) {
                                ui.add(egui::Label::new(egui::RichText::new(note).weak()).wrap());
                            }
                            let root = draft[i].clone();
                            let view = View {
                                filter: filter.as_str(),
                                changed_only: *changed_only,
                                root: &root,
                            };
                            form(ui, &mut draft[i], &defaults[i], SECTIONS[i], &view);
                        }
                    });
            });
        let enter =
            |i: &mut egui::InputState| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Enter);
        apply |= can_apply && ctx.input_mut(enter);
        let name = match from.as_ref().and_then(|p| p.file_stem()) {
            Some(stem) => format!("{}-edited.toml", stem.to_string_lossy()),
            None => "settings.toml".to_string(),
        };
        if revert {
            (*draft, *from) = (now, None);
        }
        self.forms.open = open;
        if apply {
            self.apply_settings(&pending);
        }
        if let Some(text) = copied {
            self.message = match text.is_empty() {
                true => "nothing differs from the defaults".to_string(),
                false => format!("copied: {text}"),
            };
            ctx.copy_text(text);
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
                let Some(path) = dialog().set_file_name(name).save_file() else {
                    return Ok(Outcome::Cancelled);
                };
                write(&path, &text).map(Outcome::Message)
            });
        }
    }

    /// Puts the edited values to use; anything the simulation rejects changes nothing. The
    /// run goes on as it was: paused stays paused.
    fn apply_settings(&mut self, pending: &[String]) {
        let Some(draft) = &self.forms.draft else {
            return;
        };
        let mut s = self.shared.lock().unwrap();
        let before = (s.cfg, s.alloc, s.nav);
        let applied = parse(draft).and_then(|new| {
            (s.cfg, s.alloc, s.nav) = new;
            s.restart()
        });
        self.message = match applied {
            Ok(()) => format!("applied: {}", pending.join(", ")),
            Err(why) => {
                (s.cfg, s.alloc, s.nav) = before;
                format!("not applied: {why}")
            }
        };
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
                }
                assert!(!hint(name, salt).is_empty(), "no tooltip for {salt}.{name}");
                if let Some(text) = v.as_str() {
                    assert!(
                        choices(name).contains(&text),
                        "no choices for {salt}.{name}"
                    );
                }
            }
        }
        check(&json(&SimConfig::default()), SECTIONS[0]);
        check(&json(&AllocConfig::default()), SECTIONS[1]);
        check(&json(&NavConfig::default()), SECTIONS[2]);
    }

    fn defaults() -> [Value; 3] {
        [
            json(&SimConfig::default()),
            json(&AllocConfig::default()),
            json(&NavConfig::default()),
        ]
    }

    /// Numbers read as they are written, and fields come in the order they are declared.
    #[test]
    fn the_form_shows_clean_numbers_in_declared_order() {
        let [sim, alloc, _] = defaults();
        assert_eq!(sim["dt_s"], 0.1);
        assert_eq!(alloc["drop_efficiency"], 0.7);
        let names: Vec<&String> = alloc.as_object().unwrap().keys().collect();
        assert_eq!(names[..3], ["max_speed_ms", "tank_l", "drop_l"]);
    }

    /// Every heading names a parameter that exists, and every section starts with one.
    #[test]
    fn headings_name_real_parameters() {
        let [sim, alloc, nav] = defaults();
        let sections = [
            ("sim", &sim),
            ("sim.spin", &sim["spin"]),
            ("allocate", &alloc),
            ("allocate.triage", &alloc["triage"]),
            ("navigate", &nav),
        ];
        for (salt, name, _) in HEADINGS {
            let (_, values) = sections.iter().find(|s| s.0 == salt).expect(salt);
            assert!(values.get(name).is_some(), "{salt}.{name} does not exist");
        }
        for (salt, values) in sections {
            let first = values.as_object().unwrap().keys().next().unwrap();
            assert!(heading(salt, first).is_some(), "{salt} starts without one");
        }
    }

    #[test]
    fn units_come_from_the_ending_of_the_name() {
        let cases = [
            ("water_crit_l_m2", " L/m²"),
            ("water_limit_kw_m", " kW/m"),
            ("max_speed_ms", " m/s"),
            ("preburn_cells", " cells"),
            ("endurance_s", " s"),
            ("sense_m", " m"),
            ("tank_l", " L"),
            ("pads", ""),
            ("max_targets", ""),
            ("conv_y_lim", ""),
        ];
        for (name, expected) in cases {
            assert_eq!(unit(name), expected, "{name}");
        }
    }

    /// What changed elsewhere reaches the draft; what the user edited stays.
    #[test]
    fn the_draft_follows_the_world_but_keeps_edits() {
        let was = defaults();
        let mut draft = was.clone();
        draft[1]["pads"] = 16.into();
        draft[0]["spin"]["lt_m"] = 90.0.into();
        let mut now = was.clone();
        now[0]["fire_model"] = "firespin".into(); // switched in the Simulation menu
        now[0]["spin"]["lt_m"] = 70.0.into();
        now[1]["pads"] = 4.into();
        now[1]["tank_l"] = 250.0.into(); // a checkpoint's task
        for i in 0..3 {
            rebase(&mut draft[i], &was[i], &now[i]);
        }
        assert_eq!(draft[0]["fire_model"], "firespin");
        assert_eq!(draft[1]["tank_l"], 250.0);
        assert_eq!(
            (&draft[1]["pads"], &draft[0]["spin"]["lt_m"]),
            (&16.into(), &90.0.into())
        );
        let pending: Vec<String> = (0..3)
            .flat_map(|i| changes(&draft[i], &now[i], ""))
            .collect();
        assert_eq!(pending, ["spin.lt_m 70.0 → 90.0", "pads 4 → 16"]);
    }

    /// The window says why a draft would be refused before Apply is pressed.
    #[test]
    fn a_draft_the_simulation_refuses_says_why() {
        let good = defaults();
        assert!(parse(&good).is_ok());
        let broken = |i: usize, path: &[&str], v: Value| {
            let mut draft = good.clone();
            let mut at = &mut draft[i];
            for name in path {
                at = &mut at[*name];
            }
            *at = v;
            parse(&draft).unwrap_err()
        };
        let why = broken(1, &["plan_on_belief"], true.into());
        assert!(why.contains("plan_on_belief needs a sensor"), "{why}");
        let why = broken(0, &["spin", "dt_s"], 0.0.into());
        assert!(why.contains("spin.dt_s must be above zero"), "{why}");
        let why = broken(2, &["goals"], 0.into());
        assert!(why.contains("goals must be above zero"), "{why}");
        let why = broken(1, &["pads"], 5_000_000_000u64.into());
        assert!(why.starts_with("[allocate]"), "{why}");
        let why = broken(0, &["fire_model"], "campfire".into());
        assert!(why.starts_with("[sim]"), "{why}");
    }

    #[test]
    fn parameters_without_effect_are_told_apart() {
        let [mut sim, mut alloc, _] = defaults();
        assert!(idle("sim.spin", "lt_m", &sim).is_some());
        assert!(idle("sim", "ros_noise", &sim).is_none());
        sim["fire_model"] = "firespin".into();
        assert!(idle("sim.spin", "lt_m", &sim).is_none());
        assert!(idle("sim", "water_per_mw_grass_l_m2", &sim).is_some());
        assert!(idle("sim", "water_crit_l_m2", &sim).is_none());
        assert!(idle("allocate", "explorers_follow", &alloc).is_some());
        assert!(idle("allocate", "preburn_cells", &alloc).is_some());
        assert!(idle("allocate", "sense_m", &alloc).is_none());
        (alloc["explorers_per_station"], alloc["preburn_s"]) = (2.into(), 600.0.into());
        assert!(idle("allocate", "explorers_follow", &alloc).is_none());
        assert!(idle("allocate", "preburn_cells", &alloc).is_none());
    }

    #[test]
    fn the_filter_finds_by_name_or_description() {
        let found = |name: &str, filter: &str| matches(name, hint(name, "allocate"), filter);
        assert!(found("sense_m", "sens") && found("sense_m", " Sees the fire "));
        assert!(found("explorer_sense_m", "sense") && !found("pads", "sense"));
        assert!(found("pads", ""));
    }

    /// Copy gives flags the evaluation accepts: only what differs, nested sections kept.
    #[test]
    fn copy_gives_the_differences_as_flags() {
        let standard = defaults();
        let mut draft = standard.clone();
        assert_eq!(overrides(&draft, &standard), "");
        draft[1]["explorers_per_station"] = 4.into();
        draft[1]["plan_on_belief"] = true.into();
        draft[2]["goals"] = 9.into(); // the fly task is not part of an evaluation
        assert_eq!(
            overrides(&draft, &standard),
            r#"--config '{"explorers_per_station":4,"plan_on_belief":true}'"#
        );
        draft[0]["spin"]["lt_m"] = 90.0.into();
        let text = overrides(&draft, &standard);
        assert!(
            text.starts_with(r#"--sim '{"spin":{"lt_m":90.0}}' --config"#),
            "{text}"
        );
        let sim: Value = serde_json::from_str(r#"{"spin":{"lt_m":90.0}}"#).unwrap();
        assert!(serde_json::from_value::<SimConfig>(sim).is_ok());

        // "All to default" puts numbers and switches back and leaves the fire model alone.
        draft[0]["fire_model"] = "firespin".into();
        to_default(&mut draft[0], &standard[0]);
        assert_eq!(draft[0]["spin"], standard[0]["spin"]);
        assert_eq!(draft[0]["fire_model"], "firespin");
    }

    /// A loaded file shows as defaults plus its values, and a name the simulation does not
    /// know is refused instead of dropped.
    #[test]
    fn experiment_files_survive_the_window() {
        let file = json_value!({"allocate": {"tank_l": 250.0}, "sim": {"spin": {"lt_m": 90.0}}});
        let mut forms = Forms::default();
        forms.load(&file.to_string(), Path::new("a.toml")).unwrap();
        let draft = forms.draft.clone().unwrap();
        assert_eq!(draft[1]["drop_l"], json(&AllocConfig::default())["drop_l"]);
        assert_eq!(draft[1]["tank_l"], 250.0);
        assert_eq!(draft[0]["spin"]["lt_m"], 90.0);
        assert_eq!(draft[2], json(&NavConfig::default()));
        assert!(
            forms
                .load(r#"{"allocate": {"tank": 1}}"#, Path::new("b.toml"))
                .is_err()
        );
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
