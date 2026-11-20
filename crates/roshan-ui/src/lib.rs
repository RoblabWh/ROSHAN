//! ROSHAN2 viewer, scenario editor and policy debugger. It knows nothing about Python: the
//! policies, file loading, saving and terrain fetching are closures.
//!
//! What runs in the window:
//! - a scenario without a ground station: the fire alone;
//! - a scenario with ground stations: the allocate task (drones are based at the stations),
//!   with a built-in planner or a policy, and optionally a learned pilot flying the drones;
//! - navigate: the fly task, with a policy or by hand.
//!
//! Where things are: this file holds the simulation side and the window loop, `panels.rs` the
//! menus and panels, `draw.rs` everything painted on the map, `settings.rs` the parameter forms
//! and what is remembered between runs.
//!
//! Threading rules, which are the reason this file is shaped the way it is:
//! - `run` blocks the calling thread in the window event loop. That thread must be the process
//!   main thread, and every later `run` must come from the same thread (winit caches its loop).
//! - The simulation lives behind a mutex and is stepped by its own thread.
//! - The UI takes the lock once per frame and copies out what it draws.
//! - The sim thread calls a policy WITHOUT holding the sim lock. A Python policy needs the
//!   interpreter lock; taking the two locks in different orders on two threads is the one
//!   deadlock this design can have, and this rule removes it.
//! - Anything slow (file dialogs, loading files, fetching terrain) runs as a job on its own
//!   thread and the UI polls for the outcome.

mod demo;
mod draw;
mod panels;
#[cfg(feature = "map")]
mod picker;
mod settings;

use eframe::egui;
use roshan_core::allocate::{self, AllocConfig, AllocEnv, Method};
use roshan_core::fire::FireModel;
use roshan_core::navigate::{self, NavConfig, NavEnv};
use roshan_core::rothermel::Moisture;
use roshan_core::scenario::Scenario;
use roshan_core::terrain::Grid;
use roshan_core::world::{SimConfig, Wind, World};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

pub use demo::demo_scenario;

/// An observation as named arrays: (name, shape, values). Masks are 0/1 floats.
pub struct Obs(pub Vec<(&'static str, Vec<usize>, Vec<f32>)>);
/// Maps an observation to an action. `Err` carries a printable traceback.
/// planner: one target slot per drone (or -1). pilot: two velocity fractions per drone.
pub type Policy = Box<dyn FnMut(&Obs) -> Result<Vec<f32>, String> + Send>;
/// Stores a scenario, at the given path or under a new name of its own choosing; returns the
/// file it wrote. It may refuse, for example to overwrite a map that experiments are run on.
pub type Save = Arc<dyn Fn(&Scenario, Option<&Path>) -> Result<PathBuf, String> + Send + Sync>;
/// Builds a scenario with real terrain for an area.
pub type Fetch = Arc<dyn Fn(Area) -> Result<Scenario, String> + Send + Sync>;
/// Reads a scenario (or anything that can become one, such as a land-cover GeoTIFF) from disk.
pub type Open = Arc<dyn Fn(&Path) -> Result<Scenario, String> + Send + Sync>;
/// Reads a trained policy from a checkpoint file.
pub type LoadPolicy = Arc<dyn Fn(&Path) -> Result<Loaded, String> + Send + Sync>;

/// Reads the `sim`, `allocate` and `navigate` sections of an experiment file as one JSON object;
/// a section that is missing keeps its defaults.
pub type LoadSettings = Arc<dyn Fn(&Path) -> Result<String, String> + Send + Sync>;
/// Writes such a JSON object as an experiment file; returns a line to show the user.
pub type SaveSettings = Arc<dyn Fn(&Path, &str) -> Result<String, String> + Send + Sync>;

/// A policy read from a checkpoint.
pub struct Loaded {
    /// A pilot (fly policy) rather than a planner.
    pub fly: bool,
    pub policy: Policy,
    /// The task it was trained on, as JSON: an `AllocConfig` for a planner, a `NavConfig` for
    /// a pilot. Missing fields keep their defaults.
    pub config: String,
    /// The simulation it was trained on, as JSON of a `SimConfig`.
    pub sim: String,
}

/// A rectangle on the globe: centre in degrees, size in kilometres.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Area {
    pub lat: f64,
    pub lon: f64,
    pub width_km: f64,
    pub height_km: f64,
}

/// What the window runs.
pub enum Session {
    /// A scenario: the fire, and the allocate task if the scenario has a ground station.
    Scenario(Scenario),
    /// The navigate task on an empty map.
    Navigate { grid: Grid, cfg: NavConfig },
}

/// Who flies the drones in the fly task.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Flight {
    /// Nobody: drones stand still unless steered by hand.
    Off,
    /// Straight at the waypoint, no avoidance.
    Straight,
    /// Straight at the waypoint, sidestepping other drones. Centralised, sees everything.
    Scripted,
    /// Each drone alone, from the observation a fly policy gets.
    Decentral,
    /// The attached fly policy.
    Policy,
}

/// Who hands out targets in the allocate task.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub enum Planner {
    /// Nobody: drones only move when sent by hand.
    Off,
    #[default]
    Greedy,
    Head,
    Flank,
    Rollout,
    /// The attached policy.
    Policy,
}

#[derive(Default)]
pub struct Hooks {
    /// A policy attached from the start: the pilot in a navigate session, the planner otherwise.
    pub policy: Option<Policy>,
    pub save: Option<Save>,
    pub fetch: Option<Fetch>,
    pub open: Option<Open>,
    pub load_policy: Option<LoadPolicy>,
    pub load_settings: Option<LoadSettings>,
    pub save_settings: Option<SaveSettings>,
}

#[derive(Default)]
pub struct Options {
    /// Close the window by itself after this long (used by automated checks). Nothing is then
    /// read from or written to the remembered settings.
    pub auto_close: Option<Duration>,
    /// Open with the street map for choosing an area instead of the simulation.
    pub start_on_map: bool,
    pub planner: Planner,
    /// Open with this drone in the inspector.
    pub select: Option<usize>,
    pub fire_model: FireModel,
    /// The fires burn this long before the drones start (`AllocConfig::preburn_s`).
    pub preburn_s: f32,
    /// The caller chose what to show. Otherwise the user's default scenario, if one is set,
    /// replaces the session's.
    pub explicit: bool,
    /// The file the session's scenario came from, which Save writes back to.
    pub path: Option<PathBuf>,
    /// The world of an experiment: JSON with `sim`, `allocate` and `navigate` objects (any may
    /// be missing). A section that is given is used as training and evaluation would use it,
    /// defaults included, instead of the viewer's own defaults; `fire_model` and `preburn_s`
    /// above then do not apply to it.
    pub settings: Option<String>,
    /// The episode to start on; Restart then replays it instead of moving to the next seed.
    pub seed: Option<u64>,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Stats {
    pub frames: u64,
    pub ticks: u64,
    pub policy_calls: u64,
}

/// The simulation behind the window.
enum Sim {
    Fire(Box<World>),
    Allocate(Box<AllocEnv>, Box<allocate::Scratch>),
    Navigate(Box<NavEnv>, Box<navigate::Scratch>),
}

impl Sim {
    fn world(&self) -> &World {
        match self {
            Sim::Fire(w) => w,
            Sim::Allocate(env, _) => &env.world,
            Sim::Navigate(env, _) => &env.world,
        }
    }

    /// What the drones have seen of the fire (allocate with a sensor).
    fn belief(&self) -> Option<&roshan_core::belief::Belief> {
        match self {
            Sim::Allocate(env, _) => env.belief(),
            _ => None,
        }
    }

    fn world_mut(&mut self) -> &mut World {
        match self {
            Sim::Fire(w) => w,
            Sim::Allocate(env, _) => &mut env.world,
            Sim::Navigate(env, _) => &mut env.world,
        }
    }

    fn term_names(&self) -> &'static [&'static str] {
        match self {
            Sim::Fire(_) => &[],
            Sim::Allocate(..) => &allocate::TERMS,
            Sim::Navigate(..) => &navigate::TERMS,
        }
    }
}

/// Everything the UI thread and the simulation thread share.
struct Shared {
    scenario: Scenario,
    /// The scenario put aside while the navigate task runs on its empty map.
    stash: Option<Scenario>,
    navigating: bool,
    nav: NavConfig,
    /// The map the navigate task runs on.
    nav_grid: Grid,
    alloc: AllocConfig,
    sim: Sim,
    cfg: SimConfig,
    seed: u64,
    /// The Restart button moves on to the next seed instead of replaying this one.
    advance_seed: bool,
    /// Rollout planner: how far it looks ahead (s) and how many alternatives it tries.
    rollout: (f32, usize),
    planner: Planner,
    flight: Flight,
    has_planner: bool,
    has_fly: bool,
    /// Policies on their way to the sim thread: (is the pilot, the policy or None to detach).
    incoming: Vec<(bool, Option<Policy>)>,
    paused: bool,
    /// Simulated seconds per wall-clock second.
    speed: f32,
    /// Advance one fire step while paused.
    step_once: bool,
    ticks: u64,
    policy_calls: u64,
    error: Option<String>,
    /// The drone shown in the inspector.
    selected: Option<usize>,
    /// navigate: the velocity the user steers the selected drone with.
    manual: Option<[f32; 2]>,
    /// Reward terms of the last tick and summed over the episode, [drones, terms].
    terms_last: Vec<f32>,
    terms_sum: Vec<f32>,
    /// How the last episode ended.
    note: String,
    restarts: u64,
    /// Counts changes made by hand to the running fire (ignite, water), so the map repaints.
    edits: u64,
}

impl Shared {
    /// (Re)builds the simulation at time zero. An invalid scenario keeps the old one and
    /// reports why.
    fn restart(&mut self) -> Result<(), String> {
        let nav = self.navigating.then_some(self.nav);
        self.sim = build(&self.scenario, nav, self.cfg, self.alloc, self.seed)?;
        self.restarts += 1;
        let n = self.sim.world().drones.pos.len() * self.sim.term_names().len();
        (self.terms_last, self.terms_sum) = (vec![0.0; n], vec![0.0; n]);
        if self
            .selected
            .is_some_and(|i| i >= self.sim.world().drones.pos.len())
        {
            self.selected = None;
        }
        self.note.clear();
        self.error = None;
        Ok(())
    }

    /// Switches between the scenario and the navigate task on its empty map.
    fn set_navigating(&mut self, on: bool) -> Result<(), String> {
        if on == self.navigating {
            return Ok(());
        }
        if on {
            let empty = empty_scenario(self.nav_grid);
            self.stash = Some(std::mem::replace(&mut self.scenario, empty));
        } else {
            self.scenario = self.stash.take().unwrap_or_else(demo_scenario);
        }
        self.navigating = on;
        self.restart()
    }
}

/// A map on which nothing burns: what the navigate task flies over.
fn empty_scenario(grid: Grid) -> Scenario {
    Scenario {
        grid,
        fuel: vec![0; grid.len()],
        elevation: None,
        landcover: None,
        ignitions: vec![],
        stations: vec![],
        moisture: Moisture {
            d1: 0.06,
            d10: 0.07,
            d100: 0.08,
            herb: 0.6,
            woody: 0.9,
        },
        wind: vec![Wind {
            t_s: 0.0,
            speed_ms: 0.0,
            to: [1.0, 0.0],
        }],
        meta: String::new(),
    }
}

fn build(
    scenario: &Scenario,
    navigate: Option<NavConfig>,
    cfg: SimConfig,
    alloc: AllocConfig,
    seed: u64,
) -> Result<Sim, String> {
    if cfg.dt_s.is_nan() || cfg.dt_s <= 0.0 || cfg.fire_every == 0 {
        return Err("the tick length and the fire interval must be positive".into());
    }
    if let Some(nav) = navigate {
        if nav.drones == 0 {
            return Err("the fly task needs at least one drone".into());
        }
        let env = NavEnv::single(scenario.grid, cfg, nav, seed);
        let mut scratch = navigate::Scratch::new(nav.drones);
        env.observe(
            &mut scratch.agent,
            &mut scratch.neighbors,
            &mut scratch.mask,
        );
        return Ok(Sim::Navigate(Box::new(env), Box::new(scratch)));
    }
    if scenario.stations.is_empty() {
        return Ok(Sim::Fire(Box::new(scenario.world(cfg, seed)?)));
    }
    let env = AllocEnv::single(scenario.clone(), cfg, alloc, seed)?;
    let scratch = allocate::Scratch::new(&env);
    Ok(Sim::Allocate(Box::new(env), Box::new(scratch)))
}

/// Longest stretch of ticks run under one lock, so the UI never waits long for it.
const MAX_TICKS_PER_SLICE: u64 = 20_000;

fn pilot_obs(n: usize, agent: Vec<f32>, neighbors: Vec<f32>, mask: &[u8]) -> Obs {
    Obs(vec![
        ("agent", vec![n, navigate::AGENT.len()], agent),
        ("neighbors", vec![n, n, navigate::NEIGHBOR.len()], neighbors),
        ("mask", vec![n, n], mask.iter().map(|&v| v as f32).collect()),
    ])
}

/// What the planner and the pilot have to be asked before the next tick, if anything.
fn asks(s: &mut Shared) -> (Option<Obs>, Option<Obs>) {
    let view_m = s.nav.view_m;
    match &mut s.sim {
        Sim::Allocate(env, scratch) if !env.ended() => {
            let planner =
                (s.has_planner && s.planner == Planner::Policy && env.pending()).then(|| {
                    env.observe(scratch.out(env));
                    let o = scratch.out(env);
                    let (n, slots) = (o.decide.len(), o.target_mask.len());
                    let floats = |m: &[u8]| m.iter().map(|&v| v as f32).collect::<Vec<f32>>();
                    Obs(vec![
                        ("drones", vec![n, allocate::DRONE.len()], o.drones.to_vec()),
                        (
                            "targets",
                            vec![slots, allocate::TARGET.len()],
                            o.targets.to_vec(),
                        ),
                        ("target_mask", vec![slots], floats(o.target_mask)),
                        ("action_mask", vec![n, slots], floats(o.action_mask)),
                        ("globals", vec![allocate::GLOBAL.len()], o.globals.to_vec()),
                        ("decide", vec![n], floats(o.decide)),
                    ])
                });
            // The pilot sees what it saw in the navigate task, with the planner's targets as
            // waypoints.
            // ponytail: built before this tick's planner decision is applied, so a new target
            // reaches the pilot one tick (0.1 s) late.
            let pilot = s.has_fly.then(|| {
                let n = env.world.drones.pos.len();
                let mut agent = vec![0.0; n * navigate::AGENT.len()];
                let mut neighbors = vec![0.0; n * n * navigate::NEIGHBOR.len()];
                let mut mask = vec![0; n * n];
                env.observe_pilot(view_m, &mut agent, &mut neighbors, &mut mask);
                pilot_obs(n, agent, neighbors, &mask)
            });
            (planner, pilot)
        }
        Sim::Navigate(env, b) if s.has_fly && s.flight == Flight::Policy => {
            let n = env.config().drones;
            let obs = pilot_obs(n, b.agent.clone(), b.neighbors.clone(), &b.mask);
            (None, Some(obs))
        }
        _ => (None, None),
    }
}

/// One tick of whatever is running, with the policies' replies to `asks`, if they were asked.
fn tick(s: &mut Shared, planned: Option<Vec<f32>>, flown: Option<Vec<f32>>) {
    s.terms_last.fill(0.0);
    match &mut s.sim {
        Sim::Fire(world) => world.tick(),
        Sim::Allocate(env, _) => {
            if env.ended() {
                s.paused = true;
                return;
            }
            let n = env.world.drones.pos.len();
            if env.pending() {
                let actions = match s.planner {
                    Planner::Off => vec![-1; n],
                    Planner::Greedy => env.assign(Method::Greedy),
                    Planner::Head => env.assign(Method::Head),
                    Planner::Flank => env.assign(Method::Flank),
                    // ponytail: the lookahead runs under the lock and can stall a frame or two.
                    Planner::Rollout => env.rollout(s.rollout.0, s.rollout.1),
                    Planner::Policy => planned.filter(|a| a.len() == n).map_or(vec![-1; n], |a| {
                        a.iter().map(|&v| v.round() as i32).collect()
                    }),
                };
                env.apply(&actions);
            }
            let (fire_out, timeout) = match flown.filter(|v| v.len() == 2 * n) {
                Some(velocity) => env.tick_flown(Some(&velocity), &mut s.terms_last),
                None => env.tick(&mut s.terms_last),
            };
            if fire_out || timeout {
                let i = env.summary();
                s.note = format!(
                    "{} after {:.0} s: {:.1} % burned, {:.0} drops, {:.0} L",
                    if fire_out { "fire out" } else { "time limit" },
                    i[1],
                    i[3] * 100.0,
                    i[5],
                    i[4]
                );
                s.paused = true;
            }
        }
        Sim::Navigate(env, b) => {
            let n = env.config().drones;
            let mut actions = match s.flight {
                Flight::Off => vec![0.0; 2 * n],
                Flight::Straight => env.pilot(navigate::Pilot::Straight),
                Flight::Scripted => env.pilot(navigate::Pilot::Scripted),
                Flight::Decentral => env.pilot(navigate::Pilot::Decentral),
                Flight::Policy => flown
                    .filter(|a| a.len() == 2 * n)
                    .unwrap_or_else(|| vec![0.0; 2 * n]),
            };
            if let (Some(i), Some(v)) = (s.selected, s.manual) {
                actions[2 * i..2 * i + 2].copy_from_slice(&v);
            }
            if b.terminated + b.truncated > 0 {
                s.terms_sum.fill(0.0); // this step starts the next episode
            }
            env.step(&actions, true, b.out());
            s.terms_last.copy_from_slice(&b.terms);
            if b.terminated + b.truncated > 0 {
                let reason =
                    ["success", "collision", "left the map", "time limit"][b.info[1] as usize];
                s.note = format!(
                    "last episode: {reason} after {:.0} s, {:.0} waypoints",
                    b.info[2], b.info[3]
                );
            }
        }
    }
    for (sum, last) in s.terms_sum.iter_mut().zip(&s.terms_last) {
        *sum += last;
    }
    s.ticks += 1;
}

fn sim_loop(shared: Arc<Mutex<Shared>>, stop: Arc<AtomicBool>) {
    // The policies live on this thread; the UI hands new ones over through `Shared::incoming`.
    let (mut planner, mut pilot): (Option<Policy>, Option<Policy>) = (None, None);
    let mut last = Instant::now();
    let mut owed_s = 0.0f64; // simulated time the sim is behind the wall clock
    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(2));
        let wall = last.elapsed().as_secs_f64();
        last = Instant::now();

        let mut s = shared.lock().unwrap();
        for (is_pilot, policy) in s.incoming.drain(..) {
            *if is_pilot { &mut pilot } else { &mut planner } = policy;
        }
        if s.paused && !s.step_once {
            owed_s = 0.0;
            continue;
        }
        let dt = s.cfg.dt_s as f64;
        let mut ticks = if s.step_once {
            s.step_once = false;
            s.cfg.fire_every as u64
        } else {
            owed_s += wall * s.speed as f64;
            let due = (owed_s / dt) as u64;
            owed_s -= due as f64 * dt;
            if due > MAX_TICKS_PER_SLICE {
                owed_s = 0.0; // cannot keep up at this speed: drop the backlog, do not spiral
            }
            due.min(MAX_TICKS_PER_SLICE)
        };
        while ticks > 0 {
            ticks -= 1;
            let (ask_planner, ask_pilot) = asks(&mut s);
            let mut answers = (None, None);
            if ask_planner.is_some() || ask_pilot.is_some() {
                // Unlock before a policy runs (see the module rules).
                drop(s);
                let call = |obs: Option<Obs>, policy: &mut Option<Policy>| {
                    obs.zip(policy.as_mut()).map(|(o, p)| p(&o)).transpose()
                };
                let reply = call(ask_planner, &mut planner)
                    .and_then(|planned| Ok((planned, call(ask_pilot, &mut pilot)?)));
                s = shared.lock().unwrap();
                match reply {
                    Ok(replies) => {
                        s.policy_calls += replies.0.is_some() as u64 + replies.1.is_some() as u64;
                        answers = replies;
                    }
                    // A failing policy pauses the sim and shows why; the window stays up.
                    Err(trace) => {
                        s.paused = true;
                        s.error = Some(trace);
                        break;
                    }
                }
            }
            tick(&mut s, answers.0, answers.1);
            if s.paused {
                break; // an episode ended
            }
        }
    }
}

/// What a left click on the map does.
#[derive(Clone, Copy, PartialEq)]
enum Tool {
    /// Pick the drone shown in the inspector.
    Select,
    /// Send the selected drone to the clicked cell (allocate).
    Send,
    /// Set the clicked cell on fire in the running simulation.
    Ignite,
    /// Drop water on the running simulation.
    Water,
    /// The rest edit the scenario and restart it: toggle a starting fire or a ground station.
    StartFire,
    Station,
    Erase,
}

/// What a finished job hands back to the UI.
enum Outcome {
    /// A scenario to show, the file it came from (if any), and what to tell the user.
    Scenario(Scenario, Option<PathBuf>, String),
    Policy(Loaded, PathBuf),
    /// Settings read from an experiment file (JSON), for the Settings window.
    Settings(String, PathBuf),
    /// The scenario was written to this file.
    Saved(PathBuf),
    Message(String),
    /// The user closed the file dialog.
    Cancelled,
}

struct Viewer<'a> {
    shared: Arc<Mutex<Shared>>,
    texture: Option<egui::TextureHandle>,
    canvas: draw::Canvas,
    /// The part of the map (in cells) that is on screen, and the map size it was set for.
    view: egui::Rect,
    view_size: [usize; 2],
    tool: Tool,
    /// navigate: steer the selected drone with the keyboard.
    steering: bool,
    /// Keep the selected drone in the middle of the view.
    follow: bool,
    /// The one slow thing in progress (file dialog, load, fetch, screenshot).
    job: Option<mpsc::Receiver<Result<Outcome, String>>>,
    /// Showing the street map to choose the area instead of the simulation.
    picking: bool,
    #[cfg(feature = "map")]
    picker: Option<picker::Picker>,
    /// Last thing worth telling the user (saved file, fetch error).
    message: String,
    /// What the pointer is over, for the status bar.
    hover: String,
    save: Option<Save>,
    fetch: Option<Fetch>,
    open: Option<Open>,
    load_policy: Option<LoadPolicy>,
    load_settings: Option<LoadSettings>,
    save_settings: Option<SaveSettings>,
    /// The file the scenario was opened from or saved to.
    path: Option<PathBuf>,
    /// The scenario was edited since it was opened or saved.
    dirty: bool,
    /// The window was asked to close with unsaved edits and waits for an answer; then the
    /// answer "close anyway".
    closing: bool,
    close_anyway: bool,
    /// The window title as last set.
    title: String,
    /// Names of the attached policies: planner, pilot.
    policy_names: [Option<String>; 2],
    prefs: settings::Prefs,
    forms: settings::Forms,
    show_fetch: bool,
    show_help: bool,
    effects: draw::Effects,
    started: Instant,
    /// Measured pace for the status bar: when and at which simulated time it was last sampled,
    /// and the simulated seconds per real second since the sample before.
    pace: (Instant, f32, f32),
    opts: Options,
    on_frame: &'a mut dyn FnMut() -> bool,
    frames: &'a mut u64,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Fire,
    Allocate,
    Navigate,
}

/// A drone as drawn. Places are in cells.
struct DroneView {
    pos: [f32; 2],
    /// Which way it moves; zero when it stands still.
    vel: [f32; 2],
    goal: Option<[f32; 2]>,
    /// What it is doing (`allocate::TASK_*`).
    kind: f32,
    deciding: bool,
    /// At a ground station (allocate): not drawn, and no body for the others.
    docked: bool,
    /// Docked, serviced and free to be sent.
    ready: bool,
    /// Litres on board, and the tank size (allocate; an explorer has no tank).
    water: Option<(f32, f32)>,
    /// An explorer of the allocate task: flies its own sweep, takes no orders.
    explorer: bool,
    /// Radius of its fire sensor in cells; 0 without one.
    sense: f32,
}

/// What the inspector shows about the selected drone.
struct Inspect {
    index: usize,
    /// Name and value of every observation column of this drone.
    rows: Vec<(String, f32)>,
    /// One line per neighbour slot (navigate).
    slots: Vec<String>,
    /// Reward term, last tick, sum over the episode.
    terms: Vec<(&'static str, f32, f32)>,
}

/// What the UI copies out of the simulation each frame.
struct Snapshot {
    mode: Mode,
    size: [usize; 2],
    cell_m: f32,
    time_s: f32,
    sources: usize,
    /// FireSPIN particles in flight: position in cells, and whether it is a radiation particle.
    particles: Vec<([f32; 2], bool)>,
    burned: f32,
    /// Fireline intensity of the most intense front burning now, kW/m (0 under FireSPIN).
    front_kw: f32,
    /// allocate: what identifies the fire the episode started with.
    fingerprint: Option<u64>,
    wind: Wind,
    paused: bool,
    speed: f32,
    seed: u64,
    advance_seed: bool,
    rollout: (f32, usize),
    fire_model: FireModel,
    planner: Planner,
    flight: Flight,
    has_planner: bool,
    has_fly: bool,
    policy_calls: u64,
    restarts: u64,
    error: Option<String>,
    note: String,
    drones: Vec<DroneView>,
    /// Fire targets on offer, in cells (allocate).
    targets: Vec<[f32; 2]>,
    stations: Vec<[u32; 2]>,
    /// Per station: drones ready to be sent, drones docked (allocate).
    fleet: Vec<(u32, u32)>,
    start_fires: Vec<[u32; 2]>,
    moisture: Moisture,
    /// The scenario's wind schedule.
    winds: Vec<Wind>,
    inspect: Option<Inspect>,
    /// navigate: view range and collision distance in cells, for the rings around the selection.
    rings: Option<(f32, f32)>,
}

fn snapshot(s: &mut Shared) -> Snapshot {
    let grid = s.sim.world().terrain.grid;
    let cells = |p: [f32; 2]| [p[0] / grid.cell_m, p[1] / grid.cell_m];
    let selected = s.selected;
    let names = s.sim.term_names();
    let terms = |i: usize| -> Vec<(&'static str, f32, f32)> {
        (0..names.len())
            .map(|t| {
                (
                    names[t],
                    s.terms_last[i * names.len() + t],
                    s.terms_sum[i * names.len() + t],
                )
            })
            .collect()
    };
    let (mode, drones, targets, inspect, rings) = match &mut s.sim {
        Sim::Fire(_) => (Mode::Fire, vec![], vec![], None, None),
        Sim::Allocate(env, scratch) => {
            let tank = env.config().tank_l;
            let fleet = &env.world.drones;
            let drones = (env.tasks().iter().enumerate())
                .map(|(i, task)| DroneView {
                    pos: cells(fleet.pos[i]),
                    vel: fleet.vel[i],
                    goal: task.1.map(cells),
                    kind: task.0,
                    deciding: env.deciding()[i],
                    docked: env.docked(i),
                    ready: env.ready(i),
                    water: (!env.explorer(i)).then_some((fleet.water[i], tank)),
                    explorer: env.explorer(i),
                    sense: env.sense_m(i) / grid.cell_m,
                })
                .collect();
            let targets = env.fire_points().into_iter().map(cells).collect();
            let inspect = selected.map(|i| {
                env.observe(scratch.out(env));
                let o = scratch.out(env);
                let width = allocate::DRONE.len();
                let mut rows: Vec<(String, f32)> = allocate::DRONE
                    .iter()
                    .zip(&o.drones[i * width..][..width])
                    .map(|(name, &v)| (name.to_string(), v))
                    .collect();
                rows.push(("needs a decision".into(), o.decide[i] as f32));
                rows.push(("water (litres)".into(), env.world.drones.water[i]));
                rows.push(("explorer".into(), env.explorer(i) as u8 as f32));
                Inspect {
                    index: i,
                    rows,
                    slots: vec![],
                    terms: terms(i),
                }
            });
            (Mode::Allocate, drones, targets, inspect, None)
        }
        Sim::Navigate(env, b) => {
            let n = env.config().drones;
            let drones = (0..n)
                .map(|i| DroneView {
                    pos: cells(env.world.drones.pos[i]),
                    vel: env.world.drones.vel[i],
                    goal: Some(cells(env.goals()[i])),
                    kind: allocate::TASK_POINT,
                    deciding: false,
                    docked: false,
                    ready: false,
                    water: None,
                    explorer: false,
                    sense: 0.0,
                })
                .collect();
            let inspect = selected.map(|i| {
                let width = navigate::AGENT.len();
                let rows = navigate::AGENT
                    .iter()
                    .zip(&b.agent[i * width..][..width])
                    .map(|(name, &v)| (name.to_string(), v))
                    .collect();
                let slots = (0..n)
                    .map(|k| {
                        let v = &b.neighbors[(i * n + k) * 4..][..4];
                        let what = if k == 0 { "map edge" } else { "drone" };
                        match b.mask[i * n + k] {
                            1 => format!(
                                "{what}: {:+.2} {:+.2}  moving {:+.2} {:+.2}",
                                v[0], v[1], v[2], v[3]
                            ),
                            _ => format!("{what}: out of view"),
                        }
                    })
                    .collect();
                Inspect {
                    index: i,
                    rows,
                    slots,
                    terms: terms(i),
                }
            });
            let cfg = env.config();
            (
                Mode::Navigate,
                drones,
                vec![],
                inspect,
                Some((cfg.view_m / grid.cell_m, cfg.collide_m / grid.cell_m)),
            )
        }
    };
    let world = s.sim.world();
    Snapshot {
        mode,
        size: [grid.nx as usize, grid.ny as usize],
        cell_m: grid.cell_m,
        time_s: world.time_s(),
        sources: world.fire.n_burning(),
        particles: (world.fire.particles())
            .map(|(p, radiation)| ([p[0] / grid.cell_m, p[1] / grid.cell_m], radiation))
            .collect(),
        burned: world.burned_fraction(),
        front_kw: world.fire.strongest_front_kw(),
        fingerprint: match &s.sim {
            Sim::Allocate(env, _) => Some(env.fingerprint()),
            _ => None,
        },
        wind: world.wind(),
        paused: s.paused,
        speed: s.speed,
        seed: s.seed,
        advance_seed: s.advance_seed,
        rollout: s.rollout,
        fire_model: s.cfg.fire_model,
        planner: s.planner,
        flight: s.flight,
        has_planner: s.has_planner,
        has_fly: s.has_fly,
        policy_calls: s.policy_calls,
        restarts: s.restarts,
        error: s.error.clone(),
        note: s.note.clone(),
        drones,
        targets,
        stations: if mode == Mode::Navigate {
            vec![]
        } else {
            s.scenario.stations.clone()
        },
        fleet: match &s.sim {
            Sim::Allocate(env, _) => env.fleet(),
            _ => vec![],
        },
        start_fires: s.scenario.ignitions.clone(),
        moisture: s.scenario.moisture,
        winds: s.scenario.wind.clone(),
        inspect,
        rings,
    }
}

impl eframe::App for Viewer<'_> {
    /// Runs before every frame and also while the window is hidden, when `ui` is not called:
    /// closing by itself and noticing Ctrl+C must not depend on the window being looked at.
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let timed_out = self
            .opts
            .auto_close
            .is_some_and(|d| self.started.elapsed() >= d);
        if timed_out || !(self.on_frame)() {
            self.close_anyway = true; // an automated run or Ctrl+C is not asked about edits
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        ctx.request_repaint_after(Duration::from_millis(100));
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        *self.frames += 1;
        let ctx = ui.ctx().clone();
        self.poll_job();
        self.title_and_closing(&ctx);

        // navigate: W A S D or the arrow keys steer the selected drone.
        let steer = self.steering.then(|| {
            ctx.input(|i| {
                let key = |a, b| (i.key_down(a) || i.key_down(b)) as i32 as f32;
                [
                    key(egui::Key::D, egui::Key::ArrowRight)
                        - key(egui::Key::A, egui::Key::ArrowLeft),
                    key(egui::Key::S, egui::Key::ArrowDown) - key(egui::Key::W, egui::Key::ArrowUp),
                ]
            })
        });

        // The one lock per frame: copy out everything that gets drawn.
        let snap = {
            let mut s = self.shared.lock().unwrap();
            s.manual = steer;
            self.canvas.update(&s, self.prefs.layers);
            snapshot(&mut s)
        };
        self.hotkeys(&ctx, &snap);
        if snap.size != self.view_size {
            // A new map: show all of it.
            self.view_size = snap.size;
            self.fit();
        }
        if let Some(pixels) = self.canvas.take() {
            let image = egui::ColorImage::new(snap.size, pixels);
            let options = egui::TextureOptions::NEAREST;
            match &mut self.texture {
                Some(texture) => texture.set(image, options),
                None => self.texture = Some(ctx.load_texture("map", image, options)),
            }
        }
        self.effects.update(&snap);
        if snap.mode != Mode::Navigate {
            self.prefs.speed = Some(snap.speed);
        }

        egui::Panel::top("menu").show(ui, |ui| self.menu_bar(ui, &snap));
        egui::Panel::top("toolbar").show(ui, |ui| self.toolbar(ui, &snap));
        egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui, &snap));
        if self.prefs.side_panel {
            egui::Panel::left("controls")
                .resizable(false)
                .exact_size(250.0)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| self.side(ui, &snap));
                });
        }
        if snap.inspect.is_some() {
            egui::Panel::right("inspector")
                .resizable(false)
                .exact_size(250.0)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| self.inspector(ui, &snap));
                });
        }
        egui::CentralPanel::default().show(ui, |ui| {
            #[cfg(feature = "map")]
            if self.picking {
                let picker = self
                    .picker
                    .get_or_insert_with(|| picker::Picker::new(ui.ctx().clone()));
                return picker.show(ui, &mut self.prefs.area);
            }
            self.map(ui, &snap)
        });
        self.windows(&ctx, &snap);

        // The view is live while the simulation runs; `logic` keeps a slow beat going otherwise.
        if !snap.paused || self.effects.busy() {
            ctx.request_repaint();
        }
    }

    fn persist_egui_memory(&self) -> bool {
        self.opts.auto_close.is_none()
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if self.opts.auto_close.is_none() {
            eframe::set_value(storage, "prefs", &self.prefs);
        }
    }
}

impl Viewer<'_> {
    /// Shows the whole map.
    fn fit(&mut self) {
        let size = egui::vec2(self.view_size[0] as f32, self.view_size[1] as f32);
        self.view = egui::Rect::from_min_size(egui::Pos2::ZERO, size);
    }

    /// Changes the scenario and restarts the simulation from it, paused.
    fn edit(&mut self, change: impl FnOnce(&mut Scenario)) {
        let mut s = self.shared.lock().unwrap();
        let before = s.scenario.clone();
        change(&mut s.scenario);
        match s.restart() {
            Ok(()) => (s.paused, self.dirty) = (true, true),
            Err(why) => {
                s.scenario = before;
                self.message = format!("not applied: {why}");
            }
        }
    }

    /// Notes that the scenario now is what `path` holds.
    fn saved(&mut self, path: PathBuf) {
        self.prefs.remember(&path);
        self.message = format!("saved {}", path.display());
        (self.path, self.dirty) = (Some(path), false);
    }

    /// Shows the scenario's file in the window title, with a star while it has unsaved edits,
    /// and asks before such edits are lost by closing the window.
    fn title_and_closing(&mut self, ctx: &egui::Context) {
        let name = self.path.as_ref().and_then(|p| p.file_name());
        let name = name.map_or("not saved yet".into(), |n| n.to_string_lossy());
        let title = format!("ROSHAN2 - {name}{}", if self.dirty { " *" } else { "" });
        if title != self.title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.title = title;
        }
        if ctx.input(|i| i.viewport().close_requested()) && self.dirty && !self.close_anyway {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.closing = true;
        }
        if !self.closing {
            return;
        }
        egui::Window::new("Unsaved changes")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                ui.label("The scenario was edited and not saved.");
                ui.horizontal(|ui| {
                    if ui
                        .button("Keep editing")
                        .on_hover_text("go back; save with Ctrl+S or File > Save as")
                        .clicked()
                    {
                        self.closing = false;
                    }
                    if ui
                        .button("Close without saving")
                        .on_hover_text("close the viewer and lose the edits")
                        .clicked()
                    {
                        self.close_anyway = true;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });
    }

    /// Replaces what the window shows by this scenario, paused. False if it is not valid.
    fn show_scenario(&mut self, scenario: Scenario) -> bool {
        let mut s = self.shared.lock().unwrap();
        let before = (
            std::mem::replace(&mut s.scenario, scenario),
            s.stash.take(),
            std::mem::replace(&mut s.navigating, false),
        );
        match s.restart() {
            Ok(()) => s.paused = true,
            Err(why) => {
                (s.scenario, s.stash, s.navigating) = before;
                self.message = format!("not applied: {why}");
                return false;
            }
        }
        true
    }

    /// Starts something slow on its own thread; `poll_job` picks up the outcome.
    fn spawn(
        &mut self,
        doing: &str,
        job: impl FnOnce() -> Result<Outcome, String> + Send + 'static,
    ) {
        if self.job.is_some() {
            self.message = "still busy with the previous request".to_string();
            return;
        }
        let (tx, rx) = mpsc::channel();
        // If the window closed meanwhile nobody is listening, which is fine.
        std::thread::spawn(move || drop(tx.send(job())));
        self.job = Some(rx);
        self.message = doing.to_string();
    }

    fn poll_job(&mut self) {
        let result = match self.job.as_ref().map(|rx| rx.try_recv()) {
            Some(Ok(result)) => result,
            Some(Err(mpsc::TryRecvError::Disconnected)) => Err("the job died".to_string()),
            _ => return,
        };
        self.job = None;
        match result {
            Ok(Outcome::Scenario(scenario, path, line)) => {
                if self.show_scenario(scenario) {
                    self.message = line;
                    self.picking = false;
                    if let Some(path) = &path {
                        self.prefs.remember(path);
                    }
                    // Only a scenario file can be saved back to where it came from.
                    self.path = path.filter(|p| p.extension().is_some_and(|e| e == "npz"));
                    self.dirty = false;
                }
            }
            Ok(Outcome::Policy(loaded, path)) => self.attach(loaded, &path),
            Ok(Outcome::Settings(text, path)) => {
                self.message = match self.forms.load(&text) {
                    Ok(()) => format!("{} loaded: Apply and restart to use it", path.display()),
                    Err(why) => format!("{}: {why}", path.display()),
                };
            }
            Ok(Outcome::Saved(path)) => self.saved(path),
            Ok(Outcome::Message(line)) => self.message = line,
            Ok(Outcome::Cancelled) => self.message.clear(),
            Err(why) => self.message = format!("failed: {why}"),
        }
    }

    /// Hands a loaded policy to the sim thread and sets the task up the way it was trained.
    fn attach(&mut self, loaded: Loaded, path: &Path) {
        let name = (path.file_name()).map_or_else(String::new, |n| n.to_string_lossy().into());
        let mut s = self.shared.lock().unwrap();
        let before = (s.alloc, s.nav, s.cfg);
        let parsed = if loaded.fly {
            serde_json::from_str(&loaded.config).map(|nav| s.nav = nav)
        } else {
            serde_json::from_str(&loaded.config).map(|alloc| s.alloc = alloc)
        };
        // A checkpoint from an older simulator names settings that no longer exist: the
        // simulation then stays as it is, and the message says so.
        let trained_sim = serde_json::from_str(&loaded.sim).map(|cfg| s.cfg = cfg);
        if let Err(why) = parsed.map_err(|e| e.to_string()).and_then(|()| s.restart()) {
            (s.alloc, s.nav, s.cfg) = before;
            self.message = format!("{name} does not fit: {why}");
            return;
        }
        s.incoming.push((loaded.fly, Some(loaded.policy)));
        if loaded.fly {
            (s.has_fly, s.flight) = (true, Flight::Policy);
        } else {
            (s.has_planner, s.planner) = (true, Planner::Policy);
        }
        s.paused = true;
        let role = if loaded.fly { "pilot" } else { "planner" };
        self.message = match trained_sim {
            Ok(()) => format!(
                "{name} is now the {role}; task and simulation are set up as it was trained"
            ),
            Err(why) => format!(
                "{name} is now the {role}; the task is set up as it was trained, but not the \
                 simulation, whose settings have changed since ({why})"
            ),
        };
        self.policy_names[loaded.fly as usize] = Some(name);
    }
}

impl Shared {
    /// The simulation at time zero, running.
    fn new(session: Session, policy: Option<Policy>, opts: &Options) -> Result<Self, String> {
        let given: serde_json::Value = match &opts.settings {
            Some(text) => serde_json::from_str(text).map_err(|e| e.to_string())?,
            None => serde_json::Value::Null,
        };
        fn section<T: serde::de::DeserializeOwned>(
            given: &serde_json::Value,
            name: &str,
        ) -> Result<Option<T>, String> {
            let read = |v: &serde_json::Value| serde_json::from_value(v.clone());
            (given.get(name).map(read).transpose()).map_err(|e| format!("[{name}]: {e}"))
        }
        // ponytail: some spread-rate noise by default so fires do not look drawn with a ruler;
        // Simulation > Settings changes it. An experiment's settings come without it.
        let cfg = section(&given, "sim")?.unwrap_or(SimConfig {
            ros_noise: 0.25,
            fire_model: opts.fire_model,
            ..SimConfig::default()
        });
        // The viewer has no time limit worth the name: a day. An experiment or a loaded
        // planner brings its own.
        let alloc = section(&given, "allocate")?.unwrap_or(AllocConfig {
            time_limit_s: 86_400.0,
            preburn_s: opts.preburn_s,
            ..AllocConfig::default()
        });
        let seed = opts.seed.unwrap_or(1);
        let (scenario, navigating, nav, nav_grid) = match session {
            Session::Scenario(scenario) => {
                let grid = Grid {
                    nx: 35,
                    ny: 36,
                    cell_m: 10.0,
                };
                let nav = section(&given, "navigate")?.unwrap_or_default();
                (scenario, false, nav, grid)
            }
            // An empty map of the right size stands in for the scenario.
            Session::Navigate { grid, cfg } => (empty_scenario(grid), true, cfg, grid),
        };
        let sim = build(&scenario, navigating.then_some(nav), cfg, alloc, seed)?;
        let n = sim.world().drones.pos.len() * sim.term_names().len();
        let n_drones = sim.world().drones.pos.len();
        let has_policy = policy.is_some();
        Ok(Self {
            scenario,
            stash: None,
            navigating,
            nav,
            nav_grid,
            alloc,
            sim,
            cfg,
            seed,
            advance_seed: opts.seed.is_none(),
            rollout: (300.0, 4),
            planner: opts.planner,
            flight: if has_policy && navigating {
                Flight::Policy
            } else {
                Flight::Scripted
            },
            has_planner: has_policy && !navigating,
            has_fly: has_policy && navigating,
            incoming: vec![(navigating, policy)],
            paused: false,
            // Flying is watched in real time, fires in fast motion.
            speed: if navigating { 2.0 } else { 30.0 },
            step_once: false,
            ticks: 0,
            policy_calls: 0,
            error: None,
            selected: opts.select.filter(|&i| i < n_drones),
            manual: None,
            terms_last: vec![0.0; n],
            terms_sum: vec![0.0; n],
            note: String::new(),
            restarts: 1,
            edits: 0,
        })
    }
}

/// Opens the window and blocks until it closes. `on_frame` runs once per frame and returns
/// false to ask for the window to close (the Python binding uses it to poll for Ctrl+C).
pub fn run(
    session: Session,
    hooks: Hooks,
    opts: Options,
    on_frame: &mut dyn FnMut() -> bool,
) -> Result<Stats, String> {
    let has_policy = hooks.policy.is_some();
    let navigating = matches!(session, Session::Navigate { .. });
    let shared = Arc::new(Mutex::new(Shared::new(session, hooks.policy, &opts)?));
    let stop = Arc::new(AtomicBool::new(false));
    let worker = {
        let (shared, stop) = (shared.clone(), stop.clone());
        std::thread::spawn(move || sim_loop(shared, stop))
    };

    let mut frames = 0u64;
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ROSHAN2")
            .with_inner_size([1280.0, 860.0]),
        renderer: eframe::Renderer::Glow,
        // Automated checks neither see nor touch what the user's sessions remembered.
        persist_window: opts.auto_close.is_none(),
        // run_and_return stays at its default (true): false would exit the whole process,
        // Python included, when the window closes.
        ..Default::default()
    };
    let result = eframe::run_native(
        "roshan2",
        native,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(panels::visuals());
            let prefs: settings::Prefs = (cc.storage)
                .filter(|_| opts.auto_close.is_none())
                .and_then(|storage| eframe::get_value(storage, "prefs"))
                .unwrap_or_default();
            if let (Some(speed), false) = (prefs.speed, navigating) {
                shared.lock().unwrap().speed = speed;
            }
            let mut viewer = Viewer {
                shared: shared.clone(),
                texture: None,
                canvas: draw::Canvas::default(),
                view: egui::Rect::ZERO,
                view_size: [0, 0],
                tool: Tool::Select,
                steering: false,
                follow: false,
                job: None,
                picking: opts.start_on_map,
                #[cfg(feature = "map")]
                picker: None,
                message: String::new(),
                hover: String::new(),
                save: hooks.save,
                fetch: hooks.fetch,
                open: hooks.open,
                load_policy: hooks.load_policy,
                load_settings: hooks.load_settings,
                save_settings: hooks.save_settings,
                path: opts.path.clone(),
                dirty: false,
                closing: false,
                close_anyway: false,
                title: String::new(),
                policy_names: [
                    (has_policy && !navigating).then(|| "given at start".to_string()),
                    (has_policy && navigating).then(|| "given at start".to_string()),
                ],
                prefs,
                forms: settings::Forms::default(),
                show_fetch: opts.start_on_map,
                show_help: false,
                effects: draw::Effects::default(),
                started: Instant::now(),
                pace: (Instant::now(), 0.0, 0.0),
                opts,
                on_frame,
                frames: &mut frames,
            };
            if let (false, false, Some(path)) = (
                viewer.opts.explicit,
                navigating,
                viewer.prefs.default.clone(),
            ) {
                viewer.open_path(path);
            }
            Ok(Box::new(viewer))
        }),
    );

    // Always stop and join the sim thread, also when the window failed to open.
    stop.store(true, Ordering::Relaxed);
    worker
        .join()
        .map_err(|_| "simulation thread panicked".to_string())?;
    result.map_err(|e| e.to_string())?;

    let s = shared.lock().unwrap();
    Ok(Stats {
        frames,
        ticks: s.ticks,
        policy_calls: s.policy_calls,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Learned flight under a planner: the pilot is asked every tick, sees one row per drone,
    /// and its velocities move the drones towards the targets the planner chose.
    #[test]
    fn a_pilot_flies_the_planners_targets() {
        let session = Session::Scenario(demo_scenario());
        let mut s = Shared::new(session, None, &Options::default()).unwrap();
        s.has_fly = true;
        let start = s.sim.world().drones.pos.clone();
        let (n, width) = (start.len(), navigate::AGENT.len());
        assert!(n > 0, "the demo has drones");
        for _ in 0..100 {
            let (planner, pilot) = asks(&mut s);
            assert!(planner.is_none(), "no planner policy is attached");
            let obs = pilot.expect("the pilot is asked every tick");
            let (name, shape, agent) = &obs.0[0];
            assert_eq!((*name, shape), ("agent", &vec![n, width]));
            // Steer straight at the waypoint: columns 2 and 3 are its direction.
            let velocity = (0..n).flat_map(|i| [agent[i * width + 2], agent[i * width + 3]]);
            tick(&mut s, None, Some(velocity.collect()));
        }
        let now = &s.sim.world().drones.pos;
        let moved = |i: usize| (now[i][0] - start[i][0]).hypot(now[i][1] - start[i][1]);
        assert!((0..n).any(|i| moved(i) > 10.0), "nobody flew anywhere");
    }

    /// Drones start docked: a planner sends them out, without one they stay where they are
    /// until sent by hand, and a drone in service cannot be sent.
    #[test]
    fn docked_drones_leave_with_a_planner_and_stay_without() {
        let flying = |s: &Shared| match &s.sim {
            Sim::Allocate(env, _) => (0..env.world.drones.pos.len())
                .filter(|&i| !env.docked(i))
                .count(),
            _ => panic!("the demo runs the allocate task"),
        };
        let session = || Session::Scenario(demo_scenario());
        let mut s = Shared::new(session(), None, &Options::default()).unwrap();
        assert_eq!((s.planner, flying(&s)), (Planner::Greedy, 0));
        (0..20).for_each(|_| tick(&mut s, None, None));
        // One burning cell needs two drops; nobody else is sent.
        assert_eq!(flying(&s), 2);
        let snap = snapshot(&mut s);
        assert_eq!(
            snap.fleet,
            [(38, 38)],
            "ready and docked at the one station"
        );
        assert_eq!(snap.drones.iter().filter(|d| d.docked).count(), 38);

        let mut s = Shared::new(session(), None, &Options::default()).unwrap();
        s.planner = Planner::Off;
        (0..20).for_each(|_| tick(&mut s, None, None));
        assert_eq!(flying(&s), 0);
        let Sim::Allocate(env, _) = &mut s.sim else {
            unreachable!()
        };
        env.send(5, [20, 60]).unwrap(); // the demo's fire, a minute's flight away
        (0..1500).for_each(|_| tick(&mut s, None, None));
        let Sim::Allocate(env, _) = &mut s.sim else {
            unreachable!()
        };
        assert_eq!(env.drops(), 1, "it flew there and dropped");
        assert!(env.docked(5) && !env.ready(5), "home again and in service");
        assert!(env.send(5, [20, 60]).is_err());
    }

    #[test]
    fn switching_to_the_fly_task_and_back_keeps_the_scenario() {
        let session = Session::Scenario(demo_scenario());
        let mut s = Shared::new(session, None, &Options::default()).unwrap();
        let cells = s.scenario.grid.len();
        s.set_navigating(true).unwrap();
        assert!(matches!(s.sim, Sim::Navigate(..)));
        // Without a policy the scripted pilot flies, and "off" leaves the drones where they are.
        assert_eq!(s.flight, Flight::Scripted);
        let moved = |s: &mut Shared| {
            let start = s.sim.world().drones.pos.clone();
            (0..50).for_each(|_| tick(s, None, None));
            start != s.sim.world().drones.pos
        };
        assert!(moved(&mut s), "the scripted pilot flies");
        s.flight = Flight::Off;
        assert!(!moved(&mut s), "nobody flies");
        s.set_navigating(false).unwrap();
        assert!(matches!(s.sim, Sim::Allocate(..)));
        assert_eq!(s.scenario.grid.len(), cells);
    }
}
