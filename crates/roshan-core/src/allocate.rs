//! The `allocate` task: a swarm of small drones based at ground stations, and a planner that
//! says which fire each one flies to.
//!
//! A drone waits docked at a station until the planner gives it a fire target. It then flies
//! there, drops its water, flies home, is serviced (battery and tank) on one of the station's
//! pads and is ready again. Only the choice of target is the planner's; the rest is automatic.
//! Targets are small square blocks of cells that hold a fire front.
//!
//! One step runs the simulation until the next *decision event*: a drone became ready, the
//! block it was flying to stopped burning, or a waiting drone's regular turn came. The step
//! returns how long that took and which drones need a decision, so a trainer can discount by
//! real time and only ask the drones that are actually free. An action is an index into the
//! target table, or -1: stay docked, or fly home when in the air.
//!
//! Episode `e` is a function of its seed alone (ignitions, wind, noise), so two policies run
//! on the same seed face the same fire.
//! ponytail: concrete types like `navigate`; the batching is shared once a third task exists.

use std::collections::BTreeMap;
use std::sync::Arc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::belief::{Belief, NEVER};
use crate::fire::{BURNING, FireModel, Front};
use crate::rng::{self, Stream};
use crate::scenario::Scenario;
use crate::terrain::{Grid, Terrain};
use crate::world::{SimConfig, Weather, World};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AllocConfig {
    pub max_speed_ms: f32,
    /// Tank size and the amount released per drop, litres. A drop lands on one cell.
    pub tank_l: f32,
    pub drop_l: f32,
    /// Share of a drop that reaches the fuel; the rest drifts, evaporates or misses.
    pub drop_efficiency: f32,
    /// Time on a pad to swap the battery and fill the tank, and how many pads a station has.
    pub service_s: f32,
    pub pads: u32,
    /// Longest time a drone may be in the air between two services, seconds.
    pub endurance_s: f32,
    /// Drones based at every ground station of the scenario.
    pub drones_per_station: u32,
    /// A fire target is reached within this distance, a ground station within `station_m`.
    pub arrive_m: f32,
    pub station_m: f32,
    /// The scripted controller keeps drones in the air this far apart. 0: no separation, the
    /// drones fly straight lines (for swarms: they are taken to fly in separate height lanes).
    pub safe_m: f32,
    /// Every drone in the air sees the fire state of the cells within this distance, once
    /// per fire step, and all share what they saw (`belief::Belief`). 0: no sensor. The
    /// planners read the true fire unless `plan_on_belief` is set.
    pub sense_m: f32,
    /// Explorer drones based at every station, besides its `drones_per_station`: the same
    /// drone with a camera instead of a tank. They take off by themselves and sweep the map
    /// for fire; no planner commands them.
    /// ponytail: same speed, endurance, pads and service as the attack drones. Give them
    /// their own speed and endurance keys when a faster scout is to be studied; the speed
    /// then has to reach `fly` and `navigate::scripted_step` per drone.
    pub explorers_per_station: u32,
    /// Radius of an explorer's fire sensor, metres (the attack drones' is `sense_m`).
    pub explorer_sense_m: f32,
    /// Top speed of an explorer, m/s (the attack drones' is `max_speed_ms`): without a tank
    /// it is the lighter, faster aircraft.
    /// ponytail: the pilot observation still scales velocities by `max_speed_ms`; give it
    /// the drone's own speed when a fly policy is to steer explorers.
    pub explorer_speed_ms: f32,
    /// Above zero: every fire is reported when the episode starts, at its ignition point
    /// moved by a random error of up to this many metres. Explorers fly to the reports
    /// first (explorer k to report k) and search around each in widening squares out to
    /// that distance before they sweep. 0: nobody is told anything.
    /// ponytail: only explorers use a report; attack drones still wait for a sighting.
    pub report_m: f32,
    /// Between two legs of its sweep an explorer visits the fire it believes to burn in its
    /// stripe: the cell whose last look is oldest for the flight it takes to get there.
    pub explorers_follow: bool,
    /// Up to this many explorers stay with a fire once one is known: each flies from one
    /// believed front cell to the next, always to the one whose last look is oldest for the
    /// flight it takes, and goes back to its sweep only when no such cell is left. The other
    /// explorers keep sweeping (and following, with `explorers_follow`).
    pub explorers_track: u32,
    /// Between looks the belief is a forecast: the fire model runs on from what was seen,
    /// with the known wind, fuel and water but without the spread-rate noise (`ros_noise`),
    /// and every look corrects it. Off: the belief is the last look. Needs a sensor;
    /// Rothermel model only.
    /// ponytail: the forecast knows wind, fuel and moisture exactly; only `ros_noise` and
    /// what no drone has seen make it wrong. Give it its own weather when forecast errors
    /// are to be studied.
    pub forecast: bool,
    /// The planners work from what the drones have seen (`belief::Belief`), not from the fire:
    /// targets, where a drop goes, the news that a target went out and the `burning` and
    /// `burned` globals. Needs a sensor (`sense_m` or explorers). Until the fire is found
    /// there is no target.
    /// ponytail: the water lying on a cell stays the true value (the swarm dropped it and
    /// knows how it evaporates), and the rollout planner's lookahead still simulates the
    /// true fire. Build its world from the belief when the fire is forecast from it.
    pub plan_on_belief: bool,
    /// Cells with a fire front are grouped into square blocks of this many cells; each block
    /// is one target. 1 makes every front cell its own target.
    pub block_cells: u32,
    /// At most this many fire targets are offered, those needing the most water first.
    pub max_targets: usize,
    /// Docked drones that were told to wait are asked again this often.
    pub max_interval_s: f32,
    pub time_limit_s: f32,
    /// If above zero, the scenario's starting fires are replaced by this many random ones.
    pub random_ignitions: u32,
    /// Turn the scenario's wind schedule by one random angle per episode.
    pub random_wind: bool,
    /// Factor on every wind speed of the scenario's schedule.
    pub wind_scale: f32,
    /// The fire burns unattended this long before the episode starts, seconds, so the drones
    /// meet a front the fire model itself produced. 0 starts with the bare ignitions.
    pub preburn_s: f32,
    /// ... or until this many cells have caught fire, whichever comes first (0 = no cap).
    /// Keeps fires comparable across fuels: a fixed time gives a huge fire on windy grass
    /// and a tiny one in forest.
    pub preburn_cells: u32,
}

impl Default for AllocConfig {
    fn default() -> Self {
        Self {
            max_speed_ms: 10.0,
            tank_l: 40.0,
            drop_l: 40.0,
            drop_efficiency: 0.7,
            service_s: 60.0,
            pads: 8,
            endurance_s: 900.0,
            drones_per_station: 40,
            arrive_m: 5.0,
            station_m: 12.0,
            safe_m: 0.0,
            sense_m: 0.0,
            explorers_per_station: 0,
            explorer_sense_m: 55.0,
            explorer_speed_ms: 15.0,
            report_m: 0.0,
            explorers_follow: true,
            explorers_track: 0,
            forecast: false,
            plan_on_belief: false,
            block_cells: 2,
            max_targets: 64,
            max_interval_s: 10.0,
            time_limit_s: 1800.0,
            random_ignitions: 0,
            random_wind: false,
            wind_scale: 1.0,
            preburn_s: 0.0,
            preburn_cells: 0,
        }
    }
}

impl AllocConfig {
    /// Refuses settings the task cannot run with.
    pub fn check(&self) -> Result<(), String> {
        let positive = [
            ("max_speed_ms", self.max_speed_ms),
            ("tank_l", self.tank_l),
            ("drop_l", self.drop_l),
            ("drop_efficiency", self.drop_efficiency),
            ("service_s", self.service_s),
            ("endurance_s", self.endurance_s),
            ("arrive_m", self.arrive_m),
            ("station_m", self.station_m),
            ("max_interval_s", self.max_interval_s),
            ("time_limit_s", self.time_limit_s),
            ("wind_scale", self.wind_scale),
            ("explorer_sense_m", self.explorer_sense_m),
            ("explorer_speed_ms", self.explorer_speed_ms),
            ("pads", self.pads as f32),
            ("drones_per_station", self.drones_per_station as f32),
            ("block_cells", self.block_cells as f32),
            ("max_targets", self.max_targets as f32),
        ];
        if let Some((name, v)) = positive.iter().find(|(_, v)| !(*v > 0.0 && v.is_finite())) {
            return Err(format!("{name} must be above zero, got {v}"));
        }
        if self.drop_efficiency > 1.0 {
            return Err("drop_efficiency is a share: at most 1".into());
        }
        if self.drop_l > self.tank_l {
            return Err("drop_l is more than the tank holds".into());
        }
        if !(self.safe_m >= 0.0 && self.safe_m.is_finite()) {
            return Err(format!("safe_m must be zero or more, got {}", self.safe_m));
        }
        if !(self.report_m >= 0.0 && self.report_m.is_finite()) {
            return Err(format!(
                "report_m must be zero or more, got {}",
                self.report_m
            ));
        }
        if !(self.sense_m >= 0.0 && self.sense_m.is_finite()) {
            return Err(format!(
                "sense_m must be zero or more, got {}",
                self.sense_m
            ));
        }
        let blind = self.sense_m == 0.0 && self.explorers_per_station == 0;
        if self.plan_on_belief && blind {
            return Err("plan_on_belief needs a sensor: sense_m or explorers_per_station".into());
        }
        if self.forecast && blind {
            return Err("forecast needs a sensor: sense_m or explorers_per_station".into());
        }
        Ok(())
    }
}

/// Per-drone columns. Positions are fractions of the map width and height.
pub const DRONE: [&str; 6] = ["x", "y", "water", "task_x", "task_y", "task_kind"];
/// Per-target columns. Stations come first, then fire targets.
pub const TARGET: [&str; 7] = [
    "x",
    "y",
    "is_station",
    "cells",     // front cells in the block / cells per block
    "need",      // drops it takes to stop the block's fronts now
    "inbound",   // drops on their way to the block
    "intensity", // highest fireline intensity of its fronts, MW/m
];
pub const GLOBAL: [&str; 5] = ["wind_x", "wind_y", "time", "burning", "burned"];
/// Reward terms per drone for the step just finished.
pub const TERMS: [&str; 5] = [
    "drop",     // drops this drone made
    "refill",   // tank fractions this drone took on
    "burned",   // cells that caught fire, as a fraction of the burnable cells (same for all)
    "fire_out", // the fire went out (all drones)
    "timeout",  // the time limit was reached (all drones)
];
/// Episode summary, valid on the step an episode ends.
pub const INFO: [&str; 20] = [
    "fire_out",
    "seconds", // since the episode started, not counting the pre-burn
    "burned_cells",
    "burned_fraction",
    "water_l",
    "drops",
    "min_separation_m", // NaN where separation was not kept (`safe_m` 0 and scripted flight)
    "start_cells",      // cells that had caught fire when the episode started
    "stopped_cells",    // burning cells that water kept from spreading until they died
    "stopped_kw_mean",  // fireline intensity of those cells, kW/m (Rothermel only)
    "stopped_kw_max",
    "peak_kw",      // highest fireline intensity of the fire, kW/m (Rothermel only)
    "start_kw",     // ... of the fire as the drones find it, when the episode starts
    "first_drop_s", // NaN if there was none
    "queue_s",      // drone-seconds spent waiting for a free pad
    "max_sortie_s", // longest time a drone was in the air between two services
    // What the drones saw of the fire (`belief::Belief`); NaN without a sensor.
    "detect_s",       // when a burning cell was first seen; NaN if none was
    "seen_share",     // share of the map's cells looked at at least once
    "belief_miss",    // share of the burning cells not believed burning, mean over fire steps
    "belief_phantom", // share of the believed-burning cells not burning, mean over fire steps
];
pub const TASK_NONE: f32 = 0.0;
pub const TASK_FIRE: f32 = 1.0;
pub const TASK_STATION: f32 = 2.0;
pub const TASK_POINT: f32 = 3.0;
/// At a station, in service or ready.
pub const TASK_DOCKED: f32 = 4.0;
/// An explorer on its sweep.
pub const TASK_EXPLORE: f32 = 5.0;

/// The planners that need no learning (see the `impl` further down).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    /// The nearest block that still needs water.
    Greedy,
    /// The most intense block that still needs water.
    Head,
    /// The least intense block that still needs water.
    Flank,
    /// Greedy improved by simulating ahead.
    Rollout,
}

impl Method {
    pub const NAMES: [&str; 4] = ["greedy", "head", "flank", "rollout"];

    pub fn parse(name: &str) -> Option<Self> {
        let all = [Self::Greedy, Self::Head, Self::Flank, Self::Rollout];
        Self::NAMES.iter().position(|n| *n == name).map(|i| all[i])
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Task {
    None,
    /// Fly to this block and drop water on its fire front.
    Fire(u32),
    /// Fly to this station (index into the scenario's list) to be serviced.
    Station(u32),
    /// Fly to this cell (sent by hand).
    Point(u32),
    /// An explorer flying to this cell of its sweep.
    Explore(u32),
    /// An explorer leaving its sweep to look at this believed-burning cell.
    Follow(u32),
}

/// A fire target on offer: a block of cells with at least one front.
#[derive(Clone, Copy, Debug)]
struct Target {
    block: u32,
    /// The block's centre, metres.
    point: [f32; 2],
    fronts: u32,
    /// Drops needed now, drops on their way.
    need: f32,
    inbound: f32,
    /// Highest intensity of its fronts, kW/m.
    kw: f32,
    /// From the block to the nearest station, metres.
    home_m: f32,
}

const AIRBORNE: u64 = u64::MAX;

/// The buffers of one environment.
pub struct Out<'a> {
    pub drones: &'a mut [f32],
    pub targets: &'a mut [f32],
    /// [slots] 1 where a target slot is filled.
    pub target_mask: &'a mut [u8],
    /// [drones, slots] 1 where that drone may choose that slot. Only the rows of drones that
    /// need a decision are filled: a fire needs water for a drop and time in the air to get
    /// there and home, a station can be chosen in the air.
    pub action_mask: &'a mut [u8],
    pub globals: &'a mut [f32],
    /// [drones] 1 where the drone needs a decision.
    pub decide: &'a mut [u8],
    pub terms: &'a mut [f32],
    pub elapsed_s: &'a mut f32,
    pub terminated: &'a mut u8,
    pub truncated: &'a mut u8,
    pub info: &'a mut [f32],
    pub seed: &'a mut u64,
    pub fingerprint: &'a mut u64,
}

#[derive(Clone)]
pub struct AllocEnv {
    pub world: World,
    scenario: Arc<Scenario>,
    terrain: Arc<Terrain>,
    cfg: AllocConfig,
    sim: SimConfig,
    base_seed: u64,
    index: u64,
    episode: u64,
    seed: u64,
    fingerprint: u64,
    task: Vec<Task>,
    decide: Vec<bool>,
    /// Where each drone is heading, metres; where it is when it has nowhere to go.
    goal: Vec<[f32; 2]>,
    /// The tick from which a docked drone is serviced and ready; `AIRBORNE` in the air.
    ready_at: Vec<u64>,
    /// The station a docked drone is at.
    at: Vec<u32>,
    /// The tick a drone last took off.
    launched: Vec<u64>,
    /// [stations, pads] the tick from which each pad is free.
    pad_free: Vec<u64>,
    /// The fire targets offered at the last decision.
    fires: Vec<Target>,
    drops: u32,
    water_l: f32,
    min_separation: f32,
    first_drop: Option<u64>,
    queue_ticks: u64,
    max_sortie: u64,
    needs_reset: bool,
    /// The world's tick at which the episode started, and the cells alight by then: both
    /// zero-ish without a pre-burn. The world's own clock counts from the ignition.
    start_tick: u64,
    start_cells: u32,
    /// Highest fireline intensity (kW/m) the fire had reached by then.
    start_kw: f32,
    /// What the drones have seen of the fire in this episode; None without a sensor.
    belief: Option<Belief>,
    /// Per explorer: the cells its sweep visits in turn, there and back again.
    sweeps: Arc<Vec<Vec<u32>>>,
    /// Per drone: how many waypoints of its sweep an explorer has reached.
    leg: Vec<u32>,
    /// Per drone: an explorer has visited the fire since it last flew a leg of its sweep.
    visited: Vec<bool>,
    /// Per drone: an explorer that stays with the fire (`explorers_track`).
    tracker: Vec<bool>,
    /// Where fires were reported at the start (`report_m`), metres, and per explorer the
    /// cells it searches around its report before it sweeps.
    reports: Vec<[f32; 2]>,
    search: Vec<Vec<u32>>,
    /// The believed fire run on by the fire model (`AllocConfig::forecast`): a world of its
    /// own without drones, corrected by every look.
    forecast: Option<World>,
}

/// Drones of both kinds. The attack drones come first in every per-drone list, station by
/// station, then the explorers, station by station.
fn fleet_size(stations: usize, cfg: &AllocConfig) -> usize {
    stations * (cfg.drones_per_station + cfg.explorers_per_station) as usize
}

/// Stripe `e` of `k` along a side of `long` cells: its first cell and its width.
fn stripe(long: u32, k: u32, e: u32) -> (u32, u32) {
    let lo = (e * long / k).min(long - 1);
    (lo, ((e + 1) * long / k).max(lo + 1) - lo)
}

/// How many cells apart an explorer's lanes are: a circle of `explorer_sense_m` that takes a
/// picture every fire step covers a band a little narrower than its diameter, and a lane
/// sits on cell centres (one cell less).
fn lane_cells(grid: Grid, cfg: &AllocConfig, sim: &SimConfig) -> u32 {
    let gap = cfg.explorer_speed_ms * sim.dt_s * sim.fire_every as f32; // flown between two pictures
    let band = 2.0
        * (cfg.explorer_sense_m.powi(2) - (gap / 2.0).powi(2))
            .max(0.0)
            .sqrt();
    ((band / grid.cell_m - 1.0).floor() as u32).max(1)
}

/// The search pattern of every explorer. The map is cut into one stripe per explorer along
/// its longer side; each stripe is swept back and forth along its own longer side, starting
/// in the corner nearest the explorer's station. Lanes are as far apart as the sensor allows:
/// a circle of `explorer_sense_m` that takes a picture every fire step covers a band a little
/// narrower than its diameter, and a lane sits on cell centres (one cell less).
/// ponytail: stripes are dealt out by drone index, not to the nearest station, and lanes cross
/// ground that cannot burn. Assign by distance and skip such lanes for maps with several
/// stations or large lakes.
fn sweeps(grid: Grid, stations: &[[u32; 2]], cfg: &AllocConfig, sim: &SimConfig) -> Vec<Vec<u32>> {
    let per = cfg.explorers_per_station as usize;
    let k = (stations.len() * per) as u32;
    let spacing = lane_cells(grid, cfg, sim);
    let stripes_along_x = grid.nx >= grid.ny;
    let (long, short) = if stripes_along_x {
        (grid.nx, grid.ny)
    } else {
        (grid.ny, grid.nx)
    };
    (0..k)
        .map(|e| {
            let (lo, wide) = stripe(long, k, e);
            // Lanes run along the stripe's longer side (`along`) and are spaced `across` it.
            let lanes_cross_stripe = short >= wide;
            let across_is_x = lanes_cross_stripe == stripes_along_x;
            let ((a0, width), (b0, length)) = if lanes_cross_stripe {
                ((lo, wide), (0, short))
            } else {
                ((0, short), (lo, wide))
            };
            let station = stations[e as usize / per];
            let (sa, sb) = if across_is_x {
                (station[0], station[1])
            } else {
                (station[1], station[0])
            };
            let (flip_across, flip_along) = (sa >= a0 + width / 2, sb >= b0 + length / 2);
            let lanes = width.div_ceil(spacing);
            let (near, far) = if flip_along {
                (b0 + length - 1, b0)
            } else {
                (b0, b0 + length - 1)
            };
            let mut cells = Vec::with_capacity(2 * lanes as usize);
            for t in 0..lanes {
                let lane = if flip_across { lanes - 1 - t } else { t };
                let a = a0 + (2 * lane + 1) * width / (2 * lanes);
                let ends = if t % 2 == 0 { [near, far] } else { [far, near] };
                for b in ends {
                    cells.push(if across_is_x {
                        grid.idx(a, b)
                    } else {
                        grid.idx(b, a)
                    } as u32);
                }
            }
            cells
        })
        .collect()
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

impl AllocEnv {
    fn new(
        scenario: Arc<Scenario>,
        terrain: Arc<Terrain>,
        sim: SimConfig,
        cfg: AllocConfig,
        base_seed: u64,
        index: u64,
    ) -> Self {
        let weather = Weather {
            moisture: scenario.moisture,
            wind: scenario.wind.clone(),
        };
        let n = fleet_size(scenario.stations.len(), &cfg);
        let attackers = scenario.stations.len() * cfg.drones_per_station as usize;
        Self {
            sweeps: Arc::new(sweeps(terrain.grid, &scenario.stations, &cfg, &sim)),
            leg: vec![0; n],
            visited: vec![false; n],
            tracker: vec![false; n],
            reports: Vec::new(),
            search: Vec::new(),
            forecast: None,
            world: World::new(terrain.clone(), weather, sim, 0),
            pad_free: vec![0; scenario.stations.len() * cfg.pads as usize],
            scenario,
            terrain,
            cfg,
            sim,
            base_seed,
            index,
            episode: 0,
            seed: 0,
            fingerprint: 0,
            task: vec![Task::None; n],
            decide: (0..n).map(|i| i < attackers).collect(),
            goal: vec![[0.0; 2]; n],
            ready_at: vec![0; n],
            at: vec![0; n],
            launched: vec![0; n],
            fires: Vec::new(),
            drops: 0,
            water_l: 0.0,
            min_separation: f32::INFINITY,
            first_drop: None,
            queue_ticks: 0,
            max_sortie: 0,
            needs_reset: true,
            start_tick: 0,
            start_cells: 0,
            start_kw: 0.0,
            belief: None,
        }
    }

    fn cell_centre(&self, cell: [u32; 2]) -> [f32; 2] {
        let m = self.terrain.grid.cell_m;
        [(cell[0] as f32 + 0.5) * m, (cell[1] as f32 + 0.5) * m]
    }

    fn station_point(&self, station: usize) -> [f32; 2] {
        self.cell_centre(self.scenario.stations[station])
    }

    /// Blocks per row of the map, and the block side in cells.
    fn blocks(&self) -> (u32, u32) {
        let block = self.cfg.block_cells.max(1);
        (self.terrain.grid.nx.div_ceil(block), block)
    }

    fn block_of(&self, cell: usize) -> u32 {
        let (per_row, block) = self.blocks();
        let (x, y) = self.terrain.grid.xy(cell);
        (y / block) * per_row + x / block
    }

    /// The cells of a block (clipped at the map's edge): x range, y range.
    fn block_cells(&self, block: u32) -> (std::ops::Range<u32>, std::ops::Range<u32>) {
        let (per_row, side) = self.blocks();
        let g = self.terrain.grid;
        let (x, y) = ((block % per_row) * side, (block / per_row) * side);
        (x..(x + side).min(g.nx), y..(y + side).min(g.ny))
    }

    /// The centre of a block, metres.
    fn block_point(&self, block: u32) -> [f32; 2] {
        let (xs, ys) = self.block_cells(block);
        let m = self.terrain.grid.cell_m;
        [
            (xs.start + xs.end) as f32 * 0.5 * m,
            (ys.start + ys.end) as f32 * 0.5 * m,
        ]
    }

    /// What the planners take to be the fire: the drones' belief with `plan_on_belief`,
    /// otherwise None, which stands for the fire itself.
    fn known(&self) -> Option<&Belief> {
        self.belief.as_ref().filter(|_| self.cfg.plan_on_belief)
    }

    /// The front of a cell as far as the planners know (`fire::FireState::front`).
    fn front(&self, cell: usize) -> Option<Front> {
        match self.known() {
            Some(belief) => belief.front(cell),
            None => self.world.fire.front(cell),
        }
    }

    /// The water load (L/m²) that stops the front of a burning cell and keeps it stopped for
    /// `ember_s` although the water evaporates meanwhile; None if the cell has no front, or
    /// one that water cannot stop.
    fn hold_load(&self, cell: usize) -> Option<f32> {
        let front = self.front(cell)?;
        if !front.water_l_m2.is_finite() {
            return None;
        }
        let margin = libm::expf(self.sim.ember_s / self.sim.water_tau_s);
        Some(self.world.water.crit(front.water_l_m2) * margin)
    }

    /// The front cell of a block that is furthest below the load that stops it (the lowest
    /// cell index among equals): where the next drop goes. None if the block has no front
    /// that water can stop.
    fn neediest(&self, block: u32) -> Option<usize> {
        let (xs, ys) = self.block_cells(block);
        let mut best: Option<(f32, usize)> = None;
        for y in ys {
            for x in xs.clone() {
                let cell = self.terrain.grid.idx(x, y);
                if let Some(hold) = self.hold_load(cell) {
                    let short = hold - self.world.water.load[cell];
                    if best.is_none_or(|(b, _)| short > b) {
                        best = Some((short, cell));
                    }
                }
            }
        }
        best.map(|(_, cell)| cell)
    }

    fn task_point(&self, task: Task) -> Option<[f32; 2]> {
        match task {
            Task::None => None,
            Task::Fire(block) => Some(self.block_point(block)),
            Task::Point(cell) | Task::Explore(cell) | Task::Follow(cell) => {
                let (x, y) = self.terrain.grid.xy(cell as usize);
                Some(self.cell_centre([x, y]))
            }
            Task::Station(i) => Some(self.station_point(i as usize)),
        }
    }

    fn set_task(&mut self, i: usize, task: Task) {
        self.task[i] = task;
        self.goal[i] = self.task_point(task).unwrap_or(self.world.drones.pos[i]);
    }

    /// One environment on its own (the viewer uses this), started on the episode of `seed`.
    /// Fails like `AllocVec::new` on a malformed scenario.
    pub fn single(
        scenario: Scenario,
        sim: SimConfig,
        cfg: AllocConfig,
        seed: u64,
    ) -> Result<Self, String> {
        let mut batch = AllocVec::new(1, scenario, sim, cfg, seed, 1)?;
        let mut env = batch.envs.remove(0);
        env.reset(seed);
        Ok(env)
    }

    pub fn config(&self) -> &AllocConfig {
        &self.cfg
    }

    /// Smallest distance between two drones in the air so far in this episode, metres.
    /// Infinite where it was not watched (`safe_m` 0 under the scripted controller).
    pub fn min_separation(&self) -> f32 {
        self.min_separation
    }

    /// Identifies the fire this episode started with (burning cells and wind): the same
    /// number for every planner on the same seed, scenario and settings.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// What the drones have seen of the fire so far; None without a sensor.
    pub fn belief(&self) -> Option<&Belief> {
        self.belief.as_ref()
    }

    /// The radius of drone `i`'s fire sensor, metres; 0 for none.
    pub fn sense_m(&self, i: usize) -> f32 {
        if self.explorer(i) {
            self.cfg.explorer_sense_m
        } else {
            self.cfg.sense_m
        }
    }

    /// The top speed of drone `i`, m/s.
    fn speed(&self, i: usize) -> f32 {
        if self.explorer(i) {
            self.cfg.explorer_speed_ms
        } else {
            self.cfg.max_speed_ms
        }
    }

    /// Where fires were reported when the episode started, metres (`report_m`).
    pub fn reports(&self) -> &[[f32; 2]] {
        &self.reports
    }

    /// The attack drones: the first this many of every per-drone list.
    pub fn attackers(&self) -> usize {
        self.scenario.stations.len() * self.cfg.drones_per_station as usize
    }

    /// Is drone `i` an explorer?
    pub fn explorer(&self, i: usize) -> bool {
        i >= self.attackers()
    }

    /// The station a drone is based at.
    fn base(&self, i: usize) -> usize {
        match i.checked_sub(self.attackers()) {
            Some(e) => e / self.cfg.explorers_per_station as usize,
            None => i / self.cfg.drones_per_station as usize,
        }
    }

    /// The cell an explorer flies to after reaching `leg` waypoints: its sweep, there and
    /// back again, for ever.
    fn waypoint(&self, i: usize, leg: u32) -> u32 {
        let e = i - self.attackers();
        // First the search around the fire reported to it, if any, then the sweep.
        let search = self.search.get(e).map_or(&[][..], |s| s);
        if let Some(&cell) = search.get(leg as usize) {
            return cell;
        }
        let (cells, leg) = (&self.sweeps[e], leg - search.len() as u32);
        let round = 2 * (cells.len() - 1); // a sweep has at least one lane: two ends
        let at = leg as usize % round;
        cells[at.min(round - at)]
    }

    /// From a point to the nearest station, metres.
    fn home_m(&self, point: [f32; 2]) -> f32 {
        (0..self.scenario.stations.len())
            .map(|s| dist(point, self.station_point(s)))
            .fold(f32::INFINITY, f32::min)
    }

    /// Can explorer `i` fly to `cell` and on to a station within its endurance?
    fn in_reach(&self, i: usize, cell: u32) -> bool {
        let (x, y) = self.terrain.grid.xy(cell as usize);
        let point = self.cell_centre([x, y]);
        let airborne_s = (self.world.tick - self.launched[i]) as f32 * self.sim.dt_s;
        let flight_s = (dist(self.world.drones.pos[i], point) + self.home_m(point)) / self.speed(i);
        airborne_s + flight_s <= self.cfg.endurance_s
    }

    /// The believed-burning cell in explorer `i`'s stripe that is most worth a visit: the
    /// highest age of its last look per second it takes to get a new one (the flight and
    /// one picture), the lowest cell index among equals. None if the stripe holds none
    /// beyond what the explorer sees from where it is. Each explorer follows the fire in its
    /// own stripe, so none has to know where another one is heading.
    /// ponytail: one visit between two sweep legs, whatever the fire does. Weigh the age with
    /// the spread rate, or let an explorer stay with a fire, when the belief is what a
    /// planner works from.
    fn stalest(&self, i: usize) -> Option<u32> {
        let (belief, g) = (self.belief.as_ref()?, self.terrain.grid);
        let (e, k) = ((i - self.attackers()) as u32, self.sweeps.len() as u32);
        let (xs, ys) = if g.nx >= g.ny {
            (stripe(g.nx, k, e), (0, g.ny))
        } else {
            ((0, g.nx), stripe(g.ny, k, e))
        };
        let (p, now, dt) = (
            self.world.drones.pos[i],
            self.world.tick as u32,
            self.sim.dt_s,
        );
        let picture_s = self.sim.fire_every as f32 * dt;
        let mut best: Option<(f32, u32)> = None;
        for y in ys.0..ys.0 + ys.1 {
            for x in xs.0..xs.0 + xs.1 {
                let cell = g.idx(x, y);
                let away = dist(p, self.cell_centre([x, y]));
                if belief.state[cell] != BURNING || away <= self.cfg.explorer_sense_m {
                    continue;
                }
                let age_s = (now - belief.seen[cell]) as f32 * dt;
                let worth = age_s / (away / self.cfg.explorer_speed_ms + picture_s);
                if best.is_none_or(|(b, _)| worth > b) {
                    best = Some((worth, cell as u32));
                }
            }
        }
        best.map(|(_, cell)| cell)
    }

    /// Sends explorer `i` on to the next cell of the fire it tracks (`explorers_track`) and
    /// says whether it did. An explorer becomes a tracker when fire is known and fewer than
    /// `explorers_track` are with it, and stops being one when there is nothing left to look
    /// at within its endurance. The next cell is a believed front cell (any believed-burning
    /// cell while none has a front): the highest age of its last look per second it takes to
    /// get a new one, the first in the belief's list among equals. Cells in view, and cells
    /// another tracker is about to see, are left out, so trackers spread along the front.
    /// ponytail: greedy, one cell ahead, and the age of a look counts the same at the head
    /// as at the back of the fire. Weigh it with the spread rate, or plan a tour, when a
    /// learned explorer is to be beaten by more than this.
    fn track(&mut self, i: usize) -> bool {
        let Some(belief) = &self.belief else {
            return false;
        };
        let explorers = self.attackers()..self.task.len();
        let others = explorers.clone().filter(|&j| j != i && self.tracker[j]);
        if !self.tracker[i] && others.clone().count() >= self.cfg.explorers_track as usize {
            return false;
        }
        let taken: Vec<[f32; 2]> = others
            .filter(|&j| matches!(self.task[j], Task::Follow(_)))
            .map(|j| self.goal[j])
            .collect();
        let (g, p, r) = (
            self.terrain.grid,
            self.world.drones.pos[i],
            self.cfg.explorer_sense_m,
        );
        let (now, dt) = (self.world.tick as u32, self.sim.dt_s);
        let picture_s = self.sim.fire_every as f32 * dt;
        let fronts = belief
            .burning()
            .iter()
            .any(|&c| belief.front(c as usize).is_some());
        let mut best: Option<(f32, u32)> = None;
        for &cell in belief.burning() {
            let (x, y) = g.xy(cell as usize);
            let point = self.cell_centre([x, y]);
            let away = dist(p, point);
            if (fronts && belief.front(cell as usize).is_none())
                || away <= r
                || taken.iter().any(|&t| dist(t, point) <= r)
            {
                continue;
            }
            // A cell the forecast set alight has never been looked at: as old as can be.
            let seen = belief.seen[cell as usize];
            let age_s = now.saturating_sub(if seen == NEVER { 0 } else { seen }) as f32 * dt;
            let worth = age_s / (away / self.cfg.explorer_speed_ms + picture_s);
            if best.is_none_or(|(b, _)| worth > b) && self.in_reach(i, cell) {
                best = Some((worth, cell));
            }
        }
        self.tracker[i] = best.is_some();
        if let Some((_, cell)) = best {
            self.set_task(i, Task::Follow(cell));
        }
        best.is_some()
    }

    /// An explorer has arrived at a waypoint of its sweep. Unless it comes back from one, it
    /// first visits the fire known in its stripe (`explorers_follow`). Otherwise on to the
    /// next waypoint, or home when the way there and on to a station would take it past its
    /// endurance. From a visit or a service it returns to the waypoint it left, so every lane
    /// is flown from end to end.
    /// ponytail: a lane that cannot be flown from and to a station within the endurance is
    /// tried again and again. Cut such lanes in two when maps get that large.
    fn explore(&mut self, i: usize) {
        if self.track(i) {
            return;
        }
        let visit = (self.cfg.explorers_follow && !self.visited[i])
            .then(|| self.stalest(i))
            .flatten()
            .filter(|&cell| self.in_reach(i, cell));
        self.visited[i] = visit.is_some();
        let next = self.waypoint(i, self.leg[i] + 1);
        if let Some(cell) = visit {
            self.set_task(i, Task::Follow(cell));
        } else if self.in_reach(i, next) {
            self.leg[i] += 1;
            self.set_task(i, Task::Explore(next));
        } else {
            self.fly_home(i);
        }
    }

    /// An explorer has looked at the fire: back to the waypoint where it left its sweep.
    fn resume(&mut self, i: usize) {
        if self.track(i) {
            return;
        }
        let back = self.waypoint(i, self.leg[i]);
        if self.in_reach(i, back) {
            self.set_task(i, Task::Explore(back));
        } else {
            self.fly_home(i);
        }
    }

    fn fly_home(&mut self, i: usize) {
        let home = self.nearest_station(i);
        self.set_task(i, Task::Station(home));
    }

    /// Drops made so far in this episode.
    pub fn drops(&self) -> u32 {
        self.drops
    }

    /// Is the drone at a station (in service or ready)?
    pub fn docked(&self, i: usize) -> bool {
        self.ready_at[i] != AIRBORNE
    }

    /// Is the drone at a station, serviced and free to be sent?
    pub fn ready(&self, i: usize) -> bool {
        self.ready_at[i] <= self.world.tick
    }

    /// Per station: attack drones ready to be sent, attack drones docked.
    pub fn fleet(&self) -> Vec<(u32, u32)> {
        let mut out = vec![(0, 0); self.scenario.stations.len()];
        for i in (0..self.attackers()).filter(|&i| self.docked(i)) {
            let s = &mut out[self.at[i] as usize];
            *s = (s.0 + self.ready(i) as u32, s.1 + 1);
        }
        out
    }

    /// Starts the episode that belongs to `seed`.
    pub fn reset(&mut self, seed: u64) {
        let (grid, s) = (self.terrain.grid, &self.scenario);
        let mut wind = s.wind.clone();
        wind.iter_mut()
            .for_each(|w| w.speed_ms *= self.cfg.wind_scale);
        if self.cfg.random_wind {
            // The whole schedule turns by one angle, so the scenario's wind shifts survive.
            let angle = std::f32::consts::TAU * rng::uniform(seed, Stream::Weather, 0, 0);
            let (cos, sin) = (angle.cos(), angle.sin());
            wind.iter_mut().for_each(|w| {
                w.to = [w.to[0] * cos - w.to[1] * sin, w.to[0] * sin + w.to[1] * cos]
            });
        }
        let weather = Weather {
            moisture: s.moisture,
            wind,
        };
        self.world = World::new(self.terrain.clone(), weather, self.sim, seed);
        if self.cfg.random_ignitions > 0 {
            // ponytail: rejection sampling; a map that is almost all rock gets fewer fires.
            let (mut lit, mut draw) = (0, 0);
            while lit < self.cfg.random_ignitions && draw < 64 * self.cfg.random_ignitions as u64 {
                let x = (rng::uniform(seed, Stream::Ignition, draw, 0) * grid.nx as f32) as u32;
                let y = (rng::uniform(seed, Stream::Ignition, draw, 1) * grid.ny as f32) as u32;
                lit += self.world.ignite(x.min(grid.nx - 1), y.min(grid.ny - 1)) as u32;
                draw += 1;
            }
        } else {
            for c in &s.ignitions {
                self.world.ignite(c[0], c[1]);
            }
        }
        // Reports: each ignition, off by up to `report_m` in a random direction.
        let (w, h) = (grid.nx as f32 * grid.cell_m, grid.ny as f32 * grid.cell_m);
        let told = self.cfg.report_m > 0.0;
        self.reports = (self.world.fire.burning().filter(|_| told).enumerate())
            .map(|(k, cell)| {
                let (x, y) = grid.xy(cell as usize);
                let p = self.cell_centre([x, y]);
                let off =
                    self.cfg.report_m * rng::uniform(seed, Stream::Report, k as u64, 0).sqrt();
                let turn = std::f32::consts::TAU * rng::uniform(seed, Stream::Report, k as u64, 1);
                [
                    (p[0] + off * turn.cos()).clamp(0.0, w - 1.0),
                    (p[1] + off * turn.sin()).clamp(0.0, h - 1.0),
                ]
            })
            .collect();
        // Explorer k searches around report k: the report itself, then squares that widen by
        // one lane until they cover the possible error. Corners off the map are pulled in.
        let lane = lane_cells(grid, &self.cfg, &self.sim) as i64;
        let reach = self.cfg.report_m - self.cfg.explorer_sense_m;
        let rings = (reach / (lane as f32 * grid.cell_m)).ceil().max(0.0) as i64;
        let explorers = self.task.len() - self.attackers();
        self.search = (self.reports.iter().take(explorers))
            .map(|r| {
                let (cx, cy) = ((r[0] / grid.cell_m) as i64, (r[1] / grid.cell_m) as i64);
                let cell = |dx: i64, dy: i64| {
                    let x = (cx + dx).clamp(0, grid.nx as i64 - 1) as u32;
                    let y = (cy + dy).clamp(0, grid.ny as i64 - 1) as u32;
                    grid.idx(x, y) as u32
                };
                let corners = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)];
                let squares = (1..=rings)
                    .flat_map(|k| corners.map(|(dx, dy)| cell(dx * k * lane, dy * k * lane)));
                std::iter::once(cell(0, 0)).chain(squares).collect()
            })
            .collect();

        // Pre-burn: the fire alone, exactly as an idle episode would have it at that time.
        // There are no drones and no water in the world yet.
        let end = (self.cfg.preburn_s / self.sim.dt_s).round() as u64;
        let cap = match self.cfg.preburn_cells {
            0 => u32::MAX,
            cells => cells,
        };
        while self.world.tick < end && self.world.fire.n_ignited < cap && !self.world.fire.is_out()
        {
            self.world.tick();
        }
        (self.start_tick, self.start_cells) = (self.world.tick, self.world.fire.n_ignited);
        self.start_kw = self.world.fire.stats.peak_kw;

        // Every drone starts docked at its station, serviced and ready. Explorers carry no
        // water and are never asked for a decision.
        let (n, attackers) = (self.task.len(), self.attackers());
        self.world.drones.pos = (0..n).map(|i| self.station_point(self.base(i))).collect();
        self.world.drones.vel = vec![[0.0; 2]; n];
        self.world.drones.water = (0..n)
            .map(|i| if i < attackers { self.cfg.tank_l } else { 0.0 })
            .collect();
        self.goal.clone_from(&self.world.drones.pos);
        self.at = (0..n).map(|i| self.base(i) as u32).collect();
        self.leg.fill(0);
        self.visited.fill(false);
        self.tracker.fill(false);
        self.ready_at.fill(self.start_tick);
        self.launched.fill(self.start_tick);
        self.pad_free.fill(self.start_tick);

        // Identifies the fire an episode started with: same value, same scenario.
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut mix = |v: u64| h = (h ^ v).wrapping_mul(0x0000_0100_0000_01b3);
        let mut cells: Vec<u32> = self.world.fire.burning().collect();
        cells.sort_unstable();
        cells.iter().for_each(|&c| mix(c as u64));
        let w = self.world.wind();
        [w.speed_ms, w.to[0], w.to[1]]
            .iter()
            .for_each(|v| mix(v.to_bits() as u64));
        // Later wind changes belong to the fire as well. A steady wind adds nothing here, so
        // its fingerprints are what they were before schedules were hashed.
        for w in &self.world.weather.wind[1..] {
            [w.t_s, w.speed_ms, w.to[0], w.to[1]]
                .iter()
                .for_each(|v| mix(v.to_bits() as u64));
        }
        self.fingerprint = h >> 33; // 31 bits: exact in a CSV float column

        self.seed = seed;
        self.task.fill(Task::None);
        (0..n).for_each(|i| self.decide[i] = i < attackers);
        (self.drops, self.water_l, self.min_separation) = (0, 0.0, f32::INFINITY);
        (self.first_drop, self.queue_ticks, self.max_sortie) = (None, 0, 0);
        self.needs_reset = false;
        let sensing = self.cfg.sense_m > 0.0 || self.cfg.explorers_per_station > 0;
        self.belief = sensing.then(|| Belief::new(grid.len()));
        // The forecast starts knowing nothing, at the same time on the same clock.
        self.forecast = (sensing && self.cfg.forecast).then(|| {
            let cfg = SimConfig {
                ros_noise: 0.0,
                ..self.sim
            };
            let weather = (*self.world.weather).clone();
            let mut world = World::new(self.terrain.clone(), weather, cfg, seed);
            world.tick = self.world.tick;
            world
        });
        self.fires = self.fire_targets();
    }

    /// The blocks with a fire front, those that need the most water first: what it takes to
    /// stop each one now and what is already on its way.
    fn fire_targets(&self) -> Vec<Target> {
        let area = self.terrain.grid.cell_m.powi(2);
        let per_drop = self.cfg.drop_efficiency * self.cfg.drop_l / area; // L/m² on one cell
        // Per block: front cells, drops needed, highest intensity, metres to the nearest
        // station, and the share of the water lying there now that is left when a drone sent
        // from that station arrives. Without it a front that needs many drops is never
        // stopped: what is ordered for the load missing now has evaporated on arrival.
        let mut blocks: BTreeMap<u32, (u32, f32, f32, f32, f32)> = BTreeMap::new();
        let burning: Box<dyn Iterator<Item = u32>> = match self.known() {
            Some(belief) => Box::new(belief.burning().iter().copied()),
            None => self.world.fire.burning(),
        };
        for cell in burning {
            let cell = cell as usize;
            let Some(front) = self.front(cell) else {
                continue; // only burning out: water changes nothing there
            };
            let block = self.block_of(cell);
            let b = blocks.entry(block).or_insert_with(|| {
                let home_m = self.home_m(self.block_point(block));
                let flight_s = home_m / self.cfg.max_speed_ms;
                let left = libm::expf(-flight_s / self.sim.water_tau_s);
                (0, 0.0, 0.0, home_m, left)
            });
            // A front too intense for water is shown (cells, intensity) but needs no drop.
            let short = self.hold_load(cell).map_or(0.0, |hold| {
                (hold - self.world.water.load[cell] * b.4).max(0.0)
            });
            *b = (
                b.0 + 1,
                b.1 + short / per_drop,
                b.2.max(front.kw_m),
                b.3,
                b.4,
            );
        }
        let mut inbound: BTreeMap<u32, f32> = BTreeMap::new();
        for (i, task) in self.task.iter().enumerate() {
            if let Task::Fire(block) = task {
                let drops = (self.world.drones.water[i] / self.cfg.drop_l)
                    .floor()
                    .min(1.0);
                *inbound.entry(*block).or_default() += drops;
            }
        }
        let mut out: Vec<Target> = blocks
            .iter()
            .map(|(&block, &(fronts, need, kw, home_m, _))| Target {
                block,
                point: self.block_point(block),
                fronts,
                need,
                inbound: inbound.get(&block).copied().unwrap_or(0.0),
                kw,
                home_m,
            })
            .collect();
        out.sort_by(|a, b| b.need.total_cmp(&a.need).then(a.block.cmp(&b.block)));
        out.truncate(self.cfg.max_targets);
        out
    }

    /// May drone `i` be sent to this target: water for a drop, and enough time in the air
    /// left to get there and on to the nearest station.
    fn can_take(&self, i: usize, target: &Target) -> bool {
        let airborne_s = if self.docked(i) {
            0.0
        } else {
            (self.world.tick - self.launched[i]) as f32 * self.sim.dt_s
        };
        let flight_s =
            (dist(self.world.drones.pos[i], target.point) + target.home_m) / self.cfg.max_speed_ms;
        self.world.drones.water[i] >= self.cfg.drop_l
            && airborne_s + flight_s <= self.cfg.endurance_s
    }

    fn nearest_station(&self, i: usize) -> u32 {
        let p = self.world.drones.pos[i];
        (0..self.scenario.stations.len())
            .min_by(|&a, &b| {
                dist(p, self.station_point(a)).total_cmp(&dist(p, self.station_point(b)))
            })
            .expect("a scenario with drones has a station") as u32
    }

    /// Sets the velocities of one tick of the scripted controller: every drone in the air
    /// flies straight at its goal. With `safe_m` above zero it never gets closer than that
    /// to another drone in the air (`navigate::scripted_step`).
    fn fly(&mut self) {
        let (dt, safe) = (self.sim.dt_s, self.cfg.safe_m);
        let n = self.task.len();
        if safe > 0.0 {
            let g = self.terrain.grid;
            let size = (g.nx as f32 * g.cell_m, g.ny as f32 * g.cell_m);
            let pos = &self.world.drones.pos;
            let next = crate::navigate::scripted_step(
                pos,
                |i| self.goal[i],
                |i| self.speed(i) * dt,
                safe,
                size,
                |i| !self.docked(i),
            );
            for (i, q) in next.iter().enumerate() {
                let p = self.world.drones.pos[i];
                self.world.drones.vel[i] = [(q[0] - p[0]) / dt, (q[1] - p[1]) / dt];
            }
            return;
        }
        for i in 0..n {
            let (p, goal) = (self.world.drones.pos[i], self.goal[i]);
            let (dx, dy) = (goal[0] - p[0], goal[1] - p[1]);
            let range = dx.hypot(dy);
            self.world.drones.vel[i] = if range > 0.0 && !self.docked(i) {
                let speed = (self.speed(i) * dt).min(range) / dt / range;
                [dx * speed, dy * speed]
            } else {
                [0.0; 2]
            };
        }
    }

    fn arrived(&self, i: usize) -> bool {
        let radius = match self.task[i] {
            Task::Station(_) => self.cfg.station_m,
            _ => self.cfg.arrive_m,
        };
        dist(self.world.drones.pos[i], self.goal[i]) <= radius
    }

    /// `actions`: [drones], a slot of the target table or -1. Only drones whose `decide`
    /// flag was set are listened to.
    pub fn step(&mut self, actions: &[i32], autoreset: bool, out: Out<'_>) {
        out.terms.fill(0.0);
        (*out.terminated, *out.truncated, *out.elapsed_s) = (0, 0, 0.0);
        if self.needs_reset {
            if autoreset {
                let seed = rng::derive(self.base_seed, self.index, self.episode);
                self.episode += 1;
                self.reset(seed);
                self.observe(out);
            }
            return; // without autoreset a finished environment stays frozen
        }

        let ignited_before = self.world.fire.n_ignited;
        let (ticks, fire_out, timeout) = self.advance(actions, out.terms);

        let n = self.task.len();
        let burnable = self.terrain.burnable_cells.max(1) as f32;
        let burned = self.world.fire.n_ignited.saturating_sub(ignited_before) as f32 / burnable;
        for i in 0..n {
            out.terms[i * TERMS.len() + 2] = burned;
            out.terms[i * TERMS.len() + 3] = fire_out as u8 as f32;
            out.terms[i * TERMS.len() + 4] = timeout as u8 as f32;
        }
        *out.elapsed_s = ticks as f32 * self.sim.dt_s;
        *out.terminated = fire_out as u8;
        *out.truncated = timeout as u8;
        if fire_out || timeout {
            out.info.copy_from_slice(&self.summary());
        }
        self.observe(out);
    }

    /// Hands out `actions` and runs to the next decision event or the end of the episode.
    /// Returns the ticks that took and (fire out, time limit reached).
    fn advance(&mut self, actions: &[i32], terms: &mut [f32]) -> (u32, bool, bool) {
        self.apply(actions);
        let mut ticks = 0u32;
        loop {
            let ended = self.tick(terms);
            ticks += 1;
            if ended.0 || ended.1 || self.pending() {
                return (ticks, ended.0, ended.1);
            }
        }
    }

    /// Gives the deciding drones their new tasks. `actions`: [drones], a slot of the target
    /// table or -1; drones that were not asked are not listened to. -1 keeps a docked drone
    /// where it is and sends a drone in the air home.
    pub fn apply(&mut self, actions: &[i32]) {
        let stations = self.scenario.stations.len();
        // The index addresses actions, flags, tasks and tanks alike.
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.task.len() {
            if !self.decide[i] {
                continue;
            }
            let docked = self.docked(i);
            let slot = actions[i];
            if slot >= 0 && (slot as usize) < stations {
                if !docked {
                    self.set_task(i, Task::Station(slot as u32));
                }
            } else if let Some(target) = (slot >= 0)
                .then(|| self.fires.get(slot as usize - stations).copied())
                .flatten()
                // A drone that cannot reach the fire and get home is not sent.
                && self.can_take(i, &target)
            {
                self.set_task(i, Task::Fire(target.block));
            }
            // In the air with nothing to do: home. (A hand-sent point is kept.)
            if !docked && self.task[i] == Task::None {
                let home = self.nearest_station(i);
                self.set_task(i, Task::Station(home));
            }
        }
        self.decide.fill(false);
    }

    /// Does any drone need a decision?
    pub fn pending(&self) -> bool {
        self.decide.iter().any(|&d| d)
    }

    /// Advances the simulation by one tick: drones fly, drop and are serviced. Adds this
    /// tick's `drop` and `refill` amounts to `terms` ([drones, TERMS]) and returns (fire out,
    /// time limit reached). When either is true the episode is over.
    pub fn tick(&mut self, terms: &mut [f32]) -> (bool, bool) {
        self.tick_flown(None, terms)
    }

    /// Where a pilot should steer each drone: its task's place, or where it is when it has
    /// nothing to do or is docked.
    pub fn flight_goals(&self) -> &[[f32; 2]] {
        &self.goal
    }

    /// What a pilot sees (`navigate::observe_flight`): docked drones neither see nor are seen.
    pub fn observe_pilot(
        &self,
        view_m: f32,
        agent: &mut [f32],
        neighbors: &mut [f32],
        mask: &mut [u8],
    ) {
        let g = self.terrain.grid;
        let d = &self.world.drones;
        crate::navigate::observe_flight(
            &d.pos,
            &d.vel,
            &self.goal,
            (g.nx as f32 * g.cell_m, g.ny as f32 * g.cell_m),
            view_m,
            self.cfg.max_speed_ms,
            agent,
            neighbors,
            mask,
            |i| !self.docked(i),
        );
    }

    /// Takes a docked drone into the air, unless `safe_m` is kept and another drone in the
    /// air is closer than that.
    /// ponytail: a drone hovering over a station blocks its launches; none does under the
    /// built-in planners. Give stations a holding pattern if that changes.
    fn launch(&mut self, i: usize) {
        let (p, safe) = (self.world.drones.pos[i], self.cfg.safe_m);
        let pos = &self.world.drones.pos;
        let clear = safe <= 0.0
            || (0..pos.len()).all(|j| j == i || self.docked(j) || dist(p, pos[j]) >= safe);
        if clear {
            (self.ready_at[i], self.launched[i]) = (AIRBORNE, self.world.tick);
        }
    }

    /// Docks drone `i` at `station`: it takes the pad that is free first and is ready after
    /// the service. Drones docking in one tick are handled in index order, so pads are given
    /// first come, first served.
    fn dock(&mut self, i: usize, station: u32, terms: &mut [f32]) {
        let (now, dt) = (self.world.tick, self.sim.dt_s);
        let pads = self.cfg.pads as usize;
        let free = &mut self.pad_free[station as usize * pads..][..pads];
        let pad = (0..pads)
            .min_by_key(|&p| free[p])
            .expect("a station has a pad");
        let start = free[pad].max(now);
        let ready = start + (self.cfg.service_s / dt).ceil() as u64;
        free[pad] = ready;
        self.queue_ticks += start - now;
        self.max_sortie = self.max_sortie.max(now - self.launched[i]);
        if !self.explorer(i) {
            let water = &mut self.world.drones.water[i];
            terms[i * TERMS.len() + 1] += (self.cfg.tank_l - *water) / self.cfg.tank_l;
            *water = self.cfg.tank_l;
        }
        (self.ready_at[i], self.at[i]) = (ready, station);
        self.world.drones.vel[i] = [0.0; 2];
        self.set_task(i, Task::None);
    }

    /// Like `tick`, but with `velocity` ([drones, 2], fractions of the maximum speed) the
    /// drones in the air are flown by an outside pilot instead of the scripted controller.
    /// Nothing then keeps them apart, on the map or within their endurance:
    /// `min_separation` and `max_sortie_s` tell how that went.
    pub fn tick_flown(&mut self, velocity: Option<&[f32]>, terms: &mut [f32]) -> (bool, bool) {
        let n = self.task.len();
        let term = |i: usize, which: usize| i * TERMS.len() + which;
        let (dt, cell_area) = (self.sim.dt_s, self.terrain.grid.cell_m.powi(2));
        match velocity {
            None => self.fly(),
            Some(v) => {
                for i in 0..n {
                    let (ax, ay) = (v[2 * i], v[2 * i + 1]);
                    let scale = self.speed(i) / ax.hypot(ay).max(1.0);
                    let fly = !self.docked(i) as u8 as f32;
                    self.world.drones.vel[i] = [ax * scale * fly, ay * scale * fly];
                }
            }
        }
        if let Some(forecast) = &mut self.forecast {
            // The swarm knows the water it dropped: the forecast burns under the same.
            if (self.world.tick + 1).is_multiple_of(self.sim.fire_every as u64) {
                forecast.water.clone_from(&self.world.water);
            }
            forecast.tick();
        }
        self.world.tick();
        let now = self.world.tick;
        // The fire only changes on a fire step: that is when a block can lose its front.
        let fire_stepped = now.is_multiple_of(self.sim.fire_every as u64);
        let interval = ((self.cfg.max_interval_s / dt).round() as u64).max(1);
        let turn = (now - self.start_tick).is_multiple_of(interval);
        let watch = self.cfg.safe_m > 0.0 || velocity.is_some();
        // The sensors take a picture when the fire has changed. Copying the truth gives the
        // same belief in any drone order.
        if fire_stepped && let Some(mut belief) = self.belief.take() {
            for i in (0..n).filter(|&i| !self.docked(i) && self.sense_m(i) > 0.0) {
                let (p, r) = (self.world.drones.pos[i], self.sense_m(i));
                let fire = &self.world.fire;
                belief.look(self.terrain.grid, &fire.state, |c| fire.front(c), p, r, now);
            }
            if let Some(forecast) = &mut self.forecast {
                // Every cell just looked at is put right in the forecast, which then is
                // the belief. ponytail: a pass over the map per fire step; collect the
                // cells in `look` if maps get large.
                for (cell, _) in (belief.seen.iter().enumerate()).filter(|s| *s.1 == now as u32) {
                    forecast.correct(cell, self.world.fire.state[cell]);
                }
                belief.adopt(&forecast.fire);
            }
            belief.score(self.world.fire.burning());
            self.belief = Some(belief);
        }
        for i in 0..n {
            if self.docked(i) {
                if self.ready_at[i] > now {
                    continue; // in service
                }
                if self.explorer(i) {
                    // Off on its own: to the fire it tracks, or to the waypoint it has
                    // not reached yet.
                    if !self.track(i) {
                        let next = self.waypoint(i, self.leg[i]);
                        self.set_task(i, Task::Explore(next));
                    }
                    self.launch(i);
                    continue;
                }
                match self.task[i] {
                    Task::Fire(_) | Task::Point(_) => self.launch(i),
                    // Ready: asked once, and again at every regular turn while it waits.
                    _ if self.ready_at[i] == now || turn => self.decide[i] = true,
                    _ => {}
                }
                continue;
            }
            if watch {
                for j in 0..i {
                    if !self.docked(j) {
                        let d = dist(self.world.drones.pos[i], self.world.drones.pos[j]);
                        self.min_separation = self.min_separation.min(d);
                    }
                }
            }
            match self.task[i] {
                Task::None => {}
                Task::Point(_) => {
                    if self.arrived(i) {
                        self.set_task(i, Task::None);
                        self.decide[i] = true;
                    }
                }
                Task::Fire(block) => {
                    let here = self.arrived(i);
                    if !here && !fire_stepped {
                        continue;
                    }
                    let Some(cell) = self.neediest(block) else {
                        // The block's fire is out, only burning down, or beyond water: the
                        // drone's water is free for another fire.
                        self.set_task(i, Task::None);
                        self.decide[i] = true;
                        continue;
                    };
                    if here {
                        let load = self.cfg.drop_efficiency * self.cfg.drop_l / cell_area;
                        self.world.water.add(cell, load);
                        self.world.drones.water[i] -= self.cfg.drop_l;
                        self.drops += 1;
                        self.water_l += self.cfg.drop_l;
                        self.first_drop.get_or_insert(now);
                        terms[term(i, 0)] += 1.0;
                        if self.world.drones.water[i] >= self.cfg.drop_l {
                            self.set_task(i, Task::None);
                            self.decide[i] = true;
                        } else {
                            let home = self.nearest_station(i);
                            self.set_task(i, Task::Station(home));
                        }
                    }
                }
                Task::Station(station) => {
                    if self.arrived(i) {
                        self.dock(i, station, terms);
                    }
                }
                Task::Explore(_) => {
                    if self.arrived(i) {
                        self.explore(i);
                    }
                }
                Task::Follow(_) => {
                    if self.arrived(i) {
                        self.resume(i);
                    }
                }
            }
        }
        let fire_out = self.world.fire.is_out();
        let limit = (self.cfg.time_limit_s / dt).ceil() as u64;
        let timeout = !fire_out && now - self.start_tick >= limit;
        if fire_out || timeout {
            self.needs_reset = true;
        }
        if fire_out || timeout || self.pending() {
            self.fires = self.fire_targets(); // decisions are made on the current fire
        }
        (fire_out, timeout)
    }

    /// The episode summary in the order of `INFO`.
    pub fn summary(&self) -> [f32; 20] {
        let (stats, dt) = (self.world.fire.stats, self.sim.dt_s);
        let or_nan = |x: f32| if x.is_finite() { x } else { f32::NAN };
        let sensed = |f: fn(&Belief) -> f32| self.belief.as_ref().map_or(f32::NAN, f);
        [
            self.world.fire.is_out() as u8 as f32,
            self.episode_s(),
            self.world.fire.n_ignited as f32,
            self.world.burned_fraction(),
            self.water_l,
            self.drops as f32,
            or_nan(self.min_separation),
            self.start_cells as f32,
            stats.stopped as f32,
            (stats.stopped_kw_sum / stats.stopped.max(1) as f64) as f32,
            stats.stopped_kw_max,
            stats.peak_kw,
            self.start_kw,
            self.first_drop
                .map_or(f32::NAN, |t| (t - self.start_tick) as f32 * dt),
            self.queue_ticks as f32 * dt,
            self.max_sortie as f32 * dt,
            self.belief
                .as_ref()
                .and_then(|b| b.detected())
                .map_or(f32::NAN, |t| (t - self.start_tick) as f32 * dt),
            sensed(Belief::seen_share),
            sensed(Belief::miss),
            sensed(Belief::phantom),
        ]
    }

    /// Seconds since the episode started; the pre-burn does not count.
    pub fn episode_s(&self) -> f32 {
        (self.world.tick - self.start_tick) as f32 * self.sim.dt_s
    }

    /// Is the episode over (fire out or time limit reached)?
    pub fn ended(&self) -> bool {
        self.needs_reset
    }

    /// Sends a drone by hand: home if the cell holds a station, to drop water if it burns
    /// and the drone has water, otherwise just to fly there. A docked drone takes off for it.
    /// Fails for a drone that is still being serviced.
    pub fn send(&mut self, drone: usize, cell: [u32; 2]) -> Result<(), &'static str> {
        if self.explorer(drone) {
            return Err("an explorer flies its own sweep");
        }
        if self.docked(drone) && !self.ready(drone) {
            return Err("this drone is still being serviced");
        }
        let index = self.terrain.grid.idx(cell[0], cell[1]);
        let task = match self.scenario.stations.iter().position(|s| *s == cell) {
            Some(_) if self.docked(drone) => Task::None,
            Some(station) => Task::Station(station as u32),
            None if self.world.fire.state[index] == BURNING
                && self.world.drones.water[drone] >= self.cfg.drop_l =>
            {
                Task::Fire(self.block_of(index))
            }
            None => Task::Point(index as u32),
        };
        self.set_task(drone, task);
        self.decide[drone] = false;
        Ok(())
    }

    /// What each drone is doing: task kind (`TASK_*`) and where, in metres.
    pub fn tasks(&self) -> Vec<(f32, Option<[f32; 2]>)> {
        (0..self.task.len())
            .map(|i| (self.kind(i), self.task_point(self.task[i])))
            .collect()
    }

    fn kind(&self, i: usize) -> f32 {
        match self.task[i] {
            Task::Fire(_) => TASK_FIRE,
            Task::Point(_) => TASK_POINT,
            Task::Explore(_) | Task::Follow(_) => TASK_EXPLORE,
            Task::Station(_) => TASK_STATION,
            Task::None if self.docked(i) => TASK_DOCKED,
            Task::None => TASK_NONE,
        }
    }

    /// The fire targets on offer (block centres in metres), in slot order after the stations.
    pub fn fire_points(&self) -> Vec<[f32; 2]> {
        self.fires.iter().map(|t| t.point).collect()
    }

    /// Which drones currently need a decision.
    pub fn deciding(&self) -> &[bool] {
        &self.decide
    }

    /// Writes the tables a planner sees. Everything but `terms`, the episode flags and `info`.
    pub fn observe(&self, out: Out<'_>) {
        let g = self.terrain.grid;
        let (w, h) = (g.nx as f32 * g.cell_m, g.ny as f32 * g.cell_m);
        let (n, stations) = (self.task.len(), self.scenario.stations.len());
        let slots = stations + self.cfg.max_targets;
        let width = TARGET.len();
        out.targets.fill(0.0);
        out.target_mask.fill(0);
        out.action_mask.fill(0);
        for s in 0..stations {
            let p = self.station_point(s);
            out.targets[s * width..][..3].copy_from_slice(&[p[0] / w, p[1] / h, 1.0]);
            out.target_mask[s] = 1;
        }
        let block_cells = (self.cfg.block_cells.max(1) as f32).powi(2);
        for (f, t) in self.fires.iter().enumerate() {
            out.targets[(stations + f) * width..][..width].copy_from_slice(&[
                t.point[0] / w,
                t.point[1] / h,
                0.0,
                t.fronts as f32 / block_cells,
                t.need,
                t.inbound,
                t.kw / 1000.0,
            ]);
            out.target_mask[stations + f] = 1;
        }
        for i in 0..n {
            let p = self.world.drones.pos[i];
            out.drones[i * 6..][..6].copy_from_slice(&[
                p[0] / w,
                p[1] / h,
                self.world.drones.water[i] / self.cfg.tank_l,
                self.goal[i][0] / w,
                self.goal[i][1] / h,
                self.kind(i),
            ]);
            out.decide[i] = self.decide[i] as u8;
            if !self.decide[i] {
                continue;
            }
            let row = &mut out.action_mask[i * slots..][..slots];
            row[..stations].fill(!self.docked(i) as u8);
            for (f, t) in self.fires.iter().enumerate() {
                row[stations + f] = self.can_take(i, t) as u8;
            }
        }
        let wind = self.world.wind();
        let fire = &self.world.fire;
        let (burning, burned) = self
            .known()
            .map_or((fire.n_burning() as u32, fire.n_ignited), Belief::counts);
        out.globals.copy_from_slice(&[
            wind.to[0] * wind.speed_ms / 20.0,
            wind.to[1] * wind.speed_ms / 20.0,
            self.episode_s() / self.cfg.time_limit_s,
            burning as f32 / g.len() as f32,
            burned as f32 / self.terrain.burnable_cells.max(1) as f32,
        ]);
        (*out.seed, *out.fingerprint) = (self.seed, self.fingerprint);
    }
}

/// Buffers for one environment used on its own: rollouts, and the viewer.
pub struct Scratch {
    f: Vec<f32>,
    b: Vec<u8>,
    ids: [u64; 2],
}

impl Scratch {
    pub fn new(env: &AllocEnv) -> Self {
        let (n, slots) = (
            env.task.len(),
            env.scenario.stations.len() + env.cfg.max_targets,
        );
        let floats = n * DRONE.len()
            + slots * TARGET.len()
            + GLOBAL.len()
            + n * TERMS.len()
            + 1
            + INFO.len();
        Self {
            f: vec![0.0; floats],
            b: vec![0; slots + n * slots + n + 2],
            ids: [0; 2],
        }
    }

    pub fn out(&mut self, env: &AllocEnv) -> Out<'_> {
        let (n, slots) = (
            env.task.len(),
            env.scenario.stations.len() + env.cfg.max_targets,
        );
        let (drones, f) = self.f.split_at_mut(n * DRONE.len());
        let (targets, f) = f.split_at_mut(slots * TARGET.len());
        let (globals, f) = f.split_at_mut(GLOBAL.len());
        let (terms, f) = f.split_at_mut(n * TERMS.len());
        let (elapsed, info) = f.split_at_mut(1);
        let (target_mask, b) = self.b.split_at_mut(slots);
        let (action_mask, b) = b.split_at_mut(n * slots);
        let (decide, flags) = b.split_at_mut(n);
        let (terminated, truncated) = flags.split_at_mut(1);
        let (seed, fingerprint) = self.ids.split_at_mut(1);
        Out {
            drones,
            targets,
            target_mask,
            action_mask,
            globals,
            decide,
            terms,
            elapsed_s: &mut elapsed[0],
            terminated: &mut terminated[0],
            truncated: &mut truncated[0],
            info,
            seed: &mut seed[0],
            fingerprint: &mut fingerprint[0],
        }
    }
}

/// Planners that need no learning. They live here because the rollout planner simulates
/// ahead on copies of the environment, which only the core can do fast.
impl AllocEnv {
    /// Sends every deciding drone to a block that still needs water after what is already
    /// on its way, one drop per drone: the nearest (`Greedy`), the most intense (`Head`) or
    /// the least intense (`Flank`), nearer first among equals. Drones left over wait at their
    /// station, or fly home when in the air. Returns a slot or -1 per drone.
    pub fn assign(&self, method: Method) -> Vec<i32> {
        let stations = self.scenario.stations.len();
        let mut actions = vec![-1; self.task.len()];
        let mut open: Vec<f32> = self.fires.iter().map(|t| t.need - t.inbound).collect();
        for i in (0..self.task.len()).filter(|&i| self.decide[i]) {
            let score = |f: usize| {
                let t = &self.fires[f];
                let d = dist(self.world.drones.pos[i], t.point);
                match method {
                    Method::Head => (-t.kw, d),
                    Method::Flank => (t.kw, d),
                    _ => (d, d),
                }
            };
            let best = (0..self.fires.len())
                .filter(|&f| open[f] > 0.0 && self.can_take(i, &self.fires[f]))
                .min_by(|&a, &b| {
                    let (a, b) = (score(a), score(b));
                    a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1))
                });
            if let Some(f) = best {
                open[f] -= 1.0;
                actions[i] = (stations + f) as i32;
            }
        }
        actions
    }

    /// Plays on from here with `first` now and greedy afterwards for `horizon_s`, and returns
    /// what there is to minimise: cells burned, then cells still burning.
    fn outcome(&self, first: &[i32], horizon_s: f32, terms: &mut [f32]) -> (u32, usize) {
        let mut sim = self.clone();
        // A lookahead needs no sensing. A planner that works from the belief plays on in the
        // forecast of the fire, if there is one, so that it does not see what no drone saw;
        // otherwise in the true fire.
        if let (Some(mut believed), true) = (sim.forecast.take(), sim.cfg.plan_on_belief) {
            believed.drones = std::mem::take(&mut sim.world.drones);
            believed.water.clone_from(&sim.world.water);
            sim.world = believed;
        }
        (sim.belief, sim.forecast, sim.cfg.plan_on_belief) = (None, None, false);
        let end = sim.world.time_s() + horizon_s;
        sim.advance(first, terms);
        while !sim.needs_reset && sim.world.time_s() < end {
            let actions = sim.assign(Method::Greedy);
            sim.advance(&actions, terms);
        }
        (sim.world.fire.n_ignited, sim.world.fire.n_burning())
    }

    /// Rollout planner (Bertsekas' multiagent rollout with greedy as the base policy). The
    /// drones that have to decide are taken one at a time; for each, a few alternatives to
    /// its greedy choice are simulated `horizon_s` ahead on a copy of the world, with greedy
    /// deciding everything later, and the best outcome wins. By construction it does at
    /// least as well as greedy on the simulated horizon.
    ///
    /// Alternatives per drone: its greedy choice, -1, and the `candidates` blocks it can
    /// reach that lack the most water.
    pub fn rollout(&self, horizon_s: f32, candidates: usize) -> Vec<i32> {
        let (n, stations) = (self.task.len(), self.scenario.stations.len());
        let mut chosen = self.assign(Method::Greedy);
        if self.needs_reset {
            return chosen;
        }
        let mut terms = vec![0.0; n * TERMS.len()];
        for i in (0..n).filter(|&i| self.decide[i]) {
            let mut options = vec![chosen[i], -1];
            // `fires` is sorted by need, so the first that fit are the neediest.
            let reach = (0..self.fires.len()).filter(|&f| {
                let t = &self.fires[f];
                t.need - t.inbound > 0.0 && self.can_take(i, t)
            });
            options.extend(reach.take(candidates).map(|f| (stations + f) as i32));
            let mut best = (u32::MAX, usize::MAX);
            let mut seen: Vec<i32> = Vec::new();
            for option in options {
                if seen.contains(&option) {
                    continue;
                }
                seen.push(option);
                let mut first = chosen.clone();
                first[i] = option;
                let score = self.outcome(&first, horizon_s, &mut terms);
                if score < best {
                    (best, chosen[i]) = (score, option); // ties keep the earlier option: greedy first
                }
            }
        }
        chosen
    }

    /// The actions of a planner: see `Method`. `horizon_s` and `candidates` belong to rollout.
    pub fn plan(&self, method: Method, horizon_s: f32, candidates: usize) -> Vec<i32> {
        match method {
            Method::Rollout => self.rollout(horizon_s, candidates),
            method => self.assign(method),
        }
    }
}

/// The buffers of a whole batch, flat and environment-major. See `Out` for the meaning.
pub struct Bufs<'a> {
    pub drones: &'a mut [f32],
    pub targets: &'a mut [f32],
    pub target_mask: &'a mut [u8],
    pub action_mask: &'a mut [u8],
    pub globals: &'a mut [f32],
    pub decide: &'a mut [u8],
    pub terms: &'a mut [f32],
    pub elapsed_s: &'a mut [f32],
    pub terminated: &'a mut [u8],
    pub truncated: &'a mut [u8],
    pub info: &'a mut [f32],
    pub seed: &'a mut [u64],
    pub fingerprint: &'a mut [u64],
}

pub struct AllocVec {
    pub envs: Vec<AllocEnv>,
    pub cfg: AllocConfig,
    pub drones: usize,
    /// Target slots: the scenario's stations plus `max_targets` fire targets.
    pub slots: usize,
    /// Reset finished environments on their next step (training). Off for evaluation.
    pub autoreset: bool,
    pool: rayon::ThreadPool,
}

impl AllocVec {
    /// Fails if the scenario is malformed, has no ground station, or the settings cannot run.
    /// `threads == 0` uses every core.
    pub fn new(
        num_envs: usize,
        scenario: Scenario,
        sim: SimConfig,
        cfg: AllocConfig,
        seed: u64,
        threads: usize,
    ) -> Result<Self, String> {
        let world = scenario.world(sim, 0)?; // validates the map and what is placed on it
        cfg.check()?;
        if cfg.forecast && sim.fire_model != FireModel::Rothermel {
            return Err("forecast works with the Rothermel fire model only".into());
        }
        if scenario.stations.is_empty() {
            return Err("the scenario has no ground station for the drones".into());
        }
        let (drones, slots) = (
            fleet_size(scenario.stations.len(), &cfg),
            scenario.stations.len() + cfg.max_targets,
        );
        let (scenario, terrain) = (Arc::new(scenario), world.terrain.clone());
        let envs = (0..num_envs as u64)
            .map(|i| AllocEnv::new(scenario.clone(), terrain.clone(), sim, cfg, seed, i))
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| e.to_string())?;
        Ok(Self {
            envs,
            cfg,
            drones,
            slots,
            autoreset: true,
            pool,
        })
    }

    /// Steps every environment to its next decision event. `actions`: [envs, drones].
    pub fn step(&mut self, actions: &[i32], b: Bufs<'_>) {
        self.run(actions, None, b);
    }

    /// Starts a fresh episode everywhere and writes the first observation. With `seeds`, each
    /// environment runs the episode of its seed (evaluation); without, the next episode of
    /// its own training sequence.
    pub fn reset(&mut self, seeds: Option<&[u64]>, b: Bufs<'_>) {
        let seeds: Vec<u64> = match seeds {
            Some(seeds) => seeds.to_vec(),
            None => self
                .envs
                .iter_mut()
                .map(|e| {
                    e.episode += 1;
                    rng::derive(e.base_seed, e.index, e.episode - 1)
                })
                .collect(),
        };
        let idle = vec![-1; self.envs.len() * self.drones];
        self.run(&idle, Some(&seeds), b);
    }

    /// One tick everywhere with the drones flown by an outside pilot (a learned fly policy).
    /// `actions` [envs, drones] are applied where drones are waiting for a decision,
    /// `velocity` [envs, drones, 2] flies every drone for one tick. Finished environments stay
    /// frozen. Afterwards the planner's tables in `b` and the pilot's observation (`agent`,
    /// `neighbors`, `mask`, as in the navigate task with `view_m` and the planner's maximum
    /// speed) are up to date. `terms` and `elapsed_s` cover this one tick.
    #[allow(clippy::too_many_arguments)]
    pub fn fly_tick(
        &mut self,
        actions: &[i32],
        velocity: &[f32],
        view_m: f32,
        b: Bufs<'_>,
        agent: &mut [f32],
        neighbors: &mut [f32],
        mask: &mut [u8],
    ) {
        let (n, slots, dt) = (self.drones, self.slots, self.envs[0].sim.dt_s);
        self.pool.install(|| {
            let planner = (
                b.drones.par_chunks_mut(n * DRONE.len()),
                b.targets.par_chunks_mut(slots * TARGET.len()),
                b.target_mask.par_chunks_mut(slots),
                b.action_mask.par_chunks_mut(n * slots),
                b.globals.par_chunks_mut(GLOBAL.len()),
                b.decide.par_chunks_mut(n),
            );
            let result = (
                b.terms.par_chunks_mut(n * TERMS.len()),
                b.elapsed_s.par_iter_mut(),
                b.terminated.par_iter_mut(),
                b.truncated.par_iter_mut(),
                b.info.par_chunks_mut(INFO.len()),
                b.seed.par_iter_mut(),
                b.fingerprint.par_iter_mut(),
            );
            let pilot = (
                agent.par_chunks_mut(n * crate::navigate::AGENT.len()),
                neighbors.par_chunks_mut(n * n * 4),
                mask.par_chunks_mut(n * n),
            );
            (
                self.envs.par_iter_mut(),
                actions.par_chunks(n),
                velocity.par_chunks(2 * n),
                planner,
                result,
                pilot,
            )
                .into_par_iter()
                .for_each(|(env, actions, velocity, o, r, p)| {
                    r.0.fill(0.0);
                    (*r.1, *r.2, *r.3) = (0.0, 0, 0);
                    if !env.needs_reset {
                        if env.pending() {
                            env.apply(actions);
                        }
                        let (fire_out, timeout) = env.tick_flown(Some(velocity), r.0);
                        (*r.1, *r.2, *r.3) = (dt, fire_out as u8, timeout as u8);
                        if fire_out || timeout {
                            r.4.copy_from_slice(&env.summary());
                        }
                    }
                    env.observe(Out {
                        drones: o.0,
                        targets: o.1,
                        target_mask: o.2,
                        action_mask: o.3,
                        globals: o.4,
                        decide: o.5,
                        terms: &mut [],
                        elapsed_s: &mut 0.0,
                        terminated: &mut 0,
                        truncated: &mut 0,
                        info: &mut [],
                        seed: r.5,
                        fingerprint: r.6,
                    });
                    env.observe_pilot(view_m, p.0, p.1, p.2);
                });
        });
    }

    /// A built-in planner's actions for every environment (see `AllocEnv::plan`),
    /// [envs, drones].
    pub fn plan(&self, method: Method, horizon_s: f32, candidates: usize, actions: &mut [i32]) {
        self.pool.install(|| {
            (self.envs.par_iter(), actions.par_chunks_mut(self.drones))
                .into_par_iter()
                .for_each(|(env, out)| {
                    out.copy_from_slice(&env.plan(method, horizon_s, candidates));
                });
        });
    }

    fn run(&mut self, actions: &[i32], seeds: Option<&[u64]>, b: Bufs<'_>) {
        let (n, slots, autoreset) = (self.drones, self.slots, self.autoreset);
        self.pool.install(|| {
            let observation = (
                b.drones.par_chunks_mut(n * DRONE.len()),
                b.targets.par_chunks_mut(slots * TARGET.len()),
                b.target_mask.par_chunks_mut(slots),
                b.action_mask.par_chunks_mut(n * slots),
                b.globals.par_chunks_mut(GLOBAL.len()),
                b.decide.par_chunks_mut(n),
            );
            let result = (
                b.terms.par_chunks_mut(n * TERMS.len()),
                b.elapsed_s.par_iter_mut(),
                b.terminated.par_iter_mut(),
                b.truncated.par_iter_mut(),
                b.info.par_chunks_mut(INFO.len()),
                b.seed.par_iter_mut(),
                b.fingerprint.par_iter_mut(),
            );
            (
                self.envs.par_iter_mut().enumerate(),
                actions.par_chunks(n),
                observation,
                result,
            )
                .into_par_iter()
                .for_each(|((e, env), actions, o, r)| {
                    let out = Out {
                        drones: o.0,
                        targets: o.1,
                        target_mask: o.2,
                        action_mask: o.3,
                        globals: o.4,
                        decide: o.5,
                        terms: r.0,
                        elapsed_s: r.1,
                        terminated: r.2,
                        truncated: r.3,
                        info: r.4,
                        seed: r.5,
                        fingerprint: r.6,
                    };
                    match seeds {
                        Some(seeds) => {
                            out.terms.fill(0.0);
                            (*out.terminated, *out.truncated, *out.elapsed_s) = (0, 0, 0.0);
                            env.reset(seeds[e]);
                            env.observe(out);
                        }
                        None => env.step(actions, autoreset, out),
                    }
                });
        });
    }
}
