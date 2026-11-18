//! The `allocate` task: a planner gives every drone a target, a scripted controller flies it
//! there. Targets are groups of burning cells (to drop water on) and ground stations (to
//! refill the tank).
//!
//! One step runs the simulation until the next *decision event*: a drone finished its task,
//! its target stopped burning, or `max_interval_s` passed. The step returns how long that
//! took and which drones need a decision, so a trainer can discount by real time and only
//! ask the drones that are actually free. An action is an index into the target table, or
//! -1 for "keep doing what you are doing".
//!
//! Episode `e` is a function of its seed alone (ignitions, wind, noise), so two policies run
//! on the same seed face the same fire.
//! ponytail: concrete types like `navigate`; the batching is shared once a third task exists.

use std::collections::BTreeMap;
use std::sync::Arc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::fire::BURNING;
use crate::rng::{self, Stream};
use crate::scenario::Scenario;
use crate::terrain::Terrain;
use crate::world::{SimConfig, Weather, World};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AllocConfig {
    pub max_speed_ms: f32,
    /// Tank size and the amount released per drop, litres. A drop lands on one cell.
    /// To calibrate together with the water constants in `SimConfig`.
    pub tank_l: f32,
    pub drop_l: f32,
    /// Time to fill an empty tank at a ground station.
    pub refill_s: f32,
    /// A fire target is reached within this distance, a ground station within `station_m`
    /// (stations are wide so several drones fit).
    pub arrive_m: f32,
    pub station_m: f32,
    /// The controller never lets two drones get closer than this.
    pub safe_m: f32,
    /// Burning cells are grouped into square blocks of this many cells; each block with fire
    /// is one target. 1 makes every burning cell its own target.
    pub block_cells: u32,
    /// At most this many fire targets are offered, largest groups first.
    pub max_targets: usize,
    /// Everyone may decide again after this long, even without an event.
    pub max_interval_s: f32,
    pub time_limit_s: f32,
    /// If above zero, the scenario's starting fires are replaced by this many random ones.
    pub random_ignitions: u32,
    /// Turn the scenario's wind schedule by one random angle per episode.
    pub random_wind: bool,
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
            tank_l: 300.0,
            drop_l: 150.0,
            refill_s: 10.0,
            arrive_m: 5.0,
            station_m: 12.0,
            safe_m: 3.0,
            block_cells: 5,
            max_targets: 32,
            max_interval_s: 10.0,
            time_limit_s: 1800.0,
            random_ignitions: 0,
            random_wind: false,
            preburn_s: 0.0,
            preburn_cells: 0,
        }
    }
}

/// Per-drone columns. Positions are fractions of the map width and height.
pub const DRONE: [&str; 6] = ["x", "y", "water", "task_x", "task_y", "task_kind"];
/// Per-target columns. Stations come first, then fire targets.
pub const TARGET: [&str; 4] = ["x", "y", "is_station", "cells"];
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
pub const INFO: [&str; 8] = [
    "fire_out",
    "seconds", // since the episode started, not counting the pre-burn
    "burned_cells",
    "burned_fraction",
    "water_l",
    "drops",
    "min_separation_m",
    "start_cells", // cells that had caught fire when the episode started
];
pub const TASK_NONE: f32 = 0.0;
pub const TASK_FIRE: f32 = 1.0;
pub const TASK_STATION: f32 = 2.0;
pub const TASK_POINT: f32 = 3.0;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Task {
    None,
    /// Fly to this cell and drop water on it.
    Fire(u32),
    /// Fly to this station (index into the scenario's list) and fill the tank.
    Station(u32),
    /// Fly to this cell and wait there (sent by hand).
    Point(u32),
}

/// The buffers of one environment.
pub struct Out<'a> {
    pub drones: &'a mut [f32],
    pub targets: &'a mut [f32],
    /// [slots] 1 where a target slot is filled.
    pub target_mask: &'a mut [u8],
    /// [drones, slots] 1 where that drone may choose that slot: a fire needs water for a
    /// drop, a station needs a tank that is not full.
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
    /// The fire targets offered at the last decision: (cell, burning cells in its block).
    fires: Vec<(u32, u32)>,
    drops: u32,
    water_l: f32,
    min_separation: f32,
    needs_reset: bool,
    /// Ticks since tasks were last handed out.
    since_decision: u32,
    /// The world's tick at which the episode started, and the cells alight by then: both
    /// zero-ish without a pre-burn. The world's own clock counts from the ignition.
    start_tick: u64,
    start_cells: u32,
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
        let n = scenario.drones.len();
        Self {
            world: World::new(terrain.clone(), weather, sim, 0),
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
            decide: vec![true; n],
            fires: Vec::new(),
            drops: 0,
            water_l: 0.0,
            min_separation: f32::INFINITY,
            needs_reset: true,
            since_decision: 0,
            start_tick: 0,
            start_cells: 0,
        }
    }

    fn cell_centre(&self, cell: [u32; 2]) -> [f32; 2] {
        let m = self.terrain.grid.cell_m;
        [(cell[0] as f32 + 0.5) * m, (cell[1] as f32 + 0.5) * m]
    }

    fn task_point(&self, task: Task) -> Option<[f32; 2]> {
        match task {
            Task::None => None,
            Task::Fire(cell) | Task::Point(cell) => {
                let (x, y) = self.terrain.grid.xy(cell as usize);
                Some(self.cell_centre([x, y]))
            }
            Task::Station(i) => Some(self.cell_centre(self.scenario.stations[i as usize])),
        }
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

    /// Smallest distance between two drones so far in this episode, metres.
    pub fn min_separation(&self) -> f32 {
        self.min_separation
    }

    /// Identifies the fire this episode started with (burning cells and wind): the same
    /// number for every planner on the same seed, scenario and settings.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Tasks finished so far in this episode (drops made).
    pub fn drops(&self) -> u32 {
        self.drops
    }

    /// Starts the episode that belongs to `seed`.
    pub fn reset(&mut self, seed: u64) {
        let (grid, s) = (self.terrain.grid, &self.scenario);
        let mut wind = s.wind.clone();
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

        let n = s.drones.len();
        self.world.drones.pos = s.drones.iter().map(|c| self.cell_centre(*c)).collect();
        self.world.drones.vel = vec![[0.0; 2]; n];
        self.world.drones.water = vec![self.cfg.tank_l; n];

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
        self.decide.fill(true);
        (self.drops, self.water_l, self.min_separation) = (0, 0.0, f32::INFINITY);
        self.needs_reset = false;
        self.since_decision = 0;
        self.fires = self.fire_targets();
    }

    /// Groups the burning cells into blocks and returns, for the largest groups, the burning
    /// cell nearest to the group's centre and the group's size.
    fn fire_targets(&self) -> Vec<(u32, u32)> {
        let (grid, block) = (self.terrain.grid, self.cfg.block_cells.max(1));
        let mut groups: BTreeMap<(u32, u32), (u32, f32, f32)> = BTreeMap::new();
        for cell in self.world.fire.burning() {
            let (x, y) = grid.xy(cell as usize);
            let g = groups.entry((y / block, x / block)).or_default();
            *g = (g.0 + 1, g.1 + x as f32, g.2 + y as f32);
        }
        let mut best: BTreeMap<(u32, u32), (f32, u32)> = BTreeMap::new();
        for cell in self.world.fire.burning() {
            let (x, y) = grid.xy(cell as usize);
            let key = (y / block, x / block);
            let (n, sx, sy) = groups[&key];
            let d = (x as f32 - sx / n as f32).hypot(y as f32 - sy / n as f32);
            let b = best.entry(key).or_insert((f32::INFINITY, 0));
            if (d, cell) < *b {
                *b = (d, cell);
            }
        }
        let mut out: Vec<(u32, u32)> = best
            .iter()
            .map(|(key, &(_, cell))| (cell, groups[key].0))
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0))); // largest first, then by cell
        out.truncate(self.cfg.max_targets);
        out
    }

    /// One tick of the scripted controller: every drone moves towards its task, never closer
    /// than `safe_m` to another drone (`navigate::scripted_step`).
    fn fly(&mut self) {
        let (dt, safe) = (self.sim.dt_s, self.cfg.safe_m);
        let g = self.terrain.grid;
        let size = (g.nx as f32 * g.cell_m, g.ny as f32 * g.cell_m);
        let pos = &self.world.drones.pos;
        let goal = |i: usize| match self.task[i] {
            // Hold position inside a station's ring while refilling.
            Task::Station(_) if self.arrived(i) => pos[i],
            task => self.task_point(task).unwrap_or(pos[i]),
        };
        let next =
            crate::navigate::scripted_step(pos, goal, self.cfg.max_speed_ms * dt, safe, size);
        for (i, q) in next.iter().enumerate() {
            let p = self.world.drones.pos[i];
            self.world.drones.vel[i] = [(q[0] - p[0]) / dt, (q[1] - p[1]) / dt];
        }
    }

    fn arrived(&self, i: usize) -> bool {
        let radius = match self.task[i] {
            Task::Station(_) => self.cfg.station_m,
            _ => self.cfg.arrive_m,
        };
        self.task_point(self.task[i])
            .is_some_and(|goal| dist(self.world.drones.pos[i], goal) <= radius)
    }

    /// `actions`: [drones], a slot of the target table or -1 to keep the current task. Only
    /// drones whose `decide` flag was set are listened to.
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

        self.apply(actions);
        let ignited_before = self.world.fire.n_ignited;
        let mut ticks = 0u32;
        let (fire_out, timeout) = loop {
            let ended = self.tick(out.terms);
            ticks += 1;
            if ended.0 || ended.1 || self.pending() {
                break ended;
            }
        };

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

    /// Gives the deciding drones their new tasks. `actions`: [drones], a slot of the target
    /// table or -1 to keep the current task; drones that were not asked are not listened to.
    pub fn apply(&mut self, actions: &[i32]) {
        let stations = self.scenario.stations.len();
        // The index addresses actions, flags, tasks and tanks alike.
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.task.len() {
            if !self.decide[i] || actions[i] < 0 {
                continue;
            }
            let slot = actions[i] as usize;
            if slot < stations {
                // Refilling a full tank would be a way to do nothing forever; it is not allowed.
                if self.world.drones.water[i] < self.cfg.tank_l {
                    self.task[i] = Task::Station(slot as u32);
                }
            } else if let Some(&(cell, _)) = self.fires.get(slot - stations) {
                // A drone without water for a drop cannot take a fire; its choice is ignored.
                if self.world.drones.water[i] >= self.cfg.drop_l {
                    self.task[i] = Task::Fire(cell);
                }
            }
        }
        self.decide.fill(false);
        self.since_decision = 0;
    }

    /// Does any drone need a decision?
    pub fn pending(&self) -> bool {
        self.decide.iter().any(|&d| d)
    }

    /// Advances the simulation by one tick: drones fly, drop and refill. Adds this tick's
    /// `drop` and `refill` amounts to `terms` ([drones, TERMS]) and returns (fire out, time
    /// limit reached). When either is true the episode is over.
    pub fn tick(&mut self, terms: &mut [f32]) -> (bool, bool) {
        self.tick_flown(None, terms)
    }

    /// Where a pilot should steer each drone: its task's place, or where it is when it has
    /// nothing to do or is refilling inside a station's ring.
    pub fn flight_goals(&self) -> Vec<[f32; 2]> {
        (0..self.task.len())
            .map(|i| match self.task[i] {
                Task::Station(_) if self.arrived(i) => self.world.drones.pos[i],
                task => self.task_point(task).unwrap_or(self.world.drones.pos[i]),
            })
            .collect()
    }

    /// Like `tick`, but with `velocity` ([drones, 2], fractions of the maximum speed) the
    /// drones are flown by an outside pilot instead of the scripted controller. Nothing then
    /// keeps them apart or on the map: `min_separation` tells how that went.
    pub fn tick_flown(&mut self, velocity: Option<&[f32]>, terms: &mut [f32]) -> (bool, bool) {
        let n = self.task.len();
        let term = |i: usize, which: usize| i * TERMS.len() + which;
        let (dt, cell_area) = (self.sim.dt_s, self.terrain.grid.cell_m.powi(2));
        match velocity {
            None => self.fly(),
            Some(v) => {
                for i in 0..n {
                    let (ax, ay) = (v[2 * i], v[2 * i + 1]);
                    let scale = self.cfg.max_speed_ms / ax.hypot(ay).max(1.0);
                    self.world.drones.vel[i] = [ax * scale, ay * scale];
                }
            }
        }
        self.world.tick();
        self.since_decision += 1;
        for i in 0..n {
            for j in 0..i {
                let d = dist(self.world.drones.pos[i], self.world.drones.pos[j]);
                self.min_separation = self.min_separation.min(d);
            }
            match self.task[i] {
                Task::None => {}
                Task::Point(_) => {
                    if self.arrived(i) {
                        (self.task[i], self.decide[i]) = (Task::None, true);
                    }
                }
                Task::Fire(cell) => {
                    if self.world.fire.state[cell as usize] != BURNING {
                        (self.task[i], self.decide[i]) = (Task::None, true); // someone else got it
                    } else if self.arrived(i) {
                        self.world
                            .water
                            .add(cell as usize, self.cfg.drop_l / cell_area);
                        self.world.drones.water[i] -= self.cfg.drop_l;
                        self.drops += 1;
                        self.water_l += self.cfg.drop_l;
                        terms[term(i, 0)] += 1.0;
                        (self.task[i], self.decide[i]) = (Task::None, true);
                    }
                }
                Task::Station(_) => {
                    if self.arrived(i) {
                        let water = &mut self.world.drones.water[i];
                        let gain = (self.cfg.tank_l * dt / self.cfg.refill_s)
                            .min(self.cfg.tank_l - *water);
                        *water += gain;
                        terms[term(i, 1)] += gain / self.cfg.tank_l;
                        if *water >= self.cfg.tank_l {
                            (self.task[i], self.decide[i]) = (Task::None, true);
                        }
                    }
                }
            }
        }
        if self.since_decision as f32 * dt >= self.cfg.max_interval_s {
            self.decide.fill(true); // a regular look at the situation
        }
        let fire_out = self.world.fire.is_out();
        let limit = (self.cfg.time_limit_s / dt).ceil() as u64;
        let timeout = !fire_out && self.world.tick - self.start_tick >= limit;
        if fire_out || timeout {
            self.needs_reset = true;
        }
        if fire_out || timeout || self.pending() {
            self.fires = self.fire_targets(); // decisions are made on the current fire
        }
        (fire_out, timeout)
    }

    /// The episode summary in the order of `INFO`.
    pub fn summary(&self) -> [f32; 8] {
        [
            self.world.fire.is_out() as u8 as f32,
            self.episode_s(),
            self.world.fire.n_ignited as f32,
            self.world.burned_fraction(),
            self.water_l,
            self.drops as f32,
            self.min_separation,
            self.start_cells as f32,
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

    /// Sends a drone by hand: to refill if the cell holds a station, to drop water if it
    /// burns and the drone has water, otherwise just to fly there.
    pub fn send(&mut self, drone: usize, cell: [u32; 2]) {
        let index = self.terrain.grid.idx(cell[0], cell[1]) as u32;
        self.task[drone] = match self.scenario.stations.iter().position(|s| *s == cell) {
            Some(station) => Task::Station(station as u32),
            None if self.world.fire.state[index as usize] == BURNING
                && self.world.drones.water[drone] >= self.cfg.drop_l =>
            {
                Task::Fire(index)
            }
            None => Task::Point(index),
        };
        self.decide[drone] = false;
    }

    /// What each drone is doing: task kind (`TASK_*`) and where, in metres.
    pub fn tasks(&self) -> Vec<(f32, Option<[f32; 2]>)> {
        self.task
            .iter()
            .map(|&t| {
                let kind = match t {
                    Task::None => TASK_NONE,
                    Task::Fire(_) => TASK_FIRE,
                    Task::Station(_) => TASK_STATION,
                    Task::Point(_) => TASK_POINT,
                };
                (kind, self.task_point(t))
            })
            .collect()
    }

    /// The fire targets on offer (cell centres in metres), in slot order after the stations.
    pub fn fire_points(&self) -> Vec<[f32; 2]> {
        self.fires
            .iter()
            .map(|&(cell, _)| {
                self.task_point(Task::Fire(cell))
                    .expect("a fire has a place")
            })
            .collect()
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
        out.targets.fill(0.0);
        out.target_mask.fill(0);
        for (s, cell) in self.scenario.stations.iter().enumerate() {
            let p = self.cell_centre(*cell);
            out.targets[s * 4..][..4].copy_from_slice(&[p[0] / w, p[1] / h, 1.0, 0.0]);
            out.target_mask[s] = 1;
        }
        let block_cells = (self.cfg.block_cells.max(1) as f32).powi(2);
        for (f, &(cell, count)) in self.fires.iter().enumerate() {
            let p = self
                .task_point(Task::Fire(cell))
                .expect("a fire has a place");
            out.targets[(stations + f) * 4..][..4].copy_from_slice(&[
                p[0] / w,
                p[1] / h,
                0.0,
                count as f32 / block_cells,
            ]);
            out.target_mask[stations + f] = 1;
        }
        for i in 0..n {
            let p = self.world.drones.pos[i];
            let water = self.world.drones.water[i];
            let (kind, point) = match self.task[i] {
                Task::None => (TASK_NONE, p),
                task @ Task::Fire(_) => (TASK_FIRE, self.task_point(task).unwrap_or(p)),
                task @ Task::Station(_) => (TASK_STATION, self.task_point(task).unwrap_or(p)),
                task @ Task::Point(_) => (TASK_POINT, self.task_point(task).unwrap_or(p)),
            };
            out.drones[i * 6..][..6].copy_from_slice(&[
                p[0] / w,
                p[1] / h,
                water / self.cfg.tank_l,
                point[0] / w,
                point[1] / h,
                kind,
            ]);
            out.decide[i] = self.decide[i] as u8;
            for slot in 0..slots {
                let fire = slot >= stations;
                let allowed = if fire {
                    water >= self.cfg.drop_l
                } else {
                    water < self.cfg.tank_l
                };
                let can = out.target_mask[slot] == 1 && allowed;
                out.action_mask[i * slots + slot] = can as u8;
            }
        }
        let wind = self.world.wind();
        out.globals.copy_from_slice(&[
            wind.to[0] * wind.speed_ms / 20.0,
            wind.to[1] * wind.speed_ms / 20.0,
            self.episode_s() / self.cfg.time_limit_s,
            self.world.fire.n_burning() as f32 / g.len() as f32,
            self.world.burned_fraction(),
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
    fn nearest_station(&self, i: usize) -> Option<i32> {
        let p = self.world.drones.pos[i];
        (0..self.scenario.stations.len())
            .min_by(|&a, &b| {
                let d = |s: usize| dist(p, self.cell_centre(self.scenario.stations[s]));
                d(a).total_cmp(&d(b))
            })
            .map(|s| s as i32)
    }

    /// Greedy assignment: repeatedly pair the closest free drone and free fire. A fire another
    /// drone is flying to is left alone; a drone that cannot drop goes to the nearest station;
    /// a drone left without a fire refills or waits. Returns a slot or -1 per drone.
    pub fn greedy(&self) -> Vec<i32> {
        let (n, stations) = (self.task.len(), self.scenario.stations.len());
        let mut actions = vec![-1; n];
        let taken = |cell: u32| (0..n).any(|j| !self.decide[j] && self.task[j] == Task::Fire(cell));
        let mut fires: Vec<usize> = (0..self.fires.len())
            .filter(|&f| !taken(self.fires[f].0))
            .collect();
        let mut ready: Vec<usize> = (0..n)
            .filter(|&i| self.decide[i] && self.world.drones.water[i] >= self.cfg.drop_l)
            .collect();
        while !ready.is_empty() && !fires.is_empty() {
            let mut best = (f32::INFINITY, 0, 0);
            for (r, &i) in ready.iter().enumerate() {
                for (c, &f) in fires.iter().enumerate() {
                    let d = dist(
                        self.world.drones.pos[i],
                        self.task_point(Task::Fire(self.fires[f].0)).unwrap(),
                    );
                    if d < best.0 {
                        best = (d, r, c);
                    }
                }
            }
            actions[ready.remove(best.1)] = (stations + fires.remove(best.2)) as i32;
        }
        #[allow(clippy::needless_range_loop)] // the index addresses flags, tanks and actions
        for i in 0..n {
            if self.decide[i] && actions[i] < 0 && self.world.drones.water[i] < self.cfg.tank_l {
                actions[i] = self.nearest_station(i).unwrap_or(-1);
            }
        }
        actions
    }

    /// Plays on from here with `first` now and greedy afterwards for `horizon_s`, and returns
    /// what there is to minimise: cells burned, then cells still burning.
    fn outcome(&self, first: &[i32], horizon_s: f32, scratch: &mut Scratch) -> (u32, usize) {
        let mut sim = self.clone();
        let end = sim.world.time_s() + horizon_s;
        sim.step(first, false, scratch.out(self));
        while !sim.needs_reset && sim.world.time_s() < end {
            let actions = sim.greedy();
            sim.step(&actions, false, scratch.out(self));
        }
        (sim.world.fire.n_ignited, sim.world.fire.n_burning())
    }

    /// Rollout planner (Bertsekas' multiagent rollout with greedy as the base policy). The
    /// drones that have to decide are taken one at a time; for each, a few alternatives to
    /// its greedy choice are simulated `horizon_s` ahead on a copy of the world, with greedy
    /// deciding everything later, and the best outcome wins. By construction it does at
    /// least as well as greedy on the simulated horizon.
    ///
    /// Alternatives per drone: its greedy choice, the `candidates` nearest fires it may take,
    /// the nearest station if its tank is not full, and waiting.
    pub fn rollout(&self, horizon_s: f32, candidates: usize) -> Vec<i32> {
        let (n, stations) = (self.task.len(), self.scenario.stations.len());
        let mut chosen = self.greedy();
        if self.needs_reset {
            return chosen;
        }
        let mut scratch = Scratch::new(self);
        for i in (0..n).filter(|&i| self.decide[i]) {
            let p = self.world.drones.pos[i];
            let water = self.world.drones.water[i];
            let mut options = vec![chosen[i], -1];
            if water >= self.cfg.drop_l {
                let mut near: Vec<(f32, i32)> = (0..self.fires.len())
                    .map(|f| {
                        (
                            dist(p, self.task_point(Task::Fire(self.fires[f].0)).unwrap()),
                            (stations + f) as i32,
                        )
                    })
                    .collect();
                near.sort_by(|a, b| a.0.total_cmp(&b.0));
                options.extend(near.iter().take(candidates).map(|o| o.1));
            }
            if water < self.cfg.tank_l {
                options.extend(self.nearest_station(i));
            }
            let mut best = (u32::MAX, usize::MAX);
            let mut seen: Vec<i32> = Vec::new();
            for option in options {
                if seen.contains(&option) {
                    continue;
                }
                seen.push(option);
                let mut first = chosen.clone();
                first[i] = option;
                let score = self.outcome(&first, horizon_s, &mut scratch);
                if score < best {
                    (best, chosen[i]) = (score, option); // ties keep the earlier option: greedy first
                }
            }
        }
        chosen
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
    /// Fails if the scenario is malformed or its drones start closer than `safe_m`.
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
        if scenario.drones.is_empty() {
            return Err("the scenario has no drones".into());
        }
        let pos = &world.drones.pos;
        if (0..pos.len()).any(|i| (0..i).any(|j| dist(pos[i], pos[j]) < cfg.safe_m)) {
            return Err(format!("two drones start closer than {} m", cfg.safe_m));
        }
        let (drones, slots) = (
            scenario.drones.len(),
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
                    let g = env.terrain.grid;
                    let size = (g.nx as f32 * g.cell_m, g.ny as f32 * g.cell_m);
                    let d = &env.world.drones;
                    crate::navigate::observe_flight(
                        &d.pos,
                        &d.vel,
                        &env.flight_goals(),
                        size,
                        view_m,
                        env.cfg.max_speed_ms,
                        p.0,
                        p.1,
                        p.2,
                    );
                });
        });
    }

    /// Rollout-planner actions for every environment (see `AllocEnv::rollout`), [envs, drones].
    pub fn rollout(&self, horizon_s: f32, candidates: usize, actions: &mut [i32]) {
        self.pool.install(|| {
            (self.envs.par_iter(), actions.par_chunks_mut(self.drones))
                .into_par_iter()
                .for_each(|(env, out)| {
                    out.copy_from_slice(&env.rollout(horizon_s, candidates));
                });
        });
    }

    /// Greedy actions for every environment (see `AllocEnv::greedy`), [envs, drones].
    pub fn greedy(&self, actions: &mut [i32]) {
        for (env, out) in self.envs.iter().zip(actions.chunks_mut(self.drones)) {
            out.copy_from_slice(&env.greedy());
        }
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
