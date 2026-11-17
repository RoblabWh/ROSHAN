//! The `navigate` task: every drone flies to a sequence of waypoints without hitting another
//! drone or leaving the map. This is the low-level skill the planner later relies on.
//!
//! One step is one tick. An episode ends in success when every drone has reached all its
//! waypoints, in failure on the first collision or map exit, and is cut off at the time limit.
//! The step after an episode ended resets the environment and ignores its action, so the final
//! observation of an episode is always visible to the trainer (needed to bootstrap cut-off
//! episodes).
//!
//! Rust writes named building blocks into the caller's buffers: observation tables and a vector
//! of reward terms per drone. Weighting the terms into a reward is the trainer's business.
//! ponytail: concrete types, no Task trait. Extract the shared batching when `allocate` (M4)
//! becomes the second task.

use std::sync::Arc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::rng::{self, Stream};
use crate::rothermel::Moisture;
use crate::terrain::{Grid, Terrain};
use crate::world::{SimConfig, Weather, World};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NavConfig {
    pub drones: usize,
    pub max_speed_ms: f32,
    /// How far a drone sees other drones and the map edge, in metres. Also the length all
    /// distances in the observation are divided by.
    pub view_m: f32,
    /// Two drones closer than this have collided.
    pub collide_m: f32,
    /// A waypoint counts as reached within this distance.
    pub arrive_m: f32,
    /// Waypoints each drone has to reach per episode.
    pub goals: u32,
    /// Probability that an episode starts with all drones at a ground station in a map corner
    /// instead of at random places.
    pub station_start: f32,
    /// Time limit = goals * map diagonal / max speed * this factor.
    pub time_factor: f32,
    /// Bands (fractions of `view_m`) inside which closing in on another drone or on the map
    /// edge produces the `approach_*` reward terms.
    pub near_drone: f32,
    pub near_edge: f32,
}

impl Default for NavConfig {
    fn default() -> Self {
        Self {
            drones: 4,
            max_speed_ms: 10.0,
            view_m: 50.0,
            collide_m: 2.0,
            arrive_m: 2.5,
            goals: 3,
            station_start: 0.5,
            time_factor: 2.5,
            near_drone: 0.4,
            near_edge: 0.5,
        }
    }
}

/// Per-drone observation columns.
pub const AGENT: [&str; 8] = [
    "vx",
    "vy",
    "goal_dx",
    "goal_dy",
    "goal_cos",
    "goal_sin",
    "speed",
    "goal_dist",
];
/// Columns of each neighbour slot. Slot 0 is the nearest map edge, the rest are other drones.
pub const NEIGHBOR: [&str; 4] = ["dx", "dy", "dvx", "dvy"];
/// Reward terms per drone. Events are 0/1; the last three are small per-step amounts.
/// `progress` is signed, so it sums to the distance actually covered and jitter earns nothing.
pub const TERMS: [&str; 8] = [
    "goal",           // this drone reached a waypoint
    "success",        // the episode succeeded (every drone)
    "collision",      // this drone hit another one
    "out_of_bounds",  // this drone left the map
    "timeout",        // time ran out before this drone was finished
    "progress", // distance gained towards the waypoint in view lengths; negative when moving away
    "approach_drone", // increase in closeness to other drones inside the band
    "approach_edge", // increase in closeness to the map edge inside the band
];
const GOAL: usize = 0;
const SUCCESS: usize = 1;
const COLLISION: usize = 2;
const OUT_OF_BOUNDS: usize = 3;
const TIMEOUT: usize = 4;
const PROGRESS: usize = 5;
const APPROACH_DRONE: usize = 6;
const APPROACH_EDGE: usize = 7;

/// Episode summary columns, valid on the step an episode ends.
pub const INFO: [&str; 4] = ["success", "reason", "seconds", "goals"];
pub const REASON_COLLISION: f32 = 1.0;
pub const REASON_OUT_OF_BOUNDS: f32 = 2.0;
pub const REASON_TIMEOUT: f32 = 3.0;

/// The buffers of one environment.
pub struct Out<'a> {
    pub agent: &'a mut [f32],
    pub neighbors: &'a mut [f32],
    pub mask: &'a mut [u8],
    pub terms: &'a mut [f32],
    pub terminated: &'a mut u8,
    pub truncated: &'a mut u8,
    pub info: &'a mut [f32],
}

pub struct NavEnv {
    pub world: World,
    cfg: NavConfig,
    seed: u64,
    index: u64,
    episode: u64,
    episode_seed: u64,
    goal: Vec<[f32; 2]>,
    goals_left: Vec<u32>,
    last_dist: Vec<f32>,
    last_near_drone: Vec<f32>,
    last_near_edge: Vec<f32>,
    goals_reached: u32,
    ticks: u32,
    limit: u32,
    needs_reset: bool,
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

impl NavEnv {
    fn new(terrain: Arc<Terrain>, sim: SimConfig, cfg: NavConfig, seed: u64, index: u64) -> Self {
        let dry = Moisture {
            d1: 0.06,
            d10: 0.07,
            d100: 0.08,
            herb: 0.6,
            woody: 0.9,
        };
        let world = World::new(terrain, Weather::steady(dry, 0.0, [1.0, 0.0]), sim, seed);
        let n = cfg.drones;
        let (w, h) = Self::size(&world);
        let limit = cfg.goals as f32 * w.hypot(h) / cfg.max_speed_ms * cfg.time_factor / sim.dt_s;
        Self {
            world,
            cfg,
            seed,
            index,
            episode: 0,
            episode_seed: 0,
            goal: vec![[0.0; 2]; n],
            goals_left: vec![0; n],
            last_dist: vec![0.0; n],
            last_near_drone: vec![0.0; n],
            last_near_edge: vec![0.0; n],
            goals_reached: 0,
            ticks: 0,
            limit: limit.ceil() as u32,
            needs_reset: true,
        }
    }

    /// Map width and height in metres.
    fn size(world: &World) -> (f32, f32) {
        let g = world.terrain.grid;
        (g.nx as f32 * g.cell_m, g.ny as f32 * g.cell_m)
    }

    /// A point at least one cell inside the map. `draw` numbers the draw within this episode.
    fn random_point(&self, stream: Stream, draw: u64) -> [f32; 2] {
        let seed = self.episode_seed;
        let (w, h) = Self::size(&self.world);
        let margin = self.world.terrain.grid.cell_m;
        [
            margin + rng::uniform(seed, stream, draw, 0) * (w - 2.0 * margin),
            margin + rng::uniform(seed, stream, draw, 1) * (h - 2.0 * margin),
        ]
    }

    /// A random point that keeps clear of `others`. ponytail: after 32 tries the last
    /// candidate is taken; raise the count if maps get crowded.
    fn clear_point(&self, stream: Stream, draw: &mut u64, others: &[[f32; 2]]) -> [f32; 2] {
        let clearance = 3.0 * self.cfg.collide_m;
        let mut p = [0.0; 2];
        for _ in 0..32 {
            p = self.random_point(stream, *draw);
            *draw += 1;
            if others.iter().all(|&o| dist(o, p) >= clearance) {
                break;
            }
        }
        p
    }

    /// Starts the next episode. Episode `e` of environment `i` is a pure function of
    /// (seed, i, e), so runs are comparable whatever happened before.
    pub fn reset(&mut self) {
        let n = self.cfg.drones;
        let seed = rng::derive(self.seed, self.index, self.episode);
        self.episode_seed = seed;
        self.episode += 1;
        let (w, h) = Self::size(&self.world);
        let cell = self.world.terrain.grid.cell_m;

        let mut pos: Vec<[f32; 2]> = Vec::with_capacity(n);
        if rng::uniform(seed, Stream::Spawn, u64::MAX, 0) < self.cfg.station_start {
            // All drones on a small circle around a ground station one cell inside a corner.
            let corner = (rng::uniform(seed, Stream::Spawn, u64::MAX, 1) * 4.0) as u32;
            let station = [
                if corner & 1 == 0 {
                    1.5 * cell
                } else {
                    w - 1.5 * cell
                },
                if corner & 2 == 0 {
                    1.5 * cell
                } else {
                    h - 1.5 * cell
                },
            ];
            let radius = 1.5 * self.cfg.collide_m * (n as f32 / 4.0).max(1.0);
            for i in 0..n {
                let angle = std::f32::consts::TAU * i as f32 / n as f32;
                pos.push([
                    station[0] + radius * angle.cos(),
                    station[1] + radius * angle.sin(),
                ]);
            }
        } else {
            let mut draw = 0;
            for _ in 0..n {
                let p = self.clear_point(Stream::Spawn, &mut draw, &pos);
                pos.push(p);
            }
        }
        self.world.drones.pos = pos;
        self.world.drones.vel = vec![[0.0; 2]; n];
        self.world.tick = 0;

        let mut draw = 0;
        for i in 0..n {
            let taken = self.goal[..i].to_vec();
            self.goal[i] = self.clear_point(Stream::Goal, &mut draw, &taken);
        }
        self.goals_left.fill(self.cfg.goals);
        self.goals_reached = 0;
        self.ticks = 0;
        self.needs_reset = false;
        for i in 0..n {
            self.last_dist[i] = dist(self.world.drones.pos[i], self.goal[i]);
            self.last_near_drone[i] = self.near_drone(i);
            self.last_near_edge[i] = self.near_edge(i);
        }
    }

    /// Distance to the nearest map edge and the vector pointing to it.
    fn edge(&self, i: usize) -> (f32, [f32; 2]) {
        nearest_edge(self.world.drones.pos[i], Self::size(&self.world))
    }

    /// How deep drone `i` is inside the closeness band of other drones (0 when clear).
    fn near_drone(&self, i: usize) -> f32 {
        let band = self.cfg.near_drone * self.cfg.view_m;
        let pos = &self.world.drones.pos;
        (0..pos.len())
            .filter(|&j| j != i)
            .map(|j| (band - dist(pos[i], pos[j])).max(0.0))
            .sum::<f32>()
            / self.cfg.view_m
    }

    fn near_edge(&self, i: usize) -> f32 {
        let band = self.cfg.near_edge * self.cfg.view_m;
        (band - self.edge(i).0).max(0.0) / self.cfg.view_m
    }

    /// `actions`: [drones * 2], each drone's wanted velocity as a fraction of the maximum.
    pub fn step(&mut self, actions: &[f32], autoreset: bool, out: Out<'_>) {
        let n = self.cfg.drones;
        out.terms.fill(0.0);
        (*out.terminated, *out.truncated) = (0, 0);
        if self.needs_reset {
            if autoreset {
                self.reset();
                self.observe(out.agent, out.neighbors, out.mask);
            }
            return; // without autoreset a finished environment stays frozen
        }

        for i in 0..n {
            let (ax, ay) = (actions[2 * i], actions[2 * i + 1]);
            let scale = self.cfg.max_speed_ms / ax.hypot(ay).max(1.0); // never above max speed
            self.world.drones.vel[i] = [ax * scale, ay * scale];
        }
        self.world.tick();
        self.ticks += 1;

        let (w, h) = Self::size(&self.world);
        let term = |i: usize, which: usize| i * TERMS.len() + which;
        let mut failure = 0.0;
        for i in 0..n {
            let p = self.world.drones.pos[i];
            if p[0] < 0.0 || p[1] < 0.0 || p[0] > w || p[1] > h {
                out.terms[term(i, OUT_OF_BOUNDS)] = 1.0;
                failure = REASON_OUT_OF_BOUNDS;
            }
            for j in 0..i {
                if dist(p, self.world.drones.pos[j]) < self.cfg.collide_m {
                    out.terms[term(i, COLLISION)] = 1.0;
                    out.terms[term(j, COLLISION)] = 1.0;
                    failure = REASON_COLLISION;
                }
            }
        }

        let mut draw = 1000 * (self.goals_reached as u64 + 1); // fresh draws for each new waypoint
        for i in 0..n {
            let d = dist(self.world.drones.pos[i], self.goal[i]);
            out.terms[term(i, PROGRESS)] = (self.last_dist[i] - d) / self.cfg.view_m;
            self.last_dist[i] = d;
            if self.goals_left[i] > 0 && d < self.cfg.arrive_m {
                out.terms[term(i, GOAL)] = 1.0;
                self.goals_left[i] -= 1;
                self.goals_reached += 1;
                if self.goals_left[i] > 0 {
                    let taken: Vec<[f32; 2]> =
                        (0..n).filter(|&j| j != i).map(|j| self.goal[j]).collect();
                    self.goal[i] = self.clear_point(Stream::Goal, &mut draw, &taken);
                    self.last_dist[i] = dist(self.world.drones.pos[i], self.goal[i]);
                }
                // A drone with no waypoints left keeps its last one and should hover there.
            }
            let (near_drone, near_edge) = (self.near_drone(i), self.near_edge(i));
            out.terms[term(i, APPROACH_DRONE)] = (near_drone - self.last_near_drone[i]).max(0.0);
            out.terms[term(i, APPROACH_EDGE)] = (near_edge - self.last_near_edge[i]).max(0.0);
            (self.last_near_drone[i], self.last_near_edge[i]) = (near_drone, near_edge);
        }

        let success = failure == 0.0 && self.goals_left.iter().all(|&g| g == 0);
        let timeout = failure == 0.0 && !success && self.ticks >= self.limit;
        if success {
            (0..n).for_each(|i| out.terms[term(i, SUCCESS)] = 1.0);
        }
        if timeout {
            (0..n)
                .filter(|&i| self.goals_left[i] > 0)
                .for_each(|i| out.terms[term(i, TIMEOUT)] = 1.0);
        }
        *out.terminated = (success || failure != 0.0) as u8;
        *out.truncated = timeout as u8;
        if success || timeout || failure != 0.0 {
            self.needs_reset = true;
            let reason = if timeout { REASON_TIMEOUT } else { failure };
            out.info.copy_from_slice(&[
                success as u8 as f32,
                reason,
                self.ticks as f32 * self.world.cfg.dt_s,
                self.goals_reached as f32,
            ]);
        }
        self.observe(out.agent, out.neighbors, out.mask);
    }

    /// Writes the observation tables: `agent` [drones, 8], `neighbors` [drones, drones, 4],
    /// `mask` [drones, drones] (1 = slot holds something in view).
    pub fn observe(&self, agent: &mut [f32], neighbors: &mut [f32], mask: &mut [u8]) {
        let d = &self.world.drones;
        let (view, vmax) = (self.cfg.view_m, self.cfg.max_speed_ms);
        observe_flight(
            &d.pos,
            &d.vel,
            &self.goal,
            Self::size(&self.world),
            view,
            vmax,
            agent,
            neighbors,
            mask,
        );
    }

    pub fn goals(&self) -> &[[f32; 2]] {
        &self.goal
    }
}

/// Distance from `p` to the nearest edge of a map of the given size, and the vector to it.
pub fn nearest_edge(p: [f32; 2], (w, h): (f32, f32)) -> (f32, [f32; 2]) {
    [
        (p[0], [-p[0], 0.0]),
        (w - p[0], [w - p[0], 0.0]),
        (p[1], [0.0, -p[1]]),
        (h - p[1], [0.0, h - p[1]]),
    ]
    .into_iter()
    .min_by(|a, b| a.0.total_cmp(&b.0))
    .expect("four edges")
}

/// What a low-level pilot sees: for drones at `pos` moving with `vel` towards `goal` on a map
/// of `size` metres, writes `agent` [drones, 8], `neighbors` [drones, drones, 4] and `mask`
/// [drones, drones] (see `AGENT` and `NEIGHBOR`). The navigate task and learned flight under
/// a planner share this, so a fly policy sees the same thing in both.
#[allow(clippy::too_many_arguments)]
pub fn observe_flight(
    pos: &[[f32; 2]],
    vel: &[[f32; 2]],
    goal: &[[f32; 2]],
    size: (f32, f32),
    view: f32,
    vmax: f32,
    agent: &mut [f32],
    neighbors: &mut [f32],
    mask: &mut [u8],
) {
    let n = pos.len();
    neighbors.fill(0.0);
    mask.fill(0);
    for i in 0..n {
        let v = [vel[i][0] / vmax, vel[i][1] / vmax];
        let to_goal = [goal[i][0] - pos[i][0], goal[i][1] - pos[i][1]];
        let (speed, range) = (v[0].hypot(v[1]), to_goal[0].hypot(to_goal[1]));
        // Is the drone heading for its waypoint (cos) and which way must it turn (sin)?
        let (cos, sin) = if speed > 1e-6 && range > 1e-6 {
            let (vx, vy, gx, gy) = (
                v[0] / speed,
                v[1] / speed,
                to_goal[0] / range,
                to_goal[1] / range,
            );
            (vx * gx + vy * gy, vx * gy - vy * gx)
        } else {
            (0.0, 0.0)
        };
        agent[i * AGENT.len()..][..AGENT.len()].copy_from_slice(&[
            v[0],
            v[1],
            (to_goal[0] / view).clamp(-1.0, 1.0),
            (to_goal[1] / view).clamp(-1.0, 1.0),
            cos,
            sin,
            speed,
            (range / view).min(1.0),
        ]);

        // Slot 0: the nearest map edge, moving relative to the drone like a wall would.
        let slots = &mut neighbors[i * n * 4..][..n * 4];
        let (edge_dist, to_edge) = nearest_edge(pos[i], size);
        if edge_dist <= view {
            slots[..4].copy_from_slice(&[
                to_edge[0] / view,
                to_edge[1] / view,
                -v[0] / 2.0,
                -v[1] / 2.0,
            ]);
            mask[i * n] = 1;
        }
        // Slots 1..: the other drones in index order, if inside the square view.
        for (slot, j) in (0..n).filter(|&j| j != i).enumerate() {
            let d = [pos[j][0] - pos[i][0], pos[j][1] - pos[i][1]];
            if d[0].abs() <= view && d[1].abs() <= view {
                let dv = [
                    (vel[j][0] - vel[i][0]) / (2.0 * vmax),
                    (vel[j][1] - vel[i][1]) / (2.0 * vmax),
                ];
                slots[(slot + 1) * 4..][..4].copy_from_slice(&[
                    d[0] / view,
                    d[1] / view,
                    dv[0],
                    dv[1],
                ]);
                mask[i * n + slot + 1] = 1;
            }
        }
    }
}

/// Buffers for one environment used on its own (the viewer).
pub struct Scratch {
    pub agent: Vec<f32>,
    pub neighbors: Vec<f32>,
    pub mask: Vec<u8>,
    pub terms: Vec<f32>,
    pub terminated: u8,
    pub truncated: u8,
    pub info: Vec<f32>,
}

impl Scratch {
    pub fn new(drones: usize) -> Self {
        Self {
            agent: vec![0.0; drones * AGENT.len()],
            neighbors: vec![0.0; drones * drones * NEIGHBOR.len()],
            mask: vec![0; drones * drones],
            terms: vec![0.0; drones * TERMS.len()],
            terminated: 0,
            truncated: 0,
            info: vec![0.0; INFO.len()],
        }
    }

    pub fn out(&mut self) -> Out<'_> {
        Out {
            agent: &mut self.agent,
            neighbors: &mut self.neighbors,
            mask: &mut self.mask,
            terms: &mut self.terms,
            terminated: &mut self.terminated,
            truncated: &mut self.truncated,
            info: &mut self.info,
        }
    }
}

impl NavEnv {
    /// One environment on its own (the viewer uses this), with its first episode started.
    pub fn single(grid: Grid, sim: SimConfig, cfg: NavConfig, seed: u64) -> Self {
        let mut env = Self::new(Arc::new(Terrain::uniform(grid, 0)), sim, cfg, seed, 0);
        env.reset();
        env
    }

    pub fn config(&self) -> &NavConfig {
        &self.cfg
    }
}

/// The buffers of a whole batch, flat and environment-major.
pub struct Bufs<'a> {
    /// [envs, drones, 8]
    pub agent: &'a mut [f32],
    /// [envs, drones, drones, 4]
    pub neighbors: &'a mut [f32],
    /// [envs, drones, drones]
    pub mask: &'a mut [u8],
    /// [envs, drones, 8]
    pub terms: &'a mut [f32],
    /// [envs]
    pub terminated: &'a mut [u8],
    pub truncated: &'a mut [u8],
    /// [envs, 4]
    pub info: &'a mut [f32],
}

pub struct NavVec {
    pub envs: Vec<NavEnv>,
    pub cfg: NavConfig,
    /// Reset finished environments on their next step (training). Off for evaluation, where
    /// every seed must run exactly once.
    pub autoreset: bool,
    pool: rayon::ThreadPool,
}

impl NavVec {
    /// `threads == 0` uses every core.
    pub fn new(
        num_envs: usize,
        grid: Grid,
        sim: SimConfig,
        cfg: NavConfig,
        seed: u64,
        threads: usize,
    ) -> Self {
        // Nothing burns in this task: an all-non-burnable map makes the fire model free.
        let terrain = Arc::new(Terrain::uniform(grid, 0));
        let envs = (0..num_envs as u64)
            .map(|i| NavEnv::new(terrain.clone(), sim, cfg, seed, i))
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("failed to start the thread pool");
        Self {
            envs,
            cfg,
            autoreset: true,
            pool,
        }
    }

    /// Starts a fresh episode in every environment and writes the first observations.
    pub fn reset(&mut self, agent: &mut [f32], neighbors: &mut [f32], mask: &mut [u8]) {
        let n = self.cfg.drones;
        self.pool.install(|| {
            (
                self.envs.par_iter_mut(),
                agent.par_chunks_mut(n * AGENT.len()),
                neighbors.par_chunks_mut(n * n * 4),
                mask.par_chunks_mut(n * n),
            )
                .into_par_iter()
                .with_min_len(64)
                .for_each(|(env, agent, neighbors, mask)| {
                    env.reset();
                    env.observe(agent, neighbors, mask);
                });
        });
    }

    /// Steps every environment. `actions`: [envs, drones, 2].
    pub fn step(&mut self, actions: &[f32], out: Bufs<'_>) {
        let (n, autoreset) = (self.cfg.drones, self.autoreset);
        self.pool.install(|| {
            (
                self.envs.par_iter_mut(),
                actions.par_chunks(n * 2),
                out.agent.par_chunks_mut(n * AGENT.len()),
                out.neighbors.par_chunks_mut(n * n * 4),
                out.mask.par_chunks_mut(n * n),
                out.terms.par_chunks_mut(n * TERMS.len()),
                out.terminated.par_iter_mut(),
                out.truncated.par_iter_mut(),
                out.info.par_chunks_mut(INFO.len()),
            )
                .into_par_iter()
                .with_min_len(64)
                .for_each(
                    |(env, actions, agent, neighbors, mask, terms, terminated, truncated, info)| {
                        env.step(
                            actions,
                            autoreset,
                            Out {
                                agent,
                                neighbors,
                                mask,
                                terms,
                                terminated,
                                truncated,
                                info,
                            },
                        );
                    },
                );
        });
    }
}
