//! Fire spread on the grid.
//!
//! Every burning cell is a *source* that carries the fire along 16 fixed directions (8
//! neighbours plus 8 knight moves). Each direction is an edge with its own progress from 0 to
//! 1; the edge's rate is one over the travel time between the two cell centres, using the
//! Rothermel ellipse of every cell the straight path crosses. A cell ignites when the first
//! edge reaches it, at the interpolated arrival time, so results do not depend on the step
//! size to first order. This is a minimum-travel-time scheme that still allows wind, moisture
//! and water to change while the fire burns.
//!
//! Why 16 directions: the simulated fire is the polygon spanned by the edge directions inside
//! the true ellipse, so too few directions make a wind-driven fire too narrow. Measured with
//! `cargo run --release -p roshan-core --example bench`: with 16 directions the
//! length-to-breadth ratio is within 7 % of the ellipse up to a ratio of 3.2 (16 m/s wind on
//! grass). For wind along a grid axis it is 25 % too high at a ratio of 4.5 and 70 % at 6.8.
//!
//! Nothing here depends on the order in which sources are stored, so results are independent
//! of history that happened elsewhere on the map.

use serde::{Deserialize, Serialize};

use crate::rng::{self, Stream};
use crate::rothermel::{FuelBed, Spread};
use crate::spin::Spin;
use crate::terrain::Terrain;
use crate::world::SimConfig;

pub const UNBURNED: u8 = 0;
pub const BURNING: u8 = 1;
pub const BURNED: u8 = 2;

pub const DIRS: [(i32, i32); 16] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
];

#[derive(Clone, Copy)]
enum Kind {
    Axis,
    Diagonal,
    Knight,
}

fn kind(k: usize) -> Kind {
    let (dx, dy) = DIRS[k];
    if dx == 0 || dy == 0 {
        Kind::Axis
    } else if dx.abs() == dy.abs() {
        Kind::Diagonal
    } else {
        Kind::Knight
    }
}

/// The two cells beside the path of edge `k` (offsets from the source). For a diagonal they
/// are the cells whose shared corner the path squeezes through; for a knight move they are
/// the cells the path crosses. Unused for axis moves.
fn via(k: usize) -> [(i32, i32); 2] {
    let (dx, dy) = DIRS[k];
    let (sx, sy) = (dx.signum(), dy.signum());
    match kind(k) {
        Kind::Axis => [(0, 0); 2],
        Kind::Diagonal => [(sx, 0), (0, sy)],
        Kind::Knight if dx.abs() == 2 => [(sx, 0), (sx, sy)],
        Kind::Knight => [(0, sy), (sx, sy)],
    }
}

/// Spread parameters of one fuel under the current weather.
#[derive(Clone, Copy)]
pub struct FuelSpread {
    pub bed: Option<FuelBed>,
    /// Midflame wind vector for this fuel, m/s.
    pub wind: [f64; 2],
    /// The ellipse on flat ground (the common case, cached).
    pub flat: Spread,
    /// Water (L/m²) needed to stop a front of this fuel per m/s of its spread rate: the
    /// configured litres per MW/m times the fuel's heat per unit area.
    pub water_per_ms: f32,
    /// Heat released per unit area by the flaming front, kJ/m²: times the front's speed in
    /// m/s it is Byram's fireline intensity in kW/m.
    pub heat_kj_m2: f32,
}

impl FuelSpread {
    pub const NONE: Self = Self {
        bed: None,
        wind: [0.0; 2],
        flat: Spread::NONE,
        water_per_ms: 0.0,
        heat_kj_m2: 0.0,
    };
}

/// Read-only inputs of a fire step.
pub struct Ctx<'a> {
    pub terrain: &'a Terrain,
    /// Indexed by fuel id.
    pub fuels: &'a [FuelSpread],
    pub seed: u64,
    pub cfg: &'a SimConfig,
    /// Wind at 10 m above ground as a vector, m/s.
    pub wind: [f32; 2],
}

/// The ellipses of the 5x5 cells around a source, filled in as edges ask for them.
type Near = [Option<Spread>; 25];

impl Ctx<'_> {
    /// The fire ellipse at a cell.
    pub fn spread(&self, cell: usize) -> Spread {
        let fuel = &self.fuels[self.terrain.fuel[cell] as usize];
        let Some(bed) = &fuel.bed else {
            return Spread::NONE;
        };
        let s = self.terrain.slope[cell];
        let mut spread = if s == [0.0, 0.0] {
            fuel.flat
        } else {
            bed.spread(fuel.wind, [s[0] as f64, s[1] as f64])
        };
        let noise = self.cfg.ros_noise;
        if noise > 0.0 {
            // Mean-one log-normal factor, fixed per cell for the whole episode.
            let z = rng::normal(self.seed, Stream::RosNoise, 0, cell as u32);
            spread.head *= libm::expf(noise * z - 0.5 * noise * noise);
        }
        spread
    }

    /// Target cell and rate (1/s, dry) of edge `k` out of `cell`, or None if the edge leaves
    /// the grid or crosses something that cannot burn. `near` holds the ellipses already
    /// computed around `cell`: the 16 edges of a source cross the same 17 cells, and on a
    /// slope an ellipse costs about 40 times what it does on flat ground.
    fn edge(&self, cell: usize, k: usize, near: &mut Near) -> Option<(usize, f32)> {
        let grid = self.terrain.grid;
        let (dx, dy) = DIRS[k];
        let target = grid.offset(cell, dx, dy)?;
        let length = ((dx * dx + dy * dy) as f32).sqrt();
        let (ux, uy) = (dx as f32 / length, dy as f32 / length);
        // `c` is the cell at offset (ox, oy) from `cell`.
        let mut slowness = |(ox, oy): (i32, i32), c: usize| {
            let spread =
                *near[((oy + 2) * 5 + ox + 2) as usize].get_or_insert_with(|| self.spread(c));
            let r = spread.rate(ux, uy);
            (r > 0.0).then_some(1.0 / r)
        };
        let sides = via(k);
        let beside =
            sides.map(|(ox, oy)| grid.offset(cell, ox, oy).expect("between two grid cells"));
        let (here, there) = ((0, 0), (dx, dy));
        // Seconds per metre, averaged over the cells the path crosses.
        let mean_slowness = match kind(k) {
            Kind::Axis => (slowness(here, cell)? + slowness(there, target)?) / 2.0,
            Kind::Diagonal => {
                // Fire squeezes through a corner unless both cells beside it cannot burn.
                if beside.iter().all(|&c| !self.terrain.burnable(c)) {
                    return None;
                }
                (slowness(here, cell)? + slowness(there, target)?) / 2.0
            }
            Kind::Knight => {
                (slowness(here, cell)?
                    + slowness(sides[0], beside[0])?
                    + slowness(sides[1], beside[1])?
                    + slowness(there, target)?)
                    / 4.0
            }
        };
        Some((target, 1.0 / (mean_slowness * length * grid.cell_m)))
    }
}

/// Water on the ground: litres per square metre per cell. It evaporates, and while it lasts it
/// slows the fire.
#[derive(Clone)]
pub struct Water {
    pub load: Vec<f32>,
    wet: Vec<u32>,
    /// The lowest critical load: what stops a fire of low intensity.
    crit: f32,
}

impl Water {
    pub fn new(cells: usize, crit_l_m2: f32) -> Self {
        Self {
            load: vec![0.0; cells],
            wet: Vec::new(),
            crit: crit_l_m2.max(1e-6), // zero would divide 0 by 0 on dry cells
        }
    }

    pub fn any(&self) -> bool {
        !self.wet.is_empty()
    }

    pub fn add(&mut self, cell: usize, litres_per_m2: f32) {
        if litres_per_m2 > 0.0 {
            if self.load[cell] == 0.0 {
                self.wet.push(cell as u32);
            }
            self.load[cell] += litres_per_m2;
        }
    }

    /// Spread-rate multiplier of a cell for a fire of low intensity: 1 when dry, 0 at or
    /// above the lowest critical load.
    pub fn mult(&self, cell: usize) -> f32 {
        self.mult_at(cell, self.crit)
    }

    /// The critical load (L/m²) for a front that needs `water` to be stopped: never below
    /// the lowest one.
    pub fn crit(&self, water: f32) -> f32 {
        self.crit.max(water)
    }

    /// Spread-rate multiplier of a cell against the critical load `crit`.
    pub fn mult_at(&self, cell: usize, crit: f32) -> f32 {
        (1.0 - self.load[cell] / crit).clamp(0.0, 1.0)
    }

    pub fn decay(&mut self, dt: f32, tau_s: f32) {
        let keep = libm::expf(-dt / tau_s);
        let load = &mut self.load;
        self.wet.retain(|&c| {
            let w = &mut load[c as usize];
            *w *= keep;
            if *w < 1e-4 {
                *w = 0.0;
            }
            *w > 0.0
        });
    }

    /// Multiplier along edge `k` from `cell` to `target` against the critical load `crit`: the
    /// wettest cell on the path decides.
    /// ponytail: exact would be a travel-time-weighted mean; the minimum is slightly too
    /// cautious for partly wet paths and exact for dry or soaked ones.
    fn along(&self, terrain: &Terrain, cell: usize, k: usize, target: usize, crit: f32) -> f32 {
        let mult = |c: usize| self.mult_at(c, crit);
        // The edge exists, so the cells beside its path are on the grid.
        let nx = terrain.grid.nx as i64;
        let beside = via(k).map(|(ox, oy)| (cell as i64 + ox as i64 + oy as i64 * nx) as usize);
        // Most paths are dry even while water lies somewhere on the map: nothing to slow.
        let dry = |c: usize| self.load[c] == 0.0;
        let axis = matches!(kind(k), Kind::Axis);
        if dry(cell) && dry(target) && (axis || (dry(beside[0]) && dry(beside[1]))) {
            return 1.0;
        }
        let m = mult(cell).min(mult(target));
        match kind(k) {
            Kind::Axis => m,
            Kind::Diagonal => {
                let pass = |c: usize| {
                    if terrain.burnable(c) { mult(c) } else { 0.0 }
                };
                m.min(pass(beside[0]).max(pass(beside[1])))
            }
            Kind::Knight => m.min(mult(beside[0])).min(mult(beside[1])),
        }
    }
}

/// A burning cell that can still set neighbours on fire.
#[derive(Clone)]
pub struct Source {
    pub cell: u32,
    /// Bit k set: edge k has not arrived yet.
    alive: u16,
    /// Seconds since any edge last made progress.
    stall_s: f32,
    /// Time at which this cell's own fuel is used up.
    burnout_t: f32,
    progress: [f32; 16],
    rate: [f32; 16],
    /// Water (L/m²) it takes to stop the front that set this cell alight: the more intense
    /// the front, the more (see `SimConfig::water_per_mw_l_m2`).
    water: f32,
    /// Fireline intensity of that front, kW/m.
    kw: f32,
}

/// What it takes to stop a burning cell that can still spread.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Front {
    /// Water (L/m²) its front needs beyond the lowest critical load (see `Water::crit`);
    /// infinite for a front too intense for water (`SimConfig::water_limit_kw_m`).
    pub water_l_m2: f32,
    /// Fireline intensity of the front that lit it, kW/m. Zero under FireSPIN.
    pub kw_m: f32,
}

/// Counters for reports. They never feed back into the fire.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FireStats {
    /// Burning cells that water kept from spreading until their flames died.
    pub stopped: u32,
    /// Sum and maximum of the fireline intensity (kW/m) of those cells.
    pub stopped_kw_sum: f64,
    pub stopped_kw_max: f32,
    /// Highest fireline intensity (kW/m) any cell was lit with.
    pub peak_kw: f32,
}

const NO_SLOT: u32 = u32::MAX;

/// The Rothermel model's own state: the burning cells and their edges.
#[derive(Clone)]
pub struct Edges {
    pub sources: Vec<Source>,
    slot: Vec<u32>,
    pending: Vec<(u32, f32)>,
    reverted: Vec<u32>,
}

impl Edges {
    fn new(cells: usize) -> Self {
        Self {
            sources: Vec::new(),
            slot: vec![NO_SLOT; cells],
            pending: Vec::new(),
            reverted: Vec::new(),
        }
    }

    /// Sets `cell` on fire at time `t`. `head_start` is how long before the end of the current
    /// step that was; its edges get that much progress. Returns false if the cell cannot burn.
    fn ignite(&mut self, out: &mut Cells, ctx: &Ctx, cell: usize, t: f32, head_start: f32) -> bool {
        let spread = ctx.spread(cell);
        if out.state[cell] != UNBURNED || spread.head <= 0.0 {
            return false;
        }
        let bed = ctx.fuels[ctx.terrain.fuel[cell] as usize]
            .bed
            .as_ref()
            .expect("burnable cell has a fuel bed");
        // How fast the front was moving when it arrived: the slowest apparent speed from any
        // neighbour that burned before. Judging by the edge that brought the fire would take
        // the flank of a wind-driven fire for a head, because edges run obliquely to the front.
        // A cell lit by hand or by a diagonal jump has no such neighbour and counts as a head.
        // ponytail: fixed at ignition; a later wind change does not update it.
        let grid = ctx.terrain.grid;
        let mut front_ms = spread.head;
        for &(dx, dy) in &DIRS[..8] {
            if let Some(n) = grid.offset(cell, dx, dy) {
                let since = t - out.t_ignite[n]; // NaN for cells that never burned
                if since > 0.0 {
                    let metres = ((dx * dx + dy * dy) as f32).sqrt() * grid.cell_m;
                    front_ms = front_ms.min(metres / since);
                }
            }
        }
        let fuel = &ctx.fuels[ctx.terrain.fuel[cell] as usize];
        let kw = fuel.heat_kj_m2 * front_ms;
        out.stats.peak_kw = out.stats.peak_kw.max(kw);
        let mut source = Source {
            // Beyond the limit no amount of water stops the front.
            water: if kw > ctx.cfg.water_limit_kw_m {
                f32::INFINITY
            } else {
                fuel.water_per_ms * front_ms
            },
            kw,
            cell: cell as u32,
            alive: 0,
            stall_s: 0.0,
            burnout_t: t + bed.residence_s() as f32 + ctx.terrain.grid.cell_m / spread.head,
            progress: [0.0; 16],
            rate: [0.0; 16],
        };
        let mut near: Near = [None; 25];
        near[12] = Some(spread);
        for k in 0..16 {
            if let Some((target, rate)) = ctx.edge(cell, k, &mut near)
                && out.state[target] == UNBURNED
            {
                source.alive |= 1 << k;
                source.rate[k] = rate;
                // ponytail: a head start never completes an edge within the step it was
                // created in; steps are sized so that this cap is not reached.
                source.progress[k] = (rate * head_start).min(0.999);
            }
        }
        out.state[cell] = BURNING;
        out.t_ignite[cell] = t;
        *out.n_ignited += 1;
        self.slot[cell] = self.sources.len() as u32;
        self.sources.push(source);
        true
    }

    /// Recomputes edge rates after the weather changed.
    fn refresh_rates(&mut self, ctx: &Ctx) {
        for s in &mut self.sources {
            let mut near: Near = [None; 25];
            for k in 0..16 {
                if s.alive & (1 << k) != 0 {
                    s.rate[k] = ctx
                        .edge(s.cell as usize, k, &mut near)
                        .map_or(0.0, |(_, rate)| rate);
                }
            }
        }
    }

    /// Advances the fire from `now` by `dt` seconds. `ember_s` is how long a source whose
    /// every edge is blocked keeps the ability to resume before its flames die.
    fn step(&mut self, out: &mut Cells, ctx: &Ctx, water: &Water, now: f32, dt: f32, ember_s: f32) {
        let nx = ctx.terrain.grid.nx as i64;
        let neighbour =
            |cell: usize, (dx, dy): (i32, i32)| (cell as i64 + dx as i64 + dy as i64 * nx) as usize;
        let wet = water.any();
        let t_end = now + dt;
        self.pending.clear();
        self.reverted.clear();

        let mut i = 0;
        while i < self.sources.len() {
            let cell = self.sources[i].cell as usize;
            let crit = water.crit(self.sources[i].water);
            let own = if wet { water.mult_at(cell, crit) } else { 1.0 };
            let mut progressed = false;
            let mut bits = self.sources[i].alive;
            while bits != 0 {
                let k = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                // In the grid: the edge was only armed if its target exists.
                let target = neighbour(cell, DIRS[k]);
                if out.state[target] != UNBURNED {
                    self.sources[i].alive &= !(1 << k); // someone else got there first
                    continue;
                }
                let mult = if wet {
                    water.along(ctx.terrain, cell, k, target, crit)
                } else {
                    1.0
                };
                let s = &mut self.sources[i];
                let rate = s.rate[k] * mult;
                if rate <= 0.0 {
                    continue;
                }
                progressed = true;
                let p = s.progress[k] + rate * dt;
                if p < 1.0 {
                    s.progress[k] = p;
                    continue;
                }
                s.alive &= !(1 << k);
                // Only the target ignites. The cells a knight move passes between catch fire
                // at their own arrival times a moment later; igniting them here would hand
                // the fire half a cell of sideways travel for free at every step.
                self.pending.push((target as u32, (p - 1.0) / rate));
            }

            let s = &mut self.sources[i];
            let fate = if s.alive == 0 {
                (t_end >= s.burnout_t).then_some(BURNED)
            } else if progressed {
                s.stall_s = 0.0;
                None
            } else {
                s.stall_s += dt;
                // Flames die. A cell knocked down by water before its fuel was used up can
                // burn again later; anything else is spent.
                let knocked_down = own <= 0.0 && t_end < s.burnout_t;
                let dead = s.stall_s >= ember_s;
                if dead {
                    // An armed edge has a positive rate, so only water stalls a source.
                    let stats = &mut *out.stats;
                    stats.stopped += 1;
                    stats.stopped_kw_sum += s.kw as f64;
                    stats.stopped_kw_max = stats.stopped_kw_max.max(s.kw);
                }
                dead.then_some(if knocked_down { UNBURNED } else { BURNED })
            };
            let Some(new_state) = fate else {
                i += 1;
                continue;
            };
            self.sources.swap_remove(i); // the moved-in source is processed next: no `i += 1`
            if let Some(moved) = self.sources.get(i) {
                self.slot[moved.cell as usize] = i as u32;
            }
            self.slot[cell] = NO_SLOT;
            out.state[cell] = new_state;
            if new_state == UNBURNED {
                out.t_ignite[cell] = f32::NAN;
                *out.n_ignited -= 1;
                self.reverted.push(cell as u32);
            }
        }

        // Burning neighbours of a knocked-down cell get their edge towards it back.
        for r in 0..self.reverted.len() {
            let cell = self.reverted[r] as usize;
            for (k, &(dx, dy)) in DIRS.iter().enumerate() {
                let Some(from) = ctx.terrain.grid.offset(cell, -dx, -dy) else {
                    continue;
                };
                let slot = self.slot[from];
                if slot == NO_SLOT {
                    continue;
                }
                if let Some((_, rate)) = ctx.edge(from, k, &mut [None; 25]) {
                    let s = &mut self.sources[slot as usize];
                    s.alive |= 1 << k;
                    s.rate[k] = rate;
                    s.progress[k] = 0.0;
                }
            }
        }

        // New ignitions. Several edges may reach one cell in the same step: the earliest wins,
        // whatever order the sources were visited in.
        let mut pending = std::mem::take(&mut self.pending);
        pending.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.total_cmp(&a.1)));
        pending.dedup_by_key(|p| p.0);
        for &(cell, head_start) in &pending {
            self.ignite(out, ctx, cell as usize, t_end - head_start, head_start);
        }
        self.pending = pending;
    }

    /// Order-independent digest of the sources, for state hashes.
    fn digest(&self) -> u64 {
        self.sources.iter().fold(0u64, |acc, s| {
            let mut h = 0xcbf2_9ce4_8422_2325u64 ^ s.cell as u64;
            let mut mix = |v: u32| h = (h ^ v as u64).wrapping_mul(0x0000_0100_0000_01b3);
            mix(s.alive as u32);
            mix(s.stall_s.to_bits());
            s.progress.iter().for_each(|p| mix(p.to_bits()));
            mix(s.water.to_bits());
            acc.wrapping_add(h)
        })
    }
}

/// The layers every fire model writes and everything else reads.
pub struct Cells<'a> {
    pub state: &'a mut [u8],
    pub t_ignite: &'a mut [f32],
    pub n_ignited: &'a mut u32,
    pub stats: &'a mut FireStats,
}

/// Which fire model a world runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FireModel {
    /// Rothermel spread rates on a 16-direction travel-time grid (this module).
    #[default]
    Rothermel,
    /// The particle model of the thesis (see `spin`).
    Firespin,
}

#[derive(Clone)]
enum Model {
    Rothermel(Edges),
    Spin(Spin),
}

/// The fire: the shared per-cell layers plus the state of whichever model drives them.
/// Tasks, planners and the viewer read only the public fields and the methods below.
#[derive(Clone)]
pub struct FireState {
    /// UNBURNED, BURNING or BURNED per cell.
    pub state: Vec<u8>,
    /// Simulated time of ignition in seconds; NaN where the cell is not burning or burned.
    pub t_ignite: Vec<f32>,
    /// Cells currently burning or burned.
    pub n_ignited: u32,
    pub stats: FireStats,
    model: Model,
}

impl FireState {
    pub fn new(cells: usize, model: FireModel) -> Self {
        Self {
            state: vec![UNBURNED; cells],
            t_ignite: vec![f32::NAN; cells],
            n_ignited: 0,
            stats: FireStats::default(),
            model: match model {
                FireModel::Rothermel => Model::Rothermel(Edges::new(cells)),
                FireModel::Firespin => Model::Spin(Spin::new(cells)),
            },
        }
    }

    fn parts(&mut self) -> (Cells<'_>, &mut Model) {
        let cells = Cells {
            state: &mut self.state,
            t_ignite: &mut self.t_ignite,
            n_ignited: &mut self.n_ignited,
            stats: &mut self.stats,
        };
        (cells, &mut self.model)
    }

    /// Sets `cell` on fire at time `t`. `head_start` is how long before the end of the current
    /// step that was. Returns false if the cell cannot burn or is not unburned.
    pub fn ignite(&mut self, ctx: &Ctx, cell: usize, t: f32, head_start: f32) -> bool {
        match self.parts() {
            (mut out, Model::Rothermel(m)) => m.ignite(&mut out, ctx, cell, t, head_start),
            (mut out, Model::Spin(m)) => m.ignite(&mut out, ctx, cell, t),
        }
    }

    /// Call after the wind changed.
    pub fn refresh_rates(&mut self, ctx: &Ctx) {
        if let Model::Rothermel(m) = &mut self.model {
            m.refresh_rates(ctx);
        }
    }

    /// Into how many equal parts a fire step of `dt` seconds must be split. `rate_bound` is
    /// the fastest any Rothermel edge can advance (1/s).
    pub fn substeps(&self, ctx: &Ctx, dt: f32, rate_bound: f32) -> u32 {
        let steps = match &self.model {
            // No edge may advance more than half its length per sub-step.
            Model::Rothermel(_) => rate_bound * dt / 0.5,
            Model::Spin(_) => dt / ctx.cfg.spin.dt_s - 1e-3,
        };
        steps.ceil().max(1.0) as u32
    }

    /// Advances the fire from `now` by `dt` seconds. `ember_s` is how long a burning cell
    /// that water keeps from spreading can still resume before its flames die.
    pub fn step(&mut self, ctx: &Ctx, water: &Water, now: f32, dt: f32, ember_s: f32) {
        match self.parts() {
            (mut out, Model::Rothermel(m)) => m.step(&mut out, ctx, water, now, dt, ember_s),
            (mut out, Model::Spin(m)) => m.step(&mut out, ctx, water, now, dt, ember_s),
        }
    }

    /// The burning cells, in no particular order.
    pub fn burning(&self) -> Box<dyn Iterator<Item = u32> + '_> {
        match &self.model {
            Model::Rothermel(m) => Box::new(m.sources.iter().map(|s| s.cell)),
            Model::Spin(m) => Box::new(m.burning.iter().map(|b| b.cell)),
        }
    }

    /// The front of a burning cell that can still set neighbours alight; None for any other
    /// cell, also for one that only burns out. Water on such a cell is what stops the fire.
    pub fn front(&self, cell: usize) -> Option<Front> {
        match &self.model {
            Model::Rothermel(m) => {
                let slot = m.slot[cell];
                let s = m.sources.get(slot as usize)?; // NO_SLOT is past the end
                (s.alive != 0).then_some(Front {
                    water_l_m2: s.water,
                    kw_m: s.kw,
                })
            }
            Model::Spin(_) => (self.state[cell] == BURNING).then_some(Front {
                water_l_m2: 0.0,
                kw_m: 0.0,
            }),
        }
    }

    pub fn n_burning(&self) -> usize {
        match &self.model {
            Model::Rothermel(m) => m.sources.len(),
            Model::Spin(m) => m.burning.len(),
        }
    }

    /// True when nothing burns and nothing can start burning again by itself.
    pub fn is_out(&self) -> bool {
        match &self.model {
            Model::Rothermel(m) => m.sources.is_empty(),
            Model::Spin(m) => m.burning.is_empty() && m.n_particles() == 0,
        }
    }

    /// Heat-carrying particles in flight (FireSPIN only), for the viewer: position in metres
    /// and whether it is a radiation particle.
    pub fn particles(&self) -> Box<dyn Iterator<Item = ([f32; 2], bool)> + '_> {
        match &self.model {
            Model::Rothermel(_) => Box::new(std::iter::empty()),
            Model::Spin(m) => Box::new(m.particles()),
        }
    }

    /// Order-independent digest of the model's own state, for state hashes.
    pub fn digest(&self) -> u64 {
        match &self.model {
            Model::Rothermel(m) => m.digest(),
            Model::Spin(m) => m.digest(),
        }
    }
}
