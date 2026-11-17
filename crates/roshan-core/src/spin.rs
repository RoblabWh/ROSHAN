//! FireSPIN, the particle fire model of the thesis, ported from the old ROSHAN
//! (`src/firespin`: `firemodel_gridmap.cpp`, `firemodel_firecell.cpp`, `particles/*.cpp`).
//!
//! A burning cell emits two kinds of heat-carrying particles. *Convection* particles are
//! carried and scattered by the wind (a Langevin equation for their velocity); *radiation*
//! particles sit on a slowly growing ring around the point they were emitted from. A particle
//! loses heat exponentially and vanishes below a limit or outside the map. Every step, an
//! unburned cell with at least one particle in it is heated for the length of the step, and
//! it ignites once its heating time reaches the ignition delay of its land-cover class. A
//! burning cell burns for the burning duration of its class and then is burned out.
//!
//! The model knows wind, but not slope or fuel moisture. Which cells can burn is decided by
//! the fuel map, as for the Rothermel model; how they burn by the CLC+ land-cover class, or by
//! the "generic" class of the configuration where a scenario has no land cover.
//!
//! Differences from the old code, on purpose:
//! - Emitted convection particles keep their position inside the cell (the old constructor
//!   truncated it to whole metres).
//! - Classes without particles emit nothing (the old code divided by their particle count).
//! - A land-cover value outside the class table counts as generic instead of indexing past it.
//! - Random numbers are counter-based per cell and per particle, so an episode is a function
//!   of its seed alone and does not depend on earlier episodes or on evaluation order.
//! - Water is the shared water layer instead of a "flooded" cell state: a wet cell heats
//!   more slowly, a soaked one not at all; particles in a soaked cell are lost; a soaked
//!   burning cell emits nothing and goes out after `ember_s`, able to burn again later.
//!
//! Emission as in the old code: with the interval `(burning duration - 1) / particles of the
//! class`, cut to whole seconds, a burning cell emits one particle of a kind whenever the
//! whole seconds it has burned reach a multiple of that interval.

use serde::{Deserialize, Serialize};

use crate::fire::{BURNED, BURNING, Cells, Ctx, UNBURNED, Water};
use crate::rng::{self, Stream};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpinConfig {
    /// Time step of the particle model, seconds. The old ROSHAN used 0.1; 0.5 gives the same
    /// spread curve within 1 % at a fifth of the cost (measurements in docs/systems.md).
    pub dt_s: f32,
    pub emit_convective: bool,
    pub emit_radiation: bool,
    /// Heat a convection particle starts with, and the limit below which it vanishes.
    pub conv_y_st: f32,
    pub conv_y_lim: f32,
    /// Time constant with which a convection particle loses heat, seconds.
    pub conv_tau_mem_s: f32,
    /// Scale between a convection particle's velocity and its movement.
    pub conv_fl: f32,
    /// Kolmogorov constant of the Langevin equation.
    pub conv_c0: f32,
    /// Turbulent length scale, metres.
    pub lt_m: f32,
    /// Turbulence intensity: velocity fluctuation = `wind_a` x wind speed.
    pub wind_a: f32,
    pub rad_y_st: f32,
    pub rad_y_lim: f32,
    /// Burning duration and ignition delay (seconds) of cells without a land-cover class.
    pub generic_burn_s: f32,
    pub generic_ignite_s: f32,
}

impl Default for SpinConfig {
    /// The values of the old ROSHAN's configuration, except for the time step.
    fn default() -> Self {
        Self {
            dt_s: 0.5,
            emit_convective: true,
            emit_radiation: true,
            conv_y_st: 1.0,
            conv_y_lim: 0.2,
            conv_tau_mem_s: 10.0,
            conv_fl: 0.15,
            conv_c0: 1.98,
            lt_m: 80.0,
            wind_a: 0.314,
            rad_y_st: 1.0,
            rad_y_lim: 0.165,
            generic_burn_s: 120.0,
            generic_ignite_s: 100.0,
        }
    }
}

/// How one land-cover class burns.
#[derive(Clone, Copy)]
pub struct Class {
    pub burn_s: f32,
    pub ignite_s: f32,
    /// Particle counts of the class; they set the emission interval (see the module text).
    pub convection: u32,
    pub radiation: u32,
    /// Range of the radiation length, metres.
    pub lr_m: [f32; 2],
    /// Mean and standard deviation of the radiation front speed, m/s.
    pub sf_ms: [f32; 2],
}

const fn class(burn_s: f32, ignite_s: f32, convection: u32, radiation: u32) -> Class {
    Class {
        burn_s,
        ignite_s,
        convection,
        radiation,
        lr_m: [9.0, 10.0],
        sf_ms: [0.1, 0.025],
    }
}

const NO_FIRE: Class = class(0.0, 0.0, 0, 0);

/// Indexed by CLC+ Backbone class (1..=11), with the constants of the old `cell_classes/*.h`.
/// Entry 0 is the generic class; its two durations come from the configuration.
#[rustfmt::skip]
pub const CLASSES: [Class; 12] = [
    Class { lr_m: [5.0, 50.0], ..class(0.0, 0.0, 10, 4) },          // generic
    Class { sf_ms: [0.1, 0.0075], ..class(15000.0, 750.0, 1000, 55) }, // 1 sealed
    class(3600.0, 300.0, 80, 6),  // 2 woody needle-leaved trees
    class(3600.0, 270.0, 60, 6),  // 3 woody broadleaved deciduous trees
    class(3600.0, 330.0, 100, 6), // 4 woody broadleaved evergreen trees
    class(720.0, 120.0, 40, 4),   // 5 low-growing woody plants
    class(360.0, 100.0, 20, 2),   // 6 permanent herbaceous
    class(360.0, 100.0, 20, 2),   // 7 periodically herbaceous
    class(240.0, 100.0, 10, 3),   // 8 lichens and mosses
    NO_FIRE,                      // 9 non- and sparsely vegetated
    NO_FIRE,                      // 10 water
    NO_FIRE,                      // 11 snow and ice
];

/// The class of a cell, with its burning duration and ignition delay jittered per cell as in
/// the old code: each, independently, stays or changes by 10 to 20 % up or down.
fn class_of(ctx: &Ctx, cell: usize) -> Class {
    let code = ctx
        .terrain
        .landcover
        .as_ref()
        .map_or(0, |l| l[cell] as usize);
    let mut c = match CLASSES.get(code) {
        Some(&c) if code != 0 => c,
        _ => Class {
            burn_s: ctx.cfg.spin.generic_burn_s,
            ignite_s: ctx.cfg.spin.generic_ignite_s,
            ..CLASSES[0]
        },
    };
    let jitter = |which: u64| {
        let u = |draw: u64| rng::uniform(ctx.seed, Stream::Spin, which * 2 + draw, cell as u32);
        let sign = ((u(0) * 3.0) as i32).min(2) - 1;
        1.0 + sign as f32 * (0.1 + 0.1 * u(1))
    };
    c.burn_s *= jitter(0);
    c.ignite_s *= jitter(1);
    c
}

#[derive(Clone)]
pub struct Burn {
    pub cell: u32,
    /// Seconds burned so far and seconds of fuel left.
    age_s: f32,
    left_s: f32,
    /// Seconds for which water has kept this cell from burning.
    soaked_s: f32,
    /// Emission interval in whole seconds for convection and radiation; 0 = never.
    every_s: [u32; 2],
    /// Radiation parameters of the class.
    lr_m: [f32; 2],
    sf_ms: [f32; 2],
}

#[derive(Clone)]
struct Convection {
    pos: [f32; 2],
    vel: [f32; 2],
    heat: f32,
    key: u64,
    age: u32,
}

#[derive(Clone)]
struct Radiation {
    origin: [f32; 2],
    pos: [f32; 2],
    radius: f32,
    angle: f32,
    heat: f32,
    /// One over the time constant of the heat loss, 1/s.
    decay: f32,
    sf_ms: [f32; 2],
    key: u64,
    age: u32,
}

#[derive(Clone)]
pub struct Spin {
    pub burning: Vec<Burn>,
    convection: Vec<Convection>,
    radiation: Vec<Radiation>,
    /// Seconds each cell has been heated so far.
    heated_s: Vec<f32>,
    /// Ignition delay per cell, filled in when a cell is first heated (0 = not yet known,
    /// infinite = cannot burn). It follows from the seed and the map alone.
    ignite_s: Vec<f32>,
    /// Step in which a cell was last heated, so that several particles count once.
    stamp: Vec<u64>,
    step: u64,
    lit: Vec<u32>,
}

/// Seed of the per-particle random streams, kept apart from everything keyed by the world
/// seed directly.
const PARTICLE_SALT: u64 = 0x5350_494e_5350_494e;

impl Spin {
    pub fn new(cells: usize) -> Self {
        Self {
            burning: Vec::new(),
            convection: Vec::new(),
            radiation: Vec::new(),
            heated_s: vec![0.0; cells],
            ignite_s: vec![0.0; cells],
            stamp: vec![0; cells],
            step: 0,
            lit: Vec::new(),
        }
    }

    pub fn n_particles(&self) -> usize {
        self.convection.len() + self.radiation.len()
    }

    pub fn particles(&self) -> impl Iterator<Item = ([f32; 2], bool)> + '_ {
        let convection = self.convection.iter().map(|p| (p.pos, false));
        convection.chain(self.radiation.iter().map(|p| (p.pos, true)))
    }

    pub fn ignite(&mut self, out: &mut Cells, ctx: &Ctx, cell: usize, t: f32) -> bool {
        let class = class_of(ctx, cell);
        if out.state[cell] != UNBURNED || !ctx.terrain.burnable(cell) || class.burn_s <= 0.0 {
            return false;
        }
        let every = |particles: u32| {
            if particles == 0 {
                0
            } else {
                (((class.burn_s - 1.0) / particles as f32) as u32).max(1)
            }
        };
        self.burning.push(Burn {
            cell: cell as u32,
            age_s: 0.0,
            left_s: class.burn_s,
            soaked_s: 0.0,
            every_s: [every(class.convection), every(class.radiation)],
            lr_m: class.lr_m,
            sf_ms: class.sf_ms,
        });
        out.state[cell] = BURNING;
        out.t_ignite[cell] = t;
        *out.n_ignited += 1;
        true
    }

    pub fn step(
        &mut self,
        out: &mut Cells,
        ctx: &Ctx,
        water: &Water,
        now: f32,
        dt: f32,
        ember_s: f32,
    ) {
        let cfg = &ctx.cfg.spin;
        let grid = ctx.terrain.grid;
        let wet = water.any();
        self.step += 1;
        let Self {
            burning,
            convection,
            radiation,
            heated_s,
            ignite_s,
            stamp,
            step,
            lit,
        } = self;
        let step = *step;
        lit.clear();

        // The cell a particle is in, if it is still on the map and not lost to water.
        let cell_at = |pos: [f32; 2]| {
            let (x, y) = (pos[0] / grid.cell_m, pos[1] / grid.cell_m);
            let inside = x >= 0.0 && y >= 0.0 && x < grid.nx as f32 && y < grid.ny as f32;
            let cell = inside.then(|| grid.idx(x as u32, y as u32))?;
            (!wet || water.mult(cell) > 0.0).then_some(cell)
        };
        // A particle heats the cell it is in; several particles in one step count once.
        let mut heat = |cell: usize| {
            if out.state[cell] != UNBURNED || stamp[cell] == step {
                return;
            }
            if ignite_s[cell] == 0.0 {
                let class = class_of(ctx, cell);
                let burns = ctx.terrain.burnable(cell) && class.burn_s > 0.0;
                ignite_s[cell] = if burns { class.ignite_s } else { f32::INFINITY };
            }
            stamp[cell] = step;
            heated_s[cell] += dt * if wet { water.mult(cell) } else { 1.0 };
            if heated_s[cell] >= ignite_s[cell] {
                lit.push(cell as u32);
            }
        };

        let turbulence = cfg.wind_a * ctx.wind[0].hypot(ctx.wind[1]);
        let relax = (2.0 + 3.0 * cfg.conv_c0) / 4.0 * turbulence / cfg.lt_m;
        let kick = (cfg.conv_c0 * turbulence.powi(3) / cfg.lt_m * dt).sqrt();
        convection.retain_mut(|p| {
            p.age += 1;
            p.heat -= p.heat / cfg.conv_tau_mem_s * dt;
            let n = rng::normal2(p.key, Stream::Spin, p.age as u64, 0);
            for (i, n) in n.into_iter().enumerate() {
                p.vel[i] += -relax * (p.vel[i] - ctx.wind[i]) * dt + kick * n;
                p.pos[i] += cfg.conv_fl * p.vel[i] * dt;
            }
            match cell_at(p.pos) {
                Some(cell) if p.heat >= cfg.conv_y_lim => heat(cell),
                _ => return false,
            }
            true
        });
        radiation.retain_mut(|p| {
            p.age += 1;
            let n = rng::normal2(p.key, Stream::Spin, p.age as u64, 0);
            p.angle += std::f32::consts::TAU * n[0];
            p.radius += p.sf_ms[0] * dt + p.sf_ms[1] * dt.sqrt() * n[1];
            p.pos = [
                p.origin[0] + p.radius * libm::cosf(p.angle),
                p.origin[1] + p.radius * libm::sinf(p.angle),
            ];
            p.heat -= p.heat * p.decay * dt;
            match cell_at(p.pos) {
                Some(cell) if p.heat >= cfg.rad_y_lim => heat(cell),
                _ => return false,
            }
            true
        });

        // Cells whose heating time is up ignite; like the old code they burn within this step.
        lit.sort_unstable(); // order of `burning` must not depend on the order of particles
        for &cell in lit.iter() {
            // Rebuilt from the parts because `self` is taken apart above.
            let mut spin = Spin {
                burning: std::mem::take(burning),
                ..Spin::new(0)
            };
            spin.ignite(out, ctx, cell as usize, now + dt);
            *burning = spin.burning;
        }

        let mut i = 0;
        while i < burning.len() {
            let b = &mut burning[i];
            let cell = b.cell as usize;
            if wet && water.mult(cell) <= 0.0 {
                b.soaked_s += dt;
                if b.soaked_s >= ember_s {
                    // ponytail: a knocked-down cell forgets how much of its fuel had burned
                    // and can burn its full duration again; track `left` per cell if that
                    // matters.
                    out.state[cell] = UNBURNED;
                    out.t_ignite[cell] = f32::NAN;
                    *out.n_ignited -= 1;
                    heated_s[cell] = 0.0;
                    burning.swap_remove(i);
                } else {
                    i += 1;
                }
                continue;
            }
            b.soaked_s = 0.0;
            let before = b.age_s as u32;
            b.age_s += dt;
            b.left_s -= dt;
            let second = b.age_s as u32;
            let (x, y) = grid.xy(cell);
            for kind in 0..2 {
                let enabled = [cfg.emit_convective, cfg.emit_radiation][kind];
                if !enabled || b.every_s[kind] == 0 {
                    continue;
                }
                // One particle per whole second reached in this step that is a multiple of
                // the interval.
                for j in (before + 1..=second).filter(|s| s.is_multiple_of(b.every_s[kind])) {
                    let j = j as u64;
                    let entity = cell as u64 | (kind as u64) << 32 | j << 33;
                    let key = rng::derive(ctx.seed ^ PARTICLE_SALT, step, entity);
                    let u = |i| rng::uniform(key, Stream::Spin, 0, i);
                    let pos = [
                        (x as f32 + u(0)) * grid.cell_m,
                        (y as f32 + u(1)) * grid.cell_m,
                    ];
                    if kind == 0 {
                        convection.push(Convection {
                            pos,
                            vel: [0.0; 2],
                            heat: cfg.conv_y_st,
                            key,
                            age: 0,
                        });
                        continue;
                    }
                    let speed =
                        (b.sf_ms[0] + b.sf_ms[1] * rng::normal(key, Stream::Spin, 0, 2)).abs();
                    let length = b.lr_m[0] + (b.lr_m[1] - b.lr_m[0]) * u(3);
                    // The old code's cap on the lifetime of very slow particles.
                    let tau = if speed > 0.00925 {
                        length / speed
                    } else {
                        2703.0
                    };
                    radiation.push(Radiation {
                        origin: pos,
                        pos,
                        radius: 0.0,
                        angle: 0.0,
                        heat: cfg.rad_y_st,
                        decay: 1.0 / tau,
                        sf_ms: b.sf_ms,
                        key,
                        age: 0,
                    });
                }
            }
            if b.left_s <= 0.0 {
                out.state[cell] = BURNED;
                burning.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Digest that does not depend on the order of the lists.
    pub fn digest(&self) -> u64 {
        fn hash(values: &[u64]) -> u64 {
            values.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &v| {
                (h ^ v).wrapping_mul(0x0000_0100_0000_01b3)
            })
        }
        let bits = |p: [f32; 2]| (p[0].to_bits() as u64) << 32 | p[1].to_bits() as u64;
        let mut sum = hash(&[self.step]);
        for (cell, h) in self.heated_s.iter().enumerate() {
            if *h != 0.0 {
                sum = sum.wrapping_add(hash(&[cell as u64, h.to_bits() as u64]));
            }
        }
        for b in &self.burning {
            let state = (b.age_s.to_bits() as u64) << 32 | b.left_s.to_bits() as u64;
            sum = sum.wrapping_add(hash(&[b.cell as u64, state, b.soaked_s.to_bits() as u64]));
        }
        for p in &self.convection {
            sum = sum.wrapping_add(hash(&[p.key, p.age as u64, bits(p.pos), bits(p.vel)]));
        }
        for p in &self.radiation {
            sum = sum.wrapping_add(hash(&[p.key, p.age as u64, bits(p.pos)]));
        }
        sum
    }
}
