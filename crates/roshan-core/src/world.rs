//! One simulated environment: terrain, weather, fire, water and the clock.
//!
//! `World` is `Clone`, and a clone shares the terrain and weather, so snapshotting a world for
//! a rollout copies only the dynamic layers.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::fire::{Ctx, FireModel, FireState, FuelSpread, Water};
use crate::fuel::SB40;
use crate::rothermel::{FuelBed, Moisture, Spread};
use crate::spin::SpinConfig;
use crate::terrain::Terrain;

/// The read-only inputs of the fire model, borrowed field by field so that `self.fire` can be
/// changed at the same time.
macro_rules! ctx {
    ($w:expr) => {{
        let wind = $w.weather.wind[$w.wind_index];
        Ctx {
            terrain: &$w.terrain,
            fuels: &$w.fuels,
            seed: $w.seed,
            cfg: &$w.cfg,
            wind: [wind.speed_ms * wind.to[0], wind.speed_ms * wind.to[1]],
        }
    }};
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SimConfig {
    /// Length of one tick in seconds. Time is counted in whole ticks, never accumulated.
    pub dt_s: f32,
    /// The fire advances once every this many ticks.
    pub fire_every: u32,
    /// Lowest water load (litres per square metre) that stops fire: what a front of low
    /// intensity needs (Stechishen & Little 1971: 0.2-0.6 below about 360 kW/m).
    pub water_crit_l_m2: f32,
    /// Water (litres per square metre) needed per MW/m of fireline intensity to stop a front,
    /// in grass and in every other fuel (Loane & Gould 1986). Rothermel model only; FireSPIN
    /// knows no intensity and uses `water_crit_l_m2`.
    pub water_per_mw_grass_l_m2: f32,
    pub water_per_mw_l_m2: f32,
    /// The same for litter under trees (the timber models TL and TU): pine-needle litter
    /// takes several times the water of eucalypt forest (Alexander 2000).
    pub water_per_mw_timber_l_m2: f32,
    /// Fireline intensity (kW/m) above which water does not stop a front, however much of
    /// it: fires are held to be beyond control at the head above about 4 MW/m (Alexander
    /// 2000). Rothermel model only.
    pub water_limit_kw_m: f32,
    /// Time constant with which water on the ground disappears, seconds. An assumption: no
    /// measurement was found.
    pub water_tau_s: f32,
    /// How long a fully blocked burning cell can still resume before its flames die, seconds.
    /// A model parameter without a measured counterpart.
    pub ember_s: f32,
    /// Factor from the 20-ft wind to the wind at the fuel under a tree canopy (the timber
    /// models TU and TL): 0.1 in dense stands, 0.2 in open ones (Andrews 2012).
    pub sheltered_waf: f32,
    /// Standard deviation of the log-normal per-cell spread-rate noise; 0 is deterministic.
    pub ros_noise: f32,
    /// Which fire model runs: "rothermel" or "firespin".
    pub fire_model: FireModel,
    /// Parameters of the FireSPIN model.
    pub spin: SpinConfig,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            dt_s: 0.1,
            fire_every: 20,
            water_crit_l_m2: 0.4,
            water_per_mw_grass_l_m2: 0.35,
            water_per_mw_l_m2: 0.6,
            water_per_mw_timber_l_m2: 2.65,
            water_limit_kw_m: 4000.0,
            water_tau_s: 600.0,
            ember_s: 60.0,
            sheltered_waf: 0.2,
            ros_noise: 0.0,
            fire_model: FireModel::Rothermel,
            spin: SpinConfig::default(),
        }
    }
}

impl SimConfig {
    /// Refuses settings no world can run with (`World::new` would panic on them).
    pub fn check(&self) -> Result<(), String> {
        let positive = [
            ("dt_s", self.dt_s),
            ("fire_every", self.fire_every as f32),
            ("spin.dt_s", self.spin.dt_s),
            ("spin.lt_m", self.spin.lt_m),
            ("spin.conv_tau_mem_s", self.spin.conv_tau_mem_s),
        ];
        match positive.iter().find(|(_, v)| !(*v > 0.0 && v.is_finite())) {
            Some((name, v)) => Err(format!("{name} must be above zero, got {v}")),
            None => Ok(()),
        }
    }
}

/// Wind from `t_s` onward: speed at 10 m above ground and the unit vector it blows towards
/// (grid coordinates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Wind {
    pub t_s: f32,
    pub speed_ms: f32,
    pub to: [f32; 2],
}

#[derive(Clone, Debug)]
pub struct Weather {
    pub moisture: Moisture,
    /// Wind schedule sorted by time; the first entry applies from the start.
    pub wind: Vec<Wind>,
}

impl Weather {
    pub fn steady(moisture: Moisture, speed_ms: f32, to: [f32; 2]) -> Self {
        Self {
            moisture,
            wind: vec![Wind {
                t_s: 0.0,
                speed_ms,
                to,
            }],
        }
    }

    fn wind_index(&self, t_s: f32) -> usize {
        self.wind.iter().rposition(|w| w.t_s <= t_s).unwrap_or(0)
    }
}

/// The drones, as parallel arrays. Positions are in metres (x = column * cell size).
#[derive(Clone, Default)]
pub struct Drones {
    pub pos: Vec<[f32; 2]>,
    pub vel: Vec<[f32; 2]>,
    /// Litres of water on board (used by the allocate task).
    pub water: Vec<f32>,
}

#[derive(Clone)]
pub struct World {
    pub cfg: SimConfig,
    pub terrain: Arc<Terrain>,
    pub weather: Arc<Weather>,
    pub seed: u64,
    pub tick: u64,
    pub fire: FireState,
    pub water: Water,
    pub drones: Drones,
    wind_index: usize,
    /// Per fuel id, under the current wind.
    fuels: Vec<FuelSpread>,
    /// Upper bound on any edge rate (1/s); sizes the fire sub-steps. It depends only on the
    /// map and the weather, never on where the fire is, so sub-stepping cannot couple distant
    /// parts of the map.
    rate_bound: f32,
}

impl World {
    pub fn new(terrain: Arc<Terrain>, weather: Weather, cfg: SimConfig, seed: u64) -> Self {
        assert!(
            !weather.wind.is_empty(),
            "weather needs at least one wind entry"
        );
        assert!(
            cfg.dt_s > 0.0 && cfg.fire_every > 0 && cfg.spin.dt_s > 0.0,
            "tick length and fire interval must be positive"
        );
        let cells = terrain.grid.len();
        let mut world = Self {
            cfg,
            fire: FireState::new(cells, cfg.fire_model),
            water: Water::new(cells, cfg.water_crit_l_m2),
            terrain,
            weather: Arc::new(weather),
            seed,
            tick: 0,
            drones: Drones::default(),
            wind_index: 0,
            fuels: Vec::new(),
            rate_bound: 0.0,
        };
        world.apply_wind();
        world
    }

    /// Simulated time in seconds.
    pub fn time_s(&self) -> f32 {
        self.tick as f32 * self.cfg.dt_s
    }

    pub fn wind(&self) -> Wind {
        self.weather.wind[self.wind_index]
    }

    /// The fire ellipse the model uses at a cell.
    pub fn spread(&self, x: u32, y: u32) -> Spread {
        ctx!(self).spread(self.terrain.grid.idx(x, y))
    }

    /// Sets a cell on fire now. Returns false if it cannot burn or already burns.
    pub fn ignite(&mut self, x: u32, y: u32) -> bool {
        let (cell, t) = (self.terrain.grid.idx(x, y), self.time_s());
        self.fire.ignite(&ctx!(self), cell, t, 0.0)
    }

    /// Adds water to a cell, in litres per square metre.
    pub fn add_water(&mut self, x: u32, y: u32, litres_per_m2: f32) {
        self.water.add(self.terrain.grid.idx(x, y), litres_per_m2);
    }

    /// Advances the world by one tick.
    pub fn tick(&mut self) {
        self.tick += 1;
        let dt = self.cfg.dt_s;
        for (p, v) in self.drones.pos.iter_mut().zip(&self.drones.vel) {
            *p = [p[0] + v[0] * dt, p[1] + v[1] * dt];
        }
        if self.tick.is_multiple_of(self.cfg.fire_every as u64) {
            self.fire_step();
        }
    }

    /// Advances the world by (at least) `seconds`.
    pub fn run(&mut self, seconds: f32) {
        for _ in 0..(seconds / self.cfg.dt_s).ceil() as u64 {
            self.tick();
        }
    }

    /// Share of the burnable cells that are burning or burned.
    pub fn burned_fraction(&self) -> f32 {
        self.fire.n_ignited as f32 / self.terrain.burnable_cells.max(1) as f32
    }

    /// Digest of the whole dynamic state. Equal hashes mean bit-identical worlds.
    pub fn state_hash(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64 ^ self.tick;
        let mut mix = |v: u32| h = (h ^ v as u64).wrapping_mul(0x0000_0100_0000_01b3);
        self.fire.state.iter().for_each(|&s| mix(s as u32));
        self.fire.t_ignite.iter().for_each(|t| mix(t.to_bits()));
        self.water.load.iter().for_each(|w| mix(w.to_bits()));
        h ^ self.fire.digest()
    }

    fn fire_step(&mut self) {
        let dt = self.cfg.dt_s * self.cfg.fire_every as f32;
        let start = self.time_s() - dt;
        let index = self.weather.wind_index(start);
        if index != self.wind_index {
            self.wind_index = index;
            self.apply_wind();
        }
        self.water.decay(dt, self.cfg.water_tau_s);

        let ctx = ctx!(self);
        let steps = self.fire.substeps(&ctx, dt, self.rate_bound);
        let sub = dt / steps as f32;
        for i in 0..steps {
            self.fire.step(
                &ctx,
                &self.water,
                start + i as f32 * sub,
                sub,
                self.cfg.ember_s,
            );
        }
    }

    /// Rebuilds everything that depends on the wind.
    fn apply_wind(&mut self) {
        let wind = self.wind();
        let mut bound = 0.0f64;
        self.fuels = (0..=SB40.len())
            .map(|id| {
                if id == 0 || !self.terrain.present[id] {
                    return FuelSpread::NONE;
                }
                let model = &SB40[id - 1];
                let bed = FuelBed::new(model, &self.weather.moisture);
                if bed.r0 <= 0.0 {
                    return FuelSpread::NONE;
                }
                let family = &model.name[..2];
                // Timber models describe fuel under trees, where the canopy shelters it.
                let waf = if matches!(family, "TU" | "TL") {
                    self.cfg.sheltered_waf as f64
                } else {
                    bed.wind_adjustment()
                };
                let per_mw = match family {
                    "GR" => self.cfg.water_per_mw_grass_l_m2,
                    "TL" | "TU" => self.cfg.water_per_mw_timber_l_m2,
                    _ => self.cfg.water_per_mw_l_m2,
                };
                // 10 m wind -> 20 ft wind -> midflame height for this fuel.
                let midflame = wind.speed_ms as f64 / 1.15 * waf;
                let vector = [midflame * wind.to[0] as f64, midflame * wind.to[1] as f64];
                let flat = bed.spread(vector, [0.0, 0.0]);
                // Wind blowing straight up the steepest slope: nothing spreads faster.
                let steepest = bed.spread([midflame, 0.0], [self.terrain.max_tan as f64, 0.0]);
                bound = bound.max(steepest.head as f64);
                FuelSpread {
                    bed: Some(bed),
                    wind: vector,
                    flat,
                    water_per_ms: per_mw * bed.heat_per_area_kj_m2() as f32 / 1000.0,
                    heat_kj_m2: bed.heat_per_area_kj_m2() as f32,
                }
            })
            .collect();
        self.rate_bound = bound as f32 / self.terrain.grid.cell_m;
        let ctx = ctx!(self);
        self.fire.refresh_rates(&ctx);
    }
}
