//! Rough single-thread timings of the fire model and of the allocate task.
//!     cargo run --release -p roshan-core --example bench
//! The first lines are a flat, dry grass map. The "hills" lines are what training runs on:
//! every cell on a slope, several fuels, water on the ground, drones and a planner.
//! ponytail: wall-clock means, no statistics. Switch to criterion if regressions need tracking.

use std::sync::Arc;
use std::time::Instant;

use roshan_core::allocate::{
    AllocConfig, AllocVec, Bufs, DRONE, GLOBAL, INFO, Method, TARGET, TERMS,
};
use roshan_core::fuel;
use roshan_core::rothermel::Moisture;
use roshan_core::scenario::Scenario;
use roshan_core::terrain::{Grid, Terrain};
use roshan_core::world::{SimConfig, Weather, Wind, World};

const DRY: Moisture = Moisture {
    d1: 0.06,
    d10: 0.07,
    d100: 0.08,
    herb: 0.60,
    woody: 0.90,
};

fn grass(nx: u32, ny: u32, cfg: SimConfig) -> World {
    let terrain = Terrain::uniform(
        Grid {
            nx,
            ny,
            cell_m: 10.0,
        },
        fuel::id("GR2").unwrap(),
    );
    World::new(
        Arc::new(terrain),
        Weather::steady(DRY, 5.0, [1.0, 0.0]),
        cfg,
        1,
    )
}

/// 1.6 km x 1.2 km of rolling hills with grass, shrub and timber in bands, a road that does
/// not burn, four drones and one station: the size and kind of map the planner trains on.
fn hills() -> Scenario {
    let grid = Grid {
        nx: 160,
        ny: 120,
        cell_m: 10.0,
    };
    let fuels = ["GR2", "GS2", "TU1", "TL3"].map(|name| fuel::id(name).unwrap());
    let (mut fuel, mut elevation) = (vec![0u8; grid.len()], vec![0.0f32; grid.len()]);
    for i in 0..grid.len() {
        let (x, y) = grid.xy(i);
        let (fx, fy) = (x as f32, y as f32);
        elevation[i] =
            40.0 * (fx / 23.0).sin() * (fy / 17.0).cos() + 15.0 * ((fx + fy) / 9.0).sin();
        fuel[i] = if x == 100 {
            0
        } else {
            fuels[((x / 20 + y / 15) % 4) as usize]
        };
    }
    Scenario {
        grid,
        fuel,
        elevation: Some(elevation),
        landcover: None,
        ignitions: vec![[40, 60]],
        stations: vec![[5, 8]],
        moisture: DRY,
        wind: vec![Wind {
            t_s: 0.0,
            speed_ms: 5.0,
            to: [1.0, 0.0],
        }],
        meta: String::new(),
    }
}

/// Timings on the hilly map: the fire alone, dry and with water lying on the map, then the
/// allocate task with the greedy and the rollout planner.
fn hills_bench() {
    let scenario = hills();
    for wet in [false, true] {
        let mut w = scenario.world(SimConfig::default(), 1).unwrap();
        for (x, y) in [
            (20, 20),
            (80, 30),
            (130, 50),
            (60, 90),
            (120, 100),
            (30, 70),
        ] {
            w.ignite(x, y);
        }
        if wet {
            w.add_water(150, 110, 1.5); // one drop, far from every fire
        }
        let start = Instant::now();
        w.run(1800.0);
        println!(
            "30-minute fire, 160x120 hills, {}: {:.1} ms, {:.0}% burned",
            if wet { "one wet cell" } else { "dry" },
            start.elapsed().as_secs_f64() * 1e3,
            w.burned_fraction() * 100.0
        );
    }

    // The allocate task: one station, random fires that burn two minutes before the drones
    // start. A step runs to the next decision event.
    // `explorers` above zero also gives the attack drones a 20 m sensor.
    for (planner, team, explorers, steps) in [
        (Method::Greedy, 40, 0, 4000),
        (Method::Greedy, 100, 0, 4000),
        (Method::Greedy, 40, 4, 4000),
        (Method::Rollout, 40, 0, 60),
        (Method::Rollout, 100, 0, 60),
        (Method::Rollout, 40, 4, 60),
    ] {
        let cfg = AllocConfig {
            random_ignitions: 3,
            random_wind: true,
            preburn_s: 120.0,
            drones_per_station: team,
            explorers_per_station: explorers,
            sense_m: if explorers > 0 { 20.0 } else { 0.0 },
            ..AllocConfig::default()
        };
        let mut v = AllocVec::new(16, scenario.clone(), SimConfig::default(), cfg, 0, 1).unwrap();
        let (e, n, slots) = (v.envs.len(), v.drones, v.slots);
        let (mut drones, mut targets) = (
            vec![0.0; e * n * DRONE.len()],
            vec![0.0; e * slots * TARGET.len()],
        );
        let (mut target_mask, mut action_mask) = (vec![0; e * slots], vec![0; e * n * slots]);
        let (mut globals, mut decide) = (vec![0.0; e * GLOBAL.len()], vec![0; e * n]);
        let (mut terms, mut elapsed_s) = (vec![0.0; e * n * TERMS.len()], vec![0.0; e]);
        let (mut terminated, mut truncated) = (vec![0; e], vec![0; e]);
        let (mut info, mut seed, mut fingerprint) =
            (vec![0.0; e * INFO.len()], vec![0; e], vec![0; e]);
        let mut actions = vec![-1; e * n];
        let (mut asked, mut simulated, mut planning) = (0usize, 0.0f64, 0.0);
        let start = Instant::now();
        for step in 0..=steps {
            asked += decide.iter().filter(|&&d| d == 1).count();
            let bufs = Bufs {
                drones: &mut drones,
                targets: &mut targets,
                target_mask: &mut target_mask,
                action_mask: &mut action_mask,
                globals: &mut globals,
                decide: &mut decide,
                terms: &mut terms,
                elapsed_s: &mut elapsed_s,
                terminated: &mut terminated,
                truncated: &mut truncated,
                info: &mut info,
                seed: &mut seed,
                fingerprint: &mut fingerprint,
            };
            if step == 0 {
                v.reset(None, bufs);
                continue;
            }
            let plan = Instant::now();
            v.plan(planner, 300.0, 4, &mut actions);
            planning += plan.elapsed().as_secs_f64();
            v.step(&actions, bufs);
            simulated += elapsed_s.iter().map(|&s| s as f64).sum::<f64>();
        }
        let total = start.elapsed().as_secs_f64();
        println!(
            "allocate, 160x120 hills, {team} drones, {explorers} explorers, {planner:?}: {:.1} ms \
             per 30 simulated minutes, {:.3} ms planning per drone asked",
            total * 1e3 * 1800.0 / simulated,
            planning * 1e3 / asked as f64
        );
    }
}

fn main() {
    hills_bench();

    // Cost of one fire step with a large front: many small fires on a 200x200 map.
    let mut w = grass(
        200,
        200,
        SimConfig {
            dt_s: 2.0,
            fire_every: 1,
            ..SimConfig::default()
        },
    );
    for y in (5..200).step_by(10) {
        for x in (5..200).step_by(10) {
            w.ignite(x, y);
        }
    }
    while w.fire.n_burning() < 2000 {
        w.tick();
    }
    let (steps, sources, start) = (200, w.fire.n_burning(), Instant::now());
    for _ in 0..steps {
        w.tick();
    }
    let per_step = start.elapsed().as_secs_f64() / steps as f64;
    println!(
        "fire step, 200x200, {sources}..{} sources: {:.0} µs/step, {:.0} ns/source",
        w.fire.n_burning(),
        per_step * 1e6,
        per_step * 1e9 / ((sources + w.fire.n_burning()) / 2) as f64
    );

    // A 60-minute mission on a 1 km map with the default clock (0.1 s ticks, fire every 2 s).
    let mut w = grass(100, 100, SimConfig::default());
    w.ignite(20, 50);
    let start = Instant::now();
    w.run(3600.0);
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "60-minute mission, 100x100: {:.1} ms for {} ticks ({:.0} ns/tick), {:.0}% burned, {} sources at the end",
        elapsed * 1e3,
        w.tick,
        elapsed * 1e9 / w.tick as f64,
        w.burned_fraction() * 100.0,
        w.fire.n_burning()
    );

    // Shape accuracy against the Rothermel ellipse the model is built from, for a wind along
    // the grid and one at 30 degrees to it.
    println!("ellipse accuracy (10 m wind -> model length/breadth, simulated/model):");
    for wind_ms in [4.0, 8.0, 12.0, 16.0, 20.0, 25.0] {
        let ratios: Vec<String> = [[1.0, 0.0], [0.866_025_4, 0.5]]
            .iter()
            .map(|&to: &[f32; 2]| {
                let terrain = Terrain::uniform(
                    Grid {
                        nx: 301,
                        ny: 221,
                        cell_m: 10.0,
                    },
                    fuel::id("GR2").unwrap(),
                );
                let cfg = SimConfig {
                    dt_s: 1.0,
                    fire_every: 2,
                    ..SimConfig::default()
                };
                let mut w =
                    World::new(Arc::new(terrain), Weather::steady(DRY, wind_ms, to), cfg, 1);
                // Start far enough from every edge that the fire is never clipped.
                let start = if to[1] == 0.0 { (50, 100) } else { (60, 60) };
                let spread = w.spread(start.0, start.1);
                w.ignite(start.0, start.1);
                w.run(1500.0 / spread.head);
                let (mut along, mut across) = ((f32::MAX, f32::MIN), (f32::MAX, f32::MIN));
                for (i, &s) in w.fire.state.iter().enumerate() {
                    if s != 0 {
                        let (x, y) = w.terrain.grid.xy(i);
                        let (a, c) = (
                            x as f32 * to[0] + y as f32 * to[1],
                            -(x as f32) * to[1] + y as f32 * to[0],
                        );
                        along = (along.0.min(a), along.1.max(a));
                        across = (across.0.min(c), across.1.max(c));
                    }
                }
                let simulated = (along.1 - along.0 + 1.0) / (across.1 - across.0 + 1.0);
                format!("{:.2}", simulated / spread.length_to_breadth())
            })
            .collect();
        let model = World::new(
            Arc::new(Terrain::uniform(
                Grid {
                    nx: 3,
                    ny: 3,
                    cell_m: 10.0,
                },
                fuel::id("GR2").unwrap(),
            )),
            Weather::steady(DRY, wind_ms, [1.0, 0.0]),
            SimConfig::default(),
            1,
        )
        .spread(1, 1);
        println!(
            "  {wind_ms:>4.0} m/s -> {:.2}   along grid {}, at 30 degrees {}   (head {:.2} m/s)",
            model.length_to_breadth(),
            ratios[0],
            ratios[1],
            model.head
        );
    }

    // Cost of a snapshot for rollout planners.
    let start = Instant::now();
    let clones: Vec<World> = (0..1000).map(|_| w.clone()).collect();
    println!(
        "clone, 100x100: {:.1} µs",
        start.elapsed().as_secs_f64() * 1e6 / clones.len() as f64
    );
}
