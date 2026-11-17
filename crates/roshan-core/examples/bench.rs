//! Rough single-thread timings of the fire model.
//!     cargo run --release -p roshan-core --example bench
//! ponytail: wall-clock means, no statistics. Switch to criterion if regressions need tracking.

use std::sync::Arc;
use std::time::Instant;

use roshan_core::fuel;
use roshan_core::rothermel::Moisture;
use roshan_core::terrain::{Grid, Terrain};
use roshan_core::world::{SimConfig, Weather, World};

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

fn main() {
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
