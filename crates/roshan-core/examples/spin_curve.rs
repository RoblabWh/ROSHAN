//! Spread curve of the FireSPIN model on the old ROSHAN's uniform map: 50 x 50 cells of 10 m,
//! generic class, a square block of ignitions in the centre. Prints seconds, burning cells, ignited cells,
//! particles; the last line is the wall-clock time.
//!
//!     cargo run --release -p roshan-core --example spin_curve -- [dt_s] [wind_ms] [seed] [end_s] [block]

use std::sync::Arc;
use std::time::Instant;

use roshan_core::fire::FireModel;
use roshan_core::rothermel::Moisture;
use roshan_core::spin::SpinConfig;
use roshan_core::terrain::{Grid, Terrain};
use roshan_core::world::{SimConfig, Weather, World};

fn main() {
    let arg = |i: usize, default: f32| {
        let value = std::env::args()
            .nth(i)
            .map(|a| a.parse().expect("a number"));
        value.unwrap_or(default)
    };
    let (dt, wind, seed, end) = (
        arg(1, 0.1),
        arg(2, 10.0),
        arg(3, 1.0) as u64,
        arg(4, 1500.0),
    );
    let grid = Grid {
        nx: 50,
        ny: 50,
        cell_m: 10.0,
    };
    let cfg = SimConfig {
        dt_s: 1.0,
        fire_every: 1,
        fire_model: FireModel::Firespin,
        spin: SpinConfig {
            dt_s: dt,
            ..SpinConfig::default()
        },
        ..SimConfig::default()
    };
    let moisture = Moisture {
        d1: 0.06,
        d10: 0.07,
        d100: 0.08,
        herb: 0.6,
        woody: 0.9,
    };
    let weather = Weather::steady(moisture, wind, [1.0, 0.0]);
    let mut w = World::new(Arc::new(Terrain::uniform(grid, 1)), weather, cfg, seed);
    let block = arg(5, 1.0) as u32;
    for x in 0..block {
        for y in 0..block {
            w.ignite(25 - block / 2 + x, 25 - block / 2 + y);
        }
    }
    let start = Instant::now();
    println!("t_s,burning,ignited,particles");
    while w.time_s() < end {
        w.run(500.0);
        let fire = &w.fire;
        println!(
            "{},{},{},{}",
            w.time_s(),
            fire.n_burning(),
            fire.n_ignited,
            fire.particles().count()
        );
    }
    eprintln!("{:.2} s wall clock", start.elapsed().as_secs_f64());
}
