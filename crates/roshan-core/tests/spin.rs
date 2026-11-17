//! The FireSPIN particle model (M8). The spread curve against the old code is measured with
//! `examples/spin_curve.rs` and `spikes/firespin_reference.cpp`; here are the properties.

use std::sync::Arc;

use roshan_core::fire::{BURNING, FireModel, UNBURNED};
use roshan_core::rothermel::Moisture;
use roshan_core::terrain::{Grid, Terrain};
use roshan_core::world::{SimConfig, Weather, World};

const GRID: Grid = Grid {
    nx: 60,
    ny: 40,
    cell_m: 10.0,
};

fn cfg() -> SimConfig {
    SimConfig {
        dt_s: 1.0,
        fire_every: 1,
        fire_model: FireModel::Firespin,
        ..SimConfig::default()
    }
}

fn world(terrain: Terrain, wind_ms: f32, seed: u64) -> World {
    let moisture = Moisture {
        d1: 0.06,
        d10: 0.07,
        d100: 0.08,
        herb: 0.6,
        woody: 0.9,
    };
    let weather = Weather::steady(moisture, wind_ms, [1.0, 0.0]);
    World::new(Arc::new(terrain), weather, cfg(), seed)
}

/// Ignites a 3 x 3 block around (x, y): a single cell often fails to spread in this model.
fn light(w: &mut World, x: u32, y: u32) {
    for dx in 0..3 {
        for dy in 0..3 {
            w.ignite(x + dx - 1, y + dy - 1);
        }
    }
}

fn uniform(seed: u64) -> World {
    let mut w = world(Terrain::uniform(GRID, 1), 5.0, seed);
    light(&mut w, 30, 20);
    w
}

#[test]
fn fire_spreads_burns_out_behind_and_is_reproducible() {
    let mut w = uniform(1);
    w.run(1500.0);
    assert!(w.fire.n_ignited > 150, "{} cells ignited", w.fire.n_ignited);
    let centre = w.terrain.grid.idx(30, 20);
    assert!(w.fire.state[centre] != BURNING && w.fire.n_burning() > 0 && !w.fire.is_out());
    assert!(w.fire.particles().count() > 0);

    let mut again = uniform(1);
    again.run(1500.0);
    assert_eq!(w.state_hash(), again.state_hash());
    let mut other = uniform(2);
    other.run(1500.0);
    assert_ne!(w.state_hash(), other.state_hash());
}

#[test]
fn a_clone_continues_identically() {
    let mut w = uniform(3);
    w.run(600.0);
    let mut copy = w.clone();
    w.run(600.0);
    copy.run(600.0);
    assert_eq!(w.state_hash(), copy.state_hash());
}

#[test]
fn the_step_size_changes_the_spread_only_slightly() {
    let burned = |dt_s: f32| {
        let total: u32 = (1..=6)
            .map(|seed| {
                let mut w = uniform(seed);
                w.cfg.spin.dt_s = dt_s;
                w.run(1500.0);
                w.fire.n_ignited
            })
            .sum();
        total as f32
    };
    let (fine, coarse) = (burned(0.1), burned(1.0));
    assert!((coarse / fine - 1.0).abs() < 0.05, "{fine} vs {coarse}");
}

#[test]
fn putting_out_one_fire_leaves_a_distant_fire_bit_identical() {
    let two = |soak: bool| {
        let mut w = world(Terrain::uniform(GRID, 1), 5.0, 7);
        light(&mut w, 8, 20);
        light(&mut w, 50, 20);
        if soak {
            for x in 0..20 {
                for y in 0..GRID.ny {
                    w.add_water(x, y, 50.0);
                }
            }
        }
        w.run(900.0);
        w
    };
    let (free, soaked) = (two(false), two(true));
    let left = |w: &World| (0..GRID.ny).any(|y| w.fire.state[GRID.idx(8, y)] != UNBURNED);
    assert!(
        left(&free) && !left(&soaked),
        "water puts the left fire out"
    );
    for y in 0..GRID.ny {
        for x in 35..GRID.nx {
            let i = GRID.idx(x, y);
            assert_eq!(free.fire.state[i], soaked.fire.state[i]);
            assert_eq!(
                free.fire.t_ignite[i].to_bits(),
                soaked.fire.t_ignite[i].to_bits()
            );
        }
    }
}

#[test]
fn a_soaked_strip_holds_the_fire_and_bare_ground_does_not_burn() {
    let mut fuel = vec![1u8; GRID.len()];
    for y in 0..GRID.ny {
        fuel[GRID.idx(45, y)] = 0;
        fuel[GRID.idx(46, y)] = 0;
        fuel[GRID.idx(47, y)] = 0;
    }
    let mut w = world(Terrain::new(GRID, fuel, None).unwrap(), 5.0, 4);
    w.cfg.water_tau_s = 1e9; // the water stays
    light(&mut w, 30, 20);
    for x in 12..16 {
        for y in 0..GRID.ny {
            w.add_water(x, y, 50.0);
        }
    }
    w.run(4000.0);
    let touched = |xs: std::ops::Range<u32>| {
        xs.flat_map(|x| (0..GRID.ny).map(move |y| GRID.idx(x, y)))
            .filter(|&i| w.fire.state[i] != UNBURNED)
            .count()
    };
    assert!(touched(16..45) > 800, "the fire fills the middle");
    assert_eq!(touched(0..16), 0, "nothing in or behind the wet strip");
    assert_eq!(touched(45..48), 0, "bare ground does not burn");
}

#[test]
fn land_cover_classes_set_how_fast_a_cell_ignites() {
    // Left half grass (class 6, ignition delay 100 s), right half forest (class 2, 300 s).
    let mut terrain = Terrain::uniform(GRID, 1);
    let cover = (0..GRID.len()).map(|i| if GRID.xy(i).0 < 30 { 6 } else { 2 });
    terrain.landcover = Some(cover.collect());
    // Wind helps: with these classes the fire hardly spreads in calm air.
    let mut w = world(terrain, 10.0, 5);
    light(&mut w, 15, 20);
    light(&mut w, 45, 20);
    w.run(4000.0);
    let count = |xs: std::ops::Range<u32>| {
        xs.flat_map(|x| (0..GRID.ny).map(move |y| GRID.idx(x, y)))
            .filter(|&i| w.fire.state[i] != UNBURNED)
            .count()
    };
    let (grass, forest) = (count(0..30), count(30..60));
    assert!(
        grass > 2 * forest && forest >= 9,
        "grass {grass}, forest {forest}"
    );
}

#[test]
fn the_model_is_chosen_in_the_configuration() {
    let cfg: SimConfig =
        serde_json::from_str(r#"{"fire_model": "firespin", "spin": {"dt_s": 0.25}}"#).unwrap();
    assert!(cfg.fire_model == FireModel::Firespin && cfg.spin.dt_s == 0.25);
    assert!(
        cfg.spin.conv_tau_mem_s == 10.0,
        "other values keep their defaults"
    );
    assert!(serde_json::from_str::<SimConfig>(r#"{"fire_model": "magic"}"#).is_err());
    assert!(SimConfig::default().fire_model == FireModel::Rothermel);
}
