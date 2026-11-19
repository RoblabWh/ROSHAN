//! M1 acceptance checks for fire spread on the grid. Every map here is non-square on purpose.

use std::sync::Arc;

use rayon::prelude::*;
use roshan_core::fire::{BURNED, UNBURNED};
use roshan_core::fuel;
use roshan_core::rothermel::Moisture;
use roshan_core::terrain::{Grid, Terrain};
use roshan_core::world::{SimConfig, Weather, Wind, World};

const DRY: Moisture = Moisture {
    d1: 0.06,
    d10: 0.07,
    d100: 0.08,
    herb: 0.60,
    woody: 0.90,
};
const GRID: Grid = Grid {
    nx: 161,
    ny: 101,
    cell_m: 10.0,
};

/// Coarse clock so long burns stay cheap: 1 s ticks, fire every 2 s. The critical water load
/// is fixed at 1 L/m² whatever the fire's intensity, so the water tests check the mechanics
/// with round numbers; `calibrated` has the real thresholds.
fn cfg() -> SimConfig {
    SimConfig {
        water_crit_l_m2: 1.0,
        water_per_mw_grass_l_m2: 0.0,
        water_per_mw_l_m2: 0.0,
        water_per_mw_timber_l_m2: 0.0,
        water_limit_kw_m: f32::INFINITY,
        ..calibrated()
    }
}

fn calibrated() -> SimConfig {
    SimConfig {
        dt_s: 1.0,
        fire_every: 2,
        ..SimConfig::default()
    }
}

fn grass(wind_ms: f32, to: [f32; 2], cfg: SimConfig) -> World {
    let terrain = Terrain::uniform(GRID, fuel::id("GR2").unwrap());
    World::new(Arc::new(terrain), Weather::steady(DRY, wind_ms, to), cfg, 1)
}

fn state(w: &World, x: u32, y: u32) -> u8 {
    w.fire.state[w.terrain.grid.idx(x, y)]
}

fn arrival(w: &World, x: u32, y: u32) -> f32 {
    w.fire.t_ignite[w.terrain.grid.idx(x, y)]
}

fn run_until(w: &mut World, limit_s: f32, mut done: impl FnMut(&World) -> bool) {
    while !done(w) {
        assert!(
            w.time_s() < limit_s,
            "condition not reached within {limit_s} s"
        );
        w.tick();
    }
}

/// Extent of the fire along `axis` and across it, in metres, measured over burning and
/// burned cells.
fn extents(w: &World, axis: [f32; 2]) -> (f32, f32) {
    let grid = w.terrain.grid;
    let (mut along, mut across) = ((f32::MAX, f32::MIN), (f32::MAX, f32::MIN));
    for (i, &s) in w.fire.state.iter().enumerate() {
        if s == UNBURNED {
            continue;
        }
        let (x, y) = grid.xy(i);
        let (px, py) = (x as f32 * grid.cell_m, y as f32 * grid.cell_m);
        let (a, c) = (px * axis[0] + py * axis[1], -px * axis[1] + py * axis[0]);
        along = (along.0.min(a), along.1.max(a));
        across = (across.0.min(c), across.1.max(c));
    }
    (
        along.1 - along.0 + grid.cell_m,
        across.1 - across.0 + grid.cell_m,
    )
}

#[test]
fn head_fire_travels_at_the_rothermel_rate() {
    let mut w = grass(6.0, [1.0, 0.0], cfg());
    let head = w.spread(20, 50).head;
    assert!(head > 0.05, "grass in wind should move: {head} m/s");
    w.ignite(20, 50);
    run_until(&mut w, 1e5, |w| state(w, 120, 50) != UNBURNED);
    let measured = 800.0 / (arrival(&w, 120, 50) - arrival(&w, 40, 50));
    assert!(
        (measured / head - 1.0).abs() < 0.01,
        "measured {measured} m/s, model {head} m/s"
    );
}

#[test]
fn fire_without_wind_is_round() {
    let mut w = grass(
        0.0,
        [1.0, 0.0],
        SimConfig {
            dt_s: 1.0,
            fire_every: 5,
            ..SimConfig::default()
        },
    );
    let r0 = w.spread(80, 50).head;
    w.ignite(80, 50);
    run_until(&mut w, 1e6, |w| {
        state(w, 110, 50) != UNBURNED && state(w, 101, 71) != UNBURNED
    });
    let speed = |dx: u32, dy: u32| {
        let metres = ((dx * dx + dy * dy) as f32).sqrt() * GRID.cell_m;
        metres / arrival(&w, 80 + dx, 50 + dy)
    };
    // Along the 16 edge directions the travel time is exact.
    for (dx, dy) in [(30, 0), (21, 21), (26, 13)] {
        assert!(
            (speed(dx, dy) / r0 - 1.0).abs() < 0.01,
            "({dx},{dy}): {} vs {r0}",
            speed(dx, dy)
        );
    }
    // Between them the path zigzags, which costs at most a few percent.
    let between = speed(28, 7) / r0;
    assert!(
        (0.96..=1.001).contains(&between),
        "between directions: {between}"
    );
}

#[test]
fn wind_driven_fire_has_the_length_to_breadth_of_the_ellipse() {
    let diagonal = [0.866_025_4, 0.5];
    for (wind_ms, to, start) in [
        (10.0, [1.0, 0.0], (30, 50)),
        (14.0, [1.0, 0.0], (30, 50)),
        (10.0, diagonal, (20, 20)),
        (14.0, diagonal, (20, 20)),
    ] {
        let mut w = grass(wind_ms, to, cfg());
        let spread = w.spread(start.0, start.1);
        w.ignite(start.0, start.1);
        w.run(1000.0 / spread.head);
        let (length, width) = extents(&w, to);
        let (got, want) = (length / width, spread.length_to_breadth());
        assert!(
            (1.8..3.2).contains(&want),
            "test should cover ratios around 2-3, got {want}"
        );
        assert!(
            (got / want - 1.0).abs() < 0.10,
            "wind {wind_ms} m/s to {to:?}: {got} vs {want}"
        );
        // The head itself must be where the ellipse says, within a cell or two.
        assert!(
            (length / (1000.0 * (1.0 + (1.0 - spread.ecc) / (1.0 + spread.ecc))) - 1.0).abs()
                < 0.05
        );
    }
}

/// A map with several fuels, hills, a road and per-cell noise.
fn patchy(seed: u64) -> World {
    let grid = Grid {
        nx: 90,
        ny: 60,
        cell_m: 10.0,
    };
    let ids = ["GR2", "GS2", "SH5", "TL3"].map(|n| fuel::id(n).unwrap());
    let fuel: Vec<u8> = (0..grid.len())
        .map(|i| {
            let (x, y) = grid.xy(i);
            if x == 60 && y % 17 != 0 {
                0
            } else {
                ids[((x / 15 + y / 12) % 4) as usize]
            }
        })
        .collect();
    let elevation: Vec<f32> = (0..grid.len())
        .map(|i| {
            let (x, y) = grid.xy(i);
            40.0 * (x as f32 / 14.0).sin() + 25.0 * (y as f32 / 9.0).cos()
        })
        .collect();
    let terrain = Terrain::new(grid, fuel, Some(&elevation)).unwrap();
    let weather = Weather::steady(DRY, 5.0, [0.8, 0.6]);
    let mut w = World::new(
        Arc::new(terrain),
        weather,
        SimConfig {
            ros_noise: 0.3,
            ..cfg()
        },
        seed,
    );
    assert!(w.ignite(10, 10) && w.ignite(40, 30));
    w
}

#[test]
fn same_seed_same_world_for_any_thread_count() {
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut worlds: Vec<World> = (0..12).map(patchy).collect();
        pool.install(|| worlds.par_iter_mut().for_each(|w| w.run(900.0)));
        worlds.iter().map(World::state_hash).collect::<Vec<_>>()
    };
    let single = run(1);
    assert_eq!(
        single,
        run(std::thread::available_parallelism().map_or(8, |n| n.get()))
    );
    assert_ne!(
        single[0], single[1],
        "different seeds must give different fires"
    );
    assert!(!patchy(0).fire.is_out());
}

#[test]
fn a_clone_is_independent_and_identical() {
    let mut a = patchy(3);
    a.run(200.0);
    let mut b = a.clone();
    assert!(
        Arc::ptr_eq(&a.terrain, &b.terrain),
        "terrain must be shared, not copied"
    );
    a.run(300.0);
    b.run(300.0);
    assert_eq!(a.state_hash(), b.state_hash());

    let mut c = a.clone();
    c.add_water(45, 32, 5.0);
    c.ignite(80, 50);
    let before = a.state_hash();
    c.run(100.0);
    assert_eq!(
        a.state_hash(),
        before,
        "changing a clone must not touch the original"
    );
    a.run(100.0);
    assert_ne!(a.state_hash(), c.state_hash());
}

#[test]
fn suppressing_one_fire_leaves_a_distant_fire_bit_identical() {
    let make = || {
        let mut w = grass(
            6.0,
            [0.0, 1.0],
            SimConfig {
                ros_noise: 0.2,
                ..cfg()
            },
        );
        assert!(w.ignite(20, 20) && w.ignite(120, 20));
        w.run(60.0);
        w
    };
    let (mut plain, mut doused) = (make(), make());
    for y in 15..45 {
        for x in 10..31 {
            doused.add_water(x, y, 3.0);
        }
    }
    plain.run(540.0);
    doused.run(540.0);

    let grid = plain.terrain.grid;
    let column = |w: &World, x0: u32, x1: u32| -> Vec<(u8, u32)> {
        (0..grid.len())
            .filter(|&i| (x0..x1).contains(&grid.xy(i).0))
            .map(|i| (w.fire.state[i], w.fire.t_ignite[i].to_bits()))
            .collect()
    };
    assert_ne!(
        column(&plain, 0, 60),
        column(&doused, 0, 60),
        "the water must have changed the near fire"
    );
    assert_eq!(
        column(&plain, 90, 161),
        column(&doused, 90, 161),
        "the far fire must be untouched, bit for bit"
    );
}

/// A line fire across the whole map heading for column 50, stopped at column 49.
fn line_fire_at_strip(cfg: SimConfig) -> World {
    let mut w = grass(6.0, [1.0, 0.0], cfg);
    for y in 0..GRID.ny {
        w.ignite(30, y);
    }
    run_until(&mut w, 1e5, |w| state(w, 49, 50) != UNBURNED);
    for y in 0..GRID.ny {
        w.add_water(50, y, 2.0);
    }
    w
}

#[test]
fn a_wet_strip_holds_the_front_until_it_dries() {
    // `cross` is the time the dry front needs for one cell. Twice the critical load blocks for
    // tau*ln(2); tau is chosen so that this is ten crossing times.
    let cross = GRID.cell_m / grass(6.0, [1.0, 0.0], cfg()).spread(50, 50).head;
    let tau = 10.0 * cross / std::f32::consts::LN_2;
    let mut w = line_fire_at_strip(SimConfig {
        water_tau_s: tau,
        ember_s: 1e9,
        ..cfg()
    });
    let wetted = w.time_s();
    let mut control = grass(6.0, [1.0, 0.0], cfg());
    for y in 0..GRID.ny {
        control.ignite(30, y);
    }
    control.run(wetted + 3.0 * cross);
    assert!(
        (0..GRID.ny).all(|y| state(&control, 51, y) != UNBURNED),
        "without water the front is past"
    );

    w.run(8.0 * cross);
    assert!(
        (0..GRID.ny).all(|y| state(&w, 50, y) == UNBURNED && state(&w, 51, y) == UNBURNED),
        "held while wet"
    );
    w.run(32.0 * cross);
    assert!(
        (0..GRID.ny).all(|y| state(&w, 52, y) != UNBURNED),
        "released once dry"
    );
    // Dry at 10 crossing times; the drying strip then slows the front for a few more.
    let crossed = (arrival(&w, 50, 50) - wetted) / cross;
    assert!(
        (10.0..25.0).contains(&crossed),
        "crossed after {crossed} crossing times"
    );
}

#[test]
fn a_lasting_wet_strip_contains_the_fire() {
    let mut w = line_fire_at_strip(SimConfig {
        water_tau_s: 1e7,
        ember_s: 30.0,
        ..cfg()
    });
    // The slow backing fire behind the line needs a while to reach the map edge.
    run_until(&mut w, 2e5, |w| w.fire.is_out());
    for y in 0..GRID.ny {
        assert_eq!(state(&w, 49, y), BURNED);
        assert!(
            (50..GRID.nx).all(|x| state(&w, x, y) == UNBURNED),
            "nothing beyond the strip may burn"
        );
    }
}

#[test]
fn water_on_a_burning_cell_puts_it_out_and_it_can_reignite() {
    // Put out a lone fresh ignition for good.
    let mut w = grass(
        0.0,
        [1.0, 0.0],
        SimConfig {
            water_tau_s: 1e7,
            ember_s: 20.0,
            ..cfg()
        },
    );
    w.ignite(80, 50);
    w.add_water(80, 50, 5.0);
    w.run(40.0);
    assert_eq!(state(&w, 80, 50), UNBURNED);
    assert!(w.fire.is_out() && w.fire.n_ignited == 0 && arrival(&w, 80, 50).is_nan());

    // Knock down one cell on the edge of a live fire: once the water is gone its burning
    // neighbours set it alight again.
    let mut w = grass(
        0.0,
        [1.0, 0.0],
        SimConfig {
            water_tau_s: 50.0,
            ember_s: 20.0,
            ..cfg()
        },
    );
    w.ignite(80, 50);
    run_until(&mut w, 1e6, |w| state(w, 82, 50) != UNBURNED);
    let doused = w.time_s();
    w.add_water(82, 50, 2.0);
    w.run(26.0);
    assert_eq!(
        state(&w, 82, 50),
        UNBURNED,
        "held long enough for its flames to die"
    );
    w.run(3000.0);
    assert_ne!(
        state(&w, 82, 50),
        UNBURNED,
        "re-ignited from its neighbours"
    );
    assert!(arrival(&w, 82, 50) > doused + 30.0);
}

#[test]
fn fire_does_not_cross_ground_that_cannot_burn() {
    let grass_id = fuel::id("GR2").unwrap();
    let with_barrier = |blocked: &dyn Fn(u32, u32) -> bool| {
        let fuel = (0..GRID.len())
            .map(|i| {
                let (x, y) = GRID.xy(i);
                if blocked(x, y) { 0 } else { grass_id }
            })
            .collect();
        let terrain = Terrain::new(GRID, fuel, None).unwrap();
        let mut w = World::new(
            Arc::new(terrain),
            Weather::steady(DRY, 8.0, [1.0, 0.0]),
            cfg(),
            1,
        );
        assert!(w.ignite(20, 80));
        // Until everything reachable has burned; flanks and the backing fire are slow.
        run_until(&mut w, 5e5, |w| w.fire.is_out());
        w
    };

    // A straight one-cell road.
    let w = with_barrier(&|x, _| x == 50);
    assert!((0..GRID.len()).all(|i| GRID.xy(i).0 < 50 || w.fire.state[i] == UNBURNED));
    assert_eq!(state(&w, 49, 3), BURNED);

    // A one-cell road running diagonally: neither diagonal nor knight moves may slip through.
    let w = with_barrier(&|x, y| x == 40 + y);
    assert!((0..GRID.len()).all(|i| {
        let (x, y) = GRID.xy(i);
        x <= 40 + y || w.fire.state[i] == UNBURNED
    }));
    assert_eq!(state(&w, 39 + 100, 100), BURNED);

    // The same straight road with one missing cell lets the fire through.
    let w = with_barrier(&|x, y| x == 50 && y != 50);
    assert_eq!(state(&w, 150, 10), BURNED);
}

#[test]
fn a_wind_shift_turns_the_fire() {
    let make = |wind: Vec<Wind>| {
        let terrain = Terrain::uniform(GRID, fuel::id("GR2").unwrap());
        let mut w = World::new(
            Arc::new(terrain),
            Weather {
                moisture: DRY,
                wind,
            },
            cfg(),
            1,
        );
        w.ignite(40, 20);
        w.run(900.0);
        w
    };
    let east = Wind {
        t_s: 0.0,
        speed_ms: 8.0,
        to: [1.0, 0.0],
    };
    let steady = make(vec![east]);
    let shifted = make(vec![
        east,
        Wind {
            t_s: 300.0,
            speed_ms: 8.0,
            to: [0.0, 1.0],
        },
    ]);
    assert_eq!(shifted.wind().to, [0.0, 1.0]);
    let (steady_x, steady_y) = (
        extents(&steady, [1.0, 0.0]).0,
        extents(&steady, [0.0, 1.0]).0,
    );
    let (shifted_x, shifted_y) = (
        extents(&shifted, [1.0, 0.0]).0,
        extents(&shifted, [0.0, 1.0]).0,
    );
    // After the shift the old head only creeps on as a flank, and the new head runs south.
    assert!(
        shifted_x < 0.8 * steady_x,
        "eastward run cut short: {shifted_x} vs {steady_x}"
    );
    let southward_run = steady.spread(40, 20).head * 600.0;
    assert!(
        shifted_y > southward_run && shifted_y > 1.3 * steady_y,
        "new head runs south: {shifted_y} vs {steady_y}"
    );
}

#[test]
fn water_that_holds_the_flank_does_not_hold_the_head() {
    // Shrub in wind: the head is several MW/m, the flank a fraction of that. The same load
    // lies ahead of the fire (column 60) and beside it (row 60) and does not evaporate.
    let terrain = Terrain::uniform(GRID, fuel::id("SH5").unwrap());
    let cfg = SimConfig {
        water_tau_s: 1e9,
        ember_s: 1e9,
        ..calibrated()
    };
    let weather = Weather::steady(DRY, 5.0, [1.0, 0.0]);
    let mut w = World::new(Arc::new(terrain), weather, cfg, 1);
    for y in 0..GRID.ny {
        w.add_water(60, y, 2.0);
    }
    for x in 0..GRID.nx {
        w.add_water(x, 60, 2.0);
    }
    w.ignite(40, 50);
    // The head gets through its strip ...
    run_until(&mut w, 1e5, |w| state(w, 100, 50) != UNBURNED);
    // ... while beside the origin, where the fire is all flank, the row holds. (Further
    // downwind the fire reaches the row with its shoulder, which is nearly as intense as the
    // head, and gets through there too.)
    let beyond = (0..=50).filter(|&x| state(&w, x, 62) != UNBURNED);
    assert_eq!(beyond.count(), 0, "the flank crossed the wet row");
    assert_ne!(state(&w, 45, 59), UNBURNED, "the flank reached the row");
}

#[test]
fn fuel_under_trees_feels_less_wind() {
    let head = |name: &str, sheltered_waf: f32| {
        let terrain = Terrain::uniform(GRID, fuel::id(name).unwrap());
        let cfg = SimConfig {
            sheltered_waf,
            ..calibrated()
        };
        let weather = Weather::steady(DRY, 8.0, [1.0, 0.0]);
        World::new(Arc::new(terrain), weather, cfg, 1)
            .spread(5, 5)
            .head
    };
    // Timber understory: sheltered at 0.2 against 0.36 in the open.
    assert!(head("TU2", 0.2) < 0.7 * head("TU2", 0.36));
    assert_eq!(
        head("GR2", 0.2),
        head("GR2", 0.36),
        "grass stands in the open"
    );
}

#[test]
fn stopped_fronts_count_with_their_intensity() {
    // A fire left alone stops no front, and its peak is the head's intensity.
    let mut idle = grass(5.0, [1.0, 0.0], calibrated());
    idle.ignite(40, 50);
    idle.run(900.0);
    assert_eq!(idle.fire.stats.stopped, 0);
    let strongest = idle.fire.strongest_front_kw();
    assert!(strongest > 100.0 && strongest <= idle.fire.stats.peak_kw);
    assert!(
        (250.0..450.0).contains(&idle.fire.stats.peak_kw),
        "GR2 head at 5 m/s is about 330 kW/m, got {}",
        idle.fire.stats.peak_kw
    );

    // Soak every front cell: each one stalls, dies after `ember_s` and is counted once with
    // the intensity it was lit with, whether it ends unburned or burned out.
    let mut w = idle.clone();
    let cells = w.terrain.grid.len();
    let fronts: Vec<f32> = (0..cells)
        .filter_map(|c| w.fire.front(c).map(|f| f.kw_m))
        .collect();
    assert!(fronts.len() > 5 && fronts.len() <= w.fire.n_burning());
    for c in 0..cells {
        if w.fire.front(c).is_some() {
            let (x, y) = w.terrain.grid.xy(c);
            w.add_water(x, y, 5.0);
        }
    }
    w.run(calibrated().ember_s + 10.0);
    let stats = w.fire.stats;
    assert_eq!(stats.stopped as usize, fronts.len());
    let sum: f64 = fronts.iter().map(|&k| k as f64).sum();
    assert!((stats.stopped_kw_sum - sum).abs() < 1e-3 * sum);
    assert_eq!(
        stats.stopped_kw_max,
        fronts.iter().copied().fold(0.0, f32::max)
    );
    assert!(stats.stopped_kw_max <= stats.peak_kw);
    assert!(w.fire.is_out() || w.fire.n_burning() < idle.fire.n_burning());
}

#[test]
fn litter_under_trees_takes_more_water_than_shrub() {
    // Per MW/m a front needs `water_per_mw_timber_l_m2` in TU and `water_per_mw_l_m2` in SH.
    let need = |name: &str| {
        let terrain = Terrain::uniform(GRID, fuel::id(name).unwrap());
        let weather = Weather::steady(DRY, 10.0, [1.0, 0.0]);
        let mut w = World::new(Arc::new(terrain), weather, calibrated(), 1);
        w.ignite(40, 50);
        let f = w.fire.front(w.terrain.grid.idx(40, 50)).unwrap();
        f.water_l_m2 / (f.kw_m / 1000.0)
    };
    let cfg = SimConfig::default();
    assert!((need("TU3") - cfg.water_per_mw_timber_l_m2).abs() < 1e-3);
    assert!((need("SH2") - cfg.water_per_mw_l_m2).abs() < 1e-3);
    assert!((need("GR2") - cfg.water_per_mw_grass_l_m2).abs() < 1e-3);
}

#[test]
fn a_front_beyond_the_limit_is_not_stopped_by_any_amount_of_water() {
    // A shrub head in wind runs at about 6 MW/m. Under a limit of 4 MW/m a flooded strip
    // does not hold it; with the limit lifted the same strip does.
    let crossed = |water_limit_kw_m: f32| {
        let terrain = Terrain::uniform(GRID, fuel::id("SH5").unwrap());
        let cfg = SimConfig {
            water_tau_s: 1e9,
            water_limit_kw_m,
            ..calibrated()
        };
        let weather = Weather::steady(DRY, 5.0, [1.0, 0.0]);
        let mut w = World::new(Arc::new(terrain), weather, cfg, 1);
        for y in 0..GRID.ny {
            for x in 60..64 {
                w.add_water(x, y, 100.0);
            }
        }
        w.ignite(40, 50);
        w.run(900.0);
        let head = w.fire.stats.peak_kw;
        assert!(head > 5000.0, "the head is intense: {head} kW/m");
        (state(&w, 70, 50) != UNBURNED, w.fire.stats)
    };
    let (through, stats) = crossed(4000.0);
    assert!(through, "water does not stop a 6 MW/m head");
    assert!(stats.stopped_kw_max <= 4000.0, "{stats:?}");
    let (through, stats) = crossed(f32::INFINITY);
    assert!(!through && stats.stopped > 0, "without the limit it does");
}
