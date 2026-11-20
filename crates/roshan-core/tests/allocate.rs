//! Checks for the allocate task: separation, determinism, the drones' life cycle, decisions
//! and suppression.

use roshan_core::allocate::{
    AllocConfig, AllocVec, Bufs, DRONE, GLOBAL, INFO, Method, TARGET, TASK_DOCKED, TASK_EXPLORE,
    TASK_FIRE, TASK_STATION, TERMS,
};
use roshan_core::fuel;
use roshan_core::navigate::AGENT;
use roshan_core::rothermel::Moisture;
use roshan_core::scenario::Scenario;
use roshan_core::terrain::Grid;
use roshan_core::world::{SimConfig, Wind};

/// Grassland, 600 m x 400 m, with two stations in opposite corners.
fn scenario() -> Scenario {
    let grid = Grid {
        nx: 60,
        ny: 40,
        cell_m: 10.0,
    };
    Scenario {
        grid,
        fuel: vec![fuel::id("GR2").unwrap(); grid.len()],
        elevation: None,
        landcover: None,
        ignitions: vec![[30, 20]],
        stations: vec![[3, 5], [56, 36]],
        moisture: Moisture {
            d1: 0.06,
            d10: 0.07,
            d100: 0.08,
            herb: 0.60,
            woody: 0.90,
        },
        wind: vec![Wind {
            t_s: 0.0,
            speed_ms: 5.0,
            to: [1.0, 0.0],
        }],
        meta: String::new(),
    }
}

struct Buffers {
    drones: Vec<f32>,
    targets: Vec<f32>,
    target_mask: Vec<u8>,
    action_mask: Vec<u8>,
    globals: Vec<f32>,
    decide: Vec<u8>,
    terms: Vec<f32>,
    elapsed_s: Vec<f32>,
    terminated: Vec<u8>,
    truncated: Vec<u8>,
    info: Vec<f32>,
    seed: Vec<u64>,
    fingerprint: Vec<u64>,
}

impl Buffers {
    fn new(v: &AllocVec) -> Self {
        let (e, n, s) = (v.envs.len(), v.drones, v.slots);
        Self {
            drones: vec![0.0; e * n * DRONE.len()],
            targets: vec![0.0; e * s * TARGET.len()],
            target_mask: vec![0; e * s],
            action_mask: vec![0; e * n * s],
            globals: vec![0.0; e * GLOBAL.len()],
            decide: vec![0; e * n],
            terms: vec![0.0; e * n * TERMS.len()],
            elapsed_s: vec![0.0; e],
            terminated: vec![0; e],
            truncated: vec![0; e],
            info: vec![0.0; e * INFO.len()],
            seed: vec![0; e],
            fingerprint: vec![0; e],
        }
    }

    fn all(&mut self) -> Bufs<'_> {
        Bufs {
            drones: &mut self.drones,
            targets: &mut self.targets,
            target_mask: &mut self.target_mask,
            action_mask: &mut self.action_mask,
            globals: &mut self.globals,
            decide: &mut self.decide,
            terms: &mut self.terms,
            elapsed_s: &mut self.elapsed_s,
            terminated: &mut self.terminated,
            truncated: &mut self.truncated,
            info: &mut self.info,
            seed: &mut self.seed,
            fingerprint: &mut self.fingerprint,
        }
    }
}

/// 16 drones (8 at each station) on four random fires for ten minutes.
fn cfg() -> AllocConfig {
    AllocConfig {
        time_limit_s: 600.0,
        random_ignitions: 4,
        random_wind: true,
        drones_per_station: 8,
        ..AllocConfig::default()
    }
}

/// One fire at the scenario's ignition, `per_station` drones at each of the two stations.
fn one_fire(per_station: u32) -> AllocConfig {
    AllocConfig {
        random_ignitions: 0,
        random_wind: false,
        drones_per_station: per_station,
        ..cfg()
    }
}

fn info(name: &str) -> usize {
    INFO.iter().position(|&n| n == name).unwrap()
}

fn target(name: &str) -> usize {
    TARGET.iter().position(|&n| n == name).unwrap()
}

const KIND: usize = 5; // DRONE column "task_kind"
const WATER: usize = 2;

/// A simple deterministic planner: a deciding drone takes the fire slot `pick` chooses among
/// those it may take, and otherwise answers -1.
fn plan(v: &AllocVec, b: &Buffers, pick: impl Fn(usize, usize, usize) -> usize) -> Vec<i32> {
    let (n, slots) = (v.drones, v.slots);
    let stations = slots - v.cfg.max_targets;
    (0..v.envs.len() * n)
        .map(|k| {
            let (e, i) = (k / n, k % n);
            let mask = &b.action_mask[k * slots..][..slots];
            let fires: Vec<usize> = (stations..slots).filter(|&s| mask[s] == 1).collect();
            match (b.decide[k], fires.is_empty()) {
                (1, false) => fires[pick(e, i, fires.len())] as i32,
                _ => -1,
            }
        })
        .collect()
}

/// Runs the given seeds to the end, with the planner above or without anyone acting; returns
/// each environment's episode summary and fingerprint.
fn run(seeds: &[u64], threads: usize, config: AllocConfig, act: bool) -> Vec<(Vec<f32>, u64)> {
    let mut v = AllocVec::new(
        seeds.len(),
        scenario(),
        SimConfig::default(),
        config,
        0,
        threads,
    )
    .unwrap();
    v.autoreset = false;
    let mut b = Buffers::new(&v);
    v.reset(Some(seeds), b.all());
    let mut done: Vec<Option<(Vec<f32>, u64)>> = vec![None; seeds.len()];
    while done.iter().any(Option::is_none) {
        let actions = if act {
            plan(&v, &b, |_, i, n| i % n)
        } else {
            vec![-1; seeds.len() * v.drones]
        };
        v.step(&actions, b.all());
        #[allow(clippy::needless_range_loop)] // the index addresses three buffers
        for e in 0..seeds.len() {
            if done[e].is_none() && b.terminated[e] + b.truncated[e] > 0 {
                done[e] = Some((
                    b.info[e * INFO.len()..][..INFO.len()].to_vec(),
                    b.fingerprint[e],
                ));
            }
        }
    }
    done.into_iter().map(Option::unwrap).collect()
}

/// NaN-safe equality of two result lists (a summary holds NaN where nothing was measured).
fn same(a: &[(Vec<f32>, u64)], b: &[(Vec<f32>, u64)]) -> bool {
    let bits = |r: &[(Vec<f32>, u64)]| -> Vec<(Vec<u32>, u64)> {
        r.iter()
            .map(|(i, f)| (i.iter().map(|x| x.to_bits()).collect(), *f))
            .collect()
    };
    bits(a) == bits(b)
}

#[test]
fn sixteen_drones_never_get_closer_than_the_safe_distance() {
    // Separation kept, quick service on plenty of pads: as many drones in the air as can be.
    let config = AllocConfig {
        safe_m: 3.0,
        service_s: 1.0,
        pads: 16,
        ..cfg()
    };
    let mut v = AllocVec::new(24, scenario(), SimConfig::default(), config, 3, 0).unwrap();
    let mut b = Buffers::new(&v);
    v.reset(None, b.all());
    let (mut tasks, mut closest, mut steps) = (0.0, f32::INFINITY, 0u64);
    while tasks < 10_000.0 {
        // Spread the drones over the offered fires differently every step: lots of crossing.
        let actions = plan(&v, &b, |e, i, n| (e * 7 + i * 3 + steps as usize) % n);
        v.step(&actions, b.all());
        steps += 1;
        for e in 0..v.envs.len() {
            let terms = &b.terms[e * 16 * TERMS.len()..][..16 * TERMS.len()];
            tasks += terms.chunks(TERMS.len()).map(|t| t[0]).sum::<f32>(); // drops
            closest = closest.min(v.envs[e].min_separation());
        }
        assert!(
            steps < 2_000_000,
            "only {tasks} tasks finished: the drones are stuck"
        );
    }
    println!("{tasks} drops in {steps} steps, closest approach {closest:.2} m");
    assert!(
        closest >= config.safe_m - 1e-3,
        "closest approach {closest} m"
    );
}

/// The task with fires that have already burned for five minutes when the episode starts.
fn preburned() -> AllocConfig {
    AllocConfig {
        preburn_s: 300.0,
        ..cfg()
    }
}

/// `cfg` with an explorer at each station and a sensor on the attack drones.
fn explored() -> AllocConfig {
    AllocConfig {
        explorers_per_station: 1,
        sense_m: 20.0,
        ..cfg()
    }
}

/// `explored` with planners that only know what the drones have seen.
fn believing() -> AllocConfig {
    AllocConfig {
        plan_on_belief: true,
        ..explored()
    }
}

#[test]
fn an_episode_depends_only_on_its_seed() {
    for cfg in [cfg(), preburned(), explored(), believing()] {
        let seeds: Vec<u64> = (100..164).collect();
        let all = run(&seeds, 0, cfg, true);
        assert!(
            same(&all, &run(&seeds, 1, cfg, true)),
            "1 thread vs all threads"
        );
        for k in [0, 7, 63] {
            assert!(
                same(&run(&seeds[k..=k], 1, cfg, true), &all[k..=k]),
                "seed {} alone vs in a batch of 64",
                seeds[k]
            );
        }
        // Different seeds are different fires; the same seed is the same fire whatever the
        // policy.
        assert_ne!(all[0].1, all[1].1);
        let idle = run(&seeds, 0, cfg, false);
        assert!(
            all.iter().zip(&idle).all(|(a, b)| a.1 == b.1),
            "fingerprints must not depend on the policy"
        );
    }
}

#[test]
fn a_preburn_is_the_fire_left_alone() {
    let seeds: Vec<u64> = (200..208).collect();
    let make = |config| {
        let mut v =
            AllocVec::new(seeds.len(), scenario(), SimConfig::default(), config, 0, 1).unwrap();
        let mut b = Buffers::new(&v);
        v.reset(Some(&seeds), b.all());
        (v, b)
    };
    let (burned, b) = make(preburned());
    let (mut plain, _) = make(cfg());
    let time = GLOBAL.iter().position(|&g| g == "time").unwrap();
    for (e, (a, p)) in burned.envs.iter().zip(&mut plain.envs).enumerate() {
        // 300 s of the default 0.1 s ticks.
        (0..3000).for_each(|_| p.world.tick());
        assert_eq!(
            a.world.state_hash(),
            p.world.state_hash(),
            "seed {}",
            seeds[e]
        );
        assert!(
            a.world.fire.n_ignited > 4,
            "the fires spread before the start"
        );
        assert_eq!(a.episode_s(), 0.0);
        assert_eq!(
            b.globals[e * GLOBAL.len() + time],
            0.0,
            "the episode's clock starts at 0"
        );
    }
}

#[test]
fn the_time_limit_and_the_summary_count_from_the_start_of_the_episode() {
    let seeds: Vec<u64> = (300..316).collect();
    let (seconds, start, burned) = (info("seconds"), info("start_cells"), info("burned_cells"));
    let plain = run(&seeds, 0, cfg(), false);
    for (k, (info, _)) in run(&seeds, 0, preburned(), false).iter().enumerate() {
        assert_eq!(
            info[seconds], 600.0,
            "an idle episode lasts the whole time limit"
        );
        assert!(info[start] > plain[k].0[start] && info[start] < info[burned]);
    }
    assert!(plain.iter().all(|(info, _)| info[start] <= 4.0));

    // A cap on the cells cuts the pre-burn short; the overshoot is at most one fire step.
    let capped = AllocConfig {
        preburn_s: 3000.0,
        preburn_cells: 30,
        ..cfg()
    };
    for (info, _) in run(&seeds, 0, capped, false) {
        assert!(
            (30.0..60.0).contains(&info[start]),
            "started with {} cells",
            info[start]
        );
    }
}

#[test]
fn a_fire_that_dies_in_the_preburn_ends_the_episode_at_once() {
    // One burnable cell in bare ground: it burns out long before the pre-burn is over.
    let mut lonely = scenario();
    lonely.fuel.fill(0);
    let cell = lonely.grid.idx(30, 20);
    lonely.fuel[cell] = fuel::id("GR2").unwrap();
    let config = AllocConfig {
        preburn_s: 600.0,
        ..AllocConfig::default()
    };
    let mut v = AllocVec::new(1, lonely, SimConfig::default(), config, 0, 1).unwrap();
    v.autoreset = false;
    let mut b = Buffers::new(&v);
    v.reset(Some(&[1]), b.all());
    assert!(
        v.envs[0].world.time_s() < 600.0,
        "the pre-burn stops when the fire is out"
    );
    v.step(&vec![-1; v.drones], b.all());
    assert_eq!((b.terminated[0], b.truncated[0]), (1, 0));
    assert_eq!(
        b.info[..3],
        [1.0, 0.1, 1.0],
        "fire out after one tick, one cell burned"
    );
}

#[test]
fn dropping_water_saves_area_and_what_is_stopped_is_reported() {
    let seeds: Vec<u64> = (0..48).collect();
    let (fought, idle) = (run(&seeds, 0, cfg(), true), run(&seeds, 0, cfg(), false));
    let mean = |r: &[(Vec<f32>, u64)], name: &str| {
        r.iter().map(|x| x.0[info(name)]).sum::<f32>() / r.len() as f32
    };
    let (burned_fought, burned_idle) = (
        mean(&fought, "burned_fraction"),
        mean(&idle, "burned_fraction"),
    );
    println!(
        "burned fraction: {burned_fought:.4} with drops, {burned_idle:.4} without; per episode \
         {:.1} drops, {:.1} fronts stopped at {:.0} kW/m, first drop after {:.0} s",
        mean(&fought, "drops"),
        mean(&fought, "stopped_cells"),
        mean(&fought, "stopped_kw_mean"),
        mean(&fought, "first_drop_s"),
    );
    assert!(
        burned_fought < 0.8 * burned_idle,
        "{burned_fought} vs {burned_idle}"
    );
    assert!(
        mean(&fought, "drops") > 4.0 && mean(&fought, "stopped_cells") > 1.0,
        "drones must have dropped water and stopped fronts"
    );
    let config = cfg();
    for (x, _) in &fought {
        assert_eq!(x[info("water_l")], x[info("drops")] * config.drop_l);
        assert!(x[info("stopped_kw_max")] <= x[info("peak_kw")]);
        assert!(
            x[info("max_sortie_s")] <= config.endurance_s,
            "no drone outstays its endurance"
        );
        assert!(
            x[info("min_separation_m")].is_nan(),
            "not watched without safe_m"
        );
    }
    assert!(
        idle.iter().all(|(x, _)| x[info("drops")] == 0.0
            && x[info("stopped_cells")] == 0.0
            && x[info("first_drop_s")].is_nan()),
        "no action, no drops, nothing stopped"
    );
    // In no episode may fighting the fire burn more than doing nothing.
    let burned = info("burned_cells");
    assert!(
        fought
            .iter()
            .zip(&idle)
            .all(|(a, b)| a.0[burned] <= b.0[burned])
    );
}

#[test]
fn a_drone_flies_out_drops_comes_home_and_is_serviced() {
    let config = AllocConfig {
        service_s: 20.0,
        ..one_fire(1)
    };
    let mut v = AllocVec::new(1, scenario(), SimConfig::default(), config, 0, 1).unwrap();
    let mut b = Buffers::new(&v);
    v.reset(None, b.all());
    let (slots, stations) = (v.slots, 2);
    assert_eq!(b.decide, [1, 1], "everyone decides at the start");
    assert_eq!(
        &b.target_mask[..3],
        [1, 1, 1],
        "two stations and the one fire"
    );
    assert_eq!(b.target_mask[3..].iter().sum::<u8>(), 0);
    assert_eq!(b.targets[2], 1.0, "stations come first");
    assert_eq!(
        (b.drones[WATER], b.drones[KIND]),
        (1.0, TASK_DOCKED),
        "full tank, docked"
    );
    assert_eq!(
        &b.action_mask[..3],
        [0, 0, 1],
        "a docked drone can only take a fire"
    );

    // Drone 0 takes the fire, drone 1 waits and is asked again on the clock.
    v.step(&[stations as i32, -1], b.all());
    assert_eq!(b.elapsed_s[0], config.max_interval_s);
    assert_eq!(
        (b.decide[0], b.decide[1]),
        (0, 1),
        "only the waiting drone is asked"
    );
    assert_eq!(b.drones[KIND], TASK_FIRE);
    assert_eq!(b.drones[DRONE.len() + KIND], TASK_DOCKED);
    assert_eq!(
        b.targets[stations * TARGET.len() + target("inbound")],
        1.0,
        "one drop is on its way to the fire"
    );
    let start = (b.drones[DRONE.len()], b.drones[DRONE.len() + 1]);

    // On until the drop: the tank is empty and the drone turns home by itself.
    let (mut clock, mut dropped_at, mut ready_at, mut refilled) = (b.elapsed_s[0], None, None, 0.0);
    for _ in 0..200 {
        v.step(&[-1, -1], b.all());
        clock += b.elapsed_s[0];
        refilled += b.terms[1];
        if b.terms[0] > 0.0 {
            dropped_at = Some(clock);
        }
        if dropped_at.is_some() && b.decide[0] == 1 {
            ready_at = Some(clock);
            break;
        }
        if dropped_at.is_some() && b.drones[KIND] != TASK_DOCKED {
            assert_eq!(
                (b.drones[WATER], b.drones[KIND]),
                (0.0, TASK_STATION),
                "empty and on its way home"
            );
        }
        assert_eq!(b.terminated[0] + b.truncated[0], 0);
    }
    let (dropped_at, ready_at) = (dropped_at.unwrap(), ready_at.unwrap());
    // Station at (35, 55) m, the fire's block at (310, 210) m: 316 m, 31.6 s each way minus
    // the arrival radii. The test's clock only ticks when the waiting drone is asked (10 s).
    let first_drop = v.envs[0].summary()[info("first_drop_s")];
    assert!((30.5..31.6).contains(&first_drop), "{first_drop} s");
    assert!(dropped_at >= first_drop && dropped_at < first_drop + config.max_interval_s);
    // Home is the nearest station: the other one, 298 m from the block.
    let flight_home = ready_at - first_drop - config.service_s;
    assert!(
        (28.5..30.5).contains(&flight_home),
        "home and serviced after {ready_at} s"
    );
    assert_eq!(
        (b.drones[WATER], b.drones[KIND]),
        (1.0, TASK_DOCKED),
        "serviced"
    );
    assert!(
        (refilled - 1.0).abs() < 1e-6,
        "the refill term is one tank: {refilled}"
    );
    assert_eq!(
        (b.drones[DRONE.len()], b.drones[DRONE.len() + 1]),
        start,
        "the waiting drone never moved"
    );
    let summary = v.envs[0].summary();
    assert_eq!(summary[info("drops")], 1.0);
    assert!((summary[info("max_sortie_s")] - (ready_at - config.service_s)).abs() < 0.2);
    assert_eq!(b.action_mask[slots + stations], 0, "rows are per drone");
}

#[test]
fn need_counts_front_cells_with_the_hold_margin() {
    let config = one_fire(4);
    let sim = SimConfig::default();
    let mut v = AllocVec::new(1, scenario(), sim, config, 0, 1).unwrap();
    let mut b = Buffers::new(&v);
    v.reset(None, b.all());
    // One burning cell of grass in a light fire: the lowest critical load applies. To hold
    // it for `ember_s` while the water evaporates it takes exp(ember_s / water_tau_s) of it.
    let load = sim.water_crit_l_m2 * (sim.ember_s / sim.water_tau_s).exp();
    let per_drop = config.drop_efficiency * config.drop_l / 100.0;
    let fire = &b.targets[2 * TARGET.len()..][..TARGET.len()];
    assert!(
        (fire[target("need")] - load / per_drop).abs() < 1e-4,
        "{fire:?}"
    );
    assert_eq!(
        fire[target("cells")],
        0.25,
        "one front cell in a 2 x 2 block"
    );
    assert!((0.2..0.5).contains(&fire[target("intensity")]), "{fire:?}");

    // Water already on the cell counts with what is left of it when a drone sent now arrives
    // from the nearest station (298 m, 29.8 s): a cell holding exactly the load that stops it
    // still needs a little, or a front that takes many drops would never be stopped.
    let cell = v.envs[0].world.terrain.grid.idx(30, 20);
    let mut wet = v.envs[0].clone();
    wet.world.water.add(cell, load);
    let mut scratch = roshan_core::allocate::Scratch::new(&wet);
    wet.step(&vec![-1; v.drones], false, scratch.out(&wet));
    let seen = scratch.out(&wet).targets[2 * TARGET.len() + target("need")];
    assert!(seen > 0.05 && seen < 0.5, "needs {seen} drops");

    // Greedy sends what is needed and no more: two drops, from the nearer station.
    let mut actions = vec![0; v.drones];
    v.plan(Method::Greedy, 0.0, 0, &mut actions);
    assert_eq!(actions, [2, 2, -1, -1, -1, -1, -1, -1]);
    // The other rules agree when there is one block.
    for method in [Method::Head, Method::Flank] {
        let mut other = vec![0; v.drones];
        v.plan(method, 0.0, 0, &mut other);
        assert_eq!(other, actions);
    }
}

#[test]
fn a_drop_goes_to_the_neediest_front_cell() {
    // A fire a few cells large; three drones leave together for the same block and drop in
    // the same tick.
    let config = AllocConfig {
        preburn_cells: 12,
        preburn_s: 3000.0,
        ..one_fire(3)
    };
    let mut v = AllocVec::new(1, scenario(), SimConfig::default(), config, 0, 1).unwrap();
    let mut b = Buffers::new(&v);
    v.reset(None, b.all());
    let mut actions = vec![-1; v.drones];
    actions[..3].fill(2); // slot 2: the block that needs the most water
    v.step(&actions, b.all());
    let mut guard = 0;
    while b.terms.chunks(TERMS.len()).map(|t| t[0]).sum::<f32>() == 0.0 {
        v.step(&vec![-1; v.drones], b.all());
        guard += 1;
        assert!(guard < 100, "no drop");
    }
    let world = &v.envs[0].world;
    let wet: Vec<usize> = (0..world.water.load.len())
        .filter(|&c| world.water.load[c] > 0.0)
        .collect();
    let per_drop = config.drop_efficiency * config.drop_l / 100.0;
    let total: f32 = wet.iter().map(|&c| world.water.load[c]).sum();
    // A little of it has evaporated by the time the step returns.
    assert!(
        total <= 3.0 * per_drop && total > 2.9 * per_drop,
        "three drops landed: {total}"
    );
    let grid = world.terrain.grid;
    let block = |c: usize| (grid.xy(c).0 / 2, grid.xy(c).1 / 2);
    for &c in &wet {
        assert!(world.fire.front(c).is_some(), "water only on a front");
        assert_eq!(block(c), block(wet[0]), "all in the chosen block");
    }
    // Every front cell of the block got a drop before any got a second one.
    let fronts = (0..world.water.load.len())
        .filter(|&c| block(c) == block(wet[0]) && world.fire.front(c).is_some())
        .count();
    assert_eq!(wet.len(), fronts.min(3));
}

#[test]
fn fronts_too_intense_for_water_get_no_drops() {
    // Shrub in wind: the head runs far above the 4 MW/m limit, flanks and back stay below.
    let mut shrub = scenario();
    shrub.fuel.fill(fuel::id("SH5").unwrap());
    let config = AllocConfig {
        preburn_cells: 30,
        preburn_s: 3000.0,
        drones_per_station: 20,
        ..one_fire(20)
    };
    let sim = SimConfig::default();
    let mut v = AllocVec::new(1, shrub, sim, config, 0, 1).unwrap();
    v.autoreset = false;
    let mut b = Buffers::new(&v);
    v.reset(None, b.all());
    let (need, mw) = (target("need"), target("intensity"));
    let fires = |b: &Buffers| -> Vec<(f32, f32)> {
        (b.targets.chunks(TARGET.len()).zip(&b.target_mask).skip(2))
            .filter(|(_, m)| **m == 1)
            .map(|(t, _)| (t[need], t[mw]))
            .collect()
    };
    let limit = sim.water_limit_kw_m / 1000.0;
    assert!(fires(&b).iter().any(|f| f.1 > limit), "a head is offered");
    assert!(fires(&b).iter().all(|f| f.0.is_finite()));
    let mut actions = vec![0; v.drones];
    while b.terminated[0] + b.truncated[0] == 0 {
        v.plan(Method::Head, 0.0, 0, &mut actions);
        v.step(&actions, b.all());
    }
    let summary = &b.info[..INFO.len()];
    assert!(summary[info("drops")] > 50.0 && summary[info("stopped_cells")] > 5.0);
    assert!(
        summary[info("stopped_kw_max")] <= sim.water_limit_kw_m,
        "{summary:?}"
    );
    assert!(summary[info("peak_kw")] > sim.water_limit_kw_m);
}

#[test]
fn pads_serve_in_docking_order() {
    // One pad, three drones that come home in the same tick: they are ready one service
    // after the other, lowest index first, and the waiting is counted.
    // The fire is large enough to outlive three drops.
    let config = AllocConfig {
        pads: 1,
        service_s: 30.0,
        preburn_cells: 40,
        preburn_s: 3000.0,
        ..one_fire(3)
    };
    let mut near = scenario();
    near.stations.truncate(1);
    let mut v = AllocVec::new(1, near, SimConfig::default(), config, 0, 1).unwrap();
    let mut b = Buffers::new(&v);
    v.reset(None, b.all());
    v.step(&[1, 1, 1], b.all());
    let (mut clock, mut ready) = (b.elapsed_s[0], [0.0f32; 3]);
    // Nobody waits, so the first step only returns when a drone is back and serviced.
    for _ in 0..400 {
        for (at, &asked) in ready.iter_mut().zip(&b.decide) {
            if *at == 0.0 && asked == 1 {
                *at = clock;
            }
        }
        if ready[2] > 0.0 {
            break;
        }
        v.step(&[-1, -1, -1], b.all());
        clock += b.elapsed_s[0];
    }
    assert!(ready[0] > 0.0, "{ready:?}");
    assert!((ready[1] - ready[0] - 30.0).abs() < 0.11, "{ready:?}");
    assert!((ready[2] - ready[1] - 30.0).abs() < 0.11, "{ready:?}");
    let queue = v.envs[0].summary()[info("queue_s")];
    assert!((queue - 90.0).abs() < 0.3, "queued {queue} s");
}

#[test]
fn endurance_masks_far_fires() {
    // The fire's block is 316 m from station 0 and 298 m from station 1. A sortie is the way
    // there plus the way to the nearest station (station 1's 298 m): 61.4 s and 59.7 s.
    let sortie = |endurance_s: f32| {
        let config = AllocConfig {
            endurance_s,
            ..one_fire(1)
        };
        let mut v = AllocVec::new(1, scenario(), SimConfig::default(), config, 0, 1).unwrap();
        let mut b = Buffers::new(&v);
        v.reset(None, b.all());
        let mask = (b.action_mask[2], b.action_mask[v.slots + 2]);
        // Choosing the fire anyway is ignored: the drone stays docked and makes no drop.
        v.step(&[2, 2], b.all());
        (mask, (b.terms[0], b.terms[TERMS.len()]))
    };
    assert_eq!(sortie(900.0), ((1, 1), (1.0, 1.0)));
    assert_eq!(sortie(60.5).0, (0, 1));
    assert_eq!(sortie(59.0), ((0, 0), (0.0, 0.0)));
}

#[test]
fn docked_drones_are_neither_flown_nor_seen() {
    let config = one_fire(2);
    let mut v = AllocVec::new(1, scenario(), SimConfig::default(), config, 0, 1).unwrap();
    let mut b = Buffers::new(&v);
    v.reset(None, b.all());
    let n = v.drones;
    let (mut agent, mut neighbors, mut mask) = (
        vec![0.0; n * AGENT.len()],
        vec![0.0; n * n * 4],
        vec![0u8; n * n],
    );
    let velocity = vec![1.0; 2 * n]; // everyone is told to fly south-east
    let mut actions = vec![-1; n];
    actions[0] = 2; // only drone 0 is sent to the fire
    let start = b.drones.clone();
    for _ in 0..50 {
        v.fly_tick(
            &actions,
            &velocity,
            50.0,
            b.all(),
            &mut agent,
            &mut neighbors,
            &mut mask,
        );
        actions.fill(-1);
    }
    assert_ne!(b.drones[..2], start[..2], "the drone in the air moved");
    for i in 1..n {
        let row = &b.drones[i * DRONE.len()..][..DRONE.len()];
        assert_eq!(row[..2], start[i * DRONE.len()..][..2], "drone {i} stayed");
        assert_eq!(row[KIND], TASK_DOCKED);
        assert!(
            agent[i * AGENT.len()..][..AGENT.len()]
                .iter()
                .all(|&x| x == 0.0),
            "a docked drone sees nothing"
        );
    }
    // Drone 1 sits docked 5 m from where drone 0 took off: drone 0 does not see it. Slot 0
    // of a row is the map edge.
    assert_eq!(mask[1..n].iter().sum::<u8>(), 0);
    assert!(
        v.envs[0].min_separation().is_infinite(),
        "docked drones are no bodies"
    );
}

#[test]
fn bad_scenarios_and_settings_are_refused() {
    let none = Scenario {
        stations: vec![],
        ..scenario()
    };
    assert!(AllocVec::new(1, none, SimConfig::default(), cfg(), 0, 1).is_err());
    for bad in [
        AllocConfig { pads: 0, ..cfg() },
        AllocConfig {
            drop_l: 50.0,
            ..cfg()
        },
        AllocConfig {
            drop_efficiency: 1.5,
            ..cfg()
        },
        AllocConfig {
            safe_m: -1.0,
            ..cfg()
        },
        AllocConfig {
            sense_m: f32::NAN,
            ..cfg()
        },
        AllocConfig {
            plan_on_belief: true, // nothing to believe without a sensor
            ..cfg()
        },
        AllocConfig {
            drones_per_station: 0,
            ..cfg()
        },
    ] {
        assert!(AllocVec::new(1, scenario(), SimConfig::default(), bad, 0, 1).is_err());
    }
}

#[test]
fn rollout_is_repeatable_and_sometimes_differs_from_greedy() {
    let config = AllocConfig {
        drones_per_station: 3,
        preburn_cells: 20,
        preburn_s: 3000.0,
        ..cfg()
    };
    let mut v = AllocVec::new(6, scenario(), SimConfig::default(), config, 11, 0).unwrap();
    let mut b = Buffers::new(&v);
    v.reset(None, b.all());
    let n = v.envs.len() * v.drones;
    let (mut greedy, mut rollout, mut again) = (vec![0; n], vec![0; n], vec![0; n]);
    let mut differed = false;
    for _ in 0..40 {
        v.plan(Method::Greedy, 0.0, 0, &mut greedy);
        v.plan(Method::Rollout, 200.0, 4, &mut rollout);
        v.plan(Method::Rollout, 200.0, 4, &mut again);
        assert_eq!(rollout, again, "same state, same plan");
        differed |= rollout != greedy;
        v.step(&rollout, b.all());
    }
    assert!(
        differed,
        "lookahead should sometimes prefer something other than greedy"
    );
}

/// A random wind turns the whole schedule: the scenario's wind shift keeps its size, its time
/// and its speed, and a steady wind gets the direction it always got.
#[test]
fn a_random_wind_keeps_the_scenarios_wind_shifts() {
    let mut shifting = scenario();
    shifting.wind.push(Wind {
        t_s: 300.0,
        speed_ms: 8.0,
        to: [0.0, 1.0],
    });
    let build = |s: Scenario| {
        let mut v = AllocVec::new(1, s, SimConfig::default(), cfg(), 7, 1).unwrap();
        let mut b = Buffers::new(&v);
        v.reset(None, b.all());
        v.envs[0].world.weather.wind.clone()
    };
    let (steady, turned) = (build(scenario()), build(shifting));
    assert_eq!(turned[0], steady[0], "the first entry is drawn as before");
    assert_ne!(steady[0].to, [1.0, 0.0], "the wind was turned");
    assert_eq!((turned[1].t_s, turned[1].speed_ms), (300.0, 8.0));
    let (a, b) = (turned[0].to, turned[1].to);
    assert!(
        (a[0] * b[0] + a[1] * b[1]).abs() < 1e-6,
        "still a quarter turn apart"
    );
    assert!(
        (a[0] * b[1] - a[1] * b[0] - 1.0).abs() < 1e-6,
        "and in the same sense"
    );
}

#[test]
fn the_wind_scale_changes_every_speed_of_the_schedule() {
    let mut shifting = scenario();
    shifting.wind.push(Wind {
        t_s: 300.0,
        speed_ms: 8.0,
        to: [0.0, 1.0],
    });
    let config = AllocConfig {
        wind_scale: 0.5,
        ..one_fire(1)
    };
    let mut v = AllocVec::new(1, shifting, SimConfig::default(), config, 7, 1).unwrap();
    let mut b = Buffers::new(&v);
    v.reset(None, b.all());
    let wind = &v.envs[0].world.weather.wind;
    assert_eq!((wind[0].speed_ms, wind[1].speed_ms), (2.5, 4.0));
    assert_eq!((wind[0].to, wind[1].t_s), ([1.0, 0.0], 300.0));
}

#[test]
fn sensing_reports_what_was_seen_and_never_changes_the_fire() {
    let seeds: Vec<u64> = (0..16).collect();
    let sensing = AllocConfig {
        sense_m: 20.0,
        ..cfg()
    };
    let (blind, seeing) = (run(&seeds, 0, cfg(), true), run(&seeds, 0, sensing, true));
    let first = info("detect_s");
    let share = |x: f32| (0.0..=1.0).contains(&x);
    for (b, s) in blind.iter().zip(&seeing) {
        let cut = |r: &(Vec<f32>, u64)| (r.0[..first].to_vec(), r.1);
        assert!(same(&[cut(b)], &[cut(s)]), "a sensor changed the episode");
        assert!(
            b.0[first..].iter().all(|x| x.is_nan()),
            "no sensor: nothing measured"
        );
        // The drones fly to the fire, so they see it, and little else of the map.
        let at = |name: &str| s.0[info(name)];
        assert!(at("detect_s") > 0.0 && at("detect_s") < at("seconds"));
        assert!(at("seen_share") > 0.0 && at("seen_share") < 0.5);
        assert!(share(at("belief_miss")) && share(at("belief_phantom")));
    }
    // Docked drones see nothing: with nobody sent, all of the fire is missed.
    for (s, _) in run(&seeds[..4], 0, sensing, false) {
        assert!(s[first].is_nan() && s[info("seen_share")] == 0.0);
        assert!(s[info("belief_miss")] == 1.0 && s[info("belief_phantom")].is_nan());
    }
}

#[test]
fn explorers_fly_alone_carry_no_water_and_come_home_in_time() {
    // Two attack drones and one explorer per station; the explorers have to come home early.
    let config = AllocConfig {
        explorers_per_station: 1,
        endurance_s: 120.0,
        service_s: 20.0,
        ..one_fire(2)
    };
    let mut v = AllocVec::new(1, scenario(), SimConfig::default(), config, 0, 1).unwrap();
    v.autoreset = false;
    let mut b = Buffers::new(&v);
    v.reset(Some(&[5]), b.all());
    let (n, slots) = (v.drones, v.slots);
    assert_eq!(n, 6, "four attack drones, then two explorers");
    assert_eq!(
        v.envs[0].fleet(),
        vec![(2, 2), (2, 2)],
        "stations count attack drones"
    );
    assert!(
        v.envs[0].send(4, [30, 20]).is_err(),
        "nobody sends an explorer"
    );
    // Per explorer: 0 at the start, 1 once seen on its sweep, 2 docked again, 3 off again.
    let mut stage = [0; 2];
    while b.terminated[0] + b.truncated[0] == 0 {
        for (e, i) in (4..n).enumerate() {
            let kind = b.drones[i * DRONE.len() + KIND];
            assert!([TASK_EXPLORE, TASK_STATION, TASK_DOCKED].contains(&kind));
            assert_eq!(b.decide[i], 0, "an explorer is never asked");
            assert_eq!(
                b.drones[i * DRONE.len() + WATER],
                0.0,
                "and carries no water"
            );
            assert!(b.action_mask[i * slots..][..slots].iter().all(|&m| m == 0));
            assert_eq!(
                b.terms[i * TERMS.len()..][..2],
                [0.0, 0.0],
                "no drop, no refill"
            );
            let wanted = [TASK_EXPLORE, TASK_DOCKED, TASK_EXPLORE];
            if stage[e] < 3 && kind == wanted[stage[e]] {
                stage[e] += 1;
            }
        }
        // The attack drones wait; an explorer told to fly to a fire does not listen.
        let mut actions = vec![-1; n];
        actions[4..].fill(2);
        v.step(&actions, b.all());
    }
    assert_eq!(stage, [3, 3], "out, home for a service, out again");
    assert_eq!(b.info[info("drops")], 0.0);
    let sortie = b.info[info("max_sortie_s")];
    assert!(
        sortie > 60.0 && sortie <= 120.0,
        "longest sortie {sortie} s"
    );
}

#[test]
fn the_first_sweep_sees_the_whole_map_and_finds_the_fire() {
    // One explorer per station, nobody else in the air, one fire somewhere. Each explorer
    // has 300 m x 400 m to sweep: four lanes, about 220 s with the way there.
    let config = AllocConfig {
        explorers_per_station: 1,
        explorers_follow: false,
        random_ignitions: 1,
        time_limit_s: 260.0,
        ..cfg()
    };
    for (seed, (s, _)) in run(&(0..16).collect::<Vec<u64>>(), 0, config, false)
        .iter()
        .enumerate()
    {
        let at = |name: &str| s[info(name)];
        assert_eq!(at("seen_share"), 1.0, "seed {seed}: a gap in the sweep");
        assert!(
            at("detect_s") < at("seconds"),
            "seed {seed}: the fire was not found"
        );
        assert!(at("belief_miss") < 1.0 && at("belief_phantom") <= 1.0);
        assert_eq!(at("drops"), 0.0);
    }
}

#[test]
fn following_the_fire_keeps_the_belief_fresher_and_the_sweep_still_ends() {
    let seeds: Vec<u64> = (0..16).collect();
    let sweep = AllocConfig {
        explorers_per_station: 1,
        explorers_follow: false,
        random_ignitions: 1,
        ..cfg()
    };
    let follow = AllocConfig {
        explorers_follow: true,
        ..sweep
    };
    let (swept, followed) = (run(&seeds, 0, sweep, false), run(&seeds, 0, follow, false));
    let mean = |r: &[(Vec<f32>, u64)], name: &str| {
        r.iter().map(|x| x.0[info(name)]).sum::<f32>() / r.len() as f32
    };
    let (miss_swept, miss_followed) = (mean(&swept, "belief_miss"), mean(&followed, "belief_miss"));
    println!(
        "burning cells not known: {miss_swept:.3} sweeping, {miss_followed:.3} following; \
         phantoms {:.3} and {:.3}",
        mean(&swept, "belief_phantom"),
        mean(&followed, "belief_phantom")
    );
    assert!(miss_followed < miss_swept);
    for (a, b) in swept.iter().zip(&followed) {
        assert_eq!(
            a.0[info("burned_cells")],
            b.0[info("burned_cells")],
            "the same fire"
        );
        assert_eq!(
            b.0[info("seen_share")],
            1.0,
            "visits slow the sweep but leave no gap"
        );
        assert!(b.0[info("max_sortie_s")] <= 900.0);
    }
}

#[test]
fn planning_on_the_belief_waits_for_the_fire_to_be_found() {
    let seeds: Vec<u64> = (0..24).collect();
    let (truth, belief) = (
        run(&seeds, 0, explored(), true),
        run(&seeds, 0, believing(), true),
    );
    let idle = run(&seeds, 0, explored(), false);
    let mean = |r: &[(Vec<f32>, u64)], name: &str| {
        r.iter().map(|x| x.0[info(name)]).sum::<f32>() / r.len() as f32
    };
    let burned = |r: &[(Vec<f32>, u64)]| mean(r, "burned_fraction");
    println!(
        "burned: {:.4} idle, {:.4} planning on the fire, {:.4} on the belief; first drop after \
         {:.0} s and {:.0} s",
        burned(&idle),
        burned(&truth),
        burned(&belief),
        mean(&truth, "first_drop_s"),
        mean(&belief, "first_drop_s")
    );
    for ((t, b), seed) in truth.iter().zip(&belief).zip(&seeds) {
        assert_eq!(t.1, b.1, "the same fire");
        let (found, drop) = (b.0[info("detect_s")], b.0[info("first_drop_s")]);
        assert!(
            found < drop,
            "seed {seed}: a drop at {drop} s before the fire was found at {found} s"
        );
        assert!(
            t.0[info("first_drop_s")] < drop,
            "seed {seed}: knowing the fire is faster"
        );
    }
    // Late and half blind, the drones still save area, and less than with the true fire.
    assert!(burned(&belief) < burned(&idle));
    assert!(burned(&truth) < burned(&belief));

    // Nothing is known when the episode starts: no fire target is on offer.
    let v = AllocVec::new(1, scenario(), SimConfig::default(), believing(), 0, 1).unwrap();
    let mut v = v;
    let mut b = Buffers::new(&v);
    v.reset(Some(&[3]), b.all());
    assert_eq!(
        b.target_mask.iter().map(|&m| m as usize).sum::<usize>(),
        2,
        "the stations"
    );
    assert_eq!(
        b.globals[GLOBAL.iter().position(|&g| g == "burning").unwrap()],
        0.0
    );
}
