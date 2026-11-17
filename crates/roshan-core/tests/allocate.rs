//! M4 checks for the allocate task: separation, determinism, decisions and suppression.

use roshan_core::allocate::{AllocConfig, AllocVec, Bufs, DRONE, GLOBAL, INFO, TARGET, TERMS};
use roshan_core::fuel;
use roshan_core::rothermel::Moisture;
use roshan_core::scenario::Scenario;
use roshan_core::terrain::Grid;
use roshan_core::world::{SimConfig, Wind};

/// Grassland, 600 m x 400 m, with two stations and `drones` drones parked in a row.
fn scenario(drones: u32) -> Scenario {
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
        drones: (0..drones).map(|i| [2 + i % 8, 2 + i / 8]).collect(),
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

fn cfg() -> AllocConfig {
    AllocConfig {
        time_limit_s: 600.0,
        random_ignitions: 4,
        random_wind: true,
        ..AllocConfig::default()
    }
}

/// A simple deterministic planner: a drone with water takes the fire slot `pick` chooses among
/// those it may take; a drone without goes to station 0.
fn plan(v: &AllocVec, b: &Buffers, pick: impl Fn(usize, usize, usize) -> usize) -> Vec<i32> {
    let (n, slots) = (v.drones, v.slots);
    let stations = slots - v.cfg.max_targets;
    (0..v.envs.len() * n)
        .map(|k| {
            let (e, i) = (k / n, k % n);
            let mask = &b.action_mask[k * slots..][..slots];
            let fires: Vec<usize> = (stations..slots).filter(|&s| mask[s] == 1).collect();
            match (b.decide[k], fires.is_empty()) {
                (0, _) => -1,
                (_, false) => fires[pick(e, i, fires.len())] as i32,
                (_, true) if b.drones[k * DRONE.len() + 2] < 1.0 => 0,
                _ => -1,
            }
        })
        .collect()
}

/// Runs the given seeds to the end with the largest-fire-first planner; returns each
/// environment's episode summary and fingerprint.
fn run(seeds: &[u64], threads: usize, config: AllocConfig, act: bool) -> Vec<(Vec<f32>, u64)> {
    let mut v = AllocVec::new(
        seeds.len(),
        scenario(4),
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

#[test]
fn sixteen_drones_never_get_closer_than_the_safe_distance() {
    let config = AllocConfig {
        max_interval_s: 5.0,
        ..cfg()
    };
    let mut v = AllocVec::new(24, scenario(16), SimConfig::default(), config, 3, 0).unwrap();
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

#[test]
fn an_episode_depends_only_on_its_seed() {
    let seeds: Vec<u64> = (100..164).collect();
    let all = run(&seeds, 0, cfg(), true);
    assert_eq!(all, run(&seeds, 1, cfg(), true), "1 thread vs all threads");
    for k in [0, 7, 63] {
        assert_eq!(
            run(&seeds[k..=k], 1, cfg(), true)[0],
            all[k],
            "seed {} alone vs in a batch of 64",
            seeds[k]
        );
    }
    // Different seeds are different fires; the same seed is the same fire whatever the policy.
    assert_ne!(all[0].1, all[1].1);
    let idle = run(&seeds, 0, cfg(), false);
    assert!(
        all.iter().zip(&idle).all(|(a, b)| a.1 == b.1),
        "fingerprints must not depend on the policy"
    );
}

#[test]
fn dropping_water_saves_area_and_idle_drones_stay_put() {
    let seeds: Vec<u64> = (0..48).collect();
    let (fought, idle) = (run(&seeds, 0, cfg(), true), run(&seeds, 0, cfg(), false));
    let mean = |r: &[(Vec<f32>, u64)], col: usize| {
        r.iter().map(|x| x.0[col]).sum::<f32>() / r.len() as f32
    };
    let (burned_fought, burned_idle) = (mean(&fought, 3), mean(&idle, 3));
    println!(
        "burned fraction: {burned_fought:.4} with drops, {burned_idle:.4} without; drops per episode {:.1}",
        mean(&fought, 5)
    );
    assert!(
        burned_fought < 0.8 * burned_idle,
        "{burned_fought} vs {burned_idle}"
    );
    assert!(
        mean(&fought, 5) > 4.0 && mean(&fought, 4) > 0.0,
        "drones must have dropped water"
    );
    assert!(
        idle.iter().all(|x| x.0[5] == 0.0 && x.0[4] == 0.0),
        "no action, no drops"
    );
    // In no episode may fighting the fire burn more than doing nothing.
    assert!(fought.iter().zip(&idle).all(|(a, b)| a.0[2] <= b.0[2]));
}

#[test]
fn decisions_masks_and_tanks_follow_the_rules() {
    let config = AllocConfig {
        random_ignitions: 0,
        random_wind: false,
        ..cfg()
    };
    let mut v = AllocVec::new(1, scenario(2), SimConfig::default(), config, 0, 1).unwrap();
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
    assert_eq!((b.drones[2], b.drones[5]), (1.0, 0.0), "full tank, no task");

    // Drone 0 takes the fire, drone 1 keeps doing nothing.
    v.step(&[stations as i32, -1], b.all());
    assert!(b.elapsed_s[0] > 0.0 && b.elapsed_s[0] <= config.max_interval_s + 0.2);
    assert_eq!(b.drones[5], 1.0, "drone 0 is on a fire task");
    assert_eq!(b.drones[DRONE.len() + 5], 0.0, "drone 1 has none");
    let start = (b.drones[DRONE.len()], b.drones[DRONE.len() + 1]);

    // Fly on until drone 0 has made two drops: then its tank is empty.
    let (mut drops, mut guard) = (0.0, 0);
    while drops < 2.0 {
        let a0 = if b.decide[0] == 1 && b.action_mask[stations] == 1 {
            stations as i32
        } else {
            -1
        };
        v.step(&[a0, -1], b.all());
        drops += b.terms[0];
        guard += 1;
        assert!(
            guard < 2000 && b.terminated[0] == 0,
            "two drops must happen before the fire is out"
        );
    }
    assert_eq!(b.drones[2], 0.0, "two drops empty the tank");
    assert!(
        b.action_mask[stations..slots].iter().all(|m| *m == 0),
        "no water, no fire target"
    );
    assert_eq!(
        &b.action_mask[..stations],
        [1, 1],
        "stations stay available"
    );
    assert_eq!(
        (b.drones[DRONE.len()], b.drones[DRONE.len() + 1]),
        start,
        "the idle drone never moved"
    );

    // Choosing a fire anyway is ignored; going to the station refills the tank.
    v.step(&[stations as i32, -1], b.all());
    assert_eq!(b.drones[5], 0.0, "the forbidden choice was ignored");
    let mut refilled = 0.0;
    for _ in 0..400 {
        let a0 = if b.decide[0] == 1 && b.drones[2] < 1.0 {
            0
        } else {
            -1
        };
        v.step(&[a0, -1], b.all());
        refilled += b.terms[1];
        if b.drones[2] >= 1.0 {
            break;
        }
    }
    assert_eq!(b.drones[2], 1.0);
    assert!(
        (refilled - 1.0).abs() < 1e-3,
        "refill term adds up to one tank: {refilled}"
    );
}

#[test]
fn bad_scenarios_are_refused() {
    let none = Scenario {
        drones: vec![],
        ..scenario(1)
    };
    assert!(AllocVec::new(1, none, SimConfig::default(), cfg(), 0, 1).is_err());
    let stacked = Scenario {
        drones: vec![[2, 2], [2, 2]],
        ..scenario(1)
    };
    assert!(AllocVec::new(1, stacked, SimConfig::default(), cfg(), 0, 1).is_err());
}

#[test]
fn rollout_is_never_worse_than_greedy_on_its_own_horizon_and_is_repeatable() {
    let mut v = AllocVec::new(6, scenario(3), SimConfig::default(), cfg(), 11, 0).unwrap();
    let mut b = Buffers::new(&v);
    v.reset(None, b.all());
    let n = v.envs.len() * v.drones;
    let (mut greedy, mut rollout, mut again) = (vec![0; n], vec![0; n], vec![0; n]);
    let mut differed = false;
    for _ in 0..40 {
        v.greedy(&mut greedy);
        v.rollout(120.0, 4, &mut rollout);
        v.rollout(120.0, 4, &mut again);
        assert_eq!(rollout, again, "same state, same plan");
        differed |= rollout != greedy;
        v.step(&rollout, b.all());
    }
    assert!(
        differed,
        "lookahead should sometimes prefer something other than greedy"
    );
}
