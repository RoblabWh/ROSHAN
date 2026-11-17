//! M2 checks for the navigate task: episode logic, reset semantics and determinism.

use roshan_core::navigate::{
    AGENT, Bufs, INFO, NavConfig, NavVec, REASON_COLLISION, REASON_OUT_OF_BOUNDS, REASON_TIMEOUT,
    TERMS,
};
use roshan_core::terrain::Grid;
use roshan_core::world::SimConfig;

const GRID: Grid = Grid {
    nx: 35,
    ny: 36,
    cell_m: 10.0,
};

struct Buffers {
    agent: Vec<f32>,
    neighbors: Vec<f32>,
    mask: Vec<u8>,
    terms: Vec<f32>,
    terminated: Vec<u8>,
    truncated: Vec<u8>,
    info: Vec<f32>,
}

impl Buffers {
    fn new(envs: usize, n: usize) -> Self {
        Self {
            agent: vec![0.0; envs * n * AGENT.len()],
            neighbors: vec![0.0; envs * n * n * 4],
            mask: vec![0; envs * n * n],
            terms: vec![0.0; envs * n * TERMS.len()],
            terminated: vec![0; envs],
            truncated: vec![0; envs],
            info: vec![0.0; envs * INFO.len()],
        }
    }

    fn all(&mut self) -> Bufs<'_> {
        Bufs {
            agent: &mut self.agent,
            neighbors: &mut self.neighbors,
            mask: &mut self.mask,
            terms: &mut self.terms,
            terminated: &mut self.terminated,
            truncated: &mut self.truncated,
            info: &mut self.info,
        }
    }
}

fn make(envs: usize, cfg: NavConfig, threads: usize) -> (NavVec, Buffers) {
    let mut vec = NavVec::new(envs, GRID, SimConfig::default(), cfg, 7, threads);
    let mut b = Buffers::new(envs, cfg.drones);
    vec.reset(&mut b.agent, &mut b.neighbors, &mut b.mask);
    (vec, b)
}

/// Full speed straight at the waypoint, easing off within 10 m. No avoidance at all.
fn straight_to_goal(vec: &NavVec) -> Vec<f32> {
    vec.envs
        .iter()
        .flat_map(|env| {
            env.world
                .drones
                .pos
                .iter()
                .zip(env.goals())
                .flat_map(|(p, g)| {
                    let (dx, dy) = (g[0] - p[0], g[1] - p[1]);
                    let range = dx.hypot(dy).max(1e-6);
                    let speed = (range / 10.0).min(1.0);
                    [dx / range * speed, dy / range * speed]
                })
        })
        .collect()
}

/// Runs `steps` steps with the straight-line pilot and returns (episodes, successes, reasons).
fn fly(vec: &mut NavVec, b: &mut Buffers, steps: usize) -> (u32, u32, [u32; 4]) {
    let (mut episodes, mut successes, mut reasons) = (0, 0, [0; 4]);
    for _ in 0..steps {
        let actions = straight_to_goal(vec);
        vec.step(&actions, b.all());
        for e in 0..vec.envs.len() {
            if b.terminated[e] + b.truncated[e] > 0 {
                let info = &b.info[e * INFO.len()..][..INFO.len()];
                episodes += 1;
                successes += info[0] as u32;
                reasons[info[1] as usize] += 1;
                assert_eq!(
                    info[0] == 1.0,
                    info[1] == 0.0,
                    "success means no failure reason"
                );
            }
        }
    }
    (episodes, successes, reasons)
}

#[test]
fn a_lone_drone_flying_straight_always_succeeds() {
    let cfg = NavConfig {
        drones: 1,
        ..NavConfig::default()
    };
    let (mut vec, mut b) = make(64, cfg, 2);
    let (episodes, successes, _) = fly(&mut vec, &mut b, 4000);
    assert!(episodes > 300, "only {episodes} episodes finished");
    assert_eq!(
        successes, episodes,
        "one drone cannot collide and has ample time"
    );
}

#[test]
fn four_drones_flying_straight_sometimes_collide() {
    let (mut vec, mut b) = make(256, NavConfig::default(), 4);
    let (episodes, successes, reasons) = fly(&mut vec, &mut b, 4000);
    let rate = successes as f32 / episodes as f32;
    println!(
        "straight-line pilot, 4 drones: {rate:.3} success over {episodes} episodes, reasons {reasons:?}"
    );
    assert!(
        reasons[REASON_COLLISION as usize] > 0,
        "without avoidance some drones must collide"
    );
    assert_eq!(
        reasons[REASON_OUT_OF_BOUNDS as usize], 0,
        "waypoints are inside the map"
    );
    assert!((0.3..0.999).contains(&rate), "success rate {rate}");
}

#[test]
fn leaving_the_map_ends_the_episode_and_the_next_step_resets() {
    let cfg = NavConfig {
        drones: 2,
        station_start: 0.0,
        ..NavConfig::default()
    };
    let (mut vec, mut b) = make(8, cfg, 1);
    let west = [-1.0, 0.0].repeat(8 * 2);
    let term = |name: &str| TERMS.iter().position(|t| *t == name).unwrap();
    let mut ended = [false; 8];
    for _ in 0..400 {
        vec.step(&west, b.all());
        // The index also addresses four parallel buffers, so a plain range is the clear form.
        #[allow(clippy::needless_range_loop)]
        for e in 0..8 {
            if ended[e] {
                // The step after the end: a fresh episode, no reward terms, no flags.
                assert_eq!((b.terminated[e], b.truncated[e]), (0, 0));
                assert!(
                    b.terms[e * 2 * TERMS.len()..][..2 * TERMS.len()]
                        .iter()
                        .all(|t| *t == 0.0)
                );
                ended[e] = false;
            } else if b.terminated[e] == 1 {
                let terms = &b.terms[e * 2 * TERMS.len()..][..2 * TERMS.len()];
                let out = terms[term("out_of_bounds")] + terms[TERMS.len() + term("out_of_bounds")];
                let hit = terms[term("collision")] + terms[TERMS.len() + term("collision")];
                assert!(out > 0.0 || hit > 0.0);
                let reason = b.info[e * INFO.len() + 1];
                assert!(reason == REASON_OUT_OF_BOUNDS || reason == REASON_COLLISION);
                ended[e] = true;
            }
        }
    }
    assert!(
        vec.envs.iter().all(|env| env.world.tick < 400),
        "every environment ended and restarted"
    );
}

#[test]
fn hovering_runs_into_the_time_limit_and_freezes_without_autoreset() {
    let (mut vec, mut b) = make(4, NavConfig::default(), 1);
    vec.autoreset = false;
    let hover = vec![0.0; 4 * 4 * 2];
    let mut truncated_at = None;
    for step in 0..6000 {
        vec.step(&hover, b.all());
        if b.truncated[0] == 1 {
            truncated_at = Some(step);
            assert_eq!(
                b.terminated[0], 0,
                "a cut-off episode is not a terminal state"
            );
            assert_eq!(b.info[1], REASON_TIMEOUT);
            let timeout = TERMS.iter().position(|t| *t == "timeout").unwrap();
            assert_eq!(b.terms[timeout], 1.0);
            break;
        }
    }
    let at = truncated_at.expect("the time limit must cut the episode off");
    // 3 waypoints * 502 m diagonal / 10 m/s * 2.5 = 377 s = 3765 ticks.
    assert!((3700..3800).contains(&at), "cut off at step {at}");
    let frozen = b.agent.clone();
    vec.step(&hover, b.all());
    assert_eq!((b.terminated[0], b.truncated[0]), (0, 0));
    assert_eq!(b.agent, frozen, "finished environments stay as they are");
}

#[test]
fn results_do_not_depend_on_the_thread_count() {
    let run = |threads: usize| {
        let (mut vec, mut b) = make(200, NavConfig::default(), threads);
        let mut trace = Vec::new();
        for _ in 0..600 {
            let actions = straight_to_goal(&vec);
            vec.step(&actions, b.all());
            trace.extend_from_slice(&b.terms);
            trace.extend_from_slice(&b.info);
        }
        (trace, b.agent, b.neighbors, b.mask)
    };
    assert!(run(1) == run(std::thread::available_parallelism().map_or(8, |n| n.get())));
}

#[test]
fn observations_are_normalised_and_masked() {
    let (mut vec, mut b) = make(64, NavConfig::default(), 2);
    for _ in 0..300 {
        let actions = straight_to_goal(&vec);
        vec.step(&actions, b.all());
        assert!(
            b.agent
                .iter()
                .all(|x| x.is_finite() && x.abs() <= 1.0 + 1e-5)
        );
        assert!(
            b.neighbors
                .iter()
                .all(|x| x.is_finite() && x.abs() <= 1.0 + 1e-5)
        );
        // An empty slot is all zeros.
        for (slot, m) in b.neighbors.chunks(4).zip(&b.mask) {
            assert!(*m == 1 || slot.iter().all(|x| *x == 0.0));
        }
    }
    assert!(b.mask.contains(&1) && b.mask.contains(&0));
}
