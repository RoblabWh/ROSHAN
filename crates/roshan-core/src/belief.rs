//! What the drones have seen of the fire: per cell the state at the last look and when that
//! was. One map shared by all drones. It is filled from the fire and never read by it, so
//! sensing cannot change how the fire burns.
//!
//! ponytail: the sensor is perfect inside its circle and every drone knows at once what any
//! drone saw. Add a miss probability (a new `rng::Stream`, keyed by tick and cell) and a
//! communication range when an experiment needs them.
//! ponytail: only the fire state is kept. A planner that reads the belief also needs the water
//! need and intensity of each front (`FireState::front`) and a list of the believed-burning
//! cells; add both then.

use crate::fire::{BURNING, UNBURNED};
use crate::terrain::Grid;

/// `seen` of a cell nobody has looked at yet.
pub const NEVER: u32 = u32::MAX;

#[derive(Clone, Debug)]
pub struct Belief {
    /// Per cell: the fire state (`fire::UNBURNED`...) at the last look; unburned before the first.
    pub state: Vec<u8>,
    /// Per cell: the world tick of the last look, or `NEVER`.
    pub seen: Vec<u32>,
    n_seen: u32,
    n_burning: u32,
    detected: Option<u64>,
    /// Sums of the shares `score` measured, and how often each was defined.
    miss: f64,
    miss_steps: u32,
    phantom: f64,
    phantom_steps: u32,
}

impl Belief {
    pub fn new(cells: usize) -> Self {
        Self {
            state: vec![UNBURNED; cells],
            seen: vec![NEVER; cells],
            n_seen: 0,
            n_burning: 0,
            detected: None,
            miss: 0.0,
            miss_steps: 0,
            phantom: 0.0,
            phantom_steps: 0,
        }
    }

    /// One picture: copies `truth` (the fire's state per cell) for every cell whose centre is
    /// within `radius_m` of `pos` (metres). Parts of the circle off the map are ignored.
    pub fn look(&mut self, grid: Grid, truth: &[u8], pos: [f32; 2], radius_m: f32, tick: u64) {
        let m = grid.cell_m;
        // Cells whose centre can be in reach along one axis, clipped to the map.
        let span = |p: f32, n: u32| {
            let lo = ((p - radius_m) / m - 0.5).ceil().max(0.0) as u32;
            let hi = ((p + radius_m) / m - 0.5).floor().min(n as f32 - 1.0);
            lo..if hi < 0.0 { 0 } else { hi as u32 + 1 }
        };
        let (xs, r2) = (span(pos[0], grid.nx), radius_m * radius_m);
        for y in span(pos[1], grid.ny) {
            let dy = (y as f32 + 0.5) * m - pos[1];
            for x in xs.clone() {
                let dx = (x as f32 + 0.5) * m - pos[0];
                let inside = dx * dx + dy * dy <= r2; // false for a position that is NaN
                if !inside {
                    continue;
                }
                let cell = grid.idx(x, y);
                let (old, new) = (self.state[cell], truth[cell]);
                self.n_seen += (self.seen[cell] == NEVER) as u32;
                self.n_burning += (new == BURNING) as u32;
                self.n_burning -= (old == BURNING) as u32;
                if new == BURNING {
                    self.detected.get_or_insert(tick);
                }
                (self.state[cell], self.seen[cell]) = (new, tick as u32);
            }
        }
    }

    /// Compares the belief with the fire: `burning` are the cells burning now, in any order.
    /// Call once per fire step; the averages are over the steps where a share exists.
    pub fn score(&mut self, burning: impl Iterator<Item = u32>) {
        let (mut real, mut held) = (0u32, 0u32);
        for cell in burning {
            real += 1;
            held += (self.state[cell as usize] == BURNING) as u32;
        }
        if real > 0 {
            self.miss += (real - held) as f64 / real as f64;
            self.miss_steps += 1;
        }
        if self.n_burning > 0 {
            self.phantom += (self.n_burning - held) as f64 / self.n_burning as f64;
            self.phantom_steps += 1;
        }
    }

    /// The world tick at which a burning cell was first seen.
    pub fn detected(&self) -> Option<u64> {
        self.detected
    }

    /// Share of the map's cells looked at at least once.
    pub fn seen_share(&self) -> f32 {
        self.n_seen as f32 / self.seen.len() as f32
    }

    /// Share of the burning cells the belief did not hold as burning, averaged over the fire
    /// steps so far; NaN before the first.
    pub fn miss(&self) -> f32 {
        (self.miss / self.miss_steps as f64) as f32
    }

    /// Share of the believed-burning cells that did not burn (any more), averaged over the
    /// fire steps on which any cell was believed burning; NaN if none ever was.
    pub fn phantom(&self) -> f32 {
        (self.phantom / self.phantom_steps as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fire::BURNED;

    const GRID: Grid = Grid {
        nx: 30,
        ny: 20,
        cell_m: 10.0,
    };

    #[test]
    fn the_sensor_is_a_circle() {
        let truth = vec![UNBURNED; GRID.len()];
        let mut b = Belief::new(GRID.len());
        // Over the centre of cell (15, 10) with 70 m: the cells within 7 cell widths.
        b.look(GRID, &truth, [155.0, 105.0], 70.0, 4);
        let seen = |x: u32, y: u32| b.seen[GRID.idx(x, y)] == 4;
        assert!(seen(15, 10) && seen(22, 10) && seen(15, 3) && seen(8, 10));
        assert!(!seen(23, 10), "beyond the radius");
        assert!(!seen(20, 15), "the corner of the square around the circle");
        assert!(seen(19, 15), "dx 4, dy 5: 64 m");
        let cells = b.seen.iter().filter(|&&t| t != NEVER).count();
        assert_eq!(cells, 149);
        assert!((b.seen_share() - 149.0 / 600.0).abs() < 1e-6);
    }

    #[test]
    fn a_circle_off_the_map_sees_only_what_is_on_it() {
        let truth = vec![UNBURNED; GRID.len()];
        let mut b = Belief::new(GRID.len());
        for pos in [[-500.0, 50.0], [50.0, 5000.0], [f32::NAN, 0.0]] {
            b.look(GRID, &truth, pos, 70.0, 1);
        }
        assert_eq!(b.seen_share(), 0.0);
        b.look(GRID, &truth, [-20.0, -20.0], 70.0, 1); // the corner is in reach
        assert!(b.seen[0] == 1 && b.seen_share() < 0.05);
    }

    #[test]
    fn the_belief_is_the_last_look_and_goes_stale() {
        let mut truth = vec![UNBURNED; GRID.len()];
        let (near, far) = (GRID.idx(5, 5), GRID.idx(25, 15));
        (truth[near], truth[far]) = (BURNING, BURNING);
        let mut b = Belief::new(GRID.len());
        let burning = |t: &[u8]| {
            let cells: Vec<u32> = (0..t.len() as u32)
                .filter(|&c| t[c as usize] == BURNING)
                .collect();
            cells.into_iter()
        };
        b.score(burning(&truth));
        assert!(b.detected().is_none() && b.miss() == 1.0 && b.phantom().is_nan());

        b.look(GRID, &truth, [55.0, 55.0], 20.0, 7);
        assert_eq!(b.detected(), Some(7));
        assert_eq!((b.state[near], b.state[far]), (BURNING, UNBURNED));
        b.score(burning(&truth));
        assert_eq!(b.miss(), 0.75, "all missed, then one of two");
        assert_eq!(b.phantom(), 0.0);

        // Water knocks the seen cell down and the other burns out: nobody looks, so the
        // belief keeps a fire that is gone.
        (truth[near], truth[far]) = (UNBURNED, BURNED);
        b.score(burning(&truth));
        assert_eq!(b.state[near], BURNING);
        assert_eq!(b.phantom(), 0.5, "right once, then wrong once");

        b.look(GRID, &truth, [55.0, 55.0], 20.0, 9);
        b.look(GRID, &truth, [255.0, 155.0], 20.0, 9);
        assert_eq!((b.state[near], b.state[far]), (UNBURNED, BURNED));
        assert_eq!(b.detected(), Some(7), "the first sighting stays");
        b.score(burning(&truth));
        assert_eq!(b.phantom(), 0.5, "nothing believed burning: not counted");
    }
}
