//! What the drones have seen of the fire: per cell the state at the last look and when that
//! was. One map shared by all drones. It is filled from the fire and never read by it, so
//! sensing cannot change how the fire burns.
//!
//! ponytail: the sensor is perfect inside its circle and every drone knows at once what any
//! drone saw. Add a miss probability (a new `rng::Stream`, keyed by tick and cell) and a
//! communication range when an experiment needs them.
//! Of a burning cell the sensor also reads what it takes to stop its front (`fire::Front`:
//! water need and fireline intensity), so that a planner can work from the belief instead of
//! the fire (`AllocConfig::plan_on_belief`).
//! ponytail: the intensity of a front is taken to be measured exactly. Derive it from the
//! believed state, the fuel and the wind when sensor errors are modelled.

use crate::fire::{BURNED, BURNING, FireState, Front, UNBURNED};
use crate::terrain::Grid;

/// `seen` of a cell nobody has looked at yet.
pub const NEVER: u32 = u32::MAX;

#[derive(Clone, Debug)]
pub struct Belief {
    /// Per cell: the fire state (`fire::UNBURNED`...) at the last look; unburned before the first.
    pub state: Vec<u8>,
    /// Per cell: the world tick of the last look, or `NEVER`.
    pub seen: Vec<u32>,
    /// Per cell: the front seen at the last look, as in `Front`; `water` is NaN for a cell
    /// that had no front (not burning, or only burning out).
    water: Vec<f32>,
    kw: Vec<f32>,
    /// The cells believed burning, in the order they were found. Between `look` and the next
    /// `score` it may still hold cells that were seen to have stopped burning.
    lit: Vec<u32>,
    stale: bool,
    n_seen: u32,
    n_burning: u32,
    n_burned: u32,
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
            water: vec![f32::NAN; cells],
            kw: vec![0.0; cells],
            lit: Vec::new(),
            stale: false,
            n_seen: 0,
            n_burning: 0,
            n_burned: 0,
            detected: None,
            miss: 0.0,
            miss_steps: 0,
            phantom: 0.0,
            phantom_steps: 0,
        }
    }

    /// One picture: copies `truth` (the fire's state per cell) and, of burning cells, their
    /// `front` (`FireState::front`) for every cell whose centre is within `radius_m` of `pos`
    /// (metres). Parts of the circle off the map are ignored.
    pub fn look(
        &mut self,
        grid: Grid,
        truth: &[u8],
        front: impl Fn(usize) -> Option<Front>,
        pos: [f32; 2],
        radius_m: f32,
        tick: u64,
    ) {
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
                self.n_burned += (new == BURNED) as u32;
                self.n_burned -= (old == BURNED) as u32;
                (self.water[cell], self.kw[cell]) = (f32::NAN, 0.0);
                if new == BURNING {
                    self.detected.get_or_insert(tick);
                    if old != BURNING {
                        self.lit.push(cell as u32);
                    }
                    if let Some(f) = front(cell) {
                        (self.water[cell], self.kw[cell]) = (f.water_l_m2, f.kw_m);
                    }
                } else if old == BURNING {
                    self.stale = true;
                }
                (self.state[cell], self.seen[cell]) = (new, tick as u32);
            }
        }
    }

    /// Call once per fire step, after the looks. Tidies the list of believed-burning cells
    /// and compares the belief with the fire: `burning` are the cells burning now, in any
    /// order. The averages are over the steps where a share exists.
    pub fn score(&mut self, burning: impl Iterator<Item = u32>) {
        if std::mem::take(&mut self.stale) {
            // A cell leaves and enters the list at most once per fire step, so after this
            // every believed-burning cell is in it exactly once.
            let state = &self.state;
            self.lit.retain(|&c| state[c as usize] == BURNING);
        }
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

    /// Replaces the believed fire by a forecast of it (`AllocConfig::forecast`): a fire the
    /// model has run on from what was seen, corrected by every look. When and whether a cell
    /// was looked at stays. Call after the looks and before `score`.
    pub fn adopt(&mut self, fire: &FireState) {
        self.state.copy_from_slice(&fire.state);
        for &cell in &self.lit {
            self.water[cell as usize] = f32::NAN;
        }
        self.lit.clear();
        self.lit.extend(fire.burning());
        for &cell in &self.lit {
            if let Some(f) = fire.front(cell as usize) {
                (self.water[cell as usize], self.kw[cell as usize]) = (f.water_l_m2, f.kw_m);
            }
        }
        self.stale = false;
        self.n_burning = self.lit.len() as u32;
        self.n_burned = fire.n_ignited - self.n_burning;
    }

    /// The cells believed burning (valid after `score`), in no particular order.
    pub fn burning(&self) -> &[u32] {
        &self.lit
    }

    /// The front of a believed-burning cell as it was last seen; None for a cell that is not
    /// believed burning or had no front then.
    pub fn front(&self, cell: usize) -> Option<Front> {
        (!self.water[cell].is_nan()).then_some(Front {
            water_l_m2: self.water[cell],
            kw_m: self.kw[cell],
        })
    }

    /// Cells believed burning, and cells believed burning or burned.
    pub fn counts(&self) -> (u32, u32) {
        (self.n_burning, self.n_burning + self.n_burned)
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
        b.look(GRID, &truth, |_| None, [155.0, 105.0], 70.0, 4);
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
            b.look(GRID, &truth, |_| None, pos, 70.0, 1);
        }
        assert_eq!(b.seen_share(), 0.0);
        b.look(GRID, &truth, |_| None, [-20.0, -20.0], 70.0, 1); // the corner is in reach
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

        let front = Front {
            water_l_m2: 0.3,
            kw_m: 800.0,
        };
        b.look(GRID, &truth, |_| Some(front), [55.0, 55.0], 20.0, 7);
        assert_eq!(b.detected(), Some(7));
        assert_eq!((b.front(near), b.front(far)), (Some(front), None));
        assert_eq!((b.burning(), b.counts()), (&[near as u32][..], (1, 1)));
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

        b.look(GRID, &truth, |_| None, [55.0, 55.0], 20.0, 9);
        b.look(GRID, &truth, |_| None, [255.0, 155.0], 20.0, 9);
        assert_eq!((b.state[near], b.state[far]), (UNBURNED, BURNED));
        assert_eq!(b.detected(), Some(7), "the first sighting stays");
        b.score(burning(&truth));
        assert!(b.burning().is_empty() && b.front(near).is_none());
        assert_eq!(b.counts(), (0, 1), "one cell seen burned");
        assert_eq!(b.phantom(), 0.5, "nothing believed burning: not counted");
    }
}
