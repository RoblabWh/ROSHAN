//! A scenario: a map plus everything placed on it before the clock starts.
//!
//! The core never reads files. Python loads and saves scenarios and hands plain arrays over;
//! `meta` is an opaque JSON string (georeference, name) that travels along untouched.

use std::sync::Arc;

use crate::rothermel::Moisture;
use crate::terrain::{Grid, Terrain};
use crate::world::{SimConfig, Weather, Wind, World};

#[derive(Clone, Debug)]
pub struct Scenario {
    pub grid: Grid,
    /// Fuel id per cell (see `Terrain::fuel`).
    pub fuel: Vec<u8>,
    /// Elevation per cell in metres; a flat map without it.
    pub elevation: Option<Vec<f32>>,
    /// Land-cover class per cell that the fuel was derived from. The simulation ignores it;
    /// it is kept for display and for fire models that work on land cover.
    pub landcover: Option<Vec<u8>>,
    /// Cells (x, y) that burn at time zero.
    pub ignitions: Vec<[u32; 2]>,
    /// Cells drones start in.
    pub drones: Vec<[u32; 2]>,
    /// Cells holding a ground station.
    pub stations: Vec<[u32; 2]>,
    pub moisture: Moisture,
    /// Wind schedule; the first entry applies from the start.
    pub wind: Vec<Wind>,
    pub meta: String,
}

impl Scenario {
    /// Builds the world at time zero. Fails on a malformed map or anything placed outside it.
    pub fn world(&self, cfg: SimConfig, seed: u64) -> Result<World, String> {
        let mut terrain = Terrain::new(self.grid, self.fuel.clone(), self.elevation.as_deref())?;
        if self
            .landcover
            .as_ref()
            .is_some_and(|l| l.len() != self.grid.len())
        {
            return Err("land cover does not have one value per cell".into());
        }
        terrain.landcover.clone_from(&self.landcover);
        if self.wind.is_empty() {
            return Err("the wind schedule is empty".into());
        }
        let placed = [
            ("ignition", &self.ignitions),
            ("drone", &self.drones),
            ("station", &self.stations),
        ];
        for (what, cells) in placed {
            if let Some(c) = cells
                .iter()
                .find(|c| c[0] >= self.grid.nx || c[1] >= self.grid.ny)
            {
                return Err(format!(
                    "{what} at {c:?} is outside the {}x{} map",
                    self.grid.nx, self.grid.ny
                ));
            }
        }
        let weather = Weather {
            moisture: self.moisture,
            wind: self.wind.clone(),
        };
        let mut world = World::new(Arc::new(terrain), weather, cfg, seed);
        for c in &self.ignitions {
            world.ignite(c[0], c[1]); // a cell that cannot burn simply stays unburned
        }
        let centre = |c: &[u32; 2]| {
            [
                (c[0] as f32 + 0.5) * self.grid.cell_m,
                (c[1] as f32 + 0.5) * self.grid.cell_m,
            ]
        };
        world.drones.pos = self.drones.iter().map(centre).collect();
        world.drones.vel = vec![[0.0; 2]; self.drones.len()];
        Ok(world)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario() -> Scenario {
        let grid = Grid {
            nx: 12,
            ny: 7,
            cell_m: 10.0,
        };
        Scenario {
            grid,
            fuel: vec![2; grid.len()],
            elevation: None,
            landcover: None,
            ignitions: vec![[3, 2]],
            drones: vec![[0, 0], [11, 6]],
            stations: vec![[1, 1]],
            moisture: Moisture {
                d1: 0.06,
                d10: 0.07,
                d100: 0.08,
                herb: 0.6,
                woody: 0.9,
            },
            wind: vec![Wind {
                t_s: 0.0,
                speed_ms: 4.0,
                to: [1.0, 0.0],
            }],
            meta: String::new(),
        }
    }

    #[test]
    fn builds_a_world_and_rejects_what_does_not_fit() {
        let world = scenario().world(SimConfig::default(), 1).unwrap();
        assert_eq!(world.fire.n_burning(), 1);
        assert_eq!(world.drones.pos, vec![[5.0, 5.0], [115.0, 65.0]]);

        let mut outside = scenario();
        outside.drones.push([12, 0]);
        assert!(
            outside
                .world(SimConfig::default(), 1)
                .err()
                .unwrap()
                .contains("outside")
        );
        let mut windless = scenario();
        windless.wind.clear();
        assert!(windless.world(SimConfig::default(), 1).is_err());
        let mut short = scenario();
        short.fuel.pop();
        assert!(short.world(SimConfig::default(), 1).is_err());
    }
}
