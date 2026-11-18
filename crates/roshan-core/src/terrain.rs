//! Static map data. One `Terrain` is shared (through `Arc`) by every world and clone that
//! uses the same map, so cloning a world never copies it.
//!
//! Coordinates: x grows with the column index, y with the row index. Cell index = y*nx + x.
//! Wind and slope vectors use the same axes, so no compass convention exists inside the core.

use crate::fuel::SB40;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    pub nx: u32,
    pub ny: u32,
    /// Edge length of a (square) cell in metres.
    pub cell_m: f32,
}

impl Grid {
    /// Number of cells. A grid is never empty (`Terrain::new` rejects that).
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.nx as usize * self.ny as usize
    }

    pub fn idx(&self, x: u32, y: u32) -> usize {
        (y * self.nx + x) as usize
    }

    pub fn xy(&self, idx: usize) -> (u32, u32) {
        (idx as u32 % self.nx, idx as u32 / self.nx)
    }

    /// The cell displaced by (dx, dy) from `idx`, or None if that leaves the grid.
    pub fn offset(&self, idx: usize, dx: i32, dy: i32) -> Option<usize> {
        let (x, y) = self.xy(idx);
        let (x, y) = (x as i32 + dx, y as i32 + dy);
        if x >= 0 && y >= 0 && x < self.nx as i32 && y < self.ny as i32 {
            Some(self.idx(x as u32, y as u32))
        } else {
            None
        }
    }
}

pub struct Terrain {
    pub grid: Grid,
    /// Fuel id per cell: 0 = non-burnable, 1..=40 = Scott & Burgan model (see `fuel::SB40`).
    pub fuel: Vec<u8>,
    /// Terrain gradient per cell (rise over run along x and y); it points uphill.
    pub slope: Vec<[f32; 2]>,
    /// Steepest slope on the map, used to bound how fast fire can possibly spread.
    pub max_tan: f32,
    /// Which fuel ids occur on this map.
    pub present: [bool; SB40.len() + 1],
    /// CLC+ Backbone class per cell where known; only the FireSPIN model reads it.
    pub landcover: Option<Vec<u8>>,
    /// Number of cells that can burn, counted once: burned shares divide by it at every step.
    pub burnable_cells: usize,
}

impl Terrain {
    /// Builds a terrain from per-cell arrays. Slopes are derived from `elevation_m`;
    /// without it the map is flat.
    pub fn new(grid: Grid, fuel: Vec<u8>, elevation_m: Option<&[f32]>) -> Result<Self, String> {
        let n = grid.len();
        if n == 0 || grid.cell_m.is_nan() || grid.cell_m <= 0.0 {
            return Err(format!("empty grid or non-positive cell size: {grid:?}"));
        }
        if fuel.len() != n {
            return Err(format!("fuel has {} cells, grid has {n}", fuel.len()));
        }
        if let Some(&bad) = fuel.iter().find(|&&f| f as usize > SB40.len()) {
            return Err(format!("fuel id {bad} is outside 0..={}", SB40.len()));
        }
        let mut present = [false; SB40.len() + 1];
        for &f in &fuel {
            present[f as usize] = true;
        }

        let mut slope = vec![[0.0f32; 2]; n];
        if let Some(z) = elevation_m {
            if z.len() != n {
                return Err(format!("elevation has {} cells, grid has {n}", z.len()));
            }
            let (nx, ny) = (grid.nx as i32, grid.ny as i32);
            let at = |x: i32, y: i32| z[grid.idx(x as u32, y as u32)];
            for y in 0..ny {
                for x in 0..nx {
                    // Central differences, one-sided at the map edge.
                    let (x0, x1) = ((x - 1).max(0), (x + 1).min(nx - 1));
                    let (y0, y1) = ((y - 1).max(0), (y + 1).min(ny - 1));
                    let dzdx = if x1 > x0 {
                        (at(x1, y) - at(x0, y)) / ((x1 - x0) as f32 * grid.cell_m)
                    } else {
                        0.0
                    };
                    let dzdy = if y1 > y0 {
                        (at(x, y1) - at(x, y0)) / ((y1 - y0) as f32 * grid.cell_m)
                    } else {
                        0.0
                    };
                    slope[grid.idx(x as u32, y as u32)] = [dzdx, dzdy];
                }
            }
        }
        let max_tan = slope
            .iter()
            .map(|s| (s[0] * s[0] + s[1] * s[1]).sqrt())
            .fold(0.0, f32::max);
        Ok(Self {
            grid,
            burnable_cells: fuel.iter().filter(|&&f| f != 0).count(),
            fuel,
            slope,
            max_tan,
            present,
            landcover: None,
        })
    }

    /// A flat map covered by one fuel.
    pub fn uniform(grid: Grid, fuel_id: u8) -> Self {
        Self::new(grid, vec![fuel_id; grid.len()], None).expect("valid uniform terrain")
    }

    pub fn burnable(&self, idx: usize) -> bool {
        self.fuel[idx] != crate::fuel::NON_BURNABLE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexing_and_slope_on_a_non_square_grid() {
        let grid = Grid {
            nx: 7,
            ny: 4,
            cell_m: 10.0,
        };
        assert_eq!(grid.len(), 28);
        assert_eq!(grid.idx(6, 3), 27);
        assert_eq!(grid.xy(grid.idx(5, 2)), (5, 2));
        assert_eq!(grid.offset(grid.idx(0, 0), -1, 0), None);
        assert_eq!(grid.offset(grid.idx(6, 3), 0, 1), None);
        assert_eq!(grid.offset(grid.idx(3, 1), 2, 1), Some(grid.idx(5, 2)));

        // A plane rising 1 m per 10 m along x and falling 2 m per 10 m along y.
        let z: Vec<f32> = (0..grid.len())
            .map(|i| {
                let (x, y) = grid.xy(i);
                x as f32 - 2.0 * y as f32
            })
            .collect();
        let t = Terrain::new(grid, vec![1; grid.len()], Some(&z)).unwrap();
        for s in &t.slope {
            assert!(
                (s[0] - 0.1).abs() < 1e-6 && (s[1] + 0.2).abs() < 1e-6,
                "{s:?}"
            );
        }
        assert!((t.max_tan - 0.05f32.sqrt()).abs() < 1e-6);
        assert!(t.present[1] && !t.present[2]);

        assert!(Terrain::new(grid, vec![41; grid.len()], None).is_err());
        assert!(Terrain::new(grid, vec![1; 3], None).is_err());
    }
}
