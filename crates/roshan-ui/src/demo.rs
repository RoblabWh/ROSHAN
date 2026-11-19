//! A small made-up landscape for trying the viewer without any map data.

use roshan_core::fuel;
use roshan_core::rothermel::Moisture;
use roshan_core::scenario::Scenario;
use roshan_core::terrain::Grid;
use roshan_core::world::Wind;

/// 1.6 km x 1.2 km of rolling hills: grass on open ground, shrub and forest litter in irregular
/// patches, a lake in the lowest hollow and a winding road with one gap. One fire is already
/// burning.
pub fn demo_scenario() -> Scenario {
    let grid = Grid {
        nx: 160,
        ny: 120,
        cell_m: 10.0,
    };
    let id = |name| fuel::id(name).expect("fuel model exists");
    let (grass, grass_shrub, shrub, litter) = (id("GR2"), id("GS2"), id("SH5"), id("TL3"));

    let mut elevation: Vec<f32> = (0..grid.len())
        .map(|i| {
            let (x, y) = grid.xy(i);
            90.0 * fbm(x as f32 / 70.0, y as f32 / 70.0, 1)
        })
        .collect();
    // The lake fills the lowest 4 % of the map up to a level surface.
    let mut sorted = elevation.clone();
    sorted.sort_by(f32::total_cmp);
    let lake_level = sorted[grid.len() / 25];

    let fuel: Vec<u8> = (0..grid.len())
        .map(|i| {
            let (x, y) = grid.xy(i);
            let (fx, fy) = (x as f32, y as f32);
            let lake = elevation[i] < lake_level;
            let road = (fx - 70.0 - 6.0 * (fy / 18.0).sin()).abs() < 1.0 && !(52..56).contains(&y);
            let growth = fbm(fx / 35.0, fy / 35.0, 2);
            if lake || road {
                0
            } else if growth > 0.67 {
                litter
            } else if growth > 0.62 {
                shrub
            } else if growth > 0.57 {
                grass_shrub
            } else {
                grass
            }
        })
        .collect();
    for z in &mut elevation {
        *z = z.max(lake_level);
    }

    Scenario {
        grid,
        fuel,
        elevation: Some(elevation),
        landcover: None,
        ignitions: vec![[20, 60]],
        stations: vec![[3, 3]],
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
        meta: r#"{"name": "demo"}"#.to_string(),
    }
}

/// A fixed pseudo-random value in [0, 1) for a lattice point.
fn lattice(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x9E37_79B1)
        ^ (y as u32).wrapping_mul(0x85EB_CA77)
        ^ seed.wrapping_mul(0xC2B2_AE3D);
    h = (h ^ (h >> 15)).wrapping_mul(0x2C1B_3C6D);
    h = (h ^ (h >> 12)).wrapping_mul(0x297A_2D39);
    ((h ^ (h >> 15)) >> 8) as f32 / (1 << 24) as f32
}

/// Smooth value noise in [0, 1): four octaves of interpolated lattice values.
fn fbm(x: f32, y: f32, seed: u32) -> f32 {
    let octave = |x: f32, y: f32, seed: u32| {
        let (x0, y0) = (x.floor(), y.floor());
        let (tx, ty) = (x - x0, y - y0);
        let (tx, ty) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
        let (ix, iy) = (x0 as i32, y0 as i32);
        let at = |dx, dy| lattice(ix + dx, iy + dy, seed);
        let (top, bottom) = (
            at(0, 0) + (at(1, 0) - at(0, 0)) * tx,
            at(0, 1) + (at(1, 1) - at(0, 1)) * tx,
        );
        top + (bottom - top) * ty
    };
    (0..4)
        .map(|o| octave(x * (1 << o) as f32, y * (1 << o) as f32, seed + o) / (2 << o) as f32)
        .sum::<f32>()
        / 0.9375
}
