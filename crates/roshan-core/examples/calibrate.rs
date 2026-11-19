//! Tables behind the calibrated constants (docs/systems.md, "Calibration").
//!     cargo run --release -p roshan-core --example calibrate
//! Part 1: what each fuel model does in the simulator and how much water stops its head.
//! Part 2 and 3: candidate fuel models against spread rates observed in Scots pine and heath.

use std::sync::Arc;

use roshan_core::allocate::AllocConfig;
use roshan_core::fuel::{self, SB40};
use roshan_core::rothermel::{FuelBed, Moisture};
use roshan_core::terrain::{Grid, Terrain};
use roshan_core::world::{SimConfig, Weather, World};

fn moisture(dead: f32, herb: f32, woody: f32) -> Moisture {
    Moisture {
        d1: dead,
        d10: dead + 0.01,
        d100: dead + 0.02,
        herb,
        woody,
    }
}

/// Head rate of spread (m/s) the simulator uses for a fuel on flat ground.
fn head(name: &str, m: Moisture, wind_10m_ms: f32) -> f32 {
    let grid = Grid {
        nx: 3,
        ny: 3,
        cell_m: 10.0,
    };
    let terrain = Terrain::uniform(grid, fuel::id(name).expect("fuel model exists"));
    let weather = Weather::steady(m, wind_10m_ms, [1.0, 0.0]);
    World::new(Arc::new(terrain), weather, SimConfig::default(), 1)
        .spread(1, 1)
        .head
}

fn bed(name: &str, m: Moisture) -> FuelBed {
    FuelBed::new(&SB40[fuel::id(name).unwrap() as usize - 1], &m)
}

fn rmse(pairs: &[(f32, f32)]) -> f32 {
    (pairs.iter().map(|(a, b)| (a - b).powi(2)).sum::<f32>() / pairs.len() as f32).sqrt()
}

fn main() {
    let cfg = SimConfig::default();
    let dry = moisture(0.06, 0.60, 0.90);
    println!("1. Head fire on flat ground, dead fuel 6 %, live 60/90 %, wind at 10 m.");
    // What one drop of the default drone leaves on a 10 m cell, and how much more than the
    // critical load a cell needs to stay stopped for `ember_s` while the water evaporates.
    let drone = AllocConfig::default();
    let per_drop = drone.drop_efficiency * drone.drop_l / 100.0;
    let hold = (cfg.ember_s / cfg.water_tau_s).exp();
    println!(
        "   ROS m/min | intensity kW/m | water to stop it L/m2 | drops of {} L per 10 m cell\n",
        drone.drop_l
    );
    println!("{:5} {:>30} {:>30} {:>30}", "", "2 m/s", "5 m/s", "10 m/s");
    for name in [
        "GR1", "GR2", "GR6", "GS2", "SH2", "SH3", "SH5", "SH8", "SH9", "TU1", "TU2", "TU3", "TU5",
        "TL2", "TL3", "TL8",
    ] {
        print!("{name:5}");
        for wind in [2.0, 5.0, 10.0] {
            let ros = head(name, dry, wind);
            let kw = ros * bed(name, dry).heat_per_area_kj_m2() as f32;
            let per_mw = match &name[..2] {
                "GR" => cfg.water_per_mw_grass_l_m2,
                "TL" | "TU" => cfg.water_per_mw_timber_l_m2,
                _ => cfg.water_per_mw_l_m2,
            };
            let crit = cfg.water_crit_l_m2.max(per_mw * kw / 1000.0);
            let drops = (crit * hold / per_drop).ceil();
            print!(" {:7.1} {:7.0} {:6.2} {:6.0}", ros * 60.0, kw, crit, drops);
        }
        println!();
    }

    // A check that was not an input: Ausonio et al. 2021 (Drones 5: 17) take a critical flow
    // of about 4.3 L/min per metre of front for scrub at 10 km/h wind (their Fig. 3a). Here
    // a 10 m front in grass at 5 m/s is stopped by soaking its cell while the head crosses it.
    let ros = head("GR2", dry, 5.0);
    let (crossing_s, litres) = (10.0 / ros, cfg.water_crit_l_m2 * hold * 100.0);
    println!(
        "\n   Grass head at 5 m/s: {:.0} L delivered ({:.0} L carried) stop 10 m of front; the \
         head crosses a cell in {crossing_s:.0} s,\n   so that is {:.1} L/min per metre of \
         front. Ausonio et al. 2021: about 4.3.",
        litres,
        litres / drone.drop_efficiency,
        litres / 10.0 / (crossing_s / 60.0),
    );

    // Tanskanen et al. 2007 (Int. J. Wildland Fire 16): experimental fires in Scots pine stands,
    // fitted against the wind inside the stand: ROS = 0.663 exp(2.286 U) m/min at 7-11 % fuel
    // moisture, 0.481 exp(1.221 U) at 13-15 %.
    println!(
        "\n2. Scots pine: ROS m/min against the wind inside the stand (Tanskanen et al. 2007)."
    );
    for (dead, a, b) in [(0.09, 0.663, 2.286), (0.14, 0.481, 1.221f32)] {
        let m = moisture(dead, 1.0, 1.2);
        println!("\n   dead fuel {:.0} %, live 100/120 %", dead * 100.0);
        println!(
            "   {:9} {:>6} {:>6} {:>6} {:>6}   RMSE",
            "U m/s", 0.0, 0.2, 0.4, 0.6
        );
        let winds = [0.0, 0.2, 0.4, 0.6f32];
        let observed = winds.map(|u| a * (b * u).exp());
        println!("   {:9} {}", "observed", row(&observed));
        for name in ["TL3", "TL8", "TU1", "TU2", "TU5"] {
            let model = winds.map(|u| bed(name, m).spread([u as f64, 0.0], [0.0; 2]).head * 60.0);
            let pairs: Vec<_> = model.into_iter().zip(observed).collect();
            println!("   {name:9} {}   {:.2}", row(&model), rmse(&pairs));
        }
    }

    // Davies et al. 2009 (J. Appl. Ecol. 46): 27 fires in Calluna heath,
    // ROS = 0.791 + 7.917 h^2 U m/min, h heather height in m, U wind at 10-20 m in m/s.
    // Conditions of those fires: dead fuel 15-29 %, live 55-97 %, h 0.12-0.51 m.
    println!("\n3. Heath: ROS m/min against the wind at 10 m (Davies et al. 2009), heather 0.3 m,");
    println!("   dead fuel 20 %, live 75 %.");
    let m = moisture(0.20, 0.75, 0.75);
    let winds = [2.0, 4.0, 6.0, 8.0f32];
    println!(
        "   {:9} {:>6} {:>6} {:>6} {:>6}   RMSE",
        "U m/s", 2.0, 4.0, 6.0, 8.0
    );
    let observed = winds.map(|u| 0.791 + 7.917 * 0.3 * 0.3 * u);
    println!("   {:9} {}", "observed", row(&observed));
    for name in ["SH2", "SH3", "SH5", "SH8", "SH9"] {
        let model = winds.map(|u| head(name, m, u) * 60.0);
        let pairs: Vec<_> = model.into_iter().zip(observed).collect();
        println!("   {name:9} {}   {:.2}", row(&model), rmse(&pairs));
    }
}

fn row(values: &[f32]) -> String {
    let cells: Vec<_> = values.iter().map(|v| format!("{v:6.2}")).collect();
    cells.join(" ")
}
