//! The 40 standard fire behavior fuel models (Scott & Burgan 2005, RMRS-GTR-153).
//!
//! Values are kept in the units they are published in so they can be compared with the source
//! line by line: loads in tons/acre, surface-area-to-volume ratios in 1/ft, depth in ft.
//! Cross-checked on 2026-11-12 against the US Forest Service Behave source (fuelModels.cpp).

#[derive(Clone, Copy, Debug)]
pub struct FuelModel {
    pub name: &'static str,
    /// Oven-dry loads, tons/acre: 1-h, 10-h, 100-h dead, live herbaceous, live woody.
    pub load: [f32; 5],
    /// Surface-area-to-volume ratio, 1/ft: 1-h dead, live herbaceous, live woody.
    pub sav: [f32; 3],
    pub depth_ft: f32,
    /// Dead fuel moisture of extinction, fraction.
    pub mx_dead: f32,
    /// Heat content, Btu/lb.
    pub heat: f32,
    /// Dynamic models move live herbaceous load to dead as the herbs cure.
    pub dynamic: bool,
}

const fn m(
    name: &'static str,
    load: [f32; 5],
    sav: [f32; 3],
    depth_ft: f32,
    mx_dead: f32,
    dynamic: bool,
) -> FuelModel {
    FuelModel {
        name,
        load,
        sav,
        depth_ft,
        mx_dead,
        heat: 8000.0,
        dynamic,
    }
}

/// Fuel id 0 means non-burnable; ids 1..=40 index this table (id - 1).
pub const NON_BURNABLE: u8 = 0;

// One line per model, in the order of the published table: rustfmt would spread it over 300 lines.
#[rustfmt::skip]
pub const SB40: [FuelModel; 40] = [
    m("GR1", [0.10, 0.0, 0.0, 0.30, 0.0], [2200.0, 2000.0, 1500.0], 0.4, 0.15, true),
    m("GR2", [0.10, 0.0, 0.0, 1.00, 0.0], [2000.0, 1800.0, 1500.0], 1.0, 0.15, true),
    m("GR3", [0.10, 0.40, 0.0, 1.50, 0.0], [1500.0, 1300.0, 1500.0], 2.0, 0.30, true),
    m("GR4", [0.25, 0.0, 0.0, 1.90, 0.0], [2000.0, 1800.0, 1500.0], 2.0, 0.15, true),
    m("GR5", [0.40, 0.0, 0.0, 2.50, 0.0], [1800.0, 1600.0, 1500.0], 1.5, 0.40, true),
    // GR6 is the one model with a heat content of 9000 Btu/lb.
    FuelModel { heat: 9000.0, ..m("GR6", [0.10, 0.0, 0.0, 3.40, 0.0], [2200.0, 2000.0, 1500.0], 1.5, 0.40, true) },
    m("GR7", [1.00, 0.0, 0.0, 5.40, 0.0], [2000.0, 1800.0, 1500.0], 3.0, 0.15, true),
    m("GR8", [0.50, 1.00, 0.0, 7.30, 0.0], [1500.0, 1300.0, 1500.0], 4.0, 0.30, true),
    m("GR9", [1.00, 1.00, 0.0, 9.00, 0.0], [1800.0, 1600.0, 1500.0], 5.0, 0.40, true),
    m("GS1", [0.20, 0.0, 0.0, 0.50, 0.65], [2000.0, 1800.0, 1800.0], 0.9, 0.15, true),
    m("GS2", [0.50, 0.50, 0.0, 0.60, 1.00], [2000.0, 1800.0, 1800.0], 1.5, 0.15, true),
    m("GS3", [0.30, 0.25, 0.0, 1.45, 1.25], [1800.0, 1600.0, 1600.0], 1.8, 0.40, true),
    m("GS4", [1.90, 0.30, 0.10, 3.40, 7.10], [1800.0, 1600.0, 1600.0], 2.1, 0.40, true),
    m("SH1", [0.25, 0.25, 0.0, 0.15, 1.30], [2000.0, 1800.0, 1600.0], 1.0, 0.15, true),
    m("SH2", [1.35, 2.40, 0.75, 0.0, 3.85], [2000.0, 1800.0, 1600.0], 1.0, 0.15, false),
    m("SH3", [0.45, 3.00, 0.0, 0.0, 6.20], [1600.0, 1800.0, 1400.0], 2.4, 0.40, false),
    m("SH4", [0.85, 1.15, 0.20, 0.0, 2.55], [2000.0, 1800.0, 1600.0], 3.0, 0.30, false),
    m("SH5", [3.60, 2.10, 0.0, 0.0, 2.90], [750.0, 1800.0, 1600.0], 6.0, 0.15, false),
    m("SH6", [2.90, 1.45, 0.0, 0.0, 1.40], [750.0, 1800.0, 1600.0], 2.0, 0.30, false),
    m("SH7", [3.50, 5.30, 2.20, 0.0, 3.40], [750.0, 1800.0, 1600.0], 6.0, 0.15, false),
    m("SH8", [2.05, 3.40, 0.85, 0.0, 4.35], [750.0, 1800.0, 1600.0], 3.0, 0.40, false),
    m("SH9", [4.50, 2.45, 0.0, 1.55, 7.00], [750.0, 1800.0, 1500.0], 4.4, 0.40, true),
    m("TU1", [0.20, 0.90, 1.50, 0.20, 0.90], [2000.0, 1800.0, 1600.0], 0.6, 0.20, true),
    m("TU2", [0.95, 1.80, 1.25, 0.0, 0.20], [2000.0, 1800.0, 1600.0], 1.0, 0.30, false),
    m("TU3", [1.10, 0.15, 0.25, 0.65, 1.10], [1800.0, 1600.0, 1400.0], 1.3, 0.30, true),
    m("TU4", [4.50, 0.0, 0.0, 0.0, 2.00], [2300.0, 1800.0, 2000.0], 0.5, 0.12, false),
    m("TU5", [4.00, 4.00, 3.00, 0.0, 3.00], [1500.0, 1800.0, 750.0], 1.0, 0.25, false),
    m("TL1", [1.00, 2.20, 3.60, 0.0, 0.0], [2000.0, 1800.0, 1600.0], 0.2, 0.30, false),
    m("TL2", [1.40, 2.30, 2.20, 0.0, 0.0], [2000.0, 1800.0, 1600.0], 0.2, 0.25, false),
    m("TL3", [0.50, 2.20, 2.80, 0.0, 0.0], [2000.0, 1800.0, 1600.0], 0.3, 0.20, false),
    m("TL4", [0.50, 1.50, 4.20, 0.0, 0.0], [2000.0, 1800.0, 1600.0], 0.4, 0.25, false),
    m("TL5", [1.15, 2.50, 4.40, 0.0, 0.0], [2000.0, 1800.0, 1600.0], 0.6, 0.25, false),
    m("TL6", [2.40, 1.20, 1.20, 0.0, 0.0], [2000.0, 1800.0, 1600.0], 0.3, 0.25, false),
    m("TL7", [0.30, 1.40, 8.10, 0.0, 0.0], [2000.0, 1800.0, 1600.0], 0.4, 0.25, false),
    m("TL8", [5.80, 1.40, 1.10, 0.0, 0.0], [1800.0, 1800.0, 1600.0], 0.3, 0.35, false),
    m("TL9", [6.65, 3.30, 4.15, 0.0, 0.0], [1800.0, 1800.0, 1600.0], 0.6, 0.35, false),
    m("SB1", [1.50, 3.00, 11.00, 0.0, 0.0], [2000.0, 1800.0, 1600.0], 1.0, 0.25, false),
    m("SB2", [4.50, 4.25, 4.00, 0.0, 0.0], [2000.0, 1800.0, 1600.0], 1.0, 0.25, false),
    m("SB3", [5.50, 2.75, 3.00, 0.0, 0.0], [2000.0, 1800.0, 1600.0], 1.2, 0.25, false),
    m("SB4", [5.25, 3.50, 5.25, 0.0, 0.0], [2000.0, 1800.0, 1600.0], 2.7, 0.25, false),
];

/// Fuel id (1..=40) for a model name such as "GR2".
pub fn id(name: &str) -> Option<u8> {
    SB40.iter()
        .position(|f| f.name == name)
        .map(|i| i as u8 + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_complete_and_consistent() {
        assert_eq!(id("GR1"), Some(1));
        assert_eq!(id("SB4"), Some(40));
        assert_eq!(id("XX1"), None);
        for f in &SB40 {
            // Behave asserts the same rule: a standard model is dynamic exactly when it has
            // live herbaceous load.
            assert_eq!(f.dynamic, f.load[3] > 0.0, "{}", f.name);
            assert!(
                f.depth_ft > 0.0 && f.mx_dead > 0.0 && f.load[0] > 0.0,
                "{}",
                f.name
            );
        }
    }
}
