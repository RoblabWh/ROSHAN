//! Rothermel (1972) surface fire spread with the Albini (1976) refinements, following the
//! US Forest Service Behave library step for step so results can be compared with it.
//!
//! The equations are written in the units they were published in (ft, lb, Btu, min). SI appears
//! only at this module's boundary: wind comes in as m/s, `Spread` goes out in m/s.

use crate::fuel::FuelModel;
use libm::{exp, log, pow, sqrt};

/// Fuel moisture contents as fractions of oven-dry weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Moisture {
    pub d1: f32,
    pub d10: f32,
    pub d100: f32,
    pub herb: f32,
    pub woody: f32,
}

const PARTICLE_DENSITY: f64 = 32.0; // lb/ft3
const MINERAL_TOTAL: f64 = 0.0555;
const MINERAL_EFFECTIVE: f64 = 0.01;
const LB_FT2_PER_TON_ACRE: f64 = 2000.0 / 43560.0;
const FT_MIN_PER_M_S: f64 = 196.850_393_700_787_4;
const M_S_PER_FT_MIN: f64 = 0.00508;
const FT_MIN_PER_MPH: f64 = 88.0;

struct Particle {
    load: f64,     // lb/ft2
    sav: f64,      // 1/ft
    moisture: f64, // fraction
}

/// Albini's size classes: particles in the same class share one weighting factor for net load.
fn size_class(sav: f64) -> usize {
    [1200.0, 192.0, 96.0, 48.0, 16.0]
        .iter()
        .position(|&cut| sav >= cut)
        .unwrap_or(5)
}

/// Area-weighted properties of one life category (dead or live).
#[derive(Default)]
struct Category {
    area: f64,
    net_load: f64,
    moisture: f64,
    sav: f64,
    /// sum of f_ij * exp(-138/sav) * (250 + 1116 M): this category's share of the heat sink
    heat_sink: f64,
}

fn category(particles: &[Particle]) -> Category {
    let area = |p: &Particle| p.sav * p.load / PARTICLE_DENSITY;
    let total: f64 = particles.iter().map(area).sum();
    if total <= 0.0 {
        return Category::default();
    }
    let mut class_fraction = [0.0; 6];
    for p in particles {
        class_fraction[size_class(p.sav)] += area(p) / total;
    }
    let mut c = Category {
        area: total,
        ..Category::default()
    };
    for p in particles {
        let f = area(p) / total;
        c.net_load += class_fraction[size_class(p.sav)] * p.load * (1.0 - MINERAL_TOTAL);
        c.moisture += f * p.moisture;
        c.sav += f * p.sav;
        c.heat_sink += f * exp(-138.0 / p.sav) * (250.0 + 1116.0 * p.moisture);
    }
    c
}

fn moisture_damping(moisture: f64, extinction: f64) -> f64 {
    if extinction <= 0.0 || moisture >= extinction {
        return 0.0;
    }
    let r = moisture / extinction;
    1.0 - 2.59 * r + 5.11 * r * r - 3.52 * r * r * r
}

/// Everything about a fuel model at given moistures that does not depend on wind or slope.
#[derive(Clone, Copy, Debug)]
pub struct FuelBed {
    /// Rate of spread without wind or slope, ft/min. Zero when the fuel is too moist to burn.
    pub r0: f64,
    /// Reaction intensity, Btu/ft2/min.
    pub reaction_intensity: f64,
    /// Propagating flux ratio.
    pub xi: f64,
    /// Characteristic surface-area-to-volume ratio, 1/ft.
    pub sigma: f64,
    /// Packing ratio and its ratio to the optimum packing ratio.
    pub beta: f64,
    pub beta_ratio: f64,
    /// Characteristic dead and live moisture, and the live moisture of extinction (fractions).
    pub m_dead: f64,
    pub m_live: f64,
    pub mx_live: f64,
    pub depth_ft: f64,
    wind_c: f64,
    wind_b: f64,
    wind_e: f64,
    slope_k: f64,
}

impl FuelBed {
    pub fn new(model: &FuelModel, m: &Moisture) -> Self {
        let load = model.load.map(|t| t as f64 * LB_FT2_PER_TON_ACRE);
        let (d1, herb_m) = (m.d1 as f64, m.herb as f64);

        // Dynamic models: cured herbaceous load burns as dead fuel (Behave's curing curve).
        let cured = match () {
            _ if !model.dynamic => 0.0,
            _ if herb_m < 0.30 => 1.0,
            _ if herb_m <= 1.20 => 1.333 - 1.11 * herb_m,
            _ => 0.0,
        };
        let (sav1, sav_herb, sav_woody) = (
            model.sav[0] as f64,
            model.sav[1] as f64,
            model.sav[2] as f64,
        );
        let dead = [
            Particle {
                load: load[0],
                sav: sav1,
                moisture: d1,
            },
            Particle {
                load: load[1],
                sav: 109.0,
                moisture: m.d10 as f64,
            },
            Particle {
                load: load[2],
                sav: 30.0,
                moisture: m.d100 as f64,
            },
            Particle {
                load: load[3] * cured,
                sav: sav_herb,
                moisture: d1,
            },
        ];
        let live = [
            Particle {
                load: load[3] * (1.0 - cured),
                sav: sav_herb,
                moisture: herb_m,
            },
            Particle {
                load: load[4],
                sav: sav_woody,
                moisture: m.woody as f64,
            },
        ];
        let (cd, cl) = (category(&dead), category(&live));
        let total_area = cd.area + cl.area;
        let (fd, fl) = (cd.area / total_area, cl.area / total_area);

        let depth_ft = model.depth_ft as f64;
        let sigma = fd * cd.sav + fl * cl.sav;
        let total_load: f64 = dead.iter().chain(&live).map(|p| p.load).sum();
        let bulk_density = total_load / depth_ft;
        let beta = bulk_density / PARTICLE_DENSITY;
        let beta_ratio = beta / (3.348 / pow(sigma, 0.8189));

        // Live fuel burns only if the fine dead fuel can drive off its moisture.
        let mx_dead = model.mx_dead as f64;
        let fine_dead: f64 = dead.iter().map(|p| p.load * exp(-138.0 / p.sav)).sum();
        let fine_live: f64 = live.iter().map(|p| p.load * exp(-500.0 / p.sav)).sum();
        let mx_live = if fine_live > 0.0 {
            let fine_dead_moisture = dead
                .iter()
                .map(|p| p.moisture * p.load * exp(-138.0 / p.sav))
                .sum::<f64>()
                / fine_dead;
            let mx = 2.9 * (fine_dead / fine_live) * (1.0 - fine_dead_moisture / mx_dead) - 0.226;
            mx.max(mx_dead)
        } else {
            mx_dead
        };

        let sigma15 = pow(sigma, 1.5);
        let a = 133.0 / pow(sigma, 0.7913);
        let gamma =
            sigma15 / (495.0 + 0.0594 * sigma15) * pow(beta_ratio, a) * exp(a * (1.0 - beta_ratio));
        let eta_s = (0.174 / pow(MINERAL_EFFECTIVE, 0.19)).min(1.0);
        let heat = model.heat as f64;
        let reaction_intensity = gamma
            * heat
            * eta_s
            * (cd.net_load * moisture_damping(cd.moisture, mx_dead)
                + cl.net_load * moisture_damping(cl.moisture, mx_live));

        let xi = exp((0.792 + 0.681 * sqrt(sigma)) * (beta + 0.1)) / (192.0 + 0.2595 * sigma);
        let heat_sink = bulk_density * (fd * cd.heat_sink + fl * cl.heat_sink);

        Self {
            r0: reaction_intensity * xi / heat_sink,
            reaction_intensity,
            xi,
            sigma,
            beta,
            beta_ratio,
            m_dead: cd.moisture,
            m_live: cl.moisture,
            mx_live,
            depth_ft,
            wind_c: 7.47 * exp(-0.133 * pow(sigma, 0.55)),
            wind_b: 0.02526 * pow(sigma, 0.54),
            wind_e: 0.715 * exp(-0.000359 * sigma),
            slope_k: 5.275 * pow(beta, -0.3),
        }
    }

    /// Wind factor for a midflame wind speed in ft/min.
    pub fn phi_wind(&self, midflame_ft_min: f64) -> f64 {
        if midflame_ft_min <= 0.0 {
            return 0.0;
        }
        self.wind_c * pow(midflame_ft_min, self.wind_b) * pow(self.beta_ratio, -self.wind_e)
    }

    /// Slope factor for a slope given as rise over run.
    pub fn phi_slope(&self, tan_slope: f64) -> f64 {
        self.slope_k * tan_slope * tan_slope
    }

    /// Factor from the 20-ft wind to midflame height for fuel not sheltered by a canopy
    /// (Albini & Baughman 1979). The world uses `SimConfig::sheltered_waf` for timber models.
    /// ponytail: sheltering is all or nothing per fuel model. Add a canopy-cover layer and the
    /// sheltered formula when stands of different density matter.
    pub fn wind_adjustment(&self) -> f64 {
        1.83 / log((20.0 + 0.36 * self.depth_ft) / (0.13 * self.depth_ft))
    }

    /// Time the flaming front needs to pass a point, seconds (Anderson 1969).
    pub fn residence_s(&self) -> f64 {
        384.0 / self.sigma * 60.0
    }

    /// Byram's fireline intensity, Btu/ft/s, for a rate of spread in ft/min.
    pub fn fireline_intensity(&self, ros_ft_min: f64) -> f64 {
        ros_ft_min * self.reaction_intensity * (384.0 / self.sigma) / 60.0
    }

    /// Heat released per unit area by the flaming front, kJ/m². Times the rate of spread in
    /// m/s this is Byram's fireline intensity in kW/m.
    pub fn heat_per_area_kj_m2(&self) -> f64 {
        self.reaction_intensity * (384.0 / self.sigma) * 11.3565
    }

    /// Flame length, ft, for a rate of spread in ft/min (Byram 1959).
    pub fn flame_length_ft(&self, ros_ft_min: f64) -> f64 {
        0.45 * pow(self.fireline_intensity(ros_ft_min), 0.46)
    }

    /// Combines wind and slope into the fire ellipse at one place.
    /// `wind_ms`: midflame wind, pointing where the wind blows to. `slope`: terrain gradient
    /// (rise over run), pointing uphill. Both are in grid coordinates.
    pub fn spread(&self, wind_ms: [f64; 2], slope: [f64; 2]) -> Spread {
        if self.r0 <= 0.0 {
            return Spread::NONE;
        }
        let wind_speed = sqrt(wind_ms[0] * wind_ms[0] + wind_ms[1] * wind_ms[1]);
        let tan_slope = sqrt(slope[0] * slope[0] + slope[1] * slope[1]);
        let phi_w = self.phi_wind(wind_speed * FT_MIN_PER_M_S);
        let phi_s = self.phi_slope(tan_slope);

        // Wind and slope push the fire as vectors; the sum gives direction and strength.
        let mut v = [0.0; 2];
        for i in 0..2 {
            if wind_speed > 0.0 {
                v[i] += phi_w * wind_ms[i] / wind_speed;
            }
            if tan_slope > 0.0 {
                v[i] += phi_s * slope[i] / tan_slope;
            }
        }
        let phi = sqrt(v[0] * v[0] + v[1] * v[1]);
        let (dir, lb) = if phi > 0.0 {
            // The wind speed that alone would produce the same push sets the ellipse shape.
            let effective_ft_min = pow(
                phi * pow(self.beta_ratio, self.wind_e) / self.wind_c,
                1.0 / self.wind_b,
            );
            (
                [v[0] / phi, v[1] / phi],
                length_to_breadth(effective_ft_min / FT_MIN_PER_MPH),
            )
        } else {
            ([1.0, 0.0], 1.0)
        };
        Spread {
            head: (self.r0 * (1.0 + phi) * M_S_PER_FT_MIN) as f32,
            dir: [dir[0] as f32, dir[1] as f32],
            ecc: (sqrt(lb * lb - 1.0) / lb) as f32,
        }
    }
}

/// Length-to-breadth ratio of the fire ellipse for an effective midflame wind in mph
/// (Anderson 1983, capped at 8 as in Behave and FARSITE).
pub fn length_to_breadth(effective_mph: f64) -> f64 {
    if effective_mph <= 1e-7 {
        return 1.0;
    }
    (0.936 * exp(0.1147 * effective_mph) + 0.461 * exp(-0.0692 * effective_mph) - 0.397).min(8.0)
}

/// The fire ellipse at one place: fastest spread `head` (m/s) along unit vector `dir`, with the
/// ignition point at the rear focus of an ellipse of eccentricity `ecc`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spread {
    pub head: f32,
    pub dir: [f32; 2],
    pub ecc: f32,
}

impl Spread {
    pub const NONE: Self = Self {
        head: 0.0,
        dir: [1.0, 0.0],
        ecc: 0.0,
    };

    /// Rate of spread (m/s) along the unit vector (ux, uy).
    pub fn rate(&self, ux: f32, uy: f32) -> f32 {
        let cos = ux * self.dir[0] + uy * self.dir[1];
        self.head * (1.0 - self.ecc) / (1.0 - self.ecc * cos)
    }

    pub fn length_to_breadth(&self) -> f32 {
        1.0 / (1.0 - self.ecc * self.ecc).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fuel::{self, SB40};

    fn close(got: f64, want: f64) {
        assert!(
            (got - want).abs() <= 1e-5 * want.abs(),
            "got {got}, want {want}"
        );
    }

    const MOISTURE: Moisture = Moisture {
        d1: 0.06,
        d10: 0.07,
        d100: 0.08,
        herb: 0.60,
        woody: 0.90,
    };
    const MPH: f64 = 88.0; // ft/min
    const CHAINS_PER_HOUR: f64 = 1.1; // ft/min

    fn gs4() -> FuelBed {
        FuelBed::new(&SB40[fuel::id("GS4").unwrap() as usize - 1], &MOISTURE)
    }

    /// Reference values are the regression checks in the Behave library's own test suite
    /// (firelab/behave, src/testBehave/testBehave.cpp, read 2026-11-12): fuel model GS4,
    /// moistures 6/7/8/60/90 %.
    #[test]
    fn matches_behave_fuel_bed_intermediates() {
        let bed = gs4();
        close(bed.sigma, 1631.128734);
        close(bed.m_dead, 0.06005463);
        close(bed.m_live, 0.85874007);
        close(bed.mx_live, 1.37968551);
        // Behave's heat-source check for this fuel uses a canopy-sheltered wind, which this
        // module does not model, so it is not compared here. Reaction intensity and the
        // propagating flux are covered through rate of spread, intensity and flame length below.
    }

    #[test]
    fn matches_behave_spread_with_wind_and_slope() {
        // 5 mph midflame wind blowing straight up a 30 % slope.
        let bed = gs4();
        let ros = bed.r0 * (1.0 + bed.phi_wind(5.0 * MPH) + bed.phi_slope(0.30));
        close(ros, 34.011429 * CHAINS_PER_HOUR);
        close(bed.fireline_intensity(ros), 2292.684759);
        close(bed.flame_length_ft(ros), 15.811421);

        // The vector form must agree when wind and slope point the same way.
        let s = bed.spread([0.0, 5.0 * MPH / FT_MIN_PER_M_S], [0.0, 0.30]);
        close(s.head as f64, ros * M_S_PER_FT_MIN);
        assert!((s.dir[0]).abs() < 1e-6 && (s.dir[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn matches_behave_ellipse_shape() {
        // 5 mph midflame, flat: length-to-width 1.590064, heading-to-backing ratio 7.988029.
        let s = gs4().spread([5.0 * MPH / FT_MIN_PER_M_S, 0.0], [0.0, 0.0]);
        assert!((s.length_to_breadth() as f64 - 1.590064).abs() < 1e-4);
        assert!(((s.rate(1.0, 0.0) / s.rate(-1.0, 0.0)) as f64 - 7.988029).abs() < 1e-3);
        close(length_to_breadth(5.0), 1.590064);
        assert_eq!(length_to_breadth(0.0), 1.0);
        assert_eq!(length_to_breadth(60.0), 8.0);
    }

    #[test]
    fn matches_behave_high_wind_on_short_grass() {
        // Anderson fuel model 1 (short grass), 20 mph midflame, no wind limit.
        let fm1 = FuelModel {
            name: "FM1",
            load: [(0.034 / LB_FT2_PER_TON_ACRE) as f32, 0.0, 0.0, 0.0, 0.0],
            sav: [3500.0, 1500.0, 1500.0],
            depth_ft: 1.0,
            mx_dead: 0.12,
            heat: 8000.0,
            dynamic: false,
        };
        let bed = FuelBed::new(&fm1, &MOISTURE);
        let (flat, sloped) = (
            bed.r0 * (1.0 + bed.phi_wind(20.0 * MPH)),
            bed.r0 * (1.0 + bed.phi_wind(20.0 * MPH) + bed.phi_slope(0.30)),
        );
        assert!((flat - 1747.279440).abs() < 1e-2, "{flat}");
        assert!((sloped - 1764.333376).abs() < 1e-2, "{sloped}");
        close(bed.reaction_intensity, 743.5194 / 0.9);
    }

    #[test]
    fn behaves_sensibly_for_every_model() {
        let dry = Moisture {
            d1: 0.04,
            d10: 0.05,
            d100: 0.06,
            herb: 0.40,
            woody: 0.70,
        };
        let soaked = Moisture {
            d1: 0.60,
            d10: 0.60,
            d100: 0.60,
            herb: 3.0,
            woody: 3.0,
        };
        for model in &SB40 {
            let bed = FuelBed::new(model, &dry);
            assert!(bed.r0 > 0.0 && bed.r0.is_finite(), "{}", model.name);
            assert!(bed.phi_wind(400.0) > bed.phi_wind(100.0), "{}", model.name);
            assert!(
                FuelBed::new(model, &MOISTURE).r0 <= bed.r0,
                "{}: wetter must not be faster",
                model.name
            );
            assert_eq!(
                FuelBed::new(model, &soaked).r0,
                0.0,
                "{}: must not burn above extinction",
                model.name
            );
            let waf = bed.wind_adjustment();
            assert!(
                (0.2..0.7).contains(&waf),
                "{}: wind adjustment {waf}",
                model.name
            );

            // Wind against the slope: the two pushes subtract.
            let up = bed.spread([2.0, 0.0], [0.2, 0.0]);
            let against = bed.spread([-2.0, 0.0], [0.2, 0.0]);
            assert!(against.head < up.head, "{}", model.name);
        }
    }
}
