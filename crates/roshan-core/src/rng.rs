//! Counter-based random numbers (Philox4x32-10, Salmon et al. 2011).
//!
//! A draw is a pure function of (seed, stream, a, b). There is no generator state to advance,
//! so the value a cell gets at a tick does not depend on what any other cell, drone or thread
//! did before. That is what keeps two runs comparable when only the policy differs.

/// Independent streams. Add new ones at the end; reordering changes every existing experiment.
#[derive(Clone, Copy)]
#[repr(u32)]
pub enum Stream {
    RosNoise = 1,
    Spawn = 2,
    Goal = 3,
    Weather = 4,
    Ignition = 5,
    Spin = 6,
    Report = 7,
}

const M0: u64 = 0xD251_1F53;
const M1: u64 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

/// The raw Philox4x32-10 block function.
pub fn philox(mut ctr: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for round in 0..10 {
        if round > 0 {
            key = [key[0].wrapping_add(W0), key[1].wrapping_add(W1)];
        }
        let p0 = M0 * ctr[0] as u64;
        let p1 = M1 * ctr[2] as u64;
        ctr = [
            (p1 >> 32) as u32 ^ ctr[1] ^ key[0],
            p1 as u32,
            (p0 >> 32) as u32 ^ ctr[3] ^ key[1],
            p0 as u32,
        ];
    }
    ctr
}

fn block(seed: u64, stream: Stream, a: u64, b: u32) -> [u32; 4] {
    philox(
        [a as u32, (a >> 32) as u32, b, stream as u32],
        [seed as u32, (seed >> 32) as u32],
    )
}

/// A new seed derived from `seed` and two indices, e.g. (environment, episode).
pub fn derive(seed: u64, a: u64, b: u64) -> u64 {
    let r = philox(
        [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32],
        [seed as u32, (seed >> 32) as u32],
    );
    (r[0] as u64) << 32 | r[1] as u64
}

/// Uniform in [0, 1).
pub fn uniform(seed: u64, stream: Stream, a: u64, b: u32) -> f32 {
    (block(seed, stream, a, b)[0] >> 8) as f32 / (1u32 << 24) as f32
}

/// Standard normal (Box-Muller). libm keeps the result identical on every platform.
pub fn normal(seed: u64, stream: Stream, a: u64, b: u32) -> f32 {
    normal2(seed, stream, a, b)[0]
}

/// Two independent standard normal numbers from one draw.
pub fn normal2(seed: u64, stream: Stream, a: u64, b: u32) -> [f32; 2] {
    let r = block(seed, stream, a, b);
    let u1 = ((r[0] >> 8) as f64 + 1.0) / (1u32 << 24) as f64; // (0, 1]
    let angle = std::f64::consts::TAU * ((r[1] >> 8) as f64 / (1u32 << 24) as f64);
    let radius = libm::sqrt(-2.0 * libm::log(u1));
    [
        (radius * libm::cos(angle)) as f32,
        (radius * libm::sin(angle)) as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known-answer vectors from the Random123 reference distribution (kat_vectors).
    #[test]
    fn philox_matches_reference_vectors() {
        assert_eq!(
            philox([0; 4], [0; 2]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
        assert_eq!(
            philox([u32::MAX; 4], [u32::MAX; 2]),
            [0x408f_276d, 0x41c8_3b0e, 0xa20b_c7c6, 0x6d54_51fd]
        );
        assert_eq!(
            philox(
                [0x243f_6a88, 0x85a3_08d3, 0x1319_8a2e, 0x0370_7344],
                [0xa409_3822, 0x299f_31d0]
            ),
            [0xd16c_fe09, 0x94fd_cceb, 0x5001_e420, 0x2412_6ea1]
        );
    }

    #[test]
    fn draws_are_uniform_and_normal_enough() {
        let n = 20_000;
        let u: Vec<f32> = (0..n).map(|i| uniform(7, Stream::RosNoise, 3, i)).collect();
        let z: Vec<f32> = (0..n).map(|i| normal(7, Stream::RosNoise, 3, i)).collect();
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let var = |v: &[f32]| v.iter().map(|x| (x - mean(v)).powi(2)).sum::<f32>() / v.len() as f32;
        assert!(u.iter().all(|x| (0.0..1.0).contains(x)));
        assert!((mean(&u) - 0.5).abs() < 0.01 && (var(&u) - 1.0 / 12.0).abs() < 0.005);
        assert!(mean(&z).abs() < 0.03 && (var(&z) - 1.0).abs() < 0.05);
        // Same inputs, same value; a different entity, a different value.
        assert_eq!(
            uniform(7, Stream::RosNoise, 3, 5),
            uniform(7, Stream::RosNoise, 3, 5)
        );
        assert_ne!(
            uniform(7, Stream::RosNoise, 3, 5),
            uniform(7, Stream::RosNoise, 3, 6)
        );
    }
}
