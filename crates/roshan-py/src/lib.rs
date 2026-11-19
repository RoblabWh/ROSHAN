//! Python bindings: module `roshan._core`.
//!
//! Buffer rule: Rust borrows the caller's numpy arrays for the duration of one call and never
//! stores a pointer to them. Wrong dtype, shape or memory layout is an error, never a silent copy.

use numpy::{
    PyArray1, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyReadonlyArray4,
    PyReadwriteArray1, PyReadwriteArray2, PyReadwriteArray3, PyReadwriteArray4,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use roshan_core::allocate::{self, AllocConfig, AllocVec};
use roshan_core::navigate::{self, Bufs, NavConfig, NavVec};
use roshan_core::terrain::Grid;
use roshan_core::world::SimConfig;

fn check(name: &str, shape: &[usize], want: &[usize], c_contiguous: bool) -> PyResult<()> {
    if shape != want {
        return Err(PyValueError::new_err(format!(
            "{name}: shape {shape:?}, expected {want:?}"
        )));
    }
    if !c_contiguous {
        return Err(PyValueError::new_err(format!(
            "{name}: array is not C-contiguous; pass the original buffer, not a slice or transpose"
        )));
    }
    Ok(())
}

fn invalid(what: &str, err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{what}: {err}"))
}

/// A batch of `navigate` environments (see roshan_core::navigate).
#[pyclass]
struct NavigateEnv {
    inner: NavVec,
    sim: SimConfig,
}

#[pymethods]
impl NavigateEnv {
    /// `config` and `sim` are JSON objects holding only the values that differ from the
    /// defaults; an unknown key is an error.
    #[new]
    #[pyo3(signature = (num_envs, nx=35, ny=36, cell_m=10.0, seed=0, threads=0, config="{}", sim="{}"))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_envs: usize,
        nx: u32,
        ny: u32,
        cell_m: f32,
        seed: u64,
        threads: usize,
        config: &str,
        sim: &str,
    ) -> PyResult<Self> {
        let cfg: NavConfig = serde_json::from_str(config).map_err(|e| invalid("config", e))?;
        let sim: SimConfig = serde_json::from_str(sim).map_err(|e| invalid("sim", e))?;
        if num_envs == 0 || nx < 3 || ny < 3 || cell_m.is_nan() || cell_m <= 0.0 {
            return Err(invalid(
                "map",
                "need at least one environment and a 3x3 map of positive cell size",
            ));
        }
        if cfg.drones == 0 || cfg.max_speed_ms <= 0.0 || cfg.view_m <= 0.0 || cfg.goals == 0 {
            return Err(invalid(
                "config",
                "drones, max_speed_ms, view_m and goals must be positive",
            ));
        }
        if sim.dt_s <= 0.0 || sim.fire_every == 0 {
            return Err(invalid("sim", "dt_s and fire_every must be positive"));
        }
        let grid = Grid { nx, ny, cell_m };
        Ok(Self {
            inner: NavVec::new(num_envs, grid, sim, cfg, seed, threads),
            sim,
        })
    }

    /// Default `config` and `sim` as JSON strings.
    #[staticmethod]
    fn defaults() -> (String, String) {
        let json = |v: serde_json::Result<String>| v.expect("plain structs serialise");
        (
            json(serde_json::to_string(&NavConfig::default())),
            json(serde_json::to_string(&SimConfig::default())),
        )
    }

    /// Names and sizes of everything the environment reads and writes.
    #[getter]
    fn spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let spec = PyDict::new(py);
        spec.set_item("num_envs", self.inner.envs.len())?;
        spec.set_item("drones", self.inner.cfg.drones)?;
        spec.set_item("agent", navigate::AGENT.to_vec())?;
        spec.set_item("neighbor", navigate::NEIGHBOR.to_vec())?;
        spec.set_item("terms", navigate::TERMS.to_vec())?;
        spec.set_item("info", navigate::INFO.to_vec())?;
        let config = serde_json::to_string(&self.inner.cfg).map_err(|e| invalid("config", e))?;
        let sim = serde_json::to_string(&self.sim).map_err(|e| invalid("sim", e))?;
        spec.set_item("config", config)?;
        spec.set_item("sim", sim)?;
        Ok(spec)
    }

    /// Reset finished environments on their next step. Turn off for evaluation.
    #[getter]
    fn get_autoreset(&self) -> bool {
        self.inner.autoreset
    }

    #[setter]
    fn set_autoreset(&mut self, on: bool) {
        self.inner.autoreset = on;
    }

    /// The actions an untrained pilot takes now, [envs, drones, 2]: "straight" (at the
    /// waypoint, no avoidance), "scripted" (centralised sidestepping, full information) or
    /// "decentral" (each drone alone, from the observation a fly policy gets).
    fn pilot<'py>(&self, py: Python<'py>, kind: &str) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let pilot = navigate::Pilot::named(kind).ok_or_else(|| invalid("pilot", kind))?;
        let shape = [self.inner.envs.len(), self.inner.cfg.drones, 2];
        PyArray1::from_vec(py, self.inner.pilot(pilot)).reshape(shape)
    }

    /// Starts a fresh episode everywhere and writes the first observations.
    fn reset(
        &mut self,
        py: Python<'_>,
        mut agent: PyReadwriteArray3<'_, f32>,
        mut neighbors: PyReadwriteArray4<'_, f32>,
        mut mask: PyReadwriteArray3<'_, u8>,
    ) -> PyResult<()> {
        let (e, n) = (self.inner.envs.len(), self.inner.cfg.drones);
        let width = navigate::AGENT.len();
        check(
            "agent",
            agent.shape(),
            &[e, n, width],
            agent.is_c_contiguous(),
        )?;
        check(
            "neighbors",
            neighbors.shape(),
            &[e, n, n, 4],
            neighbors.is_c_contiguous(),
        )?;
        check("mask", mask.shape(), &[e, n, n], mask.is_c_contiguous())?;
        let (agent, neighbors, mask) = (
            agent.as_slice_mut()?,
            neighbors.as_slice_mut()?,
            mask.as_slice_mut()?,
        );
        let inner = &mut self.inner;
        py.detach(|| inner.reset(agent, neighbors, mask));
        Ok(())
    }

    /// Steps every environment by one tick. `actions` [envs, drones, 2] is read; everything
    /// else is written in place. The interpreter lock is released while the environments run.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &mut self,
        py: Python<'_>,
        actions: PyReadonlyArray3<'_, f32>,
        mut agent: PyReadwriteArray3<'_, f32>,
        mut neighbors: PyReadwriteArray4<'_, f32>,
        mut mask: PyReadwriteArray3<'_, u8>,
        mut terms: PyReadwriteArray3<'_, f32>,
        mut terminated: PyReadwriteArray1<'_, u8>,
        mut truncated: PyReadwriteArray1<'_, u8>,
        mut info: PyReadwriteArray2<'_, f32>,
    ) -> PyResult<()> {
        let (e, n) = (self.inner.envs.len(), self.inner.cfg.drones);
        let (width, n_terms, n_info) = (
            navigate::AGENT.len(),
            navigate::TERMS.len(),
            navigate::INFO.len(),
        );
        check(
            "actions",
            actions.shape(),
            &[e, n, 2],
            actions.is_c_contiguous(),
        )?;
        check(
            "agent",
            agent.shape(),
            &[e, n, width],
            agent.is_c_contiguous(),
        )?;
        check(
            "neighbors",
            neighbors.shape(),
            &[e, n, n, 4],
            neighbors.is_c_contiguous(),
        )?;
        check("mask", mask.shape(), &[e, n, n], mask.is_c_contiguous())?;
        check(
            "terms",
            terms.shape(),
            &[e, n, n_terms],
            terms.is_c_contiguous(),
        )?;
        check(
            "terminated",
            terminated.shape(),
            &[e],
            terminated.is_c_contiguous(),
        )?;
        check(
            "truncated",
            truncated.shape(),
            &[e],
            truncated.is_c_contiguous(),
        )?;
        check("info", info.shape(), &[e, n_info], info.is_c_contiguous())?;
        let actions = actions.as_slice()?;
        let out = Bufs {
            agent: agent.as_slice_mut()?,
            neighbors: neighbors.as_slice_mut()?,
            mask: mask.as_slice_mut()?,
            terms: terms.as_slice_mut()?,
            terminated: terminated.as_slice_mut()?,
            truncated: truncated.as_slice_mut()?,
            info: info.as_slice_mut()?,
        };
        let inner = &mut self.inner;
        py.detach(|| inner.step(actions, out));
        Ok(())
    }
}

/// Checks a scenario dict the way the simulator will and returns it as the simulator
/// understood it. Raises ValueError or KeyError for anything malformed.
#[pyfunction]
fn check_scenario<'py>(
    py: Python<'py>,
    scenario: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    convert::to_py(py, &convert::from_py(scenario)?)
}

/// The decentralised hand-written pilot (roshan_core::navigate::decentral) on a batch of
/// pilot observations: `agent` [envs, drones, 8], `neighbors` [envs, drones, drones, 4] and
/// `mask` [envs, drones, drones] in, velocity fractions [envs, drones, 2] out.
#[pyfunction]
fn decentral_pilot<'py>(
    py: Python<'py>,
    agent: PyReadonlyArray3<'py, f32>,
    neighbors: PyReadonlyArray4<'py, f32>,
    mask: PyReadonlyArray3<'py, u8>,
    view_m: f32,
    max_speed_ms: f32,
    safe_m: f32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let (e, n) = (mask.shape()[0], mask.shape()[1]);
    check(
        "agent",
        agent.shape(),
        &[e, n, navigate::AGENT.len()],
        agent.is_c_contiguous(),
    )?;
    check(
        "neighbors",
        neighbors.shape(),
        &[e, n, n, 4],
        neighbors.is_c_contiguous(),
    )?;
    check("mask", mask.shape(), &[e, n, n], mask.is_c_contiguous())?;
    let (agent, neighbors, mask) = (agent.as_slice()?, neighbors.as_slice()?, mask.as_slice()?);
    let width = navigate::AGENT.len();
    let actions: Vec<f32> = (0..e)
        .flat_map(|i| {
            navigate::decentral(
                &agent[i * n * width..][..n * width],
                &neighbors[i * n * n * 4..][..n * n * 4],
                &mask[i * n * n..][..n * n],
                view_m,
                max_speed_ms,
                safe_m,
            )
        })
        .collect();
    PyArray1::from_vec(py, actions).reshape([e, n, 2])
}

/// What `AllocateEnv.advance` needs for a tick flown by an outside pilot.
type Flight<'py> = (
    PyReadonlyArray3<'py, f32>,
    f32,
    PyReadwriteArray3<'py, f32>,
    PyReadwriteArray4<'py, f32>,
    PyReadwriteArray3<'py, u8>,
);

/// A batch of `allocate` environments on one scenario (see roshan_core::allocate).
#[pyclass]
struct AllocateEnv {
    inner: AllocVec,
    sim: SimConfig,
}

#[pymethods]
impl AllocateEnv {
    /// `scenario` is a scenario dict; `config` and `sim` are JSON objects with the values that
    /// differ from the defaults.
    #[new]
    #[pyo3(signature = (num_envs, scenario, seed=0, threads=0, config="{}", sim="{}"))]
    fn new(
        num_envs: usize,
        scenario: &Bound<'_, PyDict>,
        seed: u64,
        threads: usize,
        config: &str,
        sim: &str,
    ) -> PyResult<Self> {
        let cfg: AllocConfig = serde_json::from_str(config).map_err(|e| invalid("config", e))?;
        let sim: SimConfig = serde_json::from_str(sim).map_err(|e| invalid("sim", e))?;
        if num_envs == 0 || sim.dt_s <= 0.0 || sim.fire_every == 0 {
            return Err(invalid(
                "sim",
                "need at least one environment and positive dt_s and fire_every",
            ));
        }
        let inner = AllocVec::new(
            num_envs,
            convert::from_py(scenario)?,
            sim,
            cfg,
            seed,
            threads,
        )
        .map_err(|e| invalid("scenario or config", e))?;
        Ok(Self { inner, sim })
    }

    /// Default `config` as a JSON string.
    #[staticmethod]
    fn defaults() -> String {
        serde_json::to_string(&AllocConfig::default()).expect("a plain struct serialises")
    }

    /// Names and sizes of everything the environment reads and writes.
    #[getter]
    fn spec<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let spec = PyDict::new(py);
        spec.set_item("num_envs", self.inner.envs.len())?;
        spec.set_item("drones", self.inner.drones)?;
        spec.set_item("slots", self.inner.slots)?;
        spec.set_item("stations", self.inner.slots - self.inner.cfg.max_targets)?;
        spec.set_item("drone", allocate::DRONE.to_vec())?;
        spec.set_item("target", allocate::TARGET.to_vec())?;
        spec.set_item("global", allocate::GLOBAL.to_vec())?;
        spec.set_item("terms", allocate::TERMS.to_vec())?;
        spec.set_item("info", allocate::INFO.to_vec())?;
        let grid = self.inner.envs[0].world.terrain.grid;
        spec.set_item(
            "size_m",
            (grid.nx as f32 * grid.cell_m, grid.ny as f32 * grid.cell_m),
        )?;
        spec.set_item(
            "config",
            serde_json::to_string(&self.inner.cfg).map_err(|e| invalid("config", e))?,
        )?;
        spec.set_item(
            "sim",
            serde_json::to_string(&self.sim).map_err(|e| invalid("sim", e))?,
        )?;
        Ok(spec)
    }

    #[getter]
    fn get_autoreset(&self) -> bool {
        self.inner.autoreset
    }

    #[setter]
    fn set_autoreset(&mut self, on: bool) {
        self.inner.autoreset = on;
    }

    /// Writes a planner's actions for the current state into `actions` [envs, drones]:
    /// "greedy", "head", "flank" (the nearest, most intense or least intense block that still
    /// needs water), or "rollout" (greedy improved by simulating `horizon_s` ahead on copies
    /// of the world, trying up to `candidates` blocks per drone). Runs in parallel with the
    /// interpreter lock released.
    #[pyo3(signature = (method, actions, horizon_s=300.0, candidates=4))]
    fn plan(
        &self,
        py: Python<'_>,
        method: &str,
        mut actions: PyReadwriteArray2<'_, i32>,
        horizon_s: f32,
        candidates: usize,
    ) -> PyResult<()> {
        let (e, n) = (self.inner.envs.len(), self.inner.drones);
        check(
            "actions",
            actions.shape(),
            &[e, n],
            actions.is_c_contiguous(),
        )?;
        let out = actions.as_slice_mut()?;
        let inner = &self.inner;
        let method = allocate::Method::parse(method)
            .filter(|m| *m != allocate::Method::Rollout || horizon_s > 0.0)
            .ok_or_else(|| {
                invalid(
                    "method",
                    format!(
                        "expected one of {:?}, rollout with a positive horizon",
                        allocate::Method::NAMES
                    ),
                )
            })?;
        py.detach(|| inner.plan(method, horizon_s, candidates, out));
        Ok(())
    }

    /// Advances every environment to its next decision event, or, with `reset`, starts fresh
    /// episodes (those of `seeds` if given). `actions` [envs, drones] is read; the rest is
    /// written in place. The interpreter lock is released meanwhile.
    ///
    /// With `flight` = (velocity [envs, drones, 2], view_m, agent, neighbors, mask) the call is
    /// instead a single tick in which an outside pilot flies the drones with `velocity`; the
    /// pilot's next observation is written into the last three arrays (shaped as in the
    /// navigate task).
    #[pyo3(signature = (reset, seeds, actions, drones, targets, target_mask, action_mask, globals, decide, terms,
                        elapsed_s, terminated, truncated, info, seed, fingerprint, flight=None))]
    #[allow(clippy::too_many_arguments)]
    fn advance(
        &mut self,
        py: Python<'_>,
        reset: bool,
        seeds: Option<Vec<u64>>,
        actions: PyReadonlyArray2<'_, i32>,
        mut drones: PyReadwriteArray3<'_, f32>,
        mut targets: PyReadwriteArray3<'_, f32>,
        mut target_mask: PyReadwriteArray2<'_, u8>,
        mut action_mask: PyReadwriteArray3<'_, u8>,
        mut globals: PyReadwriteArray2<'_, f32>,
        mut decide: PyReadwriteArray2<'_, u8>,
        mut terms: PyReadwriteArray3<'_, f32>,
        mut elapsed_s: PyReadwriteArray1<'_, f32>,
        mut terminated: PyReadwriteArray1<'_, u8>,
        mut truncated: PyReadwriteArray1<'_, u8>,
        mut info: PyReadwriteArray2<'_, f32>,
        mut seed: PyReadwriteArray1<'_, u64>,
        mut fingerprint: PyReadwriteArray1<'_, u64>,
        flight: Option<Flight<'_>>,
    ) -> PyResult<()> {
        let (e, n, s) = (self.inner.envs.len(), self.inner.drones, self.inner.slots);
        if seeds.as_ref().is_some_and(|seeds| seeds.len() != e) {
            return Err(invalid(
                "seeds",
                format!("need one seed per environment ({e})"),
            ));
        }
        check(
            "actions",
            actions.shape(),
            &[e, n],
            actions.is_c_contiguous(),
        )?;
        check(
            "drones",
            drones.shape(),
            &[e, n, allocate::DRONE.len()],
            drones.is_c_contiguous(),
        )?;
        check(
            "targets",
            targets.shape(),
            &[e, s, allocate::TARGET.len()],
            targets.is_c_contiguous(),
        )?;
        check(
            "target_mask",
            target_mask.shape(),
            &[e, s],
            target_mask.is_c_contiguous(),
        )?;
        check(
            "action_mask",
            action_mask.shape(),
            &[e, n, s],
            action_mask.is_c_contiguous(),
        )?;
        check(
            "globals",
            globals.shape(),
            &[e, allocate::GLOBAL.len()],
            globals.is_c_contiguous(),
        )?;
        check("decide", decide.shape(), &[e, n], decide.is_c_contiguous())?;
        check(
            "terms",
            terms.shape(),
            &[e, n, allocate::TERMS.len()],
            terms.is_c_contiguous(),
        )?;
        check(
            "elapsed_s",
            elapsed_s.shape(),
            &[e],
            elapsed_s.is_c_contiguous(),
        )?;
        check(
            "terminated",
            terminated.shape(),
            &[e],
            terminated.is_c_contiguous(),
        )?;
        check(
            "truncated",
            truncated.shape(),
            &[e],
            truncated.is_c_contiguous(),
        )?;
        check(
            "info",
            info.shape(),
            &[e, allocate::INFO.len()],
            info.is_c_contiguous(),
        )?;
        check("seed", seed.shape(), &[e], seed.is_c_contiguous())?;
        check(
            "fingerprint",
            fingerprint.shape(),
            &[e],
            fingerprint.is_c_contiguous(),
        )?;
        let actions = actions.as_slice()?;
        let bufs = allocate::Bufs {
            drones: drones.as_slice_mut()?,
            targets: targets.as_slice_mut()?,
            target_mask: target_mask.as_slice_mut()?,
            action_mask: action_mask.as_slice_mut()?,
            globals: globals.as_slice_mut()?,
            decide: decide.as_slice_mut()?,
            terms: terms.as_slice_mut()?,
            elapsed_s: elapsed_s.as_slice_mut()?,
            terminated: terminated.as_slice_mut()?,
            truncated: truncated.as_slice_mut()?,
            info: info.as_slice_mut()?,
            seed: seed.as_slice_mut()?,
            fingerprint: fingerprint.as_slice_mut()?,
        };
        let inner = &mut self.inner;
        match flight {
            None => py.detach(|| {
                if reset {
                    inner.reset(seeds.as_deref(), bufs)
                } else {
                    inner.step(actions, bufs)
                }
            }),
            Some((velocity, view_m, mut agent, mut neighbors, mut mask)) => {
                check(
                    "velocity",
                    velocity.shape(),
                    &[e, n, 2],
                    velocity.is_c_contiguous(),
                )?;
                check(
                    "agent",
                    agent.shape(),
                    &[e, n, navigate::AGENT.len()],
                    agent.is_c_contiguous(),
                )?;
                check(
                    "neighbors",
                    neighbors.shape(),
                    &[e, n, n, 4],
                    neighbors.is_c_contiguous(),
                )?;
                check("mask", mask.shape(), &[e, n, n], mask.is_c_contiguous())?;
                if view_m.is_nan() || view_m <= 0.0 {
                    return Err(invalid("view_m", "must be positive"));
                }
                let velocity = velocity.as_slice()?;
                let (agent, neighbors, mask) = (
                    agent.as_slice_mut()?,
                    neighbors.as_slice_mut()?,
                    mask.as_slice_mut()?,
                );
                py.detach(|| {
                    inner.fly_tick(actions, velocity, view_m, bufs, agent, neighbors, mask)
                });
            }
        }
        Ok(())
    }
}

/// Names of the fuel models; a cell's fuel id is the position in this list plus one, and 0
/// means non-burnable.
#[pyfunction]
fn fuel_models() -> Vec<&'static str> {
    roshan_core::fuel::SB40.iter().map(|f| f.name).collect()
}

mod convert {
    //! Conversion between a Python scenario dict and the core's Scenario.
    //! Dict keys: nx, ny, cell_m, fuel [ny, nx] uint8, elevation [ny, nx] float32 or None,
    //! landcover [ny, nx] uint8 or None (optional key),
    //! ignitions / drones / stations as lists of (x, y), moisture as five fractions
    //! (1-h, 10-h, 100-h, herb, woody), wind as a list of (t_s, speed_ms, to_x, to_y), meta str.

    use super::invalid;
    use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
    use pyo3::exceptions::PyKeyError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use roshan_core::rothermel::Moisture;
    use roshan_core::scenario::Scenario;
    use roshan_core::terrain::Grid;
    use roshan_core::world::Wind;

    fn get<'py>(d: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        d.get_item(key)?
            .ok_or_else(|| PyKeyError::new_err(format!("scenario has no '{key}'")))
    }

    pub fn from_py(d: &Bound<'_, PyDict>) -> PyResult<Scenario> {
        let grid = Grid {
            nx: get(d, "nx")?.extract()?,
            ny: get(d, "ny")?.extract()?,
            cell_m: get(d, "cell_m")?.extract()?,
        };
        let shape = [grid.ny as usize, grid.nx as usize];
        let fuel: PyReadonlyArray2<'_, u8> = get(d, "fuel")?.extract()?;
        if fuel.shape() != shape {
            return Err(invalid(
                "fuel",
                format!("shape {:?}, expected {shape:?}", fuel.shape()),
            ));
        }
        let elevation = get(d, "elevation")?;
        let elevation = if elevation.is_none() {
            None
        } else {
            let z: PyReadonlyArray2<'_, f32> = elevation.extract()?;
            if z.shape() != shape {
                return Err(invalid(
                    "elevation",
                    format!("shape {:?}, expected {shape:?}", z.shape()),
                ));
            }
            Some(z.as_array().iter().copied().collect())
        };
        let landcover = match d.get_item("landcover")? {
            Some(l) if !l.is_none() => {
                let l: PyReadonlyArray2<'_, u8> = l.extract()?;
                if l.shape() != shape {
                    return Err(invalid(
                        "landcover",
                        format!("shape {:?}, expected {shape:?}", l.shape()),
                    ));
                }
                Some(l.as_array().iter().copied().collect())
            }
            _ => None,
        };
        let m: [f32; 5] = get(d, "moisture")?.extract()?;
        let wind: Vec<[f32; 4]> = get(d, "wind")?.extract()?;
        let scenario = Scenario {
            grid,
            fuel: fuel.as_array().iter().copied().collect(),
            elevation,
            landcover,
            ignitions: get(d, "ignitions")?.extract()?,
            stations: get(d, "stations")?.extract()?,
            moisture: Moisture {
                d1: m[0],
                d10: m[1],
                d100: m[2],
                herb: m[3],
                woody: m[4],
            },
            wind: wind
                .iter()
                .map(|w| Wind {
                    t_s: w[0],
                    speed_ms: w[1],
                    to: [w[2], w[3]],
                })
                .collect(),
            meta: get(d, "meta")?.extract()?,
        };
        // Validate here, at the boundary, so a bad file is a Python error and not a dead window.
        scenario
            .world(Default::default(), 0)
            .map_err(|e| invalid("scenario", e))?;
        Ok(scenario)
    }

    pub fn to_py<'py>(py: Python<'py>, s: &Scenario) -> PyResult<Bound<'py, PyDict>> {
        let shape = [s.grid.ny as usize, s.grid.nx as usize];
        let d = PyDict::new(py);
        d.set_item("nx", s.grid.nx)?;
        d.set_item("ny", s.grid.ny)?;
        d.set_item("cell_m", s.grid.cell_m)?;
        d.set_item("fuel", PyArray1::from_slice(py, &s.fuel).reshape(shape)?)?;
        match &s.elevation {
            Some(z) => d.set_item("elevation", PyArray1::from_slice(py, z).reshape(shape)?)?,
            None => d.set_item("elevation", py.None())?,
        }
        match &s.landcover {
            Some(l) => d.set_item("landcover", PyArray1::from_slice(py, l).reshape(shape)?)?,
            None => d.set_item("landcover", py.None())?,
        }
        d.set_item("ignitions", s.ignitions.clone())?;
        d.set_item("stations", s.stations.clone())?;
        let m = s.moisture;
        d.set_item("moisture", [m.d1, m.d10, m.d100, m.herb, m.woody])?;
        let wind: Vec<[f32; 4]> = s
            .wind
            .iter()
            .map(|w| [w.t_s, w.speed_ms, w.to[0], w.to[1]])
            .collect();
        d.set_item("wind", wind)?;
        d.set_item("meta", &s.meta)?;
        Ok(d)
    }
}

#[cfg(feature = "ui")]
mod ui {
    //! The viewer.

    use super::convert::{from_py, to_py};
    use super::invalid;
    use numpy::{PyArray1, PyArrayMethods};
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use roshan_core::navigate::NavConfig;
    use roshan_core::scenario::Scenario;
    use roshan_core::terrain::Grid;
    use std::path::Path;
    use std::sync::Arc;

    /// A Python callable (dict of arrays -> list of floats) as a viewer policy.
    fn as_policy(callable: Py<PyAny>) -> roshan_ui::Policy {
        Box::new(move |obs: &roshan_ui::Obs| {
            Python::attach(|py| {
                let arrays = PyDict::new(py);
                for (name, shape, values) in &obs.0 {
                    let array = PyArray1::from_slice(py, values).reshape(shape.clone());
                    array
                        .and_then(|a| arrays.set_item(*name, a))
                        .map_err(|err| err.to_string())?;
                }
                callable
                    .call1(py, (arrays,))
                    .and_then(|out| out.extract::<Vec<f32>>(py))
                    .map_err(|err| traceback(py, &err))
            })
        })
    }

    fn traceback(py: Python<'_>, err: &PyErr) -> String {
        let tb = err
            .traceback(py)
            .and_then(|tb| tb.format().ok())
            .unwrap_or_default();
        format!("{tb}{err}")
    }

    /// The built-in demo landscape as a scenario dict.
    #[pyfunction]
    pub fn demo_scenario(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        to_py(py, &roshan_ui::demo_scenario())
    }

    /// Opens the viewer and blocks until it closes. Call from the main thread. Returns
    /// (frames, ticks, policy_calls); Ctrl+C closes the window and raises KeyboardInterrupt.
    ///
    /// What runs: `scenario` (a scenario dict; the demo landscape if None), or, if `navigate`
    /// is given (JSON overrides of the navigate config), the fly task on an empty nx x ny map.
    ///
    /// policy(obs: dict of np.ndarray) -> sequence of float   called from the simulation thread;
    ///     scenario with drones: one target slot per drone (-1 keeps), asked when drones need a
    ///     decision and `planner` is "policy"; navigate: two velocity fractions per drone, every tick
    /// planner: "off", "greedy", "rollout" or "policy"         who assigns targets in a scenario
    /// save(scenario: dict, path: str | None) -> str            File > Save (path None: pick a new name) and Save as;
    ///     returns the file written
    /// path: the file `scenario` came from, which Save writes back to
    /// open(path: str) -> dict                                  File > Open: a scenario from a file
    /// load_policy(path: str) -> (is_pilot, policy, config, sim)   Policy > Load a checkpoint; `config` is the JSON of
    ///     the task it was trained on (an allocate config for a planner, a navigate config for a pilot), `sim` that
    ///     of its simulation settings
    /// settings: JSON with `sim`, `allocate`, `navigate` objects: the world of an experiment, used as training and
    ///     evaluation use it instead of the viewer's own defaults
    /// seed: the episode to start on
    /// fetch(lat, lon, width_km, height_km) -> dict             called on a worker thread by "Fetch terrain"
    /// load_settings(path: str) -> str                          Settings > Load: JSON of an experiment file's
    ///     sim, allocate and navigate sections
    /// save_settings(path: str, json: str) -> str               Settings > Save: writes such sections as a file
    /// select: index of a drone to open the inspector on
    /// fire_model: "rothermel" or "firespin"
    /// preburn_s: the fires burn this long before the drones start
    #[pyfunction]
    #[pyo3(signature = (scenario=None, policy=None, auto_close_s=None, save=None, fetch=None, start_on_map=false,
                        planner="greedy", navigate=None, nx=35, ny=36, cell_m=10.0, select=None,
                        fire_model="rothermel", open=None, load_policy=None, preburn_s=0.0,
                        load_settings=None, save_settings=None, path=None, settings=None, seed=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn run_ui(
        py: Python<'_>,
        scenario: Option<Bound<'_, PyDict>>,
        policy: Option<Py<PyAny>>,
        auto_close_s: Option<f64>,
        save: Option<Py<PyAny>>,
        fetch: Option<Py<PyAny>>,
        start_on_map: bool,
        planner: &str,
        navigate: Option<&str>,
        nx: u32,
        ny: u32,
        cell_m: f32,
        select: Option<usize>,
        fire_model: &str,
        open: Option<Py<PyAny>>,
        load_policy: Option<Py<PyAny>>,
        preburn_s: f32,
        load_settings: Option<Py<PyAny>>,
        save_settings: Option<Py<PyAny>>,
        path: Option<std::path::PathBuf>,
        settings: Option<String>,
        seed: Option<u64>,
    ) -> PyResult<(u64, u64, u64)> {
        // Without a scenario or task from the caller, the user's default scenario may be shown.
        let explicit = scenario.is_some() || navigate.is_some();
        let fire_model = serde_json::from_value(fire_model.into())
            .map_err(|_| invalid("fire_model", "expected rothermel or firespin"))?;
        let session = match (navigate, scenario) {
            (Some(json), _) => {
                let cfg: NavConfig =
                    serde_json::from_str(json).map_err(|e| invalid("navigate", e))?;
                if cfg.drones == 0 || nx < 3 || ny < 3 || cell_m.is_nan() || cell_m <= 0.0 {
                    return Err(invalid(
                        "navigate",
                        "need at least one drone and a 3x3 map of positive cell size",
                    ));
                }
                roshan_ui::Session::Navigate {
                    grid: Grid { nx, ny, cell_m },
                    cfg,
                }
            }
            (None, Some(d)) => roshan_ui::Session::Scenario(from_py(&d)?),
            (None, None) => roshan_ui::Session::Scenario(roshan_ui::demo_scenario()),
        };
        let planner = match planner {
            "off" => roshan_ui::Planner::Off,
            "greedy" => roshan_ui::Planner::Greedy,
            "head" => roshan_ui::Planner::Head,
            "flank" => roshan_ui::Planner::Flank,
            "rollout" => roshan_ui::Planner::Rollout,
            "policy" if policy.is_some() => roshan_ui::Planner::Policy,
            other => {
                return Err(invalid(
                    "planner",
                    format!(
                        "{other:?}: expected off, greedy, head, flank, rollout, or policy with a policy given"
                    ),
                ));
            }
        };
        let hooks = roshan_ui::Hooks {
            policy: policy.map(as_policy),
            save: save.map(|callable| -> roshan_ui::Save {
                Arc::new(move |scenario: &Scenario, path: Option<&Path>| {
                    Python::attach(|py| {
                        to_py(py, scenario)
                            .and_then(|d| callable.call1(py, (d, path)))
                            .and_then(|out| out.extract::<std::path::PathBuf>(py))
                            .map_err(|err| err.to_string())
                    })
                })
            }),
            open: open.map(|callable| -> roshan_ui::Open {
                Arc::new(move |path: &Path| {
                    Python::attach(|py| {
                        let out = callable.call1(py, (path,));
                        out.and_then(|out| from_py(out.bind(py).cast::<PyDict>()?))
                            .map_err(|err| err.to_string())
                    })
                })
            }),
            load_policy: load_policy.map(|callable| -> roshan_ui::LoadPolicy {
                Arc::new(move |path: &Path| {
                    Python::attach(|py| {
                        let out = callable.call1(py, (path,));
                        out.and_then(|out| out.extract::<(bool, Py<PyAny>, String, String)>(py))
                            .map(|(fly, policy, config, sim)| roshan_ui::Loaded {
                                fly,
                                policy: as_policy(policy),
                                config,
                                sim,
                            })
                            .map_err(|err| err.to_string())
                    })
                })
            }),
            load_settings: load_settings.map(|callable| -> roshan_ui::LoadSettings {
                Arc::new(move |path: &Path| {
                    Python::attach(|py| {
                        let out = callable.call1(py, (path,));
                        out.and_then(|out| out.extract::<String>(py))
                            .map_err(|err| err.to_string())
                    })
                })
            }),
            save_settings: save_settings.map(|callable| -> roshan_ui::SaveSettings {
                Arc::new(move |path: &Path, text: &str| {
                    Python::attach(|py| {
                        let out = callable.call1(py, (path, text));
                        out.and_then(|out| out.extract::<String>(py))
                            .map_err(|err| err.to_string())
                    })
                })
            }),
            fetch: fetch.map(|callable| -> roshan_ui::Fetch {
                Arc::new(move |a: roshan_ui::Area| {
                    Python::attach(|py| {
                        let out = callable.call1(py, (a.lat, a.lon, a.width_km, a.height_km));
                        out.and_then(|out| from_py(out.bind(py).cast::<PyDict>()?))
                            .map_err(|err| err.to_string())
                    })
                })
            }),
        };
        let opts = roshan_ui::Options {
            auto_close: auto_close_s.map(std::time::Duration::from_secs_f64),
            start_on_map,
            planner,
            select,
            fire_model,
            preburn_s,
            explicit,
            path,
            settings,
            seed,
        };

        // The window loop runs with the interpreter lock released. Each frame takes it briefly
        // to let Python deliver pending signals; that is how Ctrl+C reaches us.
        let mut interrupt: Option<PyErr> = None;
        let mut on_frame = || match Python::attach(|py| py.check_signals()) {
            Ok(()) => true,
            Err(err) => {
                interrupt = Some(err);
                false
            }
        };
        let result = py.detach(|| roshan_ui::run(session, hooks, opts, &mut on_frame));

        if let Some(err) = interrupt {
            return Err(err);
        }
        let stats = result.map_err(PyRuntimeError::new_err)?;
        Ok((stats.frames, stats.ticks, stats.policy_calls))
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NavigateEnv>()?;
    m.add_class::<AllocateEnv>()?;
    m.add_function(wrap_pyfunction!(check_scenario, m)?)?;
    m.add_function(wrap_pyfunction!(fuel_models, m)?)?;
    m.add_function(wrap_pyfunction!(decentral_pilot, m)?)?;
    #[cfg(feature = "ui")]
    {
        m.add_function(wrap_pyfunction!(ui::run_ui, m)?)?;
        m.add_function(wrap_pyfunction!(ui::demo_scenario, m)?)?;
    }
    Ok(())
}
