use numpy::{ndarray::Array2, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

mod adaptive;
mod anneal;
mod delta;
mod pool;
mod presolve;
mod reduce;
mod types;

#[pyfunction(signature = (state, qmatrix, index, current_energy=None, symmetric=false))]
fn delta_energy(
    py: Python<'_>,
    state: PyReadonlyArray1<'_, f64>,
    qmatrix: PyReadonlyArray2<'_, f64>,
    index: usize,
    current_energy: Option<f64>,
    symmetric: bool,
) -> PyResult<f64> {
    let state_view = state.as_array();
    let q_view = qmatrix.as_array();
    let q_shape = q_view.shape();

    // Use slices directly if contiguous to avoid copying entire arrays
    let state_slice = state_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("State array must be C-contiguous"))?;
    let q_slice = q_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Q matrix must be C-contiguous"))?;

    py.allow_threads(|| {
        delta::delta_energy_impl(
            state_slice,
            q_slice,
            q_shape[0],
            q_shape[1],
            index,
            current_energy,
            symmetric,
        )
    })
    .map_err(PyValueError::new_err)
}

#[pyfunction(signature = (states, qmatrix, indices, energies, symmetric=false))]
fn batch_delta<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    qmatrix: PyReadonlyArray2<'py, f64>,
    indices: PyReadonlyArray1<'py, i64>,
    energies: PyReadonlyArray1<'py, f64>,
    symmetric: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let s_view = states.as_array();
    let q_view = qmatrix.as_array();
    let i_view = indices.as_array();
    let e_view = energies.as_array();

    let s_shape = s_view.shape();
    let q_shape = q_view.shape();

    // Use slices directly to avoid copying
    let states_slice = s_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("States array must be C-contiguous"))?;
    let q_slice = q_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Q matrix must be C-contiguous"))?;
    let energies_slice = e_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Energies array must be C-contiguous"))?;

    let mut indices_vec = Vec::with_capacity(i_view.len());
    for value in i_view.iter() {
        if *value < 0 {
            return Err(PyValueError::new_err("indices must be non-negative"));
        }
        indices_vec.push(*value as usize);
    }

    let out = py
        .allow_threads(|| {
            delta::batch_delta_impl(
                states_slice,
                s_shape[0],
                s_shape[1],
                q_slice,
                q_shape[0],
                q_shape[1],
                &indices_vec,
                energies_slice,
                symmetric,
            )
        })
        .map_err(PyValueError::new_err)?;

    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
fn aggregate_results(
    py: Python<'_>,
    states: PyReadonlyArray2<'_, f64>,
    energies: PyReadonlyArray1<'_, f64>,
    variable_names: Vec<String>,
) -> PyResult<PyObject> {
    let s_view = states.as_array();
    let s_shape = s_view.shape();
    let e_view = energies.as_array();

    if variable_names.len() != s_shape[1] {
        return Err(PyValueError::new_err(
            "variable_names length must match state dimension",
        ));
    }

    let states_slice = s_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("States array must be C-contiguous"))?;
    let energies_slice = e_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Energies array must be C-contiguous"))?;
    let entries = py
        .allow_threads(|| {
            reduce::aggregate_results_impl(states_slice, s_shape[0], s_shape[1], energies_slice)
        })
        .map_err(PyValueError::new_err)?;

    let rows = PyList::empty_bound(py);
    for entry in entries {
        let state_dict = PyDict::new_bound(py);
        for (idx, value) in entry.state.iter().enumerate() {
            state_dict.set_item(&variable_names[idx], *value)?;
        }
        let row = PyList::empty_bound(py);
        row.append(state_dict)?;
        row.append(entry.energy)?;
        row.append(entry.count)?;
        rows.append(row)?;
    }
    Ok(rows.into_any().unbind())
}

#[pyfunction(signature = (
    qmatrix,
    hard_threshold = 1.5,
    soft_threshold = 1.0,
    coupling_threshold = 0.2,
    aggregation_threshold = 0.8,
    weak_cut_threshold = 0.1,
    probing_budget = 64,
    pool_frequency = None,
    pair_correlation = None
))]
fn presolve_plan<'py>(
    py: Python<'py>,
    qmatrix: PyReadonlyArray2<'py, f64>,
    hard_threshold: f64,
    soft_threshold: f64,
    coupling_threshold: f64,
    aggregation_threshold: f64,
    weak_cut_threshold: f64,
    probing_budget: usize,
    pool_frequency: Option<PyReadonlyArray1<'py, f64>>,
    pair_correlation: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<PyObject> {
    let q_view = qmatrix.as_array();
    let q_shape = q_view.shape();
    let q_slice = q_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Q matrix must be C-contiguous"))?;
    let pool_frequency_owned = match pool_frequency {
        Some(freq) => Some(freq.as_array().iter().copied().collect::<Vec<f64>>()),
        None => None,
    };
    let pair_correlation_owned = match pair_correlation {
        Some(corr) => Some(corr.as_array().iter().copied().collect::<Vec<f64>>()),
        None => None,
    };

    let output = py
        .allow_threads(|| {
            presolve::presolve_plan_impl(
                q_slice,
                q_shape[0],
                hard_threshold,
                soft_threshold,
                coupling_threshold,
                aggregation_threshold,
                weak_cut_threshold,
                probing_budget,
                pool_frequency_owned.as_deref(),
                pair_correlation_owned.as_deref(),
            )
        })
        .map_err(PyValueError::new_err)?;

    let result = PyDict::new_bound(py);
    let reduced_array = Array2::from_shape_vec(
        (output.reduced_dim, output.reduced_dim),
        output.reduced_matrix,
    )
    .map_err(|_| PyValueError::new_err("Failed to build reduced matrix"))?;
    result.set_item(
        "reduced_matrix",
        PyArray2::from_owned_array_bound(py, reduced_array),
    )?;
    result.set_item("active_indices", output.active_indices)?;
    result.set_item("hard_fixed_indices", output.hard_fixed_indices)?;
    result.set_item("hard_fixed_values", output.hard_fixed_values)?;
    result.set_item("soft_fixed_indices", output.soft_fixed_indices)?;
    result.set_item("fix_confidence", output.fix_confidence)?;
    result.set_item("aggregation_src", output.aggregation_src)?;
    result.set_item("aggregation_dst", output.aggregation_dst)?;
    result.set_item("aggregation_relation", output.aggregation_relation)?;
    result.set_item("aggregation_strength", output.aggregation_strength)?;
    result.set_item("component_ids", output.component_ids.clone())?;
    result.set_item("block_membership", output.component_ids)?;
    result.set_item("boundary_indices", output.boundary_indices)?;
    result.set_item("frontier_indices", output.frontier_indices)?;
    result.set_item("branch_candidate_indices", output.branch_candidate_indices)?;
    result.set_item("branch_candidate_scores", output.branch_candidate_scores)?;

    let stats = PyDict::new_bound(py);
    for (key, value) in output.stats {
        stats.set_item(key, value)?;
    }
    result.set_item("stats", stats)?;
    Ok(result.into_any().unbind())
}

#[pyfunction(signature = (states, energies, qmatrix, beta, rng_state, symmetric=false))]
fn sa_step_single_flip<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    energies: PyReadonlyArray1<'py, f64>,
    qmatrix: PyReadonlyArray2<'py, f64>,
    beta: f64,
    rng_state: u64,
    symmetric: bool,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<f64>>,
    PyObject,
)> {
    let s_view = states.as_array();
    let e_view = energies.as_array();
    let q_view = qmatrix.as_array();

    let s_shape = s_view.shape();
    let q_shape = q_view.shape();
    let states_slice = s_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("States array must be C-contiguous"))?;
    let energies_slice = e_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Energies array must be C-contiguous"))?;
    let q_slice = q_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Q matrix must be C-contiguous"))?;

    let (next_states_flat, next_energies, next_rng, stats) = py
        .allow_threads(|| {
            anneal::sa_step_single_flip_impl(
                states_slice,
                s_shape[0],
                s_shape[1],
                energies_slice,
                q_slice,
                q_shape[0],
                q_shape[1],
                beta,
                rng_state,
                symmetric,
            )
        })
        .map_err(PyValueError::new_err)?;

    let states_array = Array2::from_shape_vec((s_shape[0], s_shape[1]), next_states_flat)
        .map_err(|_| PyValueError::new_err("Failed to build states array"))?;
    let states_py = PyArray2::from_owned_array_bound(py, states_array);
    let energies_py = PyArray1::from_vec_bound(py, next_energies);

    let stats_dict = PyDict::new_bound(py);
    stats_dict.set_item("rng_state", next_rng)?;
    stats_dict.set_item("proposals", stats.proposals)?;
    stats_dict.set_item("accepted", stats.accepted)?;

    Ok((states_py, energies_py, stats_dict.into_any().unbind()))
}

#[pyfunction(signature = (states, energies, qmatrix, betas, rng_state, symmetric=false))]
fn sa_step_multi_flip<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    energies: PyReadonlyArray1<'py, f64>,
    qmatrix: PyReadonlyArray2<'py, f64>,
    betas: PyReadonlyArray1<'py, f64>,
    rng_state: u64,
    symmetric: bool,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    PyObject,
)> {
    let s_view = states.as_array();
    let e_view = energies.as_array();
    let q_view = qmatrix.as_array();
    let b_view = betas.as_array();

    let s_shape = s_view.shape();
    let q_shape = q_view.shape();
    let states_slice = s_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("States array must be C-contiguous"))?;
    let energies_slice = e_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Energies array must be C-contiguous"))?;
    let q_slice = q_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Q matrix must be C-contiguous"))?;
    let betas_slice = b_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("betas array must be C-contiguous"))?;

    let (_next_states_flat, _next_energies, history_states, history_energies, next_rng, stats) = py
        .allow_threads(|| {
            anneal::sa_step_multi_flip_impl(
                states_slice,
                s_shape[0],
                s_shape[1],
                energies_slice,
                q_slice,
                q_shape[0],
                q_shape[1],
                betas_slice,
                rng_state,
                symmetric,
            )
        })
        .map_err(PyValueError::new_err)?;

    let history_states_py = PyArray1::from_vec_bound(py, history_states);
    let history_energies_py = PyArray1::from_vec_bound(py, history_energies);

    let stats_dict = PyDict::new_bound(py);
    stats_dict.set_item("rng_state", next_rng)?;
    stats_dict.set_item("proposals", stats.proposals)?;
    stats_dict.set_item("accepted", stats.accepted)?;
    stats_dict.set_item("steps", betas_slice.len())?;
    stats_dict.set_item("shots", s_shape[0])?;
    stats_dict.set_item("dims", s_shape[1])?;

    Ok((
        history_states_py,
        history_energies_py,
        stats_dict.into_any().unbind(),
    ))
}

#[pyfunction(signature = (
    qmatrix,
    index_names,
    shots = 64,
    steps = 128,
    batch_size = None,
    init_temp = 5.0,
    end_temp = 0.01,
    schedule = "linear",
    adaptive = true,
    strategies = None,
    epsilon = 0.2,
    include_diverse = true,
    pool_max_entries = 128,
    near_dup_hamming = 2,
    replace_margin = 1e-6,
    stall_steps = 20,
    restart_ratio = 0.25,
    restart_min_flips = 4,
    restart_burnin_steps = 1,
    restart_diversity_threshold = None,
    novelty_weight = 0.05,
    seed = None
))]
fn adaptive_bulk_sa<'py>(
    py: Python<'py>,
    qmatrix: PyReadonlyArray2<'py, f64>,
    index_names: Vec<String>,
    shots: usize,
    steps: usize,
    batch_size: Option<usize>,
    init_temp: f64,
    end_temp: f64,
    schedule: &str,
    adaptive: bool,
    strategies: Option<Vec<(String, String, f64)>>,
    epsilon: f64,
    include_diverse: bool,
    pool_max_entries: usize,
    near_dup_hamming: usize,
    replace_margin: f64,
    stall_steps: usize,
    restart_ratio: f64,
    restart_min_flips: usize,
    restart_burnin_steps: usize,
    restart_diversity_threshold: Option<f64>,
    novelty_weight: f64,
    seed: Option<u64>,
) -> PyResult<(PyObject, PyObject)> {
    let q_view = qmatrix.as_array();
    let q_shape = q_view.shape();
    let q_slice = q_view
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("Q matrix must be C-contiguous"))?;
    if index_names.len() != q_shape[1] {
        return Err(PyValueError::new_err(
            "index_names length must match state dimension",
        ));
    }
    let schedule_kind = match schedule.to_ascii_lowercase().as_str() {
        "linear" => adaptive::ScheduleKind::Linear,
        "exponential" => adaptive::ScheduleKind::Exponential,
        _ => {
            return Err(PyValueError::new_err(
                "schedule must be linear or exponential",
            ))
        }
    };
    let strategies = strategies.map(|items| {
        items
            .into_iter()
            .map(|(name, kind, weight)| adaptive::StrategySpec { name, kind, weight })
            .collect::<Vec<_>>()
    });
    let config = adaptive::AdaptiveBulkConfig {
        shots,
        steps,
        batch_size: batch_size.unwrap_or(shots),
        init_temp,
        end_temp,
        schedule: schedule_kind,
        adaptive,
        epsilon,
        include_diverse,
        pool_max_entries,
        near_dup_hamming,
        replace_margin,
        stall_steps,
        restart_ratio,
        restart_min_flips,
        restart_burnin_steps,
        restart_diversity_threshold,
        novelty_weight,
        seed: seed.unwrap_or(0),
    };
    let output = py
        .allow_threads(|| {
            adaptive::run_adaptive_bulk_sa_impl(
                q_slice,
                q_shape[0],
                &index_names,
                &config,
                strategies.as_deref(),
            )
        })
        .map_err(PyValueError::new_err)?;

    let rows = PyList::empty_bound(py);
    for row in output.rows {
        let state_dict = PyDict::new_bound(py);
        for (idx, value) in row.state.iter().enumerate() {
            state_dict.set_item(&index_names[idx], *value)?;
        }
        let py_row = PyList::empty_bound(py);
        py_row.append(state_dict)?;
        py_row.append(row.energy)?;
        py_row.append(row.count)?;
        rows.append(py_row)?;
    }

    let stats = PyDict::new_bound(py);
    stats.set_item("best_energy", output.stats.best_energy)?;
    stats.set_item("restart_count", output.stats.restart_count)?;
    stats.set_item(
        "pool_mean_pairwise_distance",
        output.stats.pool_mean_pairwise_distance,
    )?;
    stats.set_item("state_diversity", output.stats.state_diversity)?;
    stats.set_item("rust_core", true)?;

    let strategy_weights = PyDict::new_bound(py);
    for (name, weight) in output.stats.strategy_weights {
        strategy_weights.set_item(name, weight)?;
    }
    stats.set_item("strategy_weights", strategy_weights)?;

    let log_entries = PyList::empty_bound(py);
    for entry in output.stats.log_entries {
        let item = PyDict::new_bound(py);
        item.set_item("strategy", entry.strategy)?;
        item.set_item("temperature", entry.temperature)?;
        item.set_item("improvements", entry.improvements)?;
        log_entries.append(item)?;
    }
    stats.set_item("log_entries", log_entries)?;
    stats.set_item("clamp_mode", "none")?;

    Ok((rows.into_any().unbind(), stats.into_any().unbind()))
}

#[pymodule]
fn _tytan_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(delta_energy, m)?)?;
    m.add_function(wrap_pyfunction!(batch_delta, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_results, m)?)?;
    m.add_function(wrap_pyfunction!(presolve_plan, m)?)?;
    m.add_function(wrap_pyfunction!(sa_step_single_flip, m)?)?;
    m.add_function(wrap_pyfunction!(sa_step_multi_flip, m)?)?;
    m.add_function(wrap_pyfunction!(adaptive_bulk_sa, m)?)?;
    Ok(())
}
