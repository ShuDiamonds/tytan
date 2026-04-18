use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

mod anneal;
mod delta;
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

    let mut rows = Vec::with_capacity(s_shape[0]);
    for chunk in next_states_flat.chunks(s_shape[1]) {
        rows.push(chunk.to_vec());
    }
    let states_py = PyArray2::from_vec2_bound(py, &rows)
        .map_err(|_| PyValueError::new_err("Failed to build states array"))?;
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

#[pymodule]
fn _tytan_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(delta_energy, m)?)?;
    m.add_function(wrap_pyfunction!(batch_delta, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_results, m)?)?;
    m.add_function(wrap_pyfunction!(sa_step_single_flip, m)?)?;
    m.add_function(wrap_pyfunction!(sa_step_multi_flip, m)?)?;
    Ok(())
}
