use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

mod anneal;
mod delta;
mod reduce;
mod types;

#[pyfunction(signature = (state, qmatrix, index, current_energy=None))]
fn delta_energy(
    state: PyReadonlyArray1<'_, f64>,
    qmatrix: PyReadonlyArray2<'_, f64>,
    index: usize,
    current_energy: Option<f64>,
) -> PyResult<f64> {
    let state_view = state.as_array();
    let q_view = qmatrix.as_array();
    let q_shape = q_view.shape();
    let state_vec = state_view.to_vec();
    let q_vec: Vec<f64> = q_view.iter().copied().collect();

    delta::delta_energy_impl(
        &state_vec,
        &q_vec,
        q_shape[0],
        q_shape[1],
        index,
        current_energy,
    )
    .map_err(PyValueError::new_err)
}

#[pyfunction]
fn batch_delta<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    qmatrix: PyReadonlyArray2<'py, f64>,
    indices: PyReadonlyArray1<'py, i64>,
    energies: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let s_view = states.as_array();
    let q_view = qmatrix.as_array();
    let i_view = indices.as_array();
    let e_view = energies.as_array();

    let s_shape = s_view.shape();
    let q_shape = q_view.shape();
    let states_flat: Vec<f64> = s_view.iter().copied().collect();
    let q_flat: Vec<f64> = q_view.iter().copied().collect();
    let mut indices_vec = Vec::with_capacity(i_view.len());
    for value in i_view.iter() {
        if *value < 0 {
            return Err(PyValueError::new_err("indices must be non-negative"));
        }
        indices_vec.push(*value as usize);
    }
    let energies_vec = e_view.to_vec();

    let out = delta::batch_delta_impl(
        &states_flat,
        s_shape[0],
        s_shape[1],
        &q_flat,
        q_shape[0],
        q_shape[1],
        &indices_vec,
        &energies_vec,
    )
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
        return Err(PyValueError::new_err("variable_names length must match state dimension"));
    }

    let states_flat: Vec<f64> = s_view.iter().copied().collect();
    let energies_vec = e_view.to_vec();
    let entries = reduce::aggregate_results_impl(&states_flat, s_shape[0], s_shape[1], &energies_vec)
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

#[pyfunction]
fn sa_step_single_flip<'py>(
    py: Python<'py>,
    states: PyReadonlyArray2<'py, f64>,
    energies: PyReadonlyArray1<'py, f64>,
    qmatrix: PyReadonlyArray2<'py, f64>,
    beta: f64,
    rng_state: u64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>, PyObject)> {
    let s_view = states.as_array();
    let e_view = energies.as_array();
    let q_view = qmatrix.as_array();

    let s_shape = s_view.shape();
    let q_shape = q_view.shape();
    let states_flat: Vec<f64> = s_view.iter().copied().collect();
    let energies_vec = e_view.to_vec();
    let q_flat: Vec<f64> = q_view.iter().copied().collect();

    let (next_states_flat, next_energies, next_rng, stats) = anneal::sa_step_single_flip_impl(
        &states_flat,
        s_shape[0],
        s_shape[1],
        &energies_vec,
        &q_flat,
        q_shape[0],
        q_shape[1],
        beta,
        rng_state,
    )
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

#[pymodule]
fn _tytan_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(delta_energy, m)?)?;
    m.add_function(wrap_pyfunction!(batch_delta, m)?)?;
    m.add_function(wrap_pyfunction!(aggregate_results, m)?)?;
    m.add_function(wrap_pyfunction!(sa_step_single_flip, m)?)?;
    Ok(())
}
