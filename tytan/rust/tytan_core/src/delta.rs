use crate::types::{energy_of_state, validate_square_matrix, validate_state_len};

pub fn delta_energy_impl(
    state: &[f64],
    qmatrix_flat: &[f64],
    nrows: usize,
    ncols: usize,
    index: usize,
    current_energy: Option<f64>,
) -> Result<f64, &'static str> {
    validate_square_matrix(qmatrix_flat.len(), nrows, ncols)?;
    validate_state_len(state.len(), nrows)?;
    if index >= nrows {
        return Err("Index out of range");
    }

    let e0 = current_energy.unwrap_or_else(|| energy_of_state(state, qmatrix_flat, nrows));
    let mut flipped = state.to_vec();
    flipped[index] = 1.0 - flipped[index];
    let e1 = energy_of_state(&flipped, qmatrix_flat, nrows);
    Ok(e1 - e0)
}

pub fn batch_delta_impl(
    states_flat: &[f64],
    shots: usize,
    dims: usize,
    qmatrix_flat: &[f64],
    nrows: usize,
    ncols: usize,
    indices: &[usize],
    energies: &[f64],
) -> Result<Vec<f64>, &'static str> {
    validate_square_matrix(qmatrix_flat.len(), nrows, ncols)?;
    if dims != nrows {
        return Err("State dimension mismatch");
    }
    if states_flat.len() != shots * dims {
        return Err("States data size mismatch");
    }
    if indices.len() != shots || energies.len() != shots {
        return Err("indices and energies must match shot size");
    }

    let mut deltas = Vec::with_capacity(shots);
    for shot in 0..shots {
        let idx = indices[shot];
        if idx >= dims {
            return Err("Index out of range");
        }
        let start = shot * dims;
        let state = &states_flat[start..start + dims];
        let delta = delta_energy_impl(state, qmatrix_flat, nrows, ncols, idx, Some(energies[shot]))?;
        deltas.push(delta);
    }
    Ok(deltas)
}
