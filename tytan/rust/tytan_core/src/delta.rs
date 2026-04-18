use crate::types::{validate_square_matrix, validate_state_len};

fn delta_energy_flipped_bit(state: &[f64], qmatrix_flat: &[f64], n: usize, index: usize) -> f64 {
    let flip = 1.0 - 2.0 * state[index];
    let row_offset = index * n;
    let mut cross_term = 0.0_f64;

    for j in 0..n {
        cross_term += state[j] * (qmatrix_flat[row_offset + j] + qmatrix_flat[j * n + index]);
    }

    flip * cross_term + qmatrix_flat[row_offset + index]
}

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

    let _ = current_energy;
    Ok(delta_energy_flipped_bit(state, qmatrix_flat, nrows, index))
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
        let _ = energies[shot];
        let delta = delta_energy_flipped_bit(state, qmatrix_flat, nrows, idx);
        deltas.push(delta);
    }
    Ok(deltas)
}

#[cfg(test)]
mod tests {
    use super::{batch_delta_impl, delta_energy_impl};
    use crate::types::energy_of_state;

    fn brute_delta(state: &[f64], qmatrix_flat: &[f64], n: usize, index: usize) -> f64 {
        let mut flipped = state.to_vec();
        flipped[index] = 1.0 - flipped[index];
        energy_of_state(&flipped, qmatrix_flat, n) - energy_of_state(state, qmatrix_flat, n)
    }

    #[test]
    fn delta_energy_matches_bruteforce_for_symmetric_matrix() {
        let state = [1.0, 0.0, 1.0];
        let q = [1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 5.0, 6.0];

        for index in 0..state.len() {
            let expected = brute_delta(&state, &q, 3, index);
            let actual = delta_energy_impl(&state, &q, 3, 3, index, None).unwrap();
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn delta_energy_matches_bruteforce_for_asymmetric_matrix_and_batch_path() {
        let states = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let q = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let energies = [
            energy_of_state(&states[0..3], &q, 3),
            energy_of_state(&states[3..6], &q, 3),
        ];
        let indices = [1_usize, 2_usize];

        let batch = batch_delta_impl(&states, 2, 3, &q, 3, 3, &indices, &energies).unwrap();
        for shot in 0..2 {
            let state = &states[shot * 3..shot * 3 + 3];
            let expected = brute_delta(state, &q, 3, indices[shot]);
            assert!((batch[shot] - expected).abs() < 1e-12);
        }
    }
}
