pub fn validate_square_matrix(flat_len: usize, nrows: usize, ncols: usize) -> Result<(), &'static str> {
    if nrows == 0 || ncols == 0 {
        return Err("QUBO matrix must be non-empty");
    }
    if nrows != ncols {
        return Err("QUBO matrix must be square");
    }
    if flat_len != nrows * ncols {
        return Err("QUBO matrix data size mismatch");
    }
    Ok(())
}

pub fn validate_state_len(state_len: usize, matrix_size: usize) -> Result<(), &'static str> {
    if state_len != matrix_size {
        return Err("State dimension mismatch");
    }
    Ok(())
}

pub fn energy_of_state(state: &[f64], qmatrix_flat: &[f64], n: usize) -> f64 {
    let mut sum = 0.0_f64;
    for i in 0..n {
        let si = state[i];
        for j in 0..n {
            sum += si * qmatrix_flat[i * n + j] * state[j];
        }
    }
    sum
}
