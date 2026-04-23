pub fn build_sparse_neighbors_impl(
    qmatrix_flat: &[f64],
    n: usize,
    threshold: f64,
) -> Result<(Vec<i64>, Vec<i64>, Vec<f64>), &'static str> {
    if n == 0 {
        return Err("QUBO matrix must be non-empty");
    }
    if qmatrix_flat.len() != n * n {
        return Err("QUBO matrix data size mismatch");
    }
    if threshold < 0.0 {
        return Err("threshold cannot be negative");
    }

    let mut offsets = vec![0_i64; n + 1];
    let mut neighbors: Vec<i64> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();

    let mut nnz: i64 = 0;
    for col in 0..n {
        for row in 0..n {
            if row == col {
                continue;
            }
            let val = qmatrix_flat[row * n + col] + qmatrix_flat[col * n + row];
            let keep = if threshold > 0.0 {
                val.abs() > threshold
            } else {
                val != 0.0
            };
            if keep {
                neighbors.push(row as i64);
                weights.push(val);
                nnz += 1;
            }
        }
        offsets[col + 1] = nnz;
    }

    Ok((offsets, neighbors, weights))
}

