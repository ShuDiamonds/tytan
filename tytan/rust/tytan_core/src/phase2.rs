use crate::types::validate_square_matrix;

fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

fn select_top_k(energies: &[f64], top_k: usize) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..energies.len()).collect();
    indices.sort_by(|a, b| energies[*a].partial_cmp(&energies[*b]).unwrap_or(std::cmp::Ordering::Equal));
    indices.truncate(top_k);
    indices
}

pub struct Phase2Stats {
    pub proposals: usize,
    pub accepted: usize,
}

pub fn sa_phase2_delta_cache_impl(
    states_flat: &[f64],
    shots: usize,
    dims: usize,
    energies: &[f64],
    qmatrix_flat: &[f64],
    offsets: &[i64],
    neighbors: &[i64],
    weights: &[f64],
    betas: &[f64],
    sweeps_per_step: usize,
    rng_state: u64,
    top_k: usize,
) -> Result<(Vec<f64>, Vec<f64>, u64, Phase2Stats), &'static str> {
    if shots == 0 || dims == 0 {
        return Err("shots and dims must be positive");
    }
    if states_flat.len() != shots * dims {
        return Err("States data size mismatch");
    }
    if energies.len() != shots {
        return Err("Energy size mismatch");
    }
    validate_square_matrix(qmatrix_flat.len(), dims, dims)?;
    if offsets.len() != dims + 1 {
        return Err("offsets length must be dims + 1");
    }
    if weights.len() != neighbors.len() {
        return Err("neighbors and weights length mismatch");
    }
    if sweeps_per_step == 0 {
        return Err("sweeps_per_step must be positive");
    }
    if top_k == 0 {
        return Err("top_k must be positive");
    }
    if betas.is_empty() {
        return Err("betas must not be empty");
    }

    let selected = select_top_k(energies, top_k.min(shots));
    let k = selected.len();

    let mut states_out = states_flat.to_vec();
    let mut energies_out = energies.to_vec();

    // Build delta-cache using sparse neighbors (off-diagonal of Q+Q.T) + diagonal correction.
    let mut delta_cache = vec![0.0_f64; k * dims];
    for (r, &shot_idx) in selected.iter().enumerate() {
        let base = shot_idx * dims;
        for i in 0..dims {
            let mut cross_sparse = 0.0_f64;
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            for pos in start..end {
                let j = neighbors[pos] as usize;
                cross_sparse += weights[pos] * states_out[base + j];
            }
            let sii = states_out[base + i];
            let qii = qmatrix_flat[i * dims + i];
            let cross = cross_sparse + 2.0 * qii * sii;
            let flip_i = 1.0 - 2.0 * sii;
            delta_cache[r * dims + i] = flip_i * cross + qii;
        }
    }

    let mut rng = if rng_state == 0 {
        0x9E3779B97F4A7C15
    } else {
        rng_state
    };
    let mut proposals = 0usize;
    let mut accepted = 0usize;

    for &beta in betas {
        for _ in 0..sweeps_per_step {
            for (r, &shot_idx) in selected.iter().enumerate() {
                rng = xorshift64(rng);
                let idx = (rng as usize) % dims;
                proposals += 1;

                let delta = delta_cache[r * dims + idx];
                let accept = if delta <= 0.0 {
                    true
                } else {
                    rng = xorshift64(rng);
                    let u = ((rng >> 11) as f64) / ((1u64 << 53) as f64);
                    u < (-beta * delta).exp()
                };

                if !accept {
                    continue;
                }
                accepted += 1;

                let base = shot_idx * dims;
                let old = states_out[base + idx];
                let flip_i = 1.0 - 2.0 * old;
                states_out[base + idx] = 1.0 - old;
                energies_out[shot_idx] += delta;
                delta_cache[r * dims + idx] = -delta_cache[r * dims + idx];

                let start = offsets[idx] as usize;
                let end = offsets[idx + 1] as usize;
                for pos in start..end {
                    let j = neighbors[pos] as usize;
                    let sj = states_out[base + j];
                    let flip_j = 1.0 - 2.0 * sj;
                    delta_cache[r * dims + j] += flip_j * weights[pos] * flip_i;
                }
            }
        }
    }

    Ok((
        states_out,
        energies_out,
        rng,
        Phase2Stats {
            proposals,
            accepted,
        },
    ))
}

