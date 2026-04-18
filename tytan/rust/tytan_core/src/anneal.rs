use crate::delta::delta_energy_impl;

#[derive(Clone, Copy)]
pub struct AnnealStats {
    pub proposals: usize,
    pub accepted: usize,
}

fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

fn sa_step_once(
    next_states: &mut [f64],
    shots: usize,
    dims: usize,
    next_energies: &mut [f64],
    qmatrix_flat: &[f64],
    nrows: usize,
    ncols: usize,
    beta: f64,
    rng: &mut u64,
    symmetric: bool,
) -> Result<usize, &'static str> {
    let mut accepted = 0usize;

    for shot in 0..shots {
        *rng = xorshift64(*rng);
        let idx = (*rng as usize) % dims;
        let start = shot * dims;
        let row = &next_states[start..start + dims];
        let delta = delta_energy_impl(
            row,
            qmatrix_flat,
            nrows,
            ncols,
            idx,
            Some(next_energies[shot]),
            symmetric,
        )?;

        let accept = if delta <= 0.0 {
            true
        } else {
            *rng = xorshift64(*rng);
            let u = ((*rng >> 11) as f64) / ((1u64 << 53) as f64);
            u < (-beta * delta).exp()
        };

        if accept {
            next_states[start + idx] = 1.0 - next_states[start + idx];
            next_energies[shot] += delta;
            accepted += 1;
        }
    }

    Ok(accepted)
}

pub fn sa_step_single_flip_impl(
    states_flat: &[f64],
    shots: usize,
    dims: usize,
    energies: &[f64],
    qmatrix_flat: &[f64],
    nrows: usize,
    ncols: usize,
    beta: f64,
    rng_state: u64,
    symmetric: bool,
) -> Result<(Vec<f64>, Vec<f64>, u64, AnnealStats), &'static str> {
    if states_flat.len() != shots * dims {
        return Err("States data size mismatch");
    }
    if energies.len() != shots {
        return Err("Energy size mismatch");
    }
    if dims != nrows || nrows != ncols {
        return Err("QUBO matrix must be square and match state dimension");
    }

    let mut next_states = states_flat.to_vec();
    let mut next_energies = energies.to_vec();
    let mut rng = if rng_state == 0 {
        0x9E3779B97F4A7C15
    } else {
        rng_state
    };
    let accepted = sa_step_once(
        &mut next_states,
        shots,
        dims,
        &mut next_energies,
        qmatrix_flat,
        nrows,
        ncols,
        beta,
        &mut rng,
        symmetric,
    )?;

    Ok((
        next_states,
        next_energies,
        rng,
        AnnealStats {
            proposals: shots,
            accepted,
        },
    ))
}

pub fn sa_step_multi_flip_impl(
    states_flat: &[f64],
    shots: usize,
    dims: usize,
    energies: &[f64],
    qmatrix_flat: &[f64],
    nrows: usize,
    ncols: usize,
    betas: &[f64],
    rng_state: u64,
    symmetric: bool,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, u64, AnnealStats), &'static str> {
    if states_flat.len() != shots * dims {
        return Err("States data size mismatch");
    }
    if energies.len() != shots {
        return Err("Energy size mismatch");
    }
    if dims != nrows || nrows != ncols {
        return Err("QUBO matrix must be square and match state dimension");
    }
    if betas.is_empty() {
        return Err("betas must not be empty");
    }

    let mut next_states = states_flat.to_vec();
    let mut next_energies = energies.to_vec();
    let mut history_states = Vec::with_capacity(betas.len() * states_flat.len());
    let mut history_energies = Vec::with_capacity(betas.len() * energies.len());
    let mut rng = if rng_state == 0 {
        0x9E3779B97F4A7C15
    } else {
        rng_state
    };
    let mut accepted = 0usize;

    for &beta in betas {
        accepted += sa_step_once(
            &mut next_states,
            shots,
            dims,
            &mut next_energies,
            qmatrix_flat,
            nrows,
            ncols,
            beta,
            &mut rng,
            symmetric,
        )?;
        history_states.extend_from_slice(&next_states);
        history_energies.extend_from_slice(&next_energies);
    }

    Ok((
        next_states,
        next_energies,
        history_states,
        history_energies,
        rng,
        AnnealStats {
            proposals: shots * betas.len(),
            accepted,
        },
    ))
}
