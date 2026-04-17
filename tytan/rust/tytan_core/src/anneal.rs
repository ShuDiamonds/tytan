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
    let mut rng = if rng_state == 0 { 0x9E3779B97F4A7C15 } else { rng_state };
    let mut accepted = 0usize;

    for shot in 0..shots {
        rng = xorshift64(rng);
        let idx = (rng as usize) % dims;
        let start = shot * dims;
        let row = &next_states[start..start + dims];
        let delta = delta_energy_impl(row, qmatrix_flat, nrows, ncols, idx, Some(next_energies[shot]))?;

        let accept = if delta <= 0.0 {
            true
        } else {
            rng = xorshift64(rng);
            let u = ((rng >> 11) as f64) / ((1u64 << 53) as f64);
            u < (-beta * delta).exp()
        };

        if accept {
            next_states[start + idx] = 1.0 - next_states[start + idx];
            next_energies[shot] += delta;
            accepted += 1;
        }
    }

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
