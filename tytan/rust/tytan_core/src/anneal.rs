use crate::delta::delta_energy_impl;
use rayon::prelude::*;

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

fn derive_shot_seed(base: u64, shot: usize) -> u64 {
    xorshift64(base ^ ((shot as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
}

#[derive(Clone)]
struct ShotOutcome {
    state: Vec<f64>,
    energy: f64,
    accepted: usize,
}

fn sa_step_once_parallel(
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

    let outcomes: Result<Vec<ShotOutcome>, &'static str> = (0..shots)
        .into_par_iter()
        .map(|shot| {
            let mut rng = derive_shot_seed(rng_state, shot);
            let start = shot * dims;
            let mut state = states_flat[start..start + dims].to_vec();
            let mut energy = energies[shot];
            rng = xorshift64(rng);
            let idx = (rng as usize) % dims;
            let delta = delta_energy_impl(
                &state,
                qmatrix_flat,
                nrows,
                ncols,
                idx,
                Some(energy),
                symmetric,
            )?;

            let accept = if delta <= 0.0 {
                true
            } else {
                rng = xorshift64(rng);
                let u = ((rng >> 11) as f64) / ((1u64 << 53) as f64);
                u < (-beta * delta).exp()
            };

            let mut accepted = 0usize;
            if accept {
                state[idx] = 1.0 - state[idx];
                energy += delta;
                accepted = 1;
            }

            Ok(ShotOutcome {
                state,
                energy,
                accepted,
            })
        })
        .collect();

    let outcomes = outcomes?;
    let mut next_states = vec![0.0_f64; shots * dims];
    let mut next_energies = vec![0.0_f64; shots];
    let mut accepted = 0usize;
    for (shot, outcome) in outcomes.into_iter().enumerate() {
        let start = shot * dims;
        next_states[start..start + dims].copy_from_slice(&outcome.state);
        next_energies[shot] = outcome.energy;
        accepted += outcome.accepted;
    }
    let next_rng = xorshift64(rng_state ^ (shots as u64) ^ (accepted as u64));
    Ok((
        next_states,
        next_energies,
        next_rng,
        AnnealStats {
            proposals: shots,
            accepted,
        },
    ))
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

    let rng = if rng_state == 0 {
        0x9E3779B97F4A7C15
    } else {
        rng_state
    };
    sa_step_once_parallel(
        states_flat,
        shots,
        dims,
        energies,
        qmatrix_flat,
        nrows,
        ncols,
        beta,
        rng,
        symmetric,
    )
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

    let mut history_states = Vec::with_capacity(betas.len() * states_flat.len());
    let mut history_energies = Vec::with_capacity(betas.len() * energies.len());
    let mut current_states = states_flat.to_vec();
    let mut current_energies = energies.to_vec();
    let mut rng = if rng_state == 0 {
        0x9E3779B97F4A7C15
    } else {
        rng_state
    };
    let mut accepted = 0usize;

    for &beta in betas {
        let (next_states, next_energies, next_rng, stats) = sa_step_once_parallel(
            &current_states,
            shots,
            dims,
            &current_energies,
            qmatrix_flat,
            nrows,
            ncols,
            beta,
            rng,
            symmetric,
        )?;
        accepted += stats.accepted;
        history_states.extend_from_slice(&next_states);
        history_energies.extend_from_slice(&next_energies);
        current_states = next_states;
        current_energies = next_energies;
        rng = next_rng;
    }

    Ok((
        current_states,
        current_energies,
        history_states,
        history_energies,
        rng,
        AnnealStats {
            proposals: shots * betas.len(),
            accepted,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::sa_step_single_flip_impl;

    #[test]
    fn single_flip_parallel_is_deterministic() {
        let states = vec![0.0, 1.0, 1.0, 0.0];
        let energies = vec![0.0, 0.0];
        let q = vec![0.0, -1.0, -1.0, 0.0];
        let first =
            sa_step_single_flip_impl(&states, 2, 2, &energies, &q, 2, 2, 1.0, 1, true).unwrap();
        let second =
            sa_step_single_flip_impl(&states, 2, 2, &energies, &q, 2, 2, 1.0, 1, true).unwrap();
        assert_eq!(first.0, second.0);
        assert_eq!(first.1, second.1);
        assert_eq!(first.2, second.2);
    }
}
