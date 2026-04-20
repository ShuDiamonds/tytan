use crate::anneal::sa_step_single_flip_impl;
use crate::pool::{ResultRow, SolutionPoolCore};
use crate::types::energy_of_state;
use rayon::prelude::*;

#[derive(Clone)]
pub enum ScheduleKind {
    Linear,
    Exponential,
}

#[derive(Clone)]
pub struct StrategySpec {
    pub name: String,
    pub kind: String,
    pub weight: f64,
}

#[derive(Clone)]
pub struct AdaptiveBulkConfig {
    pub shots: usize,
    pub steps: usize,
    pub batch_size: usize,
    pub init_temp: f64,
    pub end_temp: f64,
    pub schedule: ScheduleKind,
    pub adaptive: bool,
    pub epsilon: f64,
    pub include_diverse: bool,
    pub pool_max_entries: usize,
    pub near_dup_hamming: usize,
    pub replace_margin: f64,
    pub stall_steps: usize,
    pub restart_ratio: f64,
    pub restart_min_flips: usize,
    pub restart_burnin_steps: usize,
    pub restart_diversity_threshold: Option<f64>,
    pub novelty_weight: f64,
    pub seed: u64,
}

pub struct AdaptiveLogEntry {
    pub strategy: String,
    pub temperature: f64,
    pub improvements: usize,
}

pub struct AdaptiveBulkStats {
    pub best_energy: f64,
    pub restart_count: usize,
    pub pool_mean_pairwise_distance: f64,
    pub state_diversity: f64,
    pub strategy_weights: Vec<(String, f64)>,
    pub log_entries: Vec<AdaptiveLogEntry>,
}

pub struct AdaptiveBulkOutput {
    pub rows: Vec<ResultRow>,
    pub stats: AdaptiveBulkStats,
}

fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

fn uniform01(rng: &mut u64) -> f64 {
    *rng = xorshift64(*rng);
    ((*rng >> 11) as f64) / ((1u64 << 53) as f64)
}

fn choose_index(weights: &[f64], rng: &mut u64, epsilon: f64) -> usize {
    if weights.is_empty() {
        return 0;
    }
    let explore = uniform01(rng) < epsilon;
    if explore {
        *rng = xorshift64(*rng);
        return (*rng as usize) % weights.len();
    }
    let mut best = 0usize;
    let mut best_weight = f64::MIN;
    for (idx, weight) in weights.iter().enumerate() {
        if *weight > best_weight {
            best_weight = *weight;
            best = idx;
        }
    }
    best
}

fn schedule_temperature(
    step: usize,
    steps: usize,
    init_temp: f64,
    end_temp: f64,
    schedule: &ScheduleKind,
    strategy_kind: &str,
) -> f64 {
    let progress = step as f64 / (steps.saturating_sub(1).max(1) as f64);
    match schedule {
        ScheduleKind::Exponential if strategy_kind == "exponential" => {
            let scale = end_temp / init_temp.max(1e-9);
            (init_temp * scale.powf(progress)).max(1e-8)
        }
        _ => (init_temp + (end_temp - init_temp) * progress).max(1e-8),
    }
}

fn is_symmetric(qmatrix_flat: &[f64], n: usize) -> bool {
    for i in 0..n {
        for j in 0..n {
            if (qmatrix_flat[i * n + j] - qmatrix_flat[j * n + i]).abs() > 1e-12 {
                return false;
            }
        }
    }
    true
}

fn build_default_strategies() -> Vec<StrategySpec> {
    vec![
        StrategySpec {
            name: "linear".to_string(),
            kind: "linear".to_string(),
            weight: 1.0,
        },
        StrategySpec {
            name: "exponential".to_string(),
            kind: "exponential".to_string(),
            weight: 1.0,
        },
    ]
}

fn state_diversity(states_flat: &[f64], shots: usize, dims: usize) -> f64 {
    if shots < 2 {
        return 0.0;
    }
    let (total, count) = (0..shots)
        .into_par_iter()
        .map(|i| {
            let a = &states_flat[i * dims..i * dims + dims];
            let mut local_total = 0.0_f64;
            let mut local_count = 0usize;
            for j in (i + 1)..shots {
                let b = &states_flat[j * dims..j * dims + dims];
                local_total += a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y).abs())
                    .sum::<f64>();
                local_count += 1;
            }
            (local_total, local_count)
        })
        .reduce(|| (0.0_f64, 0usize), |a, b| (a.0 + b.0, a.1 + b.1));
    total / count.max(1) as f64
}

fn init_states(shots: usize, dims: usize, rng: &mut u64) -> Vec<f64> {
    let mut states = vec![0.0_f64; shots * dims];
    for value in &mut states {
        *value = if uniform01(rng) < 0.5 { 0.0 } else { 1.0 };
    }
    states
}

fn init_energies(states_flat: &[f64], shots: usize, dims: usize, qmatrix_flat: &[f64]) -> Vec<f64> {
    (0..shots)
        .into_par_iter()
        .map(|shot| {
            let state = &states_flat[shot * dims..shot * dims + dims];
            energy_of_state(state, qmatrix_flat, dims)
        })
        .collect()
}

fn select_strategy(weights: &[f64], rng: &mut u64, epsilon: f64) -> usize {
    choose_index(weights, rng, epsilon)
}

fn record_strategy_reward(weights: &mut [f64], idx: usize, reward: f64) {
    if let Some(weight) = weights.get_mut(idx) {
        *weight = (*weight + reward).max(1e-3);
    }
}

fn restart_states(
    states_flat: &mut [f64],
    energies: &mut [f64],
    dims: usize,
    best_state: &[f64],
    best_energy: f64,
    restart_ratio: f64,
    restart_min_flips: usize,
    rng: &mut u64,
    qmatrix_flat: &[f64],
) -> (Vec<usize>, f64, Vec<f64>) {
    let shots = energies.len();
    let restart_count = ((shots as f64) * restart_ratio).ceil().max(1.0) as usize;
    let mut worst: Vec<usize> = (0..shots).collect();
    worst.sort_by(|a, b| {
        energies[*a]
            .partial_cmp(&energies[*b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    worst.reverse();
    let worst = worst.into_iter().take(restart_count).collect::<Vec<_>>();
    let flip_count = restart_min_flips.max(1).min(dims);
    let mut best_state_updated = best_state.to_vec();
    let mut best_energy_updated = best_energy;

    for &shot in &worst {
        let mut next = best_state.to_vec();
        let mut selected = Vec::new();
        while selected.len() < flip_count {
            *rng = xorshift64(*rng);
            let idx = (*rng as usize) % dims;
            if !selected.contains(&idx) {
                selected.push(idx);
                next[idx] = 1.0 - next[idx];
            }
        }
        let energy = energy_of_state(&next, qmatrix_flat, dims);
        for idx in 0..dims {
            states_flat[shot * dims + idx] = next[idx];
        }
        energies[shot] = energy;
        if energy < best_energy_updated {
            best_energy_updated = energy;
            best_state_updated = next;
        }
    }
    (worst, best_energy_updated, best_state_updated)
}

pub fn run_adaptive_bulk_sa_impl(
    qmatrix_flat: &[f64],
    dims: usize,
    _index_names: &[String],
    config: &AdaptiveBulkConfig,
    strategies: Option<&[StrategySpec]>,
) -> Result<AdaptiveBulkOutput, &'static str> {
    if qmatrix_flat.len() != dims * dims {
        return Err("QUBO matrix size mismatch");
    }
    if config.shots == 0 || config.steps == 0 {
        return Err("shots and steps must be positive");
    }
    let symmetric = is_symmetric(qmatrix_flat, dims);
    let mut rng = if config.seed == 0 {
        0x9E3779B97F4A7C15
    } else {
        config.seed
    };
    let mut states = init_states(config.shots, dims, &mut rng);
    let mut energies = init_energies(&states, config.shots, dims, qmatrix_flat);
    let best_idx = energies
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let mut best_energy = energies[best_idx];
    let mut best_state = states[best_idx * dims..best_idx * dims + dims].to_vec();
    let mut pool = SolutionPoolCore::new(
        config.batch_size.min(config.shots).max(1),
        if config.include_diverse { 2 } else { 0 },
        config.pool_max_entries,
        config.near_dup_hamming,
        config.replace_margin,
    );

    let strategies = strategies
        .map(|items| items.to_vec())
        .unwrap_or_else(build_default_strategies);
    let mut strategy_weights: Vec<f64> = strategies
        .iter()
        .map(|spec| spec.weight.max(1e-3))
        .collect();
    let mut log_entries = Vec::with_capacity(config.steps);
    let mut restart_count = 0usize;
    let mut last_best_step = 0usize;
    let mut last_restart_step = 0usize.saturating_sub(config.restart_burnin_steps);
    let diversity_threshold = config
        .restart_diversity_threshold
        .unwrap_or_else(|| (dims as f64 * 0.25).max(1.0));

    for step in 0..config.steps {
        let strategy_idx = select_strategy(&strategy_weights, &mut rng, config.epsilon);
        let strategy = &strategies[strategy_idx];
        let temperature = schedule_temperature(
            step,
            config.steps,
            config.init_temp,
            config.end_temp,
            &config.schedule,
            &strategy.kind,
        );
        let beta = 1.0 / temperature;
        let before_states = states.clone();
        let before_energies = energies.clone();
        let (next_states, next_energies, next_rng, stats) = sa_step_single_flip_impl(
            &states,
            config.shots,
            dims,
            &energies,
            qmatrix_flat,
            dims,
            dims,
            beta,
            rng,
            symmetric,
        )?;
        states = next_states;
        energies = next_energies;
        rng = next_rng;

        let mut improvements = 0usize;
        let mut total_reward = 0.0_f64;
        for shot in 0..config.shots {
            let start = shot * dims;
            let prev = &before_states[start..start + dims];
            let next = &states[start..start + dims];
            if prev != next {
                improvements += 1;
                let novelty = pool.min_distance_to_pool(next);
                let delta = energies[shot] - before_energies[shot];
                total_reward += -delta + config.novelty_weight * novelty;
                pool.offer(next, energies[shot]);
                if energies[shot] < best_energy {
                    best_energy = energies[shot];
                    best_state = next.to_vec();
                    last_best_step = step;
                }
            }
        }
        if config.adaptive {
            record_strategy_reward(&mut strategy_weights, strategy_idx, total_reward);
        }
        log_entries.push(AdaptiveLogEntry {
            strategy: strategy.name.clone(),
            temperature,
            improvements: improvements.max(stats.accepted),
        });

        let current_diversity = state_diversity(&states, config.shots, dims);
        if step >= last_best_step
            && step.saturating_sub(last_best_step) >= config.stall_steps
            && step >= last_restart_step
            && step.saturating_sub(last_restart_step) >= config.restart_burnin_steps
            && current_diversity <= diversity_threshold
        {
            let (restart_indices, new_best_energy, new_best_state) = restart_states(
                &mut states,
                &mut energies,
                dims,
                &best_state,
                best_energy,
                config.restart_ratio,
                config.restart_min_flips,
                &mut rng,
                qmatrix_flat,
            );
            for idx in restart_indices {
                let start = idx * dims;
                pool.offer(&states[start..start + dims], energies[idx]);
            }
            restart_count += 1;
            last_restart_step = step;
            best_energy = new_best_energy;
            best_state = new_best_state;
        }
    }

    pool.offer(&best_state, best_energy);
    let rows = pool.results(config.include_diverse);
    let pool_mean_pairwise_distance = pool.mean_pairwise_distance();
    let state_diversity = state_diversity(&states, config.shots, dims);
    let strategy_weights = strategies
        .iter()
        .enumerate()
        .map(|(idx, spec)| (spec.name.clone(), strategy_weights[idx]))
        .collect();

    Ok(AdaptiveBulkOutput {
        rows,
        stats: AdaptiveBulkStats {
            best_energy,
            restart_count,
            pool_mean_pairwise_distance,
            state_diversity,
            strategy_weights,
            log_entries,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::{run_adaptive_bulk_sa_impl, AdaptiveBulkConfig, ScheduleKind};

    #[test]
    fn adaptive_bulk_smoke_test() {
        let q = vec![0.0, -1.0, 0.0, 0.0];
        let index_names = vec!["x".to_string(), "y".to_string()];
        let config = AdaptiveBulkConfig {
            shots: 4,
            steps: 4,
            batch_size: 4,
            init_temp: 5.0,
            end_temp: 0.1,
            schedule: ScheduleKind::Linear,
            adaptive: true,
            epsilon: 0.2,
            include_diverse: true,
            pool_max_entries: 16,
            near_dup_hamming: 1,
            replace_margin: 1e-6,
            stall_steps: 2,
            restart_ratio: 0.5,
            restart_min_flips: 1,
            restart_burnin_steps: 0,
            restart_diversity_threshold: Some(10.0),
            novelty_weight: 0.05,
            seed: 0,
        };
        let output = run_adaptive_bulk_sa_impl(&q, 2, &index_names, &config, None).unwrap();
        assert!(!output.rows.is_empty());
        assert!(output.stats.best_energy.is_finite());
    }
}
