use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct PresolvePlanCore {
    pub reduced_matrix: Vec<f64>,
    pub reduced_dim: usize,
    pub active_indices: Vec<usize>,
    pub hard_fixed_indices: Vec<usize>,
    pub hard_fixed_values: Vec<i64>,
    pub soft_fixed_indices: Vec<usize>,
    pub fix_confidence: Vec<f64>,
    pub aggregation_src: Vec<usize>,
    pub aggregation_dst: Vec<usize>,
    pub aggregation_relation: Vec<i64>,
    pub aggregation_strength: Vec<f64>,
    pub component_ids: Vec<usize>,
    pub boundary_indices: Vec<usize>,
    pub frontier_indices: Vec<usize>,
    pub branch_candidate_indices: Vec<usize>,
    pub branch_candidate_scores: Vec<f64>,
    pub stats: Vec<(String, f64)>,
}

#[derive(Clone, Debug)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let root = self.find(self.parent[x]);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] = self.rank[ra].saturating_add(1);
        }
    }
}

fn score_from_frequency(freq: Option<&[f64]>, idx: usize, fallback: f64) -> f64 {
    if let Some(values) = freq {
        if idx < values.len() {
            let v = values[idx].clamp(0.0, 1.0);
            return 1.0 - (v - 0.5).abs() * 2.0;
        }
    }
    fallback
}

pub fn presolve_plan_impl(
    qmatrix: &[f64],
    n: usize,
    hard_threshold: f64,
    soft_threshold: f64,
    coupling_threshold: f64,
    aggregation_threshold: f64,
    weak_cut_threshold: f64,
    probing_budget: usize,
    pool_frequency: Option<&[f64]>,
    pair_correlation: Option<&[f64]>,
) -> Result<PresolvePlanCore, &'static str> {
    if qmatrix.len() != n * n {
        return Err("QUBO matrix data size mismatch");
    }

    let scores: Vec<(f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let diag = qmatrix[i * n + i];
            let mut coupling = 0.0_f64;
            for j in 0..n {
                if i != j {
                    coupling += qmatrix[i * n + j].abs();
                }
            }
            (diag, coupling)
        })
        .collect();

    let mut fix_confidence = vec![0.0_f64; n];
    let mut hard_fixed_indices = Vec::new();
    let mut hard_fixed_values = Vec::new();
    let mut soft_fixed_indices = Vec::new();

    for i in 0..n {
        let (diag, coupling) = scores[i];
        let base_score = diag.abs() / (coupling + 1e-9);
        let freq_score = score_from_frequency(pool_frequency, i, 1.0 - base_score.min(1.0));
        let conf = (0.7 * base_score.min(1.0) + 0.3 * freq_score).clamp(0.0, 1.0);
        fix_confidence[i] = conf;
        if base_score >= hard_threshold {
            hard_fixed_indices.push(i);
            hard_fixed_values.push(if diag >= 0.0 { 0 } else { 1 });
        } else if base_score >= soft_threshold {
            soft_fixed_indices.push(i);
        }
    }

    let hard_fixed_set: std::collections::HashSet<usize> =
        hard_fixed_indices.iter().copied().collect();
    let active_mask: Vec<bool> = (0..n).map(|i| !hard_fixed_set.contains(&i)).collect();
    let active_indices: Vec<usize> = (0..n).filter(|&i| active_mask[i]).collect();
    let reduced_dim = active_indices.len();
    let mut reduced_matrix = vec![0.0_f64; reduced_dim * reduced_dim];
    reduced_matrix
        .par_chunks_mut(reduced_dim)
        .enumerate()
        .for_each(|(row_pos, row)| {
            let i = active_indices[row_pos];
            for (col_pos, &j) in active_indices.iter().enumerate() {
                row[col_pos] = qmatrix[i * n + j];
            }
        });

    let mut uf = UnionFind::new(n);
    for i in 0..n {
        for j in (i + 1)..n {
            let weight = qmatrix[i * n + j].abs();
            if weight >= coupling_threshold {
                uf.union(i, j);
            }
        }
    }

    let mut root_to_component = std::collections::HashMap::new();
    let mut component_ids = vec![0usize; n];
    let mut next_component = 0usize;
    for i in 0..n {
        let root = uf.find(i);
        let entry = root_to_component.entry(root).or_insert_with(|| {
            let current = next_component;
            next_component = next_component.saturating_add(1);
            current
        });
        component_ids[i] = *entry;
    }

    let mut boundary_scores = vec![0.0_f64; n];
    let mut aggregation_src = Vec::new();
    let mut aggregation_dst = Vec::new();
    let mut aggregation_relation = Vec::new();
    let mut aggregation_strength = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let weight = qmatrix[i * n + j];
            let abs_weight = weight.abs();
            if abs_weight >= weak_cut_threshold && component_ids[i] != component_ids[j] {
                boundary_scores[i] += abs_weight;
                boundary_scores[j] += abs_weight;
            }
            if abs_weight >= aggregation_threshold {
                let relation = if weight <= 0.0 { 0 } else { 1 };
                aggregation_src.push(i);
                aggregation_dst.push(j);
                aggregation_relation.push(relation);
                aggregation_strength.push(abs_weight);
            }
        }
    }

    if let Some(pair_corr) = pair_correlation {
        if pair_corr.len() == n * n {
            aggregation_src.clear();
            aggregation_dst.clear();
            aggregation_relation.clear();
            aggregation_strength.clear();
            for i in 0..n {
                for j in (i + 1)..n {
                    let corr = pair_corr[i * n + j];
                    let abs_corr = corr.abs();
                    if abs_corr >= aggregation_threshold {
                        aggregation_src.push(i);
                        aggregation_dst.push(j);
                        aggregation_relation.push(if corr >= 0.0 { 0 } else { 1 });
                        aggregation_strength.push(abs_corr);
                    }
                }
            }
        }
    }

    let boundary_indices: Vec<usize> = (0..n).filter(|&i| boundary_scores[i] > 0.0).collect();
    let mut frontier_pairs: Vec<(usize, f64)> = boundary_indices
        .iter()
        .map(|&i| (i, boundary_scores[i]))
        .collect();
    frontier_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let frontier_indices: Vec<usize> = frontier_pairs.into_iter().map(|(i, _)| i).collect();

    let mut branch_scores: Vec<(usize, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let (diag, coupling) = scores[i];
            let uncertainty = 1.0 - fix_confidence[i];
            let boundary_pressure = if coupling > 0.0 {
                boundary_scores[i] / coupling
            } else {
                0.0
            };
            let freq_pressure = score_from_frequency(pool_frequency, i, uncertainty);
            let probing = uncertainty.max(freq_pressure);
            let score =
                0.45 * probing + 0.35 * boundary_pressure + 0.20 * (diag.abs() / (coupling + 1.0));
            (i, score)
        })
        .collect();
    branch_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let branch_limit = probing_budget.min(n).max(1);
    let branch_candidate_indices: Vec<usize> = branch_scores
        .iter()
        .take(branch_limit)
        .map(|(idx, _)| *idx)
        .collect();
    let branch_candidate_scores: Vec<f64> = branch_scores
        .iter()
        .take(branch_limit)
        .map(|(_, score)| *score)
        .collect();

    let stats = vec![
        ("original_size".to_string(), n as f64),
        ("reduced_size".to_string(), reduced_dim as f64),
        (
            "hard_fix_count".to_string(),
            hard_fixed_indices.len() as f64,
        ),
        (
            "soft_fix_count".to_string(),
            soft_fixed_indices.len() as f64,
        ),
        (
            "aggregation_count".to_string(),
            aggregation_src.len() as f64,
        ),
        ("component_count".to_string(), next_component as f64),
        ("boundary_count".to_string(), boundary_indices.len() as f64),
        ("frontier_count".to_string(), frontier_indices.len() as f64),
        (
            "branch_candidate_count".to_string(),
            branch_candidate_indices.len() as f64,
        ),
        ("probing_budget".to_string(), probing_budget as f64),
        ("coupling_threshold".to_string(), coupling_threshold),
        ("aggregation_threshold".to_string(), aggregation_threshold),
        ("weak_cut_threshold".to_string(), weak_cut_threshold),
    ];

    Ok(PresolvePlanCore {
        reduced_matrix,
        reduced_dim,
        active_indices,
        hard_fixed_indices,
        hard_fixed_values,
        soft_fixed_indices,
        fix_confidence,
        aggregation_src,
        aggregation_dst,
        aggregation_relation,
        aggregation_strength,
        component_ids,
        boundary_indices,
        frontier_indices,
        branch_candidate_indices,
        branch_candidate_scores,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::presolve_plan_impl;

    #[test]
    fn hard_fix_and_components_are_reported() {
        let q = vec![
            -4.0, 0.0, 0.0, 0.0, //
            0.0, -0.2, 0.5, 0.0, //
            0.0, 0.5, -0.2, 0.0, //
            0.0, 0.0, 0.0, -0.1,
        ];
        let plan = presolve_plan_impl(&q, 4, 1.0, 0.5, 0.4, 0.4, 0.1, 2, None, None).expect("plan");
        assert!(plan.hard_fixed_indices.contains(&0));
        assert!(plan.active_indices.len() <= 3);
        assert!(!plan.component_ids.is_empty());
        assert_eq!(plan.reduced_dim, plan.active_indices.len());
    }

    #[test]
    fn aggregation_uses_pair_correlation_when_present() {
        let q = vec![
            0.0, 0.1, 0.0, //
            0.1, 0.0, 0.0, //
            0.0, 0.0, 0.0,
        ];
        let pair_corr = vec![
            0.0, 0.9, 0.0, //
            0.9, 0.0, 0.0, //
            0.0, 0.0, 0.0,
        ];
        let plan = presolve_plan_impl(&q, 3, 10.0, 5.0, 1.0, 0.8, 0.05, 4, None, Some(&pair_corr))
            .expect("plan");
        assert_eq!(plan.aggregation_src, vec![0]);
        assert_eq!(plan.aggregation_dst, vec![1]);
    }
}
