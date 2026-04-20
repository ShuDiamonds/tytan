use std::collections::HashMap;

use rayon::prelude::*;

#[derive(Clone)]
pub struct PoolEntry {
    pub state: Vec<i64>,
    pub energy: f64,
    pub count: usize,
}

#[derive(Clone)]
pub struct ResultRow {
    pub state: Vec<i64>,
    pub energy: f64,
    pub count: usize,
}

pub struct SolutionPoolCore {
    best_k: usize,
    diverse_k: usize,
    max_entries: usize,
    near_dup_hamming: usize,
    replace_margin: f64,
    entries: HashMap<Vec<i64>, PoolEntry>,
    best: Vec<PoolEntry>,
    diverse: Vec<PoolEntry>,
    best_dirty: bool,
    diverse_dirty: bool,
}

impl SolutionPoolCore {
    pub fn new(
        best_k: usize,
        diverse_k: usize,
        max_entries: usize,
        near_dup_hamming: usize,
        replace_margin: f64,
    ) -> Self {
        Self {
            best_k,
            diverse_k,
            max_entries: max_entries.max(1),
            near_dup_hamming,
            replace_margin: replace_margin.max(0.0),
            entries: HashMap::new(),
            best: Vec::new(),
            diverse: Vec::new(),
            best_dirty: true,
            diverse_dirty: true,
        }
    }

    fn key(state: &[f64]) -> Vec<i64> {
        state.iter().map(|v| v.round() as i64).collect()
    }

    fn hamming(a: &[i64], b: &[i64]) -> usize {
        a.iter().zip(b.iter()).filter(|(x, y)| x != y).count()
    }

    fn state_distance(state: &[i64], entry: &PoolEntry) -> usize {
        Self::hamming(state, &entry.state)
    }

    fn nearest_entry(&self, state: &[i64]) -> Option<(Vec<i64>, usize)> {
        self.entries
            .par_iter()
            .map(|(key, entry)| (key.clone(), Self::state_distance(state, entry)))
            .min_by_key(|(_, dist)| *dist)
    }

    fn trim(&mut self) {
        while self.entries.len() > self.max_entries {
            if let Some(worst_key) = self
                .entries
                .iter()
                .max_by(|a, b| {
                    a.1.energy
                        .partial_cmp(&b.1.energy)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.1.count.cmp(&b.1.count))
                })
                .map(|(key, _)| key.clone())
            {
                self.entries.remove(&worst_key);
                self.best_dirty = true;
                self.diverse_dirty = true;
            } else {
                break;
            }
        }
    }

    pub fn offer(&mut self, state: &[f64], energy: f64) {
        let key = Self::key(state);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.count += 1;
            entry.energy = entry.energy.min(energy);
            self.best_dirty = true;
            self.diverse_dirty = true;
            return;
        }

        let state_i64 = key.clone();
        if let Some((nearest_key, nearest_distance)) = self.nearest_entry(&state_i64) {
            if nearest_distance <= self.near_dup_hamming {
                if let Some(nearest) = self.entries.get(&nearest_key) {
                    if energy < nearest.energy - self.replace_margin {
                        self.entries.remove(&nearest_key);
                        self.entries.insert(
                            key,
                            PoolEntry {
                                state: state_i64,
                                energy,
                                count: 1,
                            },
                        );
                        self.best_dirty = true;
                        self.diverse_dirty = true;
                        self.trim();
                    }
                }
                return;
            }
        }

        self.entries.insert(
            key,
            PoolEntry {
                state: state_i64,
                energy,
                count: 1,
            },
        );
        self.best_dirty = true;
        self.diverse_dirty = true;
        self.trim();
    }

    pub fn min_distance_to_pool(&self, state: &[f64]) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let state_i64 = Self::key(state);
        self.entries
            .par_iter()
            .map(|(_, entry)| Self::hamming(&state_i64, &entry.state))
            .min()
            .unwrap_or(0) as f64
    }

    pub fn mean_pairwise_distance(&self) -> f64 {
        let entries: Vec<&PoolEntry> = self.entries.values().collect();
        if entries.len() < 2 {
            return 0.0;
        }
        let (total, count) = (0..entries.len())
            .into_par_iter()
            .map(|i| {
                let mut local_total = 0.0_f64;
                let mut local_count = 0usize;
                for j in (i + 1)..entries.len() {
                    local_total += Self::hamming(&entries[i].state, &entries[j].state) as f64;
                    local_count += 1;
                }
                (local_total, local_count)
            })
            .reduce(|| (0.0_f64, 0usize), |a, b| (a.0 + b.0, a.1 + b.1));
        total / count.max(1) as f64
    }

    fn refresh_best(&mut self) {
        let mut entries: Vec<PoolEntry> = self.entries.values().cloned().collect();
        entries.par_sort_by(|a, b| {
            a.energy
                .partial_cmp(&b.energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.best = entries.into_iter().take(self.best_k).collect();
        self.best_dirty = false;
    }

    fn refresh_diverse(&mut self) {
        self.ensure_best();
        if self.diverse_k == 0 {
            self.diverse.clear();
            self.diverse_dirty = false;
            return;
        }

        let mut selected = self.best.clone();
        let mut selected_keys: Vec<Vec<i64>> = selected.iter().map(|entry| entry.state.clone()).collect();
        let mut candidates: Vec<PoolEntry> = self
            .entries
            .values()
            .filter(|entry| !selected_keys.iter().any(|key| *key == entry.state))
            .cloned()
            .collect();
        let mut diverse = Vec::new();
        while !candidates.is_empty() && diverse.len() < self.diverse_k {
            let (best_idx, _) = candidates
                .par_iter()
                .enumerate()
                .map(|(idx, entry)| {
                    let min_dist = if selected.is_empty() {
                        0
                    } else {
                        selected
                            .iter()
                            .map(|chosen| Self::hamming(&entry.state, &chosen.state))
                            .min()
                            .unwrap_or(0)
                    };
                    (idx, (min_dist, -entry.energy))
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            let chosen = candidates.remove(best_idx);
            selected_keys.push(chosen.state.clone());
            selected.push(chosen.clone());
            diverse.push(chosen);
        }
        self.diverse = diverse;
        self.diverse_dirty = false;
    }

    fn ensure_best(&mut self) {
        if self.best_dirty {
            self.refresh_best();
        }
    }

    fn ensure_diverse(&mut self) {
        if self.diverse_dirty {
            self.refresh_diverse();
        }
    }

    pub fn refresh(&mut self, include_diverse: bool) {
        self.refresh_best();
        if include_diverse {
            self.refresh_diverse();
        } else {
            self.diverse_dirty = true;
        }
    }

    pub fn results(&mut self, include_diverse: bool) -> Vec<ResultRow> {
        self.refresh(include_diverse);
        let mut rows = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for entry in &self.best {
            if seen.insert(entry.state.clone()) {
                rows.push(ResultRow {
                    state: entry.state.clone(),
                    energy: entry.energy,
                    count: entry.count,
                });
            }
        }
        if include_diverse {
            for entry in &self.diverse {
                if seen.insert(entry.state.clone()) {
                    rows.push(ResultRow {
                        state: entry.state.clone(),
                        energy: entry.energy,
                        count: entry.count,
                    });
                }
            }
        }
        if rows.is_empty() && !self.entries.is_empty() {
            if let Some(entry) = self.entries.values().next() {
                rows.push(ResultRow {
                    state: entry.state.clone(),
                    energy: entry.energy,
                    count: entry.count,
                });
            }
        }
        rows
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::SolutionPoolCore;

    #[test]
    fn near_duplicate_replacement_works() {
        let mut pool = SolutionPoolCore::new(2, 1, 8, 1, 1e-6);
        pool.offer(&[0.0, 0.0, 0.0], -1.0);
        pool.offer(&[0.0, 0.0, 1.0], -2.0);
        let rows = pool.results(false);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].energy, -2.0);
    }

    #[test]
    fn max_min_diversity_prefers_farthest() {
        let mut pool = SolutionPoolCore::new(1, 1, 8, 0, 1e-6);
        pool.offer(&[0.0, 0.0, 0.0], -3.0);
        pool.offer(&[0.0, 0.0, 1.0], -2.0);
        pool.offer(&[1.0, 1.0, 1.0], -1.0);
        let rows = pool.results(true);
        assert!(rows.len() >= 2);
    }
}
