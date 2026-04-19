use std::collections::HashMap;

#[derive(Clone)]
pub struct AggregatedEntry {
    pub state: Vec<i64>,
    pub energy: f64,
    pub count: usize,
    pub first_index: usize,
}

pub fn aggregate_results_impl(
    states_flat: &[f64],
    shots: usize,
    dims: usize,
    energies: &[f64],
) -> Result<Vec<AggregatedEntry>, &'static str> {
    if states_flat.len() != shots * dims {
        return Err("States data size mismatch");
    }
    if energies.len() != shots {
        return Err("Energy size mismatch");
    }

    let mut map: HashMap<Vec<i64>, (usize, f64, usize)> = HashMap::new();
    for shot in 0..shots {
        let start = shot * dims;
        let mut key = Vec::with_capacity(dims);
        for v in &states_flat[start..start + dims] {
            key.push(v.round() as i64);
        }
        map.entry(key)
            .and_modify(|entry| entry.2 += 1)
            .or_insert((shot, energies[shot], 1));
    }

    let mut entries = Vec::with_capacity(map.len());
    for (state, (first_index, energy, count)) in map {
        entries.push(AggregatedEntry {
            state,
            energy,
            count,
            first_index,
        });
    }

    entries.sort_by(|a, b| {
        a.energy
            .partial_cmp(&b.energy)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.first_index.cmp(&b.first_index))
    });

    Ok(entries)
}
