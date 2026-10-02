//! Turning label logits into typed answers, as the reference's `primitives.py`.
//!
//! - probabilities: softmax over the allowed labels only, at a calibrated temperature
//! - choice confidence: (N * max_p - 1) / (N - 1), the published System One formula
//! - score: expected level index, sum(i * p_i); may fall between levels
//! - score confidence: 1 - 2 * E|i - score| / (K - 1), LARS's own definition
//! - order sensitivity: mean total variation distance of each ordering from their mean

pub fn softmax(logits: &[f64], temperature: f64) -> Vec<f64> {
    let t = temperature.max(1e-6);
    let z: Vec<f64> = logits.iter().map(|x| x / t).collect();
    let max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|x| (x - max).exp()).collect();
    let total = pairwise_sum(&e);
    e.iter().map(|x| x / total).collect()
}

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-500.0, 500.0)).exp())
}

pub fn choice_confidence(probs: &[f64]) -> f64 {
    let n = probs.len() as f64;
    if probs.len() < 2 {
        return 1.0;
    }
    ((n * max(probs) - 1.0) / (n - 1.0)).clamp(0.0, 1.0)
}

pub fn expected_score(probs: &[f64]) -> f64 {
    pairwise_sum(&probs.iter().enumerate().map(|(i, p)| i as f64 * p).collect::<Vec<_>>())
}

pub fn score_confidence(probs: &[f64]) -> f64 {
    let k = probs.len();
    if k < 2 {
        return 1.0;
    }
    let center = expected_score(probs);
    let spread =
        pairwise_sum(&probs.iter().enumerate().map(|(i, p)| p * (i as f64 - center).abs()).collect::<Vec<_>>());
    (1.0 - 2.0 * spread / (k as f64 - 1.0)).clamp(0.0, 1.0)
}

pub fn total_variation(p: &[f64], q: &[f64]) -> f64 {
    0.5 * pairwise_sum(&p.iter().zip(q).map(|(a, b)| (a - b).abs()).collect::<Vec<_>>())
}

/// The element-wise mean of equally long distributions, summed in order as numpy reduces axis 0.
pub fn mean_rows(rows: &[Vec<f64>]) -> Vec<f64> {
    let mut mean = vec![0.0; rows[0].len()];
    for row in rows {
        for (m, x) in mean.iter_mut().zip(row) {
            *m += x;
        }
    }
    let n = rows.len() as f64;
    mean.iter().map(|m| m / n).collect()
}

pub fn order_sensitivity(distributions: &[Vec<f64>]) -> f64 {
    if distributions.len() < 2 {
        return 0.0;
    }
    let mean = mean_rows(distributions);
    let distances: Vec<f64> = distributions.iter().map(|d| total_variation(d, &mean)).collect();
    pairwise_sum(&distances) / distances.len() as f64
}

pub fn top_margin(probs: &[f64]) -> f64 {
    if probs.len() < 2 {
        return 1.0;
    }
    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    sorted[0] - sorted[1]
}

pub fn max(values: &[f64]) -> f64 {
    values.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

/// Index of the first maximum, as `numpy.argmax`.
pub fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, v) in values.iter().enumerate() {
        if *v > values[best] {
            best = i;
        }
    }
    best
}

/// `np.sum` of a small contiguous float64 array: the reduction starts from the identity, 0.0,
/// and adds numpy's pairwise sum of all elements, so the summation order is numpy's.
pub fn pairwise_sum(values: &[f64]) -> f64 {
    0.0 + pairwise(values)
}

/// numpy's pairwise summation: blocks over 128 are split in half (at a multiple of 8)
/// recursively; up to 128, eight interleaved accumulators; under 8, a plain loop.
fn pairwise(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        let mut total = 0.0f64;
        for v in values {
            total += v;
        }
        return total;
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&values[..8]);
        let mut i = 8;
        while i + 8 <= n {
            for (k, acc) in r.iter_mut().enumerate() {
                *acc += values[i + k];
            }
            i += 8;
        }
        let mut total = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for v in &values[i..] {
            total += v;
        }
        return total;
    }
    let half = (n / 2) - (n / 2) % 8;
    pairwise(&values[..half]) + pairwise(&values[half..])
}
