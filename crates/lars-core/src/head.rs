//! Per-decision heads, fitted by `calibrate` on a user's own labels.
//!
//! A head re-maps one predicate's calibrated answer:
//!
//! ```text
//! z_k = scale · ln p_k + bias_k + h · w_k        answer = softmax(z)
//! ```
//!
//! where `p` is the calibrator's distribution over the options and `h` the model's final hidden
//! state at the answer position. Without `w` (a recalibration) it needs only `p`: a Platt- or
//! temperature-style rescaling plus a bias per option, which corrects a predicate whose options
//! the pack over- or under-predicts. With `w` it is a ridge-regularized linear readout of the
//! hidden state. On typed-decisions (RESULTS.md, 2026-10-05) recalibration adds 6 to 12 points
//! from 25 labels, and a head beats it from about 50 labels: the 0.6B pack goes from 0.44 to
//! 0.72 accuracy with 300.
//!
//! Fitting mirrors sweep/heads.py. `h` is centered and projected onto the principal directions
//! of the labeled rows (a ridge solution lies in their span, so nothing is lost). `w` is shrunk
//! by lambda, `scale` and `bias` by a unit prior towards 1 and 0, and the multinomial log loss
//! against the labels (soft or hard) is minimized with L-BFGS. `choose_head` picks among
//! recalibration and heads by cross-validation, stepping to a richer model only when it beats
//! the current one significantly, the rule `calibrate` already uses against the pack.

use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::primitives::softmax;

/// Shrinkage strengths tried for `w`, strongest first.
pub const LAMBDAS: [f64; 5] = [30.0, 3.0, 0.3, 0.03, 0.003];
/// Below this many labels only recalibration is tried: heads lose to it at 25 (RESULTS.md).
pub const MIN_HEAD_LABELS: usize = 50;
/// Principal directions kept, and labeled rows they are computed from.
const RANK: usize = 128;
const BASIS_ROWS: usize = 256;
const FLOOR: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq)]
pub struct Head {
    /// Option keys in the order of `bias` and of the columns of `weights`: yes/no for noul,
    /// level indices for score, criteria keys for choice.
    pub keys: Vec<String>,
    pub scale: f64,
    pub bias: Vec<f64>,
    /// Hidden-state size; 0 when the head reads no hidden state.
    pub dim: usize,
    /// `dim` rows of `keys.len()` weights.
    pub weights: Vec<f32>,
    /// The shrinkage it was fitted with; None for a recalibration.
    pub lambda: Option<f64>,
}

impl Head {
    pub fn uses_hidden(&self) -> bool {
        self.dim > 0
    }

    /// The head's distribution, from the calibrator's `probs` (in `keys` order) and, when the head
    /// reads one, the hidden state.
    pub fn apply(&self, probs: &[f64], hidden: Option<&[f32]>) -> Result<Vec<f64>> {
        let k = self.keys.len();
        if probs.len() != k {
            return Err(Error::Invalid(format!("head expects {k} options, got {}", probs.len())));
        }
        let mut z: Vec<f64> = probs.iter().zip(&self.bias).map(|(p, b)| self.scale * p.max(FLOOR).ln() + b).collect();
        if self.uses_hidden() {
            let h = hidden.ok_or_else(|| Error::Invalid("this calibrator's head needs the hidden state".into()))?;
            if h.len() != self.dim {
                return Err(Error::Invalid(format!(
                    "head was fitted on hidden states of size {}, the model gives {}",
                    self.dim,
                    h.len()
                )));
            }
            for (j, &hj) in h.iter().enumerate() {
                let row = &self.weights[j * k..(j + 1) * k];
                for (zk, w) in z.iter_mut().zip(row) {
                    *zk += hj as f64 * *w as f64;
                }
            }
        }
        Ok(softmax(&z, 1.0))
    }

    pub fn to_json(&self) -> Value {
        let bytes: Vec<u8> = self.weights.iter().flat_map(|w| w.to_le_bytes()).collect();
        json!({"keys": self.keys, "scale": self.scale, "bias": self.bias, "dim": self.dim,
               "weights": base64_encode(&bytes), "lambda": self.lambda})
    }

    pub fn from_json(value: &Value) -> Result<Head> {
        let invalid = |what: &str| Error::Invalid(format!("calibrator head: {what}"));
        let keys: Vec<String> = value["keys"]
            .as_array()
            .and_then(|a| a.iter().map(|k| k.as_str().map(String::from)).collect())
            .ok_or_else(|| invalid("keys"))?;
        let bias: Vec<f64> = value["bias"]
            .as_array()
            .and_then(|a| a.iter().map(Value::as_f64).collect())
            .ok_or_else(|| invalid("bias"))?;
        let dim = value["dim"].as_u64().ok_or_else(|| invalid("dim"))? as usize;
        let bytes = base64_decode(value["weights"].as_str().unwrap_or("")).ok_or_else(|| invalid("weights"))?;
        let weights: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        if keys.len() < 2 || bias.len() != keys.len() || weights.len() != dim * keys.len() || bytes.len() % 4 != 0 {
            return Err(invalid("keys, bias and weights do not agree in size"));
        }
        Ok(Head {
            keys,
            scale: value["scale"].as_f64().ok_or_else(|| invalid("scale"))?,
            bias,
            dim,
            weights,
            lambda: value["lambda"].as_f64(),
        })
    }
}

// --------------------------------------------------------------------------- choosing

/// What `choose_head` picked.
#[derive(Debug, Clone)]
pub struct HeadChoice {
    /// None: keep the calibrator as it is.
    pub head: Option<Head>,
    /// Out-of-fold distributions of the chosen model for every example (the calibrator's own
    /// when no head is chosen), for honest accuracy and certification.
    pub out_of_fold: Vec<Vec<f64>>,
    /// Cross-validated mean log loss per candidate: "calibrator", "recal", "head:<lambda>".
    pub cv_log_loss: Vec<(String, f64)>,
}

/// Choose between the calibrator's own answers (`probs`), a recalibration, and heads on
/// `hidden` (from `MIN_HEAD_LABELS` labels), by cross-validated log loss against `targets`.
///
/// Folds are `i % folds`, as in `calibrate`. Candidates are tried from simplest to richest, and
/// each replaces the current choice only if the 95% upper bound on their paired per-example
/// log-loss difference is below zero.
pub fn choose_head(
    probs: &[Vec<f64>],
    hidden: Option<&[Vec<f32>]>,
    targets: &[Vec<f64>],
    keys: &[String],
    folds: usize,
) -> HeadChoice {
    let n = probs.len();
    let base: Vec<Vec<f64>> = probs.iter().map(|p| p.iter().map(|v| v.max(FLOOR).ln()).collect()).collect();
    let loss = |q: &[f64], t: &[f64]| -> f64 { -t.iter().zip(q).map(|(t, q)| t * q.max(FLOOR).ln()).sum::<f64>() };
    let mut candidates: Vec<Option<f64>> = vec![None];
    if hidden.is_some() && n >= MIN_HEAD_LABELS {
        candidates.extend(LAMBDAS.iter().map(|&l| Some(l)));
    }
    let base_losses: Vec<f64> = probs.iter().zip(targets).map(|(p, t)| loss(p, t)).collect();
    let mut cv_log_loss = vec![("calibrator".to_string(), mean(&base_losses))];
    let folds = folds.min(n);
    if folds < 2 {
        return HeadChoice { head: None, out_of_fold: probs.to_vec(), cv_log_loss };
    }
    // held[c][i]: candidate c's out-of-fold distribution for example i.
    let mut held: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); n]; candidates.len()];
    for fold in 0..folds {
        let train: Vec<usize> = (0..n).filter(|i| i % folds != fold).collect();
        let test: Vec<usize> = (0..n).filter(|i| i % folds == fold).collect();
        let pick = |rows: &[usize], m: &[Vec<f64>]| -> Vec<Vec<f64>> { rows.iter().map(|&i| m[i].clone()).collect() };
        let projection = hidden.map(|h| Projection::new(&train.iter().map(|&i| h[i].as_slice()).collect::<Vec<_>>()));
        let features = |rows: &[usize]| -> Vec<Vec<f64>> {
            match (&projection, hidden) {
                (Some(p), Some(h)) => rows.iter().map(|&i| p.features(&h[i])).collect(),
                _ => vec![Vec::new(); rows.len()],
            }
        };
        let (train_f, test_f) = (features(&train), features(&test));
        for (c, lambda) in candidates.iter().enumerate() {
            let empty = vec![Vec::new(); train.len()];
            let f = if lambda.is_some() { &train_f } else { &empty };
            let theta = fit_theta(&pick(&train, &base), f, &pick(&train, targets), *lambda);
            for (j, &i) in test.iter().enumerate() {
                let f = if lambda.is_some() { test_f[j].as_slice() } else { &[] };
                held[c][i] = predict_theta(&theta, &base[i], f);
            }
        }
    }
    let losses: Vec<Vec<f64>> =
        held.iter().map(|rows| rows.iter().zip(targets).map(|(q, t)| loss(q, t)).collect()).collect();
    for (lambda, l) in candidates.iter().zip(&losses) {
        let name = lambda.map_or("recal".to_string(), |l| format!("head:{l}"));
        cv_log_loss.push((name, mean(l)));
    }
    let mut current: Option<usize> = None;
    for c in 0..candidates.len() {
        let reference = current.map_or(&base_losses, |k| &losses[k]);
        let diff: Vec<f64> = losses[c].iter().zip(reference).map(|(a, b)| a - b).collect();
        let (m, var) = (mean(&diff), variance(&diff));
        if m + 1.96 * (var / n as f64).sqrt() < 0.0 {
            current = Some(c);
        }
    }
    let Some(chosen) = current else {
        return HeadChoice { head: None, out_of_fold: probs.to_vec(), cv_log_loss };
    };
    let all: Vec<&[f32]> = hidden.map(|h| h.iter().map(Vec::as_slice).collect()).unwrap_or_default();
    let head = fit_head(
        &base,
        if candidates[chosen].is_some() { Some(&all) } else { None },
        targets,
        keys,
        candidates[chosen],
    );
    HeadChoice { head: Some(head), out_of_fold: held.swap_remove(chosen), cv_log_loss }
}

/// Fit a head on all rows: a recalibration when `lambda` is None, else a head on `hidden`.
/// `base` holds the calibrator's log-probabilities.
pub fn fit_head(
    base: &[Vec<f64>],
    hidden: Option<&[&[f32]]>,
    targets: &[Vec<f64>],
    keys: &[String],
    lambda: Option<f64>,
) -> Head {
    let k = keys.len();
    let (features, projection) = match (lambda, hidden) {
        (Some(_), Some(h)) => {
            let p = Projection::new(h);
            (h.iter().map(|x| p.features(x)).collect(), Some(p))
        }
        _ => (vec![Vec::new(); base.len()], None),
    };
    let theta = fit_theta(base, &features, targets, lambda);
    let Some(projection) = projection else {
        return Head { keys: keys.to_vec(), scale: theta.scale, bias: theta.bias, dim: 0, weights: Vec::new(), lambda };
    };
    // Back to the hidden state's own coordinates: w_full = Σ_s x_s (coef · w)_s, with the
    // centering folded into the bias.
    let d = projection.mean.len();
    let mixed: Vec<f64> = (0..projection.rows.len() * k)
        .map(|i| {
            let (s, kk) = (i / k, i % k);
            (0..projection.rank).map(|j| projection.coef[s * projection.rank + j] * theta.w[j * k + kk]).sum()
        })
        .collect();
    let mut weights = vec![0f32; d * k];
    for (s, row) in projection.rows.iter().enumerate() {
        for (i, x) in row.iter().enumerate() {
            for kk in 0..k {
                weights[i * k + kk] += (x * mixed[s * k + kk]) as f32;
            }
        }
    }
    let mut bias = theta.bias;
    for (i, m) in projection.mean.iter().enumerate() {
        for kk in 0..k {
            bias[kk] -= m * weights[i * k + kk] as f64;
        }
    }
    Head { keys: keys.to_vec(), scale: theta.scale, bias, dim: d, weights, lambda }
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn variance(v: &[f64]) -> f64 {
    let m = mean(v);
    v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0).max(1.0)
}

// --------------------------------------------------------------------------- projection

/// Centering and the top principal directions of a set of hidden states, as coefficients over
/// (at most `BASIS_ROWS` of) the centered rows themselves: feature_j(x) = Σ_s (x - mean)·x_s coef[s][j].
struct Projection {
    mean: Vec<f64>,
    /// Centered basis rows.
    rows: Vec<Vec<f64>>,
    /// rows.len() × rank, already divided by the features' root-mean-square norm.
    coef: Vec<f64>,
    rank: usize,
}

impl Projection {
    fn new(hidden: &[&[f32]]) -> Projection {
        let d = hidden.first().map_or(0, |h| h.len());
        let n = hidden.len().max(1);
        let mut mean = vec![0.0; d];
        for h in hidden {
            for (m, v) in mean.iter_mut().zip(h.iter()) {
                *m += *v as f64;
            }
        }
        mean.iter_mut().for_each(|m| *m /= n as f64);
        let rows: Vec<Vec<f64>> =
            hidden.iter().take(BASIS_ROWS).map(|h| h.iter().zip(&mean).map(|(v, m)| *v as f64 - m).collect()).collect();
        let m = rows.len();
        let mut gram = vec![0.0; m * m];
        for a in 0..m {
            for b in a..m {
                let g = dot(&rows[a], &rows[b]);
                gram[a * m + b] = g;
                gram[b * m + a] = g;
            }
        }
        let (values, vectors) = symmetric_eigen(gram, m);
        let top = values.first().copied().unwrap_or(0.0);
        let rank = values.iter().take(RANK).filter(|&&v| top > 0.0 && v > 1e-12 * top).count();
        // Unit principal directions: v_j = Σ_s x_s u_sj / sqrt(lambda_j).
        let mut coef = vec![0.0; m * rank];
        for s in 0..m {
            for j in 0..rank {
                coef[s * rank + j] = vectors[s * m + j] / values[j].sqrt();
            }
        }
        let mut projection = Projection { mean, rows, coef, rank };
        // Scale features to unit root-mean-square norm over the rows, so lambdas are comparable.
        let norm2: f64 = hidden.iter().map(|h| projection.features(h).iter().map(|f| f * f).sum::<f64>()).sum();
        let scale = (norm2 / n as f64).sqrt();
        if scale > 0.0 {
            projection.coef.iter_mut().for_each(|c| *c /= scale);
        }
        projection
    }

    fn features(&self, h: &[f32]) -> Vec<f64> {
        let x: Vec<f64> = h.iter().zip(&self.mean).map(|(v, m)| *v as f64 - m).collect();
        let dots: Vec<f64> = self.rows.iter().map(|r| dot(r, &x)).collect();
        (0..self.rank).map(|j| dots.iter().enumerate().map(|(s, d)| d * self.coef[s * self.rank + j]).sum()).collect()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Eigenvalues (descending) and eigenvectors (columns, row-major n × n) of a symmetric matrix,
/// by cyclic Jacobi rotations.
fn symmetric_eigen(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    (0..n).for_each(|i| v[i * n + i] = 1.0);
    let total: f64 = a.iter().map(|x| x * x).sum();
    for _ in 0..100 {
        let off: f64 = (0..n).flat_map(|p| (p + 1..n).map(move |q| (p, q))).map(|(p, q)| a[p * n + q].powi(2)).sum();
        if off <= 1e-24 * total || off == 0.0 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[j * n + j].total_cmp(&a[i * n + i]));
    let values = order.iter().map(|&i| a[i * n + i]).collect();
    let mut vectors = vec![0.0; n * n];
    for (col, &i) in order.iter().enumerate() {
        for r in 0..n {
            vectors[r * n + col] = v[r * n + i];
        }
    }
    (values, vectors)
}

// --------------------------------------------------------------------------- fitting

/// scale, w (features × keys, row-major) and bias, in feature coordinates.
struct Theta {
    scale: f64,
    w: Vec<f64>,
    bias: Vec<f64>,
}

fn predict_theta(theta: &Theta, base: &[f64], f: &[f64]) -> Vec<f64> {
    let k = base.len();
    let z: Vec<f64> = (0..k)
        .map(|kk| {
            theta.scale * base[kk]
                + theta.bias[kk]
                + f.iter().enumerate().map(|(j, x)| x * theta.w[j * k + kk]).sum::<f64>()
        })
        .collect();
    softmax(&z, 1.0)
}

fn fit_theta(base: &[Vec<f64>], features: &[Vec<f64>], targets: &[Vec<f64>], lambda: Option<f64>) -> Theta {
    let n = base.len();
    let k = base.first().map_or(0, Vec::len);
    let r = if lambda.is_some() { features.first().map_or(0, Vec::len) } else { 0 };
    let lambda = lambda.unwrap_or(0.0);
    let inv_n = 1.0 / n.max(1) as f64;
    let objective = |x: &[f64], grad: &mut [f64]| -> f64 {
        let (scale, w, bias) = (x[0], &x[1..1 + r * k], &x[1 + r * k..]);
        grad.iter_mut().for_each(|g| *g = 0.0);
        let mut total = 0.0;
        for i in 0..n {
            let z: Vec<f64> = (0..k)
                .map(|kk| scale * base[i][kk] + bias[kk] + (0..r).map(|j| features[i][j] * w[j * k + kk]).sum::<f64>())
                .collect();
            let p = softmax(&z, 1.0);
            for kk in 0..k {
                total -= targets[i][kk] * p[kk].max(1e-300).ln();
                let g = (p[kk] - targets[i][kk]) * inv_n;
                grad[0] += g * base[i][kk];
                for j in 0..r {
                    grad[1 + j * k + kk] += g * features[i][j];
                }
                grad[1 + r * k + kk] += g;
            }
        }
        let mut value = total * inv_n;
        // Ridge on w; a unit prior on scale (towards 1) and the biases (towards 0).
        value += 0.5 * inv_n * ((scale - 1.0).powi(2) + bias.iter().map(|b| b * b).sum::<f64>());
        grad[0] += inv_n * (scale - 1.0);
        for kk in 0..k {
            grad[1 + r * k + kk] += inv_n * bias[kk];
        }
        value += 0.5 * lambda * inv_n * w.iter().map(|v| v * v).sum::<f64>();
        for (g, v) in grad[1..1 + r * k].iter_mut().zip(w) {
            *g += lambda * inv_n * v;
        }
        value
    };
    let mut x0 = vec![0.0; 1 + r * k + k];
    x0[0] = 1.0;
    let x = lbfgs(objective, x0, 500);
    Theta { scale: x[0], w: x[1..1 + r * k].to_vec(), bias: x[1 + r * k..].to_vec() }
}

/// Minimize a smooth function with L-BFGS (10 pairs) and a backtracking Armijo line search.
fn lbfgs(f: impl Fn(&[f64], &mut [f64]) -> f64, mut x: Vec<f64>, max_iter: usize) -> Vec<f64> {
    const MEMORY: usize = 10;
    let dim = x.len();
    let mut g = vec![0.0; dim];
    let mut fx = f(&x, &mut g);
    let mut pairs: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::new();
    for iter in 0..max_iter {
        if g.iter().fold(0.0f64, |m, v| m.max(v.abs())) < 1e-7 {
            break;
        }
        // Two-loop recursion for the search direction -H g.
        let mut q = g.clone();
        let mut alphas = Vec::with_capacity(pairs.len());
        for (s, y, rho) in pairs.iter().rev() {
            let a = rho * dot(s, &q);
            q.iter_mut().zip(y).for_each(|(qi, yi)| *qi -= a * yi);
            alphas.push(a);
        }
        if let Some((s, y, _)) = pairs.last() {
            let gamma = dot(s, y) / dot(y, y);
            q.iter_mut().for_each(|qi| *qi *= gamma);
        } else {
            let norm = dot(&g, &g).sqrt();
            q.iter_mut().for_each(|qi| *qi /= norm.max(1.0));
        }
        for ((s, y, rho), a) in pairs.iter().zip(alphas.iter().rev()) {
            let b = rho * dot(y, &q);
            q.iter_mut().zip(s).for_each(|(qi, si)| *qi += (a - b) * si);
        }
        let direction: Vec<f64> = q.iter().map(|v| -v).collect();
        let mut slope = dot(&g, &direction);
        let direction = if slope >= 0.0 {
            // Not a descent direction: restart from steepest descent.
            pairs.clear();
            slope = -dot(&g, &g);
            g.iter().map(|v| -v).collect()
        } else {
            direction
        };
        let mut step = 1.0;
        let mut next_g = vec![0.0; dim];
        let mut accepted = None;
        for _ in 0..40 {
            let candidate: Vec<f64> = x.iter().zip(&direction).map(|(xi, di)| xi + step * di).collect();
            let value = f(&candidate, &mut next_g);
            if value.is_finite() && value <= fx + 1e-4 * step * slope {
                accepted = Some((candidate, value));
                break;
            }
            step *= 0.5;
        }
        let Some((next_x, next_fx)) = accepted else { break };
        let s: Vec<f64> = next_x.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = next_g.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 {
            if pairs.len() == MEMORY {
                pairs.remove(0);
            }
            pairs.push((s, y, 1.0 / sy));
        }
        let improvement = fx - next_fx;
        x = next_x;
        fx = next_fx;
        g = next_g;
        if iter > 0 && improvement.abs() <= 1e-12 * fx.abs().max(1.0) {
            break;
        }
    }
    x
}

// --------------------------------------------------------------------------- base64

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n =
            (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn base64_decode(text: &str) -> Option<Vec<u8>> {
    let text = text.trim_end_matches('=');
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0);
    for c in text.bytes() {
        let v = ALPHABET.iter().position(|&a| a == c)? as u32;
        acc = acc << 6 | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits & 0xff) as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    /// A deterministic stream in [0, 1).
    fn unit(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn base64_round_trips() {
        for len in 0..20 {
            let bytes: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
            assert_eq!(base64_decode(&base64_encode(&bytes)).unwrap(), bytes);
        }
        assert_eq!(base64_encode(b"Man"), "TWFu");
        assert_eq!(base64_encode(b"Ma"), "TWE=");
        assert!(base64_decode("TW!u").is_none());
    }

    #[test]
    fn jacobi_reconstructs_a_symmetric_matrix() {
        let n = 6;
        let mut seed = 7;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in i..n {
                let v = unit(&mut seed) - 0.5;
                a[i * n + j] = v;
                a[j * n + i] = v;
            }
        }
        let (values, vectors) = symmetric_eigen(a.clone(), n);
        assert!(values.windows(2).all(|w| w[0] >= w[1]));
        for i in 0..n {
            for j in 0..n {
                let r: f64 = (0..n).map(|k| vectors[i * n + k] * values[k] * vectors[j * n + k]).sum();
                assert!((r - a[i * n + j]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn apply_and_json_round_trip() {
        let head = Head {
            keys: keys(&["a", "b", "c"]),
            scale: 0.8,
            bias: vec![0.1, -0.2, 0.0],
            dim: 2,
            weights: vec![1.0, 0.0, -1.0, 0.5, 0.5, 0.0],
            lambda: Some(0.3),
        };
        let p = head.apply(&[0.5, 0.3, 0.2], Some(&[1.0, -2.0])).unwrap();
        let z = [0.8 * 0.5f64.ln() + 0.1 + 1.0 - 1.0, 0.8 * 0.3f64.ln() - 0.2 - 1.0, 0.8 * 0.2f64.ln() - 1.0];
        let expected = softmax(&z, 1.0);
        assert!(p.iter().zip(&expected).all(|(a, b)| (a - b).abs() < 1e-12));
        assert_eq!(Head::from_json(&head.to_json()).unwrap(), head);
        assert!(head.apply(&[0.5, 0.5], Some(&[1.0, 1.0])).is_err());
        assert!(head.apply(&[0.5, 0.3, 0.2], None).is_err());
        assert!(head.apply(&[0.5, 0.3, 0.2], Some(&[1.0])).is_err());
    }

    /// Labels that follow one hidden direction the calibrator does not see.
    type Planted = (Vec<Vec<f64>>, Vec<Vec<f32>>, Vec<Vec<f64>>);

    fn planted(n: usize, signal: f64, seed: &mut u64) -> Planted {
        let (mut probs, mut hidden, mut targets) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..n {
            let y = (unit(seed) * 3.0) as usize;
            let mut h: Vec<f32> = (0..24).map(|_| (unit(seed) - 0.5) as f32 * 2.0).collect();
            h[5] += (signal * (y as f64 - 1.0)) as f32;
            hidden.push(h);
            let mut p = [unit(seed) + 0.2, unit(seed) + 0.2, unit(seed) + 0.2];
            p[y] += 0.3;
            let total: f64 = p.iter().sum();
            probs.push(p.iter().map(|v| v / total).collect());
            targets.push((0..3).map(|k| if k == y { 1.0 } else { 0.0 }).collect());
        }
        (probs, hidden, targets)
    }

    #[test]
    fn a_head_is_chosen_when_the_hidden_state_carries_the_label() {
        let mut seed = 3;
        let (probs, hidden, targets) = planted(150, 2.0, &mut seed);
        let choice = choose_head(&probs, Some(&hidden), &targets, &keys(&["x", "y", "z"]), 5);
        let head = choice.head.expect("a head");
        assert!(head.uses_hidden(), "{:?}", choice.cv_log_loss);
        let (probs, hidden, targets) = planted(300, 2.0, &mut seed);
        let hits = probs
            .iter()
            .zip(&hidden)
            .zip(&targets)
            .filter(|((p, h), t)| {
                let q = head.apply(p, Some(h)).unwrap();
                crate::primitives::argmax(&q) == crate::primitives::argmax(t)
            })
            .count();
        assert!(hits as f64 / 300.0 > 0.75, "{hits}/300");
    }

    #[test]
    fn nothing_is_chosen_when_there_is_nothing_to_learn() {
        let mut seed = 11;
        let (probs, hidden, _) = planted(120, 0.0, &mut seed);
        // Targets that are the calibrator's own distribution: it is already right.
        let choice = choose_head(&probs, Some(&hidden), &probs, &keys(&["x", "y", "z"]), 5);
        assert!(choice.head.is_none(), "{:?}", choice.cv_log_loss);
        assert_eq!(choice.out_of_fold, probs);
    }

    #[test]
    fn few_labels_only_recalibrate() {
        let mut seed = 5;
        let (_, hidden, targets) = planted(40, 3.0, &mut seed);
        // A calibrator that is confidently wrong: always the option after the label.
        let shifted: Vec<Vec<f64>> =
            targets.iter().map(|t| vec![0.1 + 0.8 * t[2], 0.1 + 0.8 * t[0], 0.1 + 0.8 * t[1]]).collect();
        let choice = choose_head(&shifted, Some(&hidden), &targets, &keys(&["x", "y", "z"]), 5);
        let head = choice.head.expect("a recalibration");
        assert!(!head.uses_hidden());
        assert_eq!(choice.cv_log_loss.len(), 2, "no heads tried under MIN_HEAD_LABELS");
    }
}
