//! The backend contract, and the deterministic mock backend for tests and demos.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use regex::Regex;
use sha2::{Digest, Sha256};

use crate::error::Result;
use crate::py;
use crate::render::Template;

/// Label logits per row, and each row's final hidden state.
pub type LogitsAndHidden = (Vec<Vec<f64>>, Vec<Vec<f32>>);

/// A model that can prefill a prefix once and score many suffixes against label tokens.
///
/// Implementations own tokenization, KV-cache reuse and batching; the engine never sees token
/// IDs. `label_logits` returns, for each suffix, the next-token logits restricted to its labels,
/// in label order.
pub trait Backend: Send {
    fn name(&self) -> &str;
    fn model_name(&self) -> &str;
    fn template(&self) -> &Template;
    /// True if `" " + label` encodes to exactly one token (the form read after `Answer:`).
    fn is_single_token(&self, label: &str) -> bool;
    fn count_tokens(&self, text: &str) -> usize;
    fn label_logits(&mut self, prefix: &str, suffixes: &[String], labels: &[Vec<String>]) -> Result<Vec<Vec<f64>>>;
    /// As `label_logits`, plus the model's final hidden state at each row's answer position, for
    /// calibrators with a head. None when the backend cannot read hidden states.
    fn label_logits_hidden(
        &mut self,
        _prefix: &str,
        _suffixes: &[String],
        _labels: &[Vec<String>],
    ) -> Result<Option<LogitsAndHidden>> {
        Ok(None)
    }
}

/// Size of the mock's hidden states.
pub const MOCK_HIDDEN: usize = 32;

static OPTION_LINE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^([A-Z]{1,2})\) (.+)$").unwrap());
static STATE_BLOCK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<STATE [0-9a-f]+>\n(.*)\n</STATE [0-9a-f]+>").unwrap());
static WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[a-z0-9]+").unwrap());

/// Deterministic logits from a hash of (state, option) plus a keyword bonus when an option's key
/// or description appears in the state, so demos behave sensibly without a model. The answers
/// mean nothing beyond that. Identical to the reference's `MockBackend`.
pub struct MockBackend {
    pub keyword_bonus: f64,
    pub noise: f64,
    template: Template,
}

impl Default for MockBackend {
    fn default() -> Self {
        MockBackend { keyword_bonus: 3.0, noise: 2.0, template: Template::Plain }
    }
}

fn hash_unit(parts: &[&str]) -> f64 {
    let digest = Sha256::digest(parts.join("\u{1}").as_bytes());
    let value = u64::from_be_bytes(digest[..8].try_into().unwrap());
    value as f64 / 2f64.powi(64)
}

fn words(text: &str) -> impl Iterator<Item = &str> {
    WORD.find_iter(text).map(|m| m.as_str())
}

impl Backend for MockBackend {
    fn name(&self) -> &str {
        "mock"
    }

    fn model_name(&self) -> &str {
        "lars-mock"
    }

    fn template(&self) -> &Template {
        &self.template
    }

    fn is_single_token(&self, label: &str) -> bool {
        label.chars().count() <= 2
    }

    fn count_tokens(&self, text: &str) -> usize {
        py::split_whitespace(text).count().max(1)
    }

    fn label_logits(&mut self, prefix: &str, suffixes: &[String], labels: &[Vec<String>]) -> Result<Vec<Vec<f64>>> {
        let state_text = STATE_BLOCK.captures(prefix).map_or(prefix, |c| c.get(1).unwrap().as_str()).to_lowercase();
        let state_words: HashSet<&str> = words(&state_text).collect();
        let mut results = Vec::with_capacity(suffixes.len());
        for (suffix, row_labels) in suffixes.iter().zip(labels) {
            let described: HashMap<&str, &str> = OPTION_LINE
                .captures_iter(suffix)
                .map(|c| (c.get(1).unwrap().as_str(), c.get(2).unwrap().as_str()))
                .collect();
            let question = suffix.split('\n').next().unwrap_or("").to_lowercase();
            let question_words: HashSet<&str> = words(&question).filter(|w| w.len() > 3).collect();
            let logits = row_labels
                .iter()
                .map(|label| {
                    let text = described.get(label.as_str()).copied().unwrap_or(label).to_lowercase();
                    let option_words: HashSet<&str> =
                        words(&text).filter(|w| w.len() > 2 && *w != "yes" && *w != "no").collect();
                    let mut value = (hash_unit(&[&state_text, &text]) - 0.5) * 2.0 * self.noise;
                    if !option_words.is_disjoint(&state_words) {
                        value += self.keyword_bonus;
                    }
                    if text.starts_with("yes") && !question_words.is_disjoint(&state_words) {
                        value += self.keyword_bonus;
                    }
                    value
                })
                .collect();
            results.push(logits);
        }
        Ok(results)
    }

    /// Hidden states are the state's words hashed into `MOCK_HIDDEN` counts, so a label that
    /// depends on a word the logits ignore can be read back by a head.
    fn label_logits_hidden(
        &mut self,
        prefix: &str,
        suffixes: &[String],
        labels: &[Vec<String>],
    ) -> Result<Option<LogitsAndHidden>> {
        let logits = self.label_logits(prefix, suffixes, labels)?;
        let state_text = STATE_BLOCK.captures(prefix).map_or(prefix, |c| c.get(1).unwrap().as_str()).to_lowercase();
        let mut hidden = vec![0f32; MOCK_HIDDEN];
        for word in words(&state_text) {
            hidden[(hash_unit(&[word]) * MOCK_HIDDEN as f64) as usize] += 1.0;
        }
        Ok(Some((logits, vec![hidden; suffixes.len()])))
    }
}
