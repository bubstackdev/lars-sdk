//! Answer labels: options are shown as `A`, `B`, ... and read back as one next-token logit each,
//! so a label is usable only if the tokenizer encodes `" " + label` as one token.

use crate::error::{Error, Result};

/// A..Z then AA..ZZ: 702 candidates, enough for the 255-option maximum with margin.
pub fn label_candidates() -> impl Iterator<Item = String> {
    let letters = || (b'A'..=b'Z').map(char::from);
    letters()
        .map(String::from)
        .chain(letters().flat_map(move |first| letters().map(move |second| format!("{first}{second}"))))
}

/// The first `count` candidates the tokenizer accepts, in canonical order.
pub fn assign_labels(count: usize, is_single_token: impl Fn(&str) -> bool) -> Result<Vec<String>> {
    let labels: Vec<String> = label_candidates().filter(|l| is_single_token(l)).take(count).collect();
    if labels.len() < count {
        return Err(Error::Backend(format!(
            "tokenizer offers only {} single-token labels, need {count}",
            labels.len()
        )));
    }
    Ok(labels)
}
