//! Python's text and number conventions, where the reference engine depends on them.
//!
//! Calibrators are fitted on exact prompt bytes, so the core renders numbers, whitespace and
//! JSON the way CPython does: `repr(float)`, `round(x, n)`, `str.isspace`, `str.strip`,
//! `str.split()` and `json.dumps`.

use serde_json::Value;

/// `str.isspace()` for one character: Rust's `char::is_whitespace` plus U+001C..U+001F, which
/// CPython counts as whitespace (bidirectional class B or S) and Rust does not.
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `str.strip()` with no arguments.
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `str.split()` with no arguments: runs of whitespace separate, empty parts are dropped.
pub fn split_whitespace(s: &str) -> impl Iterator<Item = &str> {
    s.split(is_space).filter(|part| !part.is_empty())
}

/// `repr(x)` for a float: the shortest round-tripping digits, fixed notation when the decimal
/// exponent is in [-4, 16), scientific otherwise (`1e-05`, `1e+16`).
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.into();
    }
    // Rust's `{:e}` is the shortest round-tripping representation, as CPython's repr is.
    let formatted = format!("{x:e}");
    let (mantissa, exponent) = formatted.split_once('e').expect("LowerExp has an exponent");
    let exponent: i32 = exponent.parse().expect("integer exponent");
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let point = exponent + 1; // value = 0.<digits> * 10^point
    let mut out = String::with_capacity(digits.len() + 8);
    if negative {
        out.push('-');
    }
    if -4 < point && point <= 16 {
        if point <= 0 {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-point) as usize));
            out.push_str(&digits);
        } else if point as usize >= digits.len() {
            out.push_str(&digits);
            out.extend(std::iter::repeat_n('0', point as usize - digits.len()));
            out.push_str(".0");
        } else {
            out.push_str(&digits[..point as usize]);
            out.push('.');
            out.push_str(&digits[point as usize..]);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exponent.abs()));
    }
    out
}

/// `round(x, digits)` for a float: correctly rounded, ties to even on the exact binary value.
pub fn round(x: f64, digits: u32) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    // An exact tie at `digits` places is an odd multiple of 2^-(digits + 1); for those, pick the
    // even neighbour. Everything else is not a tie, and exact decimal formatting rounds it right.
    let scaled = x * f64::powi(2.0, digits as i32 + 1);
    if scaled.fract() == 0.0 && scaled.abs() < 2f64.powi(52) && (scaled as i64) % 2 != 0 {
        let shifted = x * 10f64.powi(digits as i32); // exact: a half-integer well inside 2^52
        let floor = shifted.floor();
        let even = if (floor as i64) % 2 == 0 { floor } else { floor + 1.0 };
        return even / 10f64.powi(digits as i32);
    }
    format!("{:.*}", digits as usize, x).parse().expect("formatted float parses")
}

/// `str(value)` for a decoded JSON value, as `str(raw["label"])` sees it.
pub fn str_of(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::String(s) => s.clone(),
        Value::Number(n) => number_text(n),
        other => dumps(other, None, false),
    }
}

/// A JSON number as Python prints it after `json.loads`: integers keep their digits, anything
/// with a fraction or exponent becomes a float and prints as `repr(float)`.
fn number_text(n: &serde_json::Number) -> String {
    let raw = n.to_string();
    if raw.contains(['.', 'e', 'E']) {
        float_repr(raw.parse::<f64>().unwrap_or(f64::NAN))
    } else {
        let trimmed = raw.trim_start_matches('-');
        let negative = raw.starts_with('-') && !trimmed.trim_start_matches('0').is_empty();
        let digits = trimmed.trim_start_matches('0');
        let digits = if digits.is_empty() { "0" } else { digits };
        if negative { format!("-{digits}") } else { digits.to_string() }
    }
}

/// `json.dumps(value, ensure_ascii=False, indent=indent, sort_keys=sort_keys)`.
pub fn dumps(value: &Value, indent: Option<usize>, sort_keys: bool) -> String {
    let mut out = String::new();
    write_value(&mut out, value, indent, sort_keys, 0);
    out
}

fn write_value(out: &mut String, value: &Value, indent: Option<usize>, sort_keys: bool, level: usize) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&number_text(n)),
        Value::String(s) => write_string(out, s),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                separator(out, i, indent, level + 1);
                write_value(out, item, indent, sort_keys, level + 1);
            }
            close(out, indent, level);
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            if sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            out.push('{');
            for (i, (key, item)) in entries.into_iter().enumerate() {
                separator(out, i, indent, level + 1);
                write_string(out, key);
                out.push_str(": ");
                write_value(out, item, indent, sort_keys, level + 1);
            }
            close(out, indent, level);
            out.push('}');
        }
    }
}

fn separator(out: &mut String, index: usize, indent: Option<usize>, level: usize) {
    match indent {
        Some(width) => {
            if index > 0 {
                out.push(',');
            }
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', width * level));
        }
        None => {
            if index > 0 {
                out.push_str(", ");
            }
        }
    }
}

fn close(out: &mut String, indent: Option<usize>, level: usize) {
    if let Some(width) = indent {
        out.push('\n');
        out.extend(std::iter::repeat_n(' ', width * level));
    }
}

/// A JSON string with only what `ensure_ascii=False` escapes: quote, backslash, and controls.
fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        let cases = [
            (0.1, "0.1"),
            (1.0, "1.0"),
            (1e-05, "1e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123456789.125, "123456789.125"),
            (-0.0, "-0.0"),
            (2.5e-300, "2.5e-300"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (-1.5e-10, "-1.5e-10"),
            (123.0, "123.0"),
        ];
        for (x, expected) in cases {
            assert_eq!(float_repr(x), expected, "{x}");
        }
    }

    #[test]
    fn whitespace_includes_information_separators() {
        assert_eq!(strip("\u{1c} x \u{1f}"), "x");
        assert_eq!(strip("\u{200b}x\u{200b}"), "\u{200b}x\u{200b}");
        assert_eq!(split_whitespace(" a\u{1c}b  c ").collect::<Vec<_>>(), ["a", "b", "c"]);
    }
}
