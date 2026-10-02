//! Rendering parity with the reference: every prompt byte, every row field.

mod common;

use common::{fixture, labels, strings};
use lars_core::pyrandom::PyRandom;
use lars_core::render::{self, Template, compose_prompt, plan_rows, render_content};
use lars_core::schema::Request;

#[test]
fn rows_and_prompts_match_the_reference() {
    let data = fixture("render.json");
    let labels = labels();
    let mut prompts_checked = 0;
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let request = Request::from_value(&case["request"]).unwrap_or_else(|e| panic!("{name}: {e}"));
        let base_state = render_content(&request.state);
        assert_eq!(base_state, case["state_rendered"].as_str().unwrap(), "{name}: state");

        let rows = plan_rows(&request, &labels);
        let expected_rows = case["rows"].as_array().unwrap();
        assert_eq!(rows.len(), expected_rows.len(), "{name}: row count");
        for (i, (row, expected)) in rows.iter().zip(expected_rows).enumerate() {
            let at = format!("{name} row {i}");
            assert_eq!(row.question_id, expected["question_id"].as_str().unwrap(), "{at}: question_id");
            assert_eq!(row.kind.as_str(), expected["kind"].as_str().unwrap(), "{at}: kind");
            assert_eq!(row.keys, strings(&expected["keys"]), "{at}: keys");
            assert_eq!(row.labels, strings(&expected["labels"]), "{at}: labels");
            assert_eq!(row.suffix_body, expected["suffix_body"].as_str().unwrap(), "{at}: suffix_body");
            assert_eq!(row.variant, expected["variant"].as_str().unwrap(), "{at}: variant");
            assert_eq!(row.state_text.as_deref(), expected["state_text"].as_str(), "{at}: state_text");
            assert_eq!(row.ablated_span.as_deref(), expected["ablated_span"].as_str(), "{at}: ablated_span");
            let ends: Vec<usize> =
                expected["option_ends"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            assert_eq!(row.option_ends, ends, "{at}: option_ends");
        }

        for (template_name, prompts) in case["prompts"].as_object().unwrap() {
            let template = Template::named(template_name).unwrap();
            for (i, (row, pair)) in rows.iter().zip(prompts.as_array().unwrap()).enumerate() {
                let state = row.state_text.as_deref().unwrap_or(&base_state);
                let (prefix, suffix) = compose_prompt(&template, state, &row.suffix_body);
                assert_eq!(prefix, pair[0].as_str().unwrap(), "{name} row {i} {template_name}: prefix");
                assert_eq!(suffix, pair[1].as_str().unwrap(), "{name} row {i} {template_name}: suffix");
                prompts_checked += 1;
            }
        }
    }
    assert!(prompts_checked > 500, "only {prompts_checked} prompts checked");
}

#[test]
fn content_units_strip_and_fences_match() {
    let data = fixture("render.json");
    for case in data["render_content"].as_array().unwrap() {
        assert_eq!(render_content(&case["value"]), case["rendered"].as_str().unwrap(), "{}", case["value"]);
    }
    for case in data["split_units"].as_array().unwrap() {
        assert_eq!(render::split_units(case["text"].as_str().unwrap()), strings(&case["units"]), "{}", case["text"]);
    }
    for case in data["strip"].as_array().unwrap() {
        assert_eq!(lars_core::py::strip(case["text"].as_str().unwrap()), case["stripped"].as_str().unwrap());
    }
    for case in data["nonce"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        assert_eq!(render::state_nonce(text), case["nonce"].as_str().unwrap());
        assert_eq!(render::fence_state(text), case["fenced"].as_str().unwrap());
    }
}

#[test]
fn templates_match_the_reference() {
    let manifest = fixture("manifest.json");
    for (name, expected) in manifest["templates"].as_object().unwrap() {
        assert_eq!(Template::named(name).unwrap().apply("<SYSTEM>", "<USER>"), expected.as_str().unwrap(), "{name}");
    }
    assert_eq!(render::SYSTEM_PROMPT, manifest["system_prompt"].as_str().unwrap());
    assert_eq!(render::SENTINEL, manifest["sentinel"].as_str().unwrap());
}

#[test]
fn captured_templates_reproduce_the_models() {
    let data = fixture("templates.json");
    for model in data["models"].as_array().unwrap() {
        let template = Template::Captured {
            before_system: model["before_system"].as_str().unwrap().into(),
            between: model["between"].as_str().unwrap().into(),
            after_user: model["after_user"].as_str().unwrap().into(),
            trims_content: model["trims_content"].as_bool().unwrap(),
        };
        let chatml = Template::ChatMl.apply("\u{0}S\u{0}", "\u{0}U\u{0}");
        assert_eq!(chatml, data["chatml"].as_str().unwrap());
        assert_eq!(
            template.apply("\u{0}S\u{0}", "\u{0}U\u{0}") == chatml,
            model["equals_chatml"].as_bool().unwrap(),
            "{}",
            model["model"]
        );
    }
}

#[test]
fn shuffles_match_cpython() {
    let data = fixture("shuffle.json");
    for case in data["cases"].as_array().unwrap() {
        let seed = case["seed"].as_str().unwrap();
        let n = case["n"].as_u64().unwrap() as usize;
        let mut rng = PyRandom::from_str_seed(seed);
        for expected in case["orders"].as_array().unwrap() {
            let mut order: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut order);
            let expected: Vec<usize> =
                expected.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            assert_eq!(order, expected, "seed {seed}");
        }
    }
}

#[test]
fn round_matches_python() {
    let data = fixture("primitives.json");
    for case in data["round4"].as_array().unwrap() {
        let x = case["x"].as_f64().unwrap();
        let expected = case["out"].as_f64().unwrap();
        assert_eq!(lars_core::py::round(x, 4).to_bits(), expected.to_bits(), "round({x}, 4)");
    }
}
