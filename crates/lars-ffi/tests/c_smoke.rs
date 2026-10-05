//! Compile `tests/c/smoke.c` against the built library and `include/lars.h`, and run it on a mock
//! pack, under `leaks` on macOS. With `LARS_TEST_PACK` naming a real pack, run it on that too.
//! Unix only; needs a C compiler (`CC`, else `cc`).

#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;

/// Build the C library (cargo test builds only the rlib) and return its directory.
fn library_dir() -> PathBuf {
    // target/<profile>/deps/c_smoke-... -> target/<profile>
    let dir = std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().to_path_buf();
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let mut build = Command::new(cargo);
    build.args(["build", "-p", "lars-ffi", "--lib"]);
    if dir.file_name().is_some_and(|n| n == "release") {
        build.arg("--release");
    }
    assert!(build.status().unwrap().success(), "building liblars failed");
    dir
}

#[test]
fn the_c_abi_works_from_c() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let work = std::env::temp_dir().join(format!("lars-ffi-smoke-{}", std::process::id()));
    let pack = work.join("pack");
    std::fs::create_dir_all(&pack).unwrap();
    std::fs::write(
        pack.join("pack.json"),
        r#"{"format": 1, "name": "smoke", "version": "0.0.0", "backend": "mock", "template": "plain"}"#,
    )
    .unwrap();
    let eval: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(manifest.join("../../fixtures/data/eval.json")).unwrap())
            .unwrap();
    let lines: Vec<&str> =
        eval["calibration_examples"].as_array().unwrap().iter().map(|l| l.as_str().unwrap()).collect();
    let examples = work.join("calibration.jsonl");
    std::fs::write(&examples, lines.join("\n")).unwrap();

    let lib = library_dir();
    let binary = work.join("smoke");
    let compiler = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(compiler)
        .args(["-std=c11", "-Wall", "-Werror", "-I"])
        .arg(manifest.join("include"))
        .arg(manifest.join("tests/c/smoke.c"))
        .arg("-L")
        .arg(&lib)
        .arg(format!("-Wl,-rpath,{}", lib.display()))
        .args(["-llars", "-lpthread", "-o"])
        .arg(&binary)
        .status()
        .expect("a C compiler");
    assert!(status.success(), "smoke.c did not compile");

    let leaks = cfg!(target_os = "macos") && Path::new("/usr/bin/leaks").exists();
    if let Some(real) = std::env::var_os("LARS_TEST_PACK") {
        let output = Command::new(&binary).arg(&real).arg(&examples).output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        eprintln!("{}: {stdout}{}", Path::new(&real).display(), String::from_utf8_lossy(&output.stderr));
        assert!(output.status.success() && stdout.contains("ok: "), "the smoke program failed on the real pack");
    }
    let mut run = if leaks {
        let mut c = Command::new("/usr/bin/leaks");
        c.args(["--atExit", "--"]).arg(&binary);
        c
    } else {
        Command::new(&binary)
    };
    let output = run.arg(&pack).arg(&examples).output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    eprintln!("{stdout}{}", String::from_utf8_lossy(&output.stderr));
    assert!(stdout.contains("ok: "), "the smoke program failed");
    assert!(output.status.success(), "smoke failed or leaked");
    if leaks {
        assert!(stdout.contains(" 0 leaks for 0 total leaked bytes"), "leaks found");
    }
}
