//! Build the vendored llama.cpp as static libraries with CMake and link them.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let source = root.join("llama.cpp");
    if !source.join("CMakeLists.txt").exists() {
        panic!("llama.cpp is missing: run `git submodule update --init --depth 1`");
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=llama.cpp/include/llama.h");
    println!("cargo:rerun-if-env-changed=LARS_LLAMA_NATIVE");
    println!("cargo:rustc-env=LARS_LLAMA_CPP_VERSION={}", version(&source));

    let target = env::var("TARGET").unwrap();
    let apple = target.contains("apple");
    let mut config = cmake::Config::new(&source);
    config
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_APP", "OFF")
        .define("LLAMA_OPENSSL", "OFF")
        // No OpenMP: it would add a runtime dependency the host application may not have.
        .define("GGML_OPENMP", "OFF")
        // Portable by default, since the library ships in packages; LARS_LLAMA_NATIVE=1 tunes
        // for the build machine.
        .define("GGML_NATIVE", if env::var("LARS_LLAMA_NATIVE").as_deref() == Ok("1") { "ON" } else { "OFF" });
    if apple {
        config.define("GGML_METAL", "ON").define("GGML_METAL_EMBED_LIBRARY", "ON").define("GGML_BLAS", "OFF");
    }
    let out = config.build();

    for dir in ["lib", "lib64"] {
        println!("cargo:rustc-link-search=native={}", out.join(dir).display());
    }
    for lib in static_libs(&out) {
        println!("cargo:rustc-link-lib=static={lib}");
    }
    if apple {
        for framework in ["Foundation", "Metal", "MetalKit", "Accelerate"] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=stdc++");
    } else if target.contains("windows") {
        // ggml-cpu reads CPU features from the registry.
        println!("cargo:rustc-link-lib=advapi32");
    }
}

/// Every static library the build installed, llama first, so dependents resolve in order.
fn static_libs(out: &Path) -> Vec<String> {
    let mut libs = Vec::new();
    for dir in ["lib", "lib64"] {
        let Ok(entries) = std::fs::read_dir(out.join(dir)) else { continue };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let stem =
                name.strip_prefix("lib").and_then(|n| n.strip_suffix(".a")).or_else(|| name.strip_suffix(".lib"));
            if let Some(stem) = stem {
                libs.push(stem.to_string());
            }
        }
    }
    // llama needs ggml, ggml registers the backends, and everything needs ggml-base.
    let rank = |l: &str| match l {
        "llama" => 0,
        "ggml" => 1,
        "ggml-base" => 3,
        l if l.starts_with("ggml-") => 2,
        _ => 4,
    };
    libs.sort_by_key(|l| (rank(l), l.clone()));
    libs
}

fn version(source: &Path) -> String {
    Command::new("git")
        .args(["-C", &source.display().to_string(), "describe", "--tags", "--always"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}
