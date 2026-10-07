//! Pack downloads from a local HTTP server: versions, the latest index, a corrupted file, and
//! offline mode. One test, since it sets process-wide environment variables.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};

use lars_runtime::{Lars, OpenOptions};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn serve(root: PathBuf) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            let mut line = String::new();
            while reader.read_line(&mut line).unwrap() > 2 {
                line.clear();
            }
            let path = request_line.split_whitespace().nth(1).unwrap_or("/").trim_start_matches('/').to_string();
            let mut stream = stream;
            match std::fs::read(root.join(&path)) {
                Ok(body) => {
                    write!(stream, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len())
                        .unwrap();
                    stream.write_all(&body).unwrap();
                }
                Err(_) => {
                    write!(stream, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").unwrap()
                }
            }
        }
    });
    format!("http://{address}")
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// A published mock pack with a calibration file, so a file is downloaded and verified.
fn publish(root: &Path, name: &str, version: &str, corrupt: bool) {
    let dir = root.join(name).join(version);
    std::fs::create_dir_all(&dir).unwrap();
    let calibration = json!({"temperatures": {"noul": 1.0, "choice": 1.5, "score": 1.0, "multi": 1.0}, "platt": {},
                             "fusion": {}, "fitted_on": null, "escalate": null,
                             "pack": {"name": name, "version": version, "model_sha256": "mock"}})
    .to_string();
    std::fs::write(dir.join("calibration.json"), if corrupt { "tampered" } else { calibration.as_str() }).unwrap();
    let manifest = json!({"format": 1, "name": name, "version": version, "backend": "mock", "template": "plain",
                          "calibration": {"file": "calibration.json", "sha256": sha256(calibration.as_bytes()),
                                          "bytes": calibration.len()}});
    std::fs::write(dir.join("pack.json"), manifest.to_string()).unwrap();
}

#[test]
fn packs_download_verify_and_work_offline() {
    let work = std::env::temp_dir().join(format!("lars-download-test-{}", std::process::id()));
    let (published, cache) = (work.join("published"), work.join("cache"));
    publish(&published, "mockdl", "0.1.0", false);
    publish(&published, "mockdl", "0.2.0", false);
    publish(&published, "broken", "1.0.0", true);
    std::fs::write(published.join("mockdl/index.json"), r#"{"latest": "0.2.0"}"#).unwrap();
    let base = serve(published);
    unsafe {
        std::env::set_var("LARS_PACK_DIR", &cache);
        std::env::set_var("LARS_PACK_URL", &base);
        std::env::remove_var("LARS_OFFLINE");
    }

    let latest = Lars::open("mockdl", OpenOptions::default()).unwrap();
    assert_eq!(latest.pack().version, "0.2.0", "a bare name with nothing installed downloads the latest");

    let lars = Lars::open("mockdl@0.1.0", OpenOptions::default()).unwrap();
    assert_eq!(lars.pack().version, "0.1.0");
    assert_eq!(lars.pack().calibrator.temperature_for(lars_core::schema::Kind::Choice), 1.5);
    let answer: Value = serde_json::from_str(
        &lars
            .evaluate(r#"{"state": "hello", "questions": {"q": {"type": "noul", "instructions": "A greeting"}}}"#)
            .unwrap(),
    )
    .unwrap();
    assert!(answer["answers"]["q"]["noul"].is_number());

    let installed = Lars::open("mockdl", OpenOptions::default()).unwrap();
    assert_eq!(installed.pack().version, "0.2.0", "a bare name uses the newest installed version");

    let error = Lars::open("broken@1.0.0", OpenOptions::default()).err().unwrap();
    assert_eq!(error.error_type, "pack_error");
    assert!(error.message.contains("sha256"), "{}", error.message);
    assert!(Lars::open("nonexistent@1.0.0", OpenOptions::default()).err().unwrap().message.contains("404"));
    assert!(!cache.join("broken/1.0.0").exists(), "a failed download installs nothing");
    let leftovers: Vec<_> = std::fs::read_dir(cache.join("broken")).unwrap().flatten().collect();
    assert!(leftovers.is_empty(), "no staging directory is left behind: {leftovers:?}");

    // A newer release: open keeps using what is installed, pull fetches it.
    publish(&work.join("published"), "mockdl", "0.3.0", false);
    std::fs::write(work.join("published/mockdl/index.json"), r#"{"latest": "0.3.0"}"#).unwrap();
    assert_eq!(Lars::open("mockdl", OpenOptions::default()).unwrap().pack().version, "0.2.0");
    assert!(lars_runtime::pull("mockdl").unwrap().ends_with("mockdl/0.3.0"));
    assert!(lars_runtime::pull("mockdl@0.1.0").unwrap().ends_with("mockdl/0.1.0"));
    assert!(lars_runtime::pull("./no/such/pack").is_err());

    unsafe { std::env::set_var("LARS_OFFLINE", "1") };
    assert_eq!(
        Lars::open("mockdl", OpenOptions::default()).unwrap().pack().version,
        "0.3.0",
        "installed packs open offline"
    );
    let error = Lars::open("mockdl@0.4.0", OpenOptions::default()).err().unwrap();
    assert!(error.message.contains("LARS_OFFLINE"), "{}", error.message);
    let _ = std::fs::remove_dir_all(work);
}
