//! Model packs: a directory holding a model, its template, its layout and its pooled
//! calibrator, described by `pack.json`.
//!
//! A pack is the unit a calibrator is valid for. Weights, quantization, template, layout, the
//! llama.cpp build and flash attention all move logits (RESULTS.md, 2026-10-02), so a pack fixes
//! them, and a calibrator records the pack it was fitted on and loads against no other.
//!
//! ```json
//! {
//!   "format": 1, "name": "lars-4b", "version": "0.1.0", "backend": "llamacpp",
//!   "model": {"file": "model.gguf", "bytes": 2497280960, "sha256": "...", "base": "Qwen/Qwen3-4B-Instruct-2507",
//!             "license": "Apache-2.0", "quantization": "Q4_K_M"},
//!   "calibration": {"file": "calibration.json", "sha256": "..."},
//!   "template": "chatml",
//!   "layout": "v1",
//!   "runtime": {"llama_cpp": "b11349", "flash_attention": "off"},
//!   "evaluation": {}
//! }
//! ```
//!
//! `template` is a named template or a captured one (`before_system`, `between`, `after_user`,
//! `trims_content`). `backend: "mock"` packs carry no model and serve the deterministic mock
//! backend, for tests and for bindings' CI.

use std::io::Read;
use std::path::{Path, PathBuf};

use lars_core::calibration::Calibrator;
use lars_core::render::{Layout, Template};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};

pub const FORMAT: u64 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendKind {
    LlamaCpp,
    Mock,
}

#[derive(Debug, Clone)]
pub struct Pack {
    pub dir: PathBuf,
    pub manifest: Value,
    pub name: String,
    pub version: String,
    pub backend: BackendKind,
    pub model_file: Option<PathBuf>,
    pub model_sha256: String,
    pub template: Template,
    pub layout: Layout,
    pub flash_attention: bool,
    pub llama_cpp: Option<String>,
    pub calibrator: Calibrator,
}

fn field<'a>(manifest: &'a Value, path: &str) -> Result<&'a Value> {
    let value = path.split('.').fold(manifest, |v, key| &v[key]);
    if value.is_null() { Err(Error::pack(format!("pack.json has no {path}"))) } else { Ok(value) }
}

fn text(manifest: &Value, path: &str) -> Result<String> {
    field(manifest, path)?
        .as_str()
        .map(String::from)
        .ok_or_else(|| Error::pack(format!("pack.json {path} is not text")))
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path).map_err(|e| Error::pack(format!("{}: {e}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buffer).map_err(|e| Error::pack(format!("{}: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// A file a manifest lists: it must exist, have the listed size, and, with `verify`, the listed hash.
fn listed_file(dir: &Path, manifest: &Value, key: &str, verify: bool) -> Result<PathBuf> {
    let path = dir.join(text(manifest, &format!("{key}.file"))?);
    let metadata = std::fs::metadata(&path).map_err(|e| Error::pack(format!("{}: {e}", path.display())))?;
    if let Some(bytes) = manifest[key]["bytes"].as_u64() {
        if metadata.len() != bytes {
            return Err(Error::pack(format!("{}: {} bytes, pack.json lists {bytes}", path.display(), metadata.len())));
        }
    }
    if verify {
        let expected = text(manifest, &format!("{key}.sha256"))?;
        let actual = sha256_file(&path)?;
        if actual != expected {
            return Err(Error::pack(format!("{}: sha256 {actual}, pack.json lists {expected}", path.display())));
        }
    }
    Ok(path)
}

pub fn template_from_json(value: &Value) -> Result<Template> {
    if let Some(name) = value.as_str() {
        return Template::named(name).ok_or_else(|| Error::pack(format!("unknown template {name:?}")));
    }
    let part = |k: &str| value[k].as_str().map(String::from).ok_or_else(|| Error::pack(format!("template has no {k}")));
    Ok(Template::Captured {
        before_system: part("before_system")?,
        between: part("between")?,
        after_user: part("after_user")?,
        trims_content: value["trims_content"].as_bool().unwrap_or(false),
    })
}

impl Pack {
    /// Load the pack in `dir`. Sizes are always checked; hashes only with `verify`, since hashing
    /// a 2.5 GB model takes seconds (downloads are verified as they are written).
    pub fn load(dir: &Path, verify: bool) -> Result<Pack> {
        let manifest_path = dir.join("pack.json");
        let raw = std::fs::read_to_string(&manifest_path)
            .map_err(|e| Error::pack(format!("{}: {e}", manifest_path.display())))?;
        let manifest: Value =
            serde_json::from_str(&raw).map_err(|e| Error::pack(format!("{}: {e}", manifest_path.display())))?;
        if manifest["format"].as_u64() != Some(FORMAT) {
            return Err(Error::pack(format!("{}: unsupported pack format {}", dir.display(), manifest["format"])));
        }
        let backend = match text(&manifest, "backend")?.as_str() {
            "llamacpp" => BackendKind::LlamaCpp,
            "mock" => BackendKind::Mock,
            other => return Err(Error::pack(format!("unknown backend {other:?}"))),
        };
        let (model_file, model_sha256) = match backend {
            BackendKind::LlamaCpp => {
                (Some(listed_file(dir, &manifest, "model", verify)?), text(&manifest, "model.sha256")?)
            }
            BackendKind::Mock => (None, "mock".to_string()),
        };
        let layout_name = manifest["layout"].as_str().unwrap_or("v1");
        let layout =
            Layout::parse(layout_name).ok_or_else(|| Error::pack(format!("unknown layout {layout_name:?}")))?;
        let mut pack = Pack {
            dir: dir.to_path_buf(),
            name: text(&manifest, "name")?,
            version: text(&manifest, "version")?,
            backend,
            model_file,
            model_sha256,
            template: template_from_json(manifest.get("template").unwrap_or(&json!("chatml")))?,
            layout,
            flash_attention: manifest["runtime"]["flash_attention"].as_str() == Some("on"),
            llama_cpp: manifest["runtime"]["llama_cpp"].as_str().map(String::from),
            calibrator: Calibrator::default(),
            manifest,
        };
        if !pack.manifest["calibration"].is_null() {
            let path = listed_file(dir, &pack.manifest, "calibration", verify)?;
            let raw = std::fs::read_to_string(&path).map_err(|e| Error::pack(format!("{}: {e}", path.display())))?;
            let value: Value =
                serde_json::from_str(&raw).map_err(|e| Error::pack(format!("{}: {e}", path.display())))?;
            pack.calibrator = pack.bound_calibrator(&value)?;
        }
        Ok(pack)
    }

    /// What a calibrator fitted on this pack records about it.
    pub fn identity(&self) -> Value {
        json!({"name": self.name, "version": self.version, "model_sha256": self.model_sha256})
    }

    /// A calibrator from JSON, refused unless it records this pack (`"pack"`, as `calibrate` and
    /// `bind` write it): one fitted on other weights or another runtime would be silently wrong.
    pub fn bound_calibrator(&self, value: &Value) -> Result<Calibrator> {
        let recorded = &value["pack"];
        if recorded.is_null() {
            return Err(Error::invalid("calibrator records no pack; fit it with calibrate on this pack"));
        }
        if *recorded != self.identity() {
            return Err(Error::invalid(format!(
                "calibrator was fitted on pack {} {}, not {} {}",
                recorded["name"], recorded["version"], self.name, self.version
            )));
        }
        Calibrator::from_json(value).map_err(Error::from)
    }

    /// A calibrator's JSON, recording this pack.
    pub fn bind(&self, calibrator: &Calibrator) -> Value {
        let mut value = calibrator.to_json();
        value["pack"] = self.identity();
        value
    }
}

/// Where packs live: `LARS_PACK_DIR`, else the platform's cache directory.
pub fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("LARS_PACK_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    if cfg!(target_os = "macos") {
        home.join("Library/Caches/lars/packs")
    } else if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or(home).join("lars").join("packs")
    } else {
        std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from).unwrap_or_else(|| home.join(".cache")).join("lars/packs")
    }
}

/// A pack directory for `reference`: a path to a pack, or `name` / `name@version` in the cache,
/// downloaded first when it is not there (unless `LARS_OFFLINE` is set). Without a version, the
/// newest installed one is used, or the newest published one is downloaded.
pub fn resolve(reference: &str) -> Result<PathBuf> {
    let path = Path::new(reference);
    if path.join("pack.json").exists() {
        return Ok(path.to_path_buf());
    }
    if reference.contains(['/', '\\']) || reference.starts_with('.') {
        return Err(Error::pack(format!("{reference}: no pack.json there")));
    }
    let (name, version) = reference.split_once('@').map_or((reference, None), |(n, v)| (n, Some(v)));
    let root = cache_dir().join(name);
    let installed = match version {
        Some(v) => Some(v.to_string()).filter(|v| root.join(v).join("pack.json").exists()),
        None => newest_version(&root),
    };
    match installed {
        Some(v) => Ok(root.join(v)),
        None => install(name, version),
    }
}

fn newest_version(root: &Path) -> Option<String> {
    let parse = |v: &str| v.split('.').map(|p| p.parse::<u64>().unwrap_or(0)).collect::<Vec<_>>();
    std::fs::read_dir(root)
        .ok()?
        .flatten()
        .filter(|e| e.path().join("pack.json").exists())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .max_by_key(|v| parse(v))
}

// --------------------------------------------------------------------------- download

/// Where packs are downloaded from: `LARS_PACK_URL`, else the `sdk/` folder of the `moedex/lars`
/// Hub repository. A pack lives at `<base>/<name>/<version>/`, and `<base>/<name>/index.json`
/// (`{"latest": "0.1.0"}`) names the newest version.
pub fn download_base() -> String {
    std::env::var("LARS_PACK_URL").unwrap_or_else(|_| "https://huggingface.co/moedex/lars/resolve/main/sdk".into())
}

fn offline() -> bool {
    std::env::var("LARS_OFFLINE").is_ok_and(|v| !v.is_empty() && v != "0")
}

fn get(url: &str) -> Result<ureq::http::Response<ureq::Body>> {
    ureq::get(url).call().map_err(|e| Error::pack(format!("{url}: {e}")))
}

/// Stream `url` into `path`, checking its sha256 and size as it is written.
fn download_file(url: &str, path: &Path, sha256: &str, bytes: Option<u64>) -> Result<()> {
    let mut response = get(url)?;
    let mut reader = response.body_mut().as_reader();
    let mut file = std::fs::File::create(path).map_err(|e| Error::pack(format!("{}: {e}", path.display())))?;
    let (mut hasher, mut written) = (Sha256::new(), 0u64);
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buffer).map_err(|e| Error::pack(format!("{url}: {e}")))?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
        std::io::Write::write_all(&mut file, &buffer[..n])
            .map_err(|e| Error::pack(format!("{}: {e}", path.display())))?;
        written += n as u64;
    }
    let actual: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if actual != sha256 || bytes.is_some_and(|b| b != written) {
        return Err(Error::pack(format!("{url}: downloaded {written} bytes with sha256 {actual}; expected {sha256}")));
    }
    Ok(())
}

/// Download pack `name` (the newest version when `version` is None) into the cache and return
/// its directory. Files land in a temporary directory, are verified, and the directory is
/// renamed into place, so a failed download leaves nothing half-installed.
pub fn install(name: &str, version: Option<&str>) -> Result<PathBuf> {
    if offline() {
        return Err(Error::pack(format!("pack {name} is not installed and LARS_OFFLINE is set")));
    }
    let base = download_base();
    let version = match version {
        Some(v) => v.to_string(),
        None => {
            let url = format!("{base}/{name}/index.json");
            let text = get(&url)?.body_mut().read_to_string().map_err(|e| Error::pack(format!("{url}: {e}")))?;
            let index: Value = serde_json::from_str(&text).map_err(|e| Error::pack(format!("{url}: {e}")))?;
            index["latest"].as_str().ok_or_else(|| Error::pack(format!("{url} names no latest version")))?.to_string()
        }
    };
    let root = format!("{base}/{name}/{version}");
    let dir = cache_dir().join(name).join(&version);
    if dir.join("pack.json").exists() {
        return Ok(dir);
    }
    let staging = cache_dir().join(name).join(format!(".{version}.{}.part", std::process::id()));
    std::fs::create_dir_all(&staging).map_err(|e| Error::pack(format!("{}: {e}", staging.display())))?;
    let result = (|| -> Result<()> {
        let text = get(&format!("{root}/pack.json"))?
            .body_mut()
            .read_to_string()
            .map_err(|e| Error::pack(format!("{root}/pack.json: {e}")))?;
        let manifest: Value = serde_json::from_str(&text).map_err(|e| Error::pack(format!("{root}/pack.json: {e}")))?;
        if manifest["name"] != name || manifest["version"] != version.as_str() {
            return Err(Error::pack(format!(
                "{root}/pack.json describes {} {}",
                manifest["name"], manifest["version"]
            )));
        }
        for key in ["model", "calibration"] {
            if manifest[key].is_null() {
                continue;
            }
            let file = text_of(&manifest, key, "file")?;
            if file.contains(['/', '\\']) || file.starts_with('.') {
                return Err(Error::pack(format!("pack.json lists an unsafe file name {file:?}")));
            }
            download_file(
                &format!("{root}/{file}"),
                &staging.join(&file),
                &text_of(&manifest, key, "sha256")?,
                manifest[key]["bytes"].as_u64(),
            )?;
        }
        std::fs::write(staging.join("pack.json"), &text).map_err(|e| Error::pack(e.to_string()))?;
        Pack::load(&staging, false)?;
        std::fs::rename(&staging, &dir).map_err(|e| Error::pack(format!("{}: {e}", dir.display())))
    })();
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    result.map(|()| dir)
}

fn text_of(manifest: &Value, key: &str, field: &str) -> Result<String> {
    manifest[key][field]
        .as_str()
        .map(String::from)
        .ok_or_else(|| Error::pack(format!("pack.json has no {key}.{field}")))
}
