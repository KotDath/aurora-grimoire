use super::RagDeployArgs;
use crate::config::{AppConfig, DEFAULT_COLLECTION, DEFAULT_DEPLOY_BATCH_SIZE, DEFAULT_QDRANT_URL};
use anyhow::{Context, Result, anyhow};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::{Method, StatusCode, blocking::Client};
use serde::Deserialize;
use serde_json::{Value, json};
use sha1::{Digest, Sha1};
use std::{
    env,
    fs::{self, File},
    io::{BufRead, BufReader},
    path::{Component, Path, PathBuf},
    time::Duration,
};

const VECTORS_ROOT_DIRNAME: &str = "vectors_data";
const VECTOR_FILE_PREFIX: &str = "vectors-";

macro_rules! vprintln {
    ($verbose:expr, $($arg:tt)*) => {
        if $verbose {
            println!($($arg)*);
        }
    };
}

#[derive(Debug, Deserialize)]
struct VectorRecord {
    id: String,
    vector: Vec<f32>,
    payload: Value,
}

struct TempExtractDir {
    path: PathBuf,
}

impl Drop for TempExtractDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

pub fn run(args: RagDeployArgs) -> Result<()> {
    let verbose = args.verbose;
    let cfg = AppConfig::load()?;
    let data_root = cfg.data_root()?;
    let input = args
        .input
        .unwrap_or_else(|| data_root.join(VECTORS_ROOT_DIRNAME));
    let qdrant_url = args
        .url
        .clone()
        .or_else(|| cfg.deploy.qdrant_url.clone())
        .unwrap_or_else(|| DEFAULT_QDRANT_URL.to_string());
    let collection = args
        .collection
        .clone()
        .or_else(|| cfg.deploy.collection.clone())
        .unwrap_or_else(|| DEFAULT_COLLECTION.to_string());
    let batch_size = args
        .batch_size
        .or(cfg.deploy.batch_size)
        .unwrap_or(DEFAULT_DEPLOY_BATCH_SIZE)
        .max(1);

    let api_key = args.api_key.or_else(|| env::var("QDRANT_API_KEY").ok());

    let mut extracted: Option<TempExtractDir> = None;
    let vectors_root = if args.from_bundle || input.is_file() {
        let temp_path = env::temp_dir().join(format!(
            "aurora-grimoire-deploy-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        fs::create_dir_all(&temp_path)
            .with_context(|| format!("failed to create temp directory {}", temp_path.display()))?;
        extract_vectors_from_bundle(&input, &temp_path)?;
        extracted = Some(TempExtractDir {
            path: temp_path.clone(),
        });
        temp_path.join(VECTORS_ROOT_DIRNAME)
    } else {
        input
    };

    if !vectors_root.exists() {
        return Err(anyhow!(
            "vectors directory does not exist: {}",
            vectors_root.display()
        ));
    }

    let vector_files = discover_jsonl_files(&vectors_root, VECTOR_FILE_PREFIX)?;
    if vector_files.is_empty() {
        return Err(anyhow!(
            "no vector shards found in {}",
            vectors_root.display()
        ));
    }

    let total_vectors_hint = vectors_total_from_manifest(&vectors_root)?
        .unwrap_or(count_lines(&vector_files).context("failed to count vector records")?);

    let first_record = read_first_vector_record(&vector_files)?.ok_or_else(|| {
        anyhow!(
            "vector shards in {} do not contain valid records",
            vectors_root.display()
        )
    })?;
    let dim = first_record.vector.len();
    if dim == 0 {
        return Err(anyhow!("first vector record has zero-dimensional vector"));
    }

    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .context("failed to build Qdrant HTTP client")?;

    ensure_collection(
        &client,
        &qdrant_url,
        api_key.as_deref(),
        &collection,
        dim,
        args.recreate,
        verbose,
    )?;
    ensure_payload_indexes(
        &client,
        &qdrant_url,
        api_key.as_deref(),
        &collection,
        verbose,
    )?;

    let progress = if verbose {
        let pb = ProgressBar::new(total_vectors_hint as u64);
        let style =
            ProgressStyle::with_template("[deploy] [{bar:28.green/white}] {pos}/{len}: {msg}")
                .expect("progress template must be valid")
                .progress_chars("=> ");
        pb.set_style(style);
        Some(pb)
    } else {
        None
    };

    let mut uploaded = 0usize;
    let mut points_batch: Vec<Value> = Vec::with_capacity(batch_size);
    for path in &vector_files {
        let file = File::open(path)
            .with_context(|| format!("failed to open vector shard {}", path.display()))?;
        let reader = BufReader::new(file);
        for line_result in reader.lines() {
            let line = line_result
                .with_context(|| format!("failed to read line in {}", path.display()))?;
            if line.trim().is_empty() {
                continue;
            }
            let record: VectorRecord = serde_json::from_str(&line).with_context(|| {
                format!("failed to parse vector JSON record in {}", path.display())
            })?;
            if record.vector.len() != dim {
                return Err(anyhow!(
                    "vector dimension mismatch for id {}: expected {}, got {}",
                    record.id,
                    dim,
                    record.vector.len()
                ));
            }
            if let Some(pb) = &progress {
                pb.set_message(shorten_for_progress(&record.id, 56));
            }
            let point = json!({
                "id": stable_point_id(&record.id),
                "vector": record.vector,
                "payload": record.payload,
            });
            points_batch.push(point);

            if points_batch.len() >= batch_size {
                upsert_points(
                    &client,
                    &qdrant_url,
                    api_key.as_deref(),
                    &collection,
                    &points_batch,
                )?;
                uploaded += points_batch.len();
                if let Some(pb) = &progress {
                    pb.inc(points_batch.len() as u64);
                }
                points_batch.clear();
            }
        }
    }

    if !points_batch.is_empty() {
        upsert_points(
            &client,
            &qdrant_url,
            api_key.as_deref(),
            &collection,
            &points_batch,
        )?;
        uploaded += points_batch.len();
        if let Some(pb) = &progress {
            pb.inc(points_batch.len() as u64);
        }
    }

    if let Some(pb) = progress {
        pb.finish_and_clear();
    }

    println!(
        "{} vectors uploaded to {} at {}",
        uploaded, collection, qdrant_url
    );

    drop(extracted);
    Ok(())
}

fn ensure_collection(
    client: &Client,
    base_url: &str,
    api_key: Option<&str>,
    collection: &str,
    dim: usize,
    recreate: bool,
    verbose: bool,
) -> Result<()> {
    if recreate {
        delete_collection(client, base_url, api_key, collection)?;
    }

    let existing_dim = get_collection_vector_size(client, base_url, api_key, collection)?;
    if let Some(found) = existing_dim {
        if found != dim {
            return Err(anyhow!(
                "collection '{}' exists with vector size {}, but input vectors have size {}",
                collection,
                found,
                dim
            ));
        }
        vprintln!(
            verbose,
            "[deploy] collection '{}' already exists",
            collection
        );
        return Ok(());
    }

    let url = format!(
        "{}/collections/{}",
        base_url.trim_end_matches('/'),
        collection
    );
    let response = send_json_request(
        client,
        Method::PUT,
        &url,
        api_key,
        Some(&json!({
            "vectors": {
                "size": dim,
                "distance": "Cosine"
            }
        })),
    )?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(anyhow!(
            "failed to create Qdrant collection '{}': status={} body={}",
            collection,
            status,
            body
        ));
    }
    vprintln!(
        verbose,
        "[deploy] created collection '{}' (dim={})",
        collection,
        dim
    );
    Ok(())
}

fn ensure_payload_indexes(
    client: &Client,
    base_url: &str,
    api_key: Option<&str>,
    collection: &str,
    verbose: bool,
) -> Result<()> {
    let fields = ["version_bucket", "source_md", "section_anchor", "table_id"];
    for field in fields {
        let url = format!(
            "{}/collections/{}/index?wait=true",
            base_url.trim_end_matches('/'),
            collection
        );
        let response = send_json_request(
            client,
            Method::PUT,
            &url,
            api_key,
            Some(&json!({
                "field_name": field,
                "field_schema": "keyword"
            })),
        )?;
        if response.status().is_success() {
            vprintln!(verbose, "[deploy] ensured payload index: {}", field);
            continue;
        }

        // Ignore already-exists or unsupported index errors, but keep verbose note.
        let status = response.status();
        let body = response.text().unwrap_or_default();
        vprintln!(
            verbose,
            "[deploy] payload index '{}' not applied (status={}): {}",
            field,
            status,
            shorten_for_progress(&body, 120)
        );
    }
    Ok(())
}

fn upsert_points(
    client: &Client,
    base_url: &str,
    api_key: Option<&str>,
    collection: &str,
    points: &[Value],
) -> Result<()> {
    let url = format!(
        "{}/collections/{}/points?wait=true",
        base_url.trim_end_matches('/'),
        collection
    );
    let response = send_json_request(
        client,
        Method::PUT,
        &url,
        api_key,
        Some(&json!({ "points": points })),
    )?;
    if response.status().is_success() {
        return Ok(());
    }

    let status = response.status();
    let body = response.text().unwrap_or_default();
    Err(anyhow!(
        "failed to upsert points into Qdrant: status={} body={}",
        status,
        body
    ))
}

fn get_collection_vector_size(
    client: &Client,
    base_url: &str,
    api_key: Option<&str>,
    collection: &str,
) -> Result<Option<usize>> {
    let url = format!(
        "{}/collections/{}",
        base_url.trim_end_matches('/'),
        collection
    );
    let response = send_json_request(client, Method::GET, &url, api_key, None)?;
    if response.status() == StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(anyhow!(
            "failed to inspect Qdrant collection '{}': status={} body={}",
            collection,
            status,
            body
        ));
    }
    let value: Value = response
        .json()
        .context("failed to parse Qdrant collection info response")?;
    let vectors_node = value
        .get("result")
        .and_then(|v| v.get("config"))
        .and_then(|v| v.get("params"))
        .and_then(|v| v.get("vectors"))
        .ok_or_else(|| anyhow!("Qdrant collection response missing vectors config"))?;

    if let Some(size) = vectors_node.get("size").and_then(Value::as_u64) {
        return Ok(Some(size as usize));
    }
    if let Some(obj) = vectors_node.as_object() {
        for (_name, vec_cfg) in obj {
            if let Some(size) = vec_cfg.get("size").and_then(Value::as_u64) {
                return Ok(Some(size as usize));
            }
        }
    }
    Err(anyhow!(
        "unable to determine vector size from Qdrant collection config"
    ))
}

fn delete_collection(
    client: &Client,
    base_url: &str,
    api_key: Option<&str>,
    collection: &str,
) -> Result<()> {
    let url = format!(
        "{}/collections/{}",
        base_url.trim_end_matches('/'),
        collection
    );
    let response = send_json_request(client, Method::DELETE, &url, api_key, None)?;
    if response.status().is_success() || response.status() == StatusCode::NOT_FOUND {
        return Ok(());
    }
    let status = response.status();
    let body = response.text().unwrap_or_default();
    Err(anyhow!(
        "failed to delete Qdrant collection '{}': status={} body={}",
        collection,
        status,
        body
    ))
}

fn send_json_request(
    client: &Client,
    method: Method,
    url: &str,
    api_key: Option<&str>,
    body: Option<&Value>,
) -> Result<reqwest::blocking::Response> {
    let mut request = client.request(method, url);
    if let Some(key) = api_key {
        request = request.header("api-key", key);
    }
    if let Some(json_body) = body {
        request = request.json(json_body);
    }
    request
        .send()
        .with_context(|| format!("Qdrant request failed: {}", url))
}

fn discover_jsonl_files(root: &Path, prefix: &str) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let entries = fs::read_dir(root)
        .with_context(|| format!("failed to read directory: {}", root.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| {
            format!(
                "failed to read directory entry in {}",
                root.to_string_lossy()
            )
        })?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name.starts_with(prefix) && name.ends_with(".jsonl") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn count_lines(files: &[PathBuf]) -> Result<usize> {
    let mut count = 0usize;
    for path in files {
        let file = File::open(path)
            .with_context(|| format!("failed to open JSONL file {}", path.display()))?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line =
                line.with_context(|| format!("failed reading JSONL line in {}", path.display()))?;
            if !line.trim().is_empty() {
                count += 1;
            }
        }
    }
    Ok(count)
}

fn read_first_vector_record(files: &[PathBuf]) -> Result<Option<VectorRecord>> {
    for path in files {
        let file = File::open(path)
            .with_context(|| format!("failed to open vector shard {}", path.display()))?;
        let reader = BufReader::new(file);
        for line_result in reader.lines() {
            let line = line_result
                .with_context(|| format!("failed to read line in {}", path.display()))?;
            if line.trim().is_empty() {
                continue;
            }
            let record: VectorRecord = serde_json::from_str(&line)
                .with_context(|| format!("failed to parse vector record in {}", path.display()))?;
            return Ok(Some(record));
        }
    }
    Ok(None)
}

fn vectors_total_from_manifest(root: &Path) -> Result<Option<usize>> {
    let path = root.join("manifest.json");
    if !path.exists() {
        return Ok(None);
    }
    let file = File::open(&path)
        .with_context(|| format!("failed to open vectors manifest {}", path.display()))?;
    let value: Value =
        serde_json::from_reader(file).context("failed to parse vectors manifest JSON")?;
    if let Some(v) = value.get("vectors_total").and_then(Value::as_u64) {
        return Ok(Some(v as usize));
    }
    if let Some(v) = value.get("chunks_embedded").and_then(Value::as_u64) {
        return Ok(Some(v as usize));
    }
    Ok(None)
}

fn stable_point_id(id: &str) -> u64 {
    let digest = Sha1::digest(id.as_bytes());
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(buf)
}

fn extract_vectors_from_bundle(bundle_path: &Path, out_root: &Path) -> Result<()> {
    let file = File::open(bundle_path)
        .with_context(|| format!("failed to open bundle file {}", bundle_path.display()))?;
    let decoder = zstd::Decoder::new(file).with_context(|| {
        format!(
            "failed to create zstd decoder for bundle {}",
            bundle_path.display()
        )
    })?;
    let mut archive = tar::Archive::new(decoder);
    let mut extracted_files = 0usize;

    for entry_result in archive.entries().context("failed to read bundle entries")? {
        let mut entry = entry_result.context("failed to read bundle archive entry")?;
        let rel_path = entry
            .path()
            .context("failed to read bundle entry path")?
            .into_owned();
        if !is_safe_relative_path(&rel_path) {
            return Err(anyhow!(
                "unsafe path in bundle entry: {}",
                rel_path.to_string_lossy()
            ));
        }
        let target = map_bundle_entry_to_vectors_target(out_root, &rel_path);
        let Some(target_path) = target else {
            continue;
        };

        if let Some(parent) = target_path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create extraction directory {}",
                    parent.to_string_lossy()
                )
            })?;
        }
        entry.unpack(&target_path).with_context(|| {
            format!(
                "failed to extract bundle entry {} to {}",
                rel_path.display(),
                target_path.display()
            )
        })?;
        extracted_files += 1;
    }

    if extracted_files == 0 {
        return Err(anyhow!(
            "bundle {} does not contain vectors payload",
            bundle_path.display()
        ));
    }
    Ok(())
}

fn map_bundle_entry_to_vectors_target(out_root: &Path, rel_path: &Path) -> Option<PathBuf> {
    if rel_path == Path::new("bundle_manifest.json") {
        return Some(out_root.join("bundle_manifest.json"));
    }
    let mut components = rel_path.components();
    let first = components.next()?;
    if first != Component::Normal("vectors".as_ref()) {
        return None;
    }
    let rest: PathBuf = components.collect();
    Some(out_root.join(VECTORS_ROOT_DIRNAME).join(rest))
}

fn is_safe_relative_path(path: &Path) -> bool {
    if path.is_absolute() {
        return false;
    }
    !path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
}

fn shorten_for_progress(value: &str, max_len: usize) -> String {
    if max_len <= 3 {
        return value.to_string();
    }
    let char_count = value.chars().count();
    if char_count <= max_len {
        return value.to_string();
    }
    let keep = max_len - 3;
    let tail: String = value.chars().skip(char_count - keep).collect();
    format!("...{tail}")
}
