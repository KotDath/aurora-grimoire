use super::RagEmbedArgs;
use anyhow::{Context, Result, anyhow};
use chrono::{SecondsFormat, Utc};
use indicatif::{ProgressBar, ProgressStyle};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

const CHUNKS_ROOT_DIRNAME: &str = "chunks";
const VECTORS_ROOT_DIRNAME: &str = "vectors_data";
const CHUNK_FILE_PREFIX: &str = "chunks-";
const VECTOR_FILE_PREFIX: &str = "vectors-";
const SHARD_MAX_RECORDS: usize = 5000;

macro_rules! vprintln {
    ($verbose:expr, $($arg:tt)*) => {
        if $verbose {
            println!($($arg)*);
        }
    };
}

#[derive(Debug)]
struct PendingChunk {
    id: String,
    content: String,
    payload: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct VectorRecord {
    id: String,
    vector: Vec<f32>,
    payload: Value,
    embedding_provider: String,
    embedding_model: String,
    embedding_dim: usize,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct ExistingVectorId {
    id: String,
}

#[derive(Debug, Serialize)]
struct EmbedManifest {
    started_at: String,
    finished_at: String,
    input_root: String,
    output_root: String,
    ollama_url: String,
    embedding_model: String,
    batch_size: usize,
    workers: usize,
    chunks_total: usize,
    chunks_embedded: usize,
    chunks_skipped_existing: usize,
    vectors_total: usize,
    shards_total: usize,
    errors_total: usize,
    embedding_dim: Option<usize>,
}

pub fn run(args: RagEmbedArgs) -> Result<()> {
    let verbose = args.verbose;
    let batch_size = args.batch_size.max(1);
    let workers = args.workers.max(1);

    let home_dir =
        dirs::home_dir().ok_or_else(|| anyhow!("failed to resolve home directory for embed"))?;
    let input_root = args
        .input
        .unwrap_or_else(|| home_dir.join(".aurora-grimoire").join(CHUNKS_ROOT_DIRNAME));
    let output_root = args
        .output
        .unwrap_or_else(|| home_dir.join(".aurora-grimoire").join(VECTORS_ROOT_DIRNAME));

    if !input_root.exists() {
        return Err(anyhow!(
            "chunk input directory does not exist: {}",
            input_root.display()
        ));
    }
    fs::create_dir_all(&output_root).with_context(|| {
        format!(
            "failed to create vectors output directory: {}",
            output_root.display()
        )
    })?;

    if !args.resume {
        cleanup_vectors_output_dir(&output_root)?;
    }

    let chunk_files = discover_jsonl_files(&input_root, CHUNK_FILE_PREFIX)?;
    if chunk_files.is_empty() {
        return Err(anyhow!(
            "no chunk shards found in {}",
            input_root.to_string_lossy()
        ));
    }
    let chunks_total = count_lines(&chunk_files)?;

    let existing_ids = if args.resume {
        load_existing_vector_ids(&output_root)?
    } else {
        HashSet::new()
    };
    let skipped_existing = existing_ids.len();
    let existing_shards = discover_jsonl_files(&output_root, VECTOR_FILE_PREFIX)?;
    let mut next_shard_index = existing_shards
        .iter()
        .filter_map(|path| shard_index_from_path(path, VECTOR_FILE_PREFIX))
        .max()
        .unwrap_or(0);

    vprintln!(
        verbose,
        "[embed] input={} output={} total_chunks={} resume={} existing_vectors={}",
        input_root.display(),
        output_root.display(),
        chunks_total,
        args.resume,
        skipped_existing
    );

    let progress = if verbose {
        let pb = ProgressBar::new(chunks_total as u64);
        let style =
            ProgressStyle::with_template("[embed] [{bar:28.green/white}] {pos}/{len}: {msg}")
                .expect("progress template must be valid")
                .progress_chars("=> ");
        pb.set_style(style);
        Some(pb)
    } else {
        None
    };

    let started_at = now_rfc3339();
    let mut writer: Option<BufWriter<File>> = None;
    let mut records_in_current_shard = 0usize;
    let mut chunks_embedded = 0usize;
    let mut embedding_dim: Option<usize> = None;

    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .context("failed to build HTTP client for ollama")?;

    let mut pending_window: Vec<PendingChunk> = Vec::new();
    let window_target = batch_size * workers;

    for chunk_file in &chunk_files {
        let file = File::open(chunk_file)
            .with_context(|| format!("failed to open chunk shard: {}", chunk_file.display()))?;
        let reader = BufReader::new(file);
        for line_result in reader.lines() {
            let line = line_result.with_context(|| {
                format!(
                    "failed to read line in chunk shard {}",
                    chunk_file.display()
                )
            })?;
            if line.trim().is_empty() {
                if let Some(pb) = &progress {
                    pb.inc(1);
                }
                continue;
            }

            let payload: Value = serde_json::from_str(&line).with_context(|| {
                format!(
                    "failed to parse chunk JSON line in shard {}",
                    chunk_file.display()
                )
            })?;
            let id = payload
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("chunk record missing string id"))?
                .to_string();
            let content = payload
                .get("content")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("chunk record missing string content"))?
                .to_string();

            if let Some(pb) = &progress {
                pb.set_message(shorten_for_progress(&id, 56));
                pb.inc(1);
            }

            if args.resume && existing_ids.contains(&id) {
                continue;
            }

            pending_window.push(PendingChunk {
                id,
                content,
                payload,
            });

            if pending_window.len() >= window_target {
                let produced = process_window(
                    &client,
                    &args.ollama_url,
                    &args.model,
                    batch_size,
                    workers,
                    std::mem::take(&mut pending_window),
                )?;
                for mut record in produced {
                    let dim = record.vector.len();
                    if let Some(expected) = embedding_dim {
                        if expected != dim {
                            return Err(anyhow!(
                                "embedding dimension mismatch: expected {expected}, got {dim} for id {}",
                                record.id
                            ));
                        }
                    } else {
                        embedding_dim = Some(dim);
                    }

                    if writer.is_none() || records_in_current_shard >= SHARD_MAX_RECORDS {
                        if let Some(mut current) = writer.take() {
                            current
                                .flush()
                                .context("failed to flush vector shard writer")?;
                        }
                        next_shard_index += 1;
                        writer = Some(open_shard_writer(&output_root, next_shard_index)?);
                        records_in_current_shard = 0;
                    }
                    record.created_at = now_rfc3339();
                    write_jsonl_record(
                        writer
                            .as_mut()
                            .expect("vector shard writer must exist before write"),
                        &record,
                    )?;
                    chunks_embedded += 1;
                    records_in_current_shard += 1;
                }
            }
        }
    }

    if !pending_window.is_empty() {
        let produced = process_window(
            &client,
            &args.ollama_url,
            &args.model,
            batch_size,
            workers,
            pending_window,
        )?;
        for mut record in produced {
            let dim = record.vector.len();
            if let Some(expected) = embedding_dim {
                if expected != dim {
                    return Err(anyhow!(
                        "embedding dimension mismatch: expected {expected}, got {dim} for id {}",
                        record.id
                    ));
                }
            } else {
                embedding_dim = Some(dim);
            }
            if writer.is_none() || records_in_current_shard >= SHARD_MAX_RECORDS {
                if let Some(mut current) = writer.take() {
                    current
                        .flush()
                        .context("failed to flush vector shard writer")?;
                }
                next_shard_index += 1;
                writer = Some(open_shard_writer(&output_root, next_shard_index)?);
                records_in_current_shard = 0;
            }
            record.created_at = now_rfc3339();
            write_jsonl_record(
                writer
                    .as_mut()
                    .expect("vector shard writer must exist before write"),
                &record,
            )?;
            chunks_embedded += 1;
            records_in_current_shard += 1;
        }
    }

    if let Some(mut current) = writer {
        current
            .flush()
            .context("failed to flush final vector shard writer")?;
    }

    if let Some(pb) = progress {
        pb.finish_and_clear();
    }

    let vector_files = discover_jsonl_files(&output_root, VECTOR_FILE_PREFIX)?;
    let vectors_total = if args.resume {
        existing_ids.len() + chunks_embedded
    } else {
        chunks_embedded
    };
    let manifest = EmbedManifest {
        started_at,
        finished_at: now_rfc3339(),
        input_root: input_root.to_string_lossy().to_string(),
        output_root: output_root.to_string_lossy().to_string(),
        ollama_url: args.ollama_url,
        embedding_model: args.model,
        batch_size,
        workers,
        chunks_total,
        chunks_embedded,
        chunks_skipped_existing: skipped_existing,
        vectors_total,
        shards_total: vector_files.len(),
        errors_total: 0,
        embedding_dim,
    };
    write_manifest(&output_root, &manifest)?;

    println!(
        "{} embeddings created and stored in {}",
        chunks_embedded,
        output_root.display()
    );
    Ok(())
}

fn process_window(
    client: &Client,
    ollama_url: &str,
    model: &str,
    batch_size: usize,
    workers: usize,
    items: Vec<PendingChunk>,
) -> Result<Vec<VectorRecord>> {
    if items.is_empty() {
        return Ok(Vec::new());
    }

    let mut batches: Vec<Vec<PendingChunk>> = Vec::new();
    let mut current = Vec::new();
    for item in items {
        current.push(item);
        if current.len() >= batch_size {
            batches.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        batches.push(current);
    }

    let mut out = Vec::new();
    let mut idx = 0usize;
    while idx < batches.len() {
        let take = workers.min(batches.len() - idx);
        let mut handles = Vec::with_capacity(take);
        for _ in 0..take {
            let batch = std::mem::take(&mut batches[idx]);
            idx += 1;
            let batch_client = client.clone();
            let batch_url = ollama_url.to_string();
            let batch_model = model.to_string();
            handles.push(thread::spawn(move || {
                process_batch(&batch_client, &batch_url, &batch_model, batch)
            }));
        }
        for handle in handles {
            let result = handle
                .join()
                .map_err(|_| anyhow!("embedding worker panicked"))??;
            out.extend(result);
        }
    }
    Ok(out)
}

fn process_batch(
    client: &Client,
    ollama_url: &str,
    model: &str,
    batch: Vec<PendingChunk>,
) -> Result<Vec<VectorRecord>> {
    let texts: Vec<String> = batch.iter().map(|item| item.content.clone()).collect();
    let vectors = embed_batch(client, ollama_url, model, &texts)?;
    if vectors.len() != batch.len() {
        return Err(anyhow!(
            "embedding response size mismatch: got {} vectors for {} texts",
            vectors.len(),
            batch.len()
        ));
    }

    let mut records = Vec::with_capacity(batch.len());
    for (item, vector) in batch.into_iter().zip(vectors.into_iter()) {
        records.push(VectorRecord {
            id: item.id,
            embedding_dim: vector.len(),
            vector,
            payload: item.payload,
            embedding_provider: "ollama".to_string(),
            embedding_model: model.to_string(),
            created_at: String::new(),
        });
    }
    Ok(records)
}

fn embed_batch(
    client: &Client,
    ollama_url: &str,
    model: &str,
    texts: &[String],
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    let base = ollama_url.trim_end_matches('/');
    let embed_url = format!("{base}/api/embed");
    let response = client
        .post(&embed_url)
        .json(&json!({
            "model": model,
            "input": texts,
        }))
        .send();

    if let Ok(resp) = response {
        if resp.status().is_success() {
            let value: Value = resp
                .json()
                .context("failed to parse ollama /api/embed response")?;
            if let Some(arr) = value.get("embeddings").and_then(Value::as_array) {
                let mut out = Vec::with_capacity(arr.len());
                for row in arr {
                    out.push(parse_vector(row)?);
                }
                return Ok(out);
            }
            if let Some(single) = value.get("embedding") {
                return Ok(vec![parse_vector(single)?]);
            }
        }
    }

    // Fallback for older Ollama API surface.
    let old_url = format!("{base}/api/embeddings");
    let mut out = Vec::with_capacity(texts.len());
    for text in texts {
        let resp = client
            .post(&old_url)
            .json(&json!({
                "model": model,
                "prompt": text,
            }))
            .send()
            .with_context(|| format!("failed to call ollama embeddings endpoint at {old_url}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(anyhow!(
                "ollama embeddings request failed: status={} body={}",
                status,
                body
            ));
        }
        let value: Value = resp
            .json()
            .context("failed to parse ollama /api/embeddings response")?;
        let vec_value = value
            .get("embedding")
            .ok_or_else(|| anyhow!("ollama /api/embeddings response missing embedding field"))?;
        out.push(parse_vector(vec_value)?);
    }
    Ok(out)
}

fn parse_vector(value: &Value) -> Result<Vec<f32>> {
    let arr = value
        .as_array()
        .ok_or_else(|| anyhow!("embedding vector is not an array"))?;
    if arr.is_empty() {
        return Err(anyhow!("embedding vector is empty"));
    }
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        let number = v
            .as_f64()
            .ok_or_else(|| anyhow!("embedding element is not numeric"))?;
        out.push(number as f32);
    }
    Ok(out)
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

fn load_existing_vector_ids(output_root: &Path) -> Result<HashSet<String>> {
    let mut ids = HashSet::new();
    let files = discover_jsonl_files(output_root, VECTOR_FILE_PREFIX)?;
    for path in files {
        let file = File::open(&path)
            .with_context(|| format!("failed to open existing vector shard {}", path.display()))?;
        let reader = BufReader::new(file);
        for line_result in reader.lines() {
            let line = line_result
                .with_context(|| format!("failed reading vector shard line {}", path.display()))?;
            if line.trim().is_empty() {
                continue;
            }
            let record: ExistingVectorId = serde_json::from_str(&line).with_context(|| {
                format!(
                    "failed to parse existing vector record for resume from {}",
                    path.display()
                )
            })?;
            ids.insert(record.id);
        }
    }
    Ok(ids)
}

fn cleanup_vectors_output_dir(output_root: &Path) -> Result<()> {
    let entries = fs::read_dir(output_root)
        .with_context(|| format!("failed to read output directory {}", output_root.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| {
            format!(
                "failed to read output directory entry in {}",
                output_root.display()
            )
        })?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
            continue;
        };
        if name.starts_with(VECTOR_FILE_PREFIX) && name.ends_with(".jsonl") {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
            continue;
        }
        if name == "manifest.json" && path.is_file() {
            fs::remove_file(&path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
        }
    }
    Ok(())
}

fn open_shard_writer(output_root: &Path, shard_index: usize) -> Result<BufWriter<File>> {
    let path = output_root.join(format!("{VECTOR_FILE_PREFIX}{shard_index:05}.jsonl"));
    let file = File::create(&path)
        .with_context(|| format!("failed to create vector shard {}", path.display()))?;
    Ok(BufWriter::new(file))
}

fn write_jsonl_record<T: Serialize>(writer: &mut BufWriter<File>, record: &T) -> Result<()> {
    serde_json::to_writer(&mut *writer, record).context("failed to serialize vector JSONL line")?;
    writer
        .write_all(b"\n")
        .context("failed to write vector JSONL newline")?;
    Ok(())
}

fn write_manifest(output_root: &Path, manifest: &EmbedManifest) -> Result<()> {
    let path = output_root.join("manifest.json");
    let file = File::create(&path)
        .with_context(|| format!("failed to create embed manifest {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, manifest)
        .context("failed to serialize embed manifest")?;
    writer
        .write_all(b"\n")
        .context("failed to write newline after embed manifest")?;
    writer.flush().context("failed to flush embed manifest")?;
    Ok(())
}

fn shard_index_from_path(path: &Path, prefix: &str) -> Option<usize> {
    let name = path.file_name()?.to_str()?;
    if !name.starts_with(prefix) || !name.ends_with(".jsonl") {
        return None;
    }
    let middle = name.strip_prefix(prefix)?.strip_suffix(".jsonl")?;
    middle.parse::<usize>().ok()
}

fn now_rfc3339() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
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
