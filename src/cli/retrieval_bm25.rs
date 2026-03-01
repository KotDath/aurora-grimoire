use anyhow::{Context, Result, anyhow};
use chrono::{SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha1::{Digest, Sha1};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use tantivy::{
    Index, ReloadPolicy, TantivyDocument, Term, doc,
    query::{BooleanQuery, Occur, Query, QueryParser, TermQuery},
    schema::{
        Field, IndexRecordOption, STORED, STRING, Schema, TextFieldIndexing, TextOptions,
        Value as TantivyValueTrait,
    },
    tokenizer::{LowerCaser, RemoveLongFilter, SimpleTokenizer, TextAnalyzer},
};

const INDEX_DIRNAME: &str = "index";
const META_FILENAME: &str = "index_meta.json";
const BUILD_VERSION: u32 = 1;
const CHUNK_FILE_PREFIX: &str = "chunks-";

#[derive(Debug, Clone)]
pub struct Bm25SearchHit {
    pub id: String,
    pub score: f32,
    pub rank: usize,
    pub payload: Value,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
struct ChunksSnapshot {
    chunks_manifest_sha1: Option<String>,
    chunks_manifest_mtime_ms: Option<u128>,
    chunks_files_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct Bm25IndexMeta {
    build_version: u32,
    built_at: String,
    indexed_docs: usize,
    snapshot: ChunksSnapshot,
}

#[derive(Debug, Clone, Copy)]
struct IndexFields {
    chunk_id: Field,
    version_bucket: Field,
    source_title: Field,
    heading_path: Field,
    content: Field,
    payload_json: Field,
}

pub fn search_chunks(
    query: &str,
    chunks_root: &Path,
    bm25_root: &Path,
    doc_version: Option<&str>,
    limit: usize,
) -> Result<Vec<Bm25SearchHit>> {
    if query.trim().is_empty() {
        return Ok(Vec::new());
    }

    let index = ensure_index(chunks_root, bm25_root)?;
    let fields = resolve_fields(&index)?;

    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()
        .context("failed to build BM25 index reader")?;
    let searcher = reader.searcher();

    let query = build_query(&index, fields, query, doc_version)?;
    let top_docs = searcher
        .search(
            &query,
            &tantivy::collector::TopDocs::with_limit(limit.max(1)),
        )
        .context("failed to execute BM25 search")?;

    let mut out = Vec::with_capacity(top_docs.len());
    for (idx, (score, doc_addr)) in top_docs.into_iter().enumerate() {
        let doc: TantivyDocument = searcher
            .doc(doc_addr)
            .context("failed to load BM25 document")?;
        let id = doc
            .get_first(fields.chunk_id)
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        if id.is_empty() {
            continue;
        }

        let payload = doc
            .get_first(fields.payload_json)
            .and_then(|v| v.as_str())
            .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
            .unwrap_or_else(|| Value::Object(Default::default()));

        out.push(Bm25SearchHit {
            id,
            score,
            rank: idx + 1,
            payload,
        });
    }

    Ok(out)
}

fn ensure_index(chunks_root: &Path, bm25_root: &Path) -> Result<Index> {
    if !chunks_root.exists() {
        return Err(anyhow!(
            "chunks directory does not exist for BM25 indexing: {}",
            chunks_root.display()
        ));
    }

    fs::create_dir_all(bm25_root)
        .with_context(|| format!("failed to create BM25 root {}", bm25_root.display()))?;

    let snapshot = compute_chunks_snapshot(chunks_root)?;
    let index_dir = bm25_root.join(INDEX_DIRNAME);
    let meta_path = bm25_root.join(META_FILENAME);

    if index_dir.exists() && is_index_fresh(&meta_path, &snapshot)? {
        return open_index(&index_dir);
    }

    rebuild_index(chunks_root, bm25_root, &snapshot)?;
    open_index(&index_dir)
}

fn rebuild_index(chunks_root: &Path, bm25_root: &Path, snapshot: &ChunksSnapshot) -> Result<()> {
    let tmp_dir = bm25_root.join(format!(
        "{}.tmp-{}",
        INDEX_DIRNAME,
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    ));
    if tmp_dir.exists() {
        fs::remove_dir_all(&tmp_dir).with_context(|| {
            format!(
                "failed to cleanup stale BM25 temp directory {}",
                tmp_dir.display()
            )
        })?;
    }
    fs::create_dir_all(&tmp_dir)
        .with_context(|| format!("failed to create BM25 temp directory {}", tmp_dir.display()))?;

    let build_result = build_index_in_dir(chunks_root, &tmp_dir);
    let indexed_docs = match build_result {
        Ok(v) => v,
        Err(err) => {
            let _ = fs::remove_dir_all(&tmp_dir);
            return Err(err);
        }
    };

    let index_dir = bm25_root.join(INDEX_DIRNAME);
    if index_dir.exists() {
        fs::remove_dir_all(&index_dir).with_context(|| {
            format!(
                "failed to remove previous BM25 index directory {}",
                index_dir.display()
            )
        })?;
    }
    fs::rename(&tmp_dir, &index_dir).with_context(|| {
        format!(
            "failed to move BM25 index {} -> {}",
            tmp_dir.display(),
            index_dir.display()
        )
    })?;

    let meta = Bm25IndexMeta {
        build_version: BUILD_VERSION,
        built_at: now_rfc3339(),
        indexed_docs,
        snapshot: snapshot.clone(),
    };
    let meta_path = bm25_root.join(META_FILENAME);
    let bytes = serde_json::to_vec_pretty(&meta).context("failed to serialize BM25 index meta")?;
    fs::write(&meta_path, bytes).with_context(|| {
        format!(
            "failed to write BM25 index meta file {}",
            meta_path.display()
        )
    })?;

    Ok(())
}

fn build_index_in_dir(chunks_root: &Path, index_dir: &Path) -> Result<usize> {
    let (schema, fields) = build_schema();
    let index = Index::create_in_dir(index_dir, schema).with_context(|| {
        format!(
            "failed to create BM25 index in directory {}",
            index_dir.display()
        )
    })?;
    register_tokenizer(&index);

    let mut writer = index
        .writer(100_000_000)
        .context("failed to create BM25 index writer")?;

    let chunk_files = discover_chunk_jsonl_files(chunks_root)?;
    if chunk_files.is_empty() {
        return Err(anyhow!(
            "no chunk files were found for BM25 indexing in {}",
            chunks_root.display()
        ));
    }

    let mut indexed_docs = 0usize;
    for path in &chunk_files {
        let file = File::open(path)
            .with_context(|| format!("failed to open chunk file {}", path.display()))?;
        let reader = BufReader::new(file);
        for (line_idx, line) in reader.lines().enumerate() {
            let line = line.with_context(|| {
                format!(
                    "failed to read line {} from {}",
                    line_idx + 1,
                    path.display()
                )
            })?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let payload: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let Some(obj) = payload.as_object() else {
                continue;
            };

            let id = obj.get("id").and_then(Value::as_str).unwrap_or("").trim();
            if id.is_empty() {
                continue;
            }

            let version_bucket = obj
                .get("version_bucket")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let source_title = obj
                .get("source_title")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let heading_path = obj
                .get("heading_path")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_str)
                        .map(str::trim)
                        .filter(|v| !v.is_empty())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default();
            let content = obj
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let payload_json =
                serde_json::to_string(obj).context("failed to serialize chunk payload")?;

            writer
                .add_document(doc!(
                    fields.chunk_id => id,
                    fields.version_bucket => version_bucket,
                    fields.source_title => source_title,
                    fields.heading_path => heading_path,
                    fields.content => content,
                    fields.payload_json => payload_json
                ))
                .context("failed to add BM25 document to index")?;
            indexed_docs += 1;
        }
    }

    writer
        .commit()
        .context("failed to commit BM25 index writer")?;
    writer
        .wait_merging_threads()
        .context("failed to finalize BM25 index merges")?;
    Ok(indexed_docs)
}

fn build_query(
    index: &Index,
    fields: IndexFields,
    query_text: &str,
    doc_version: Option<&str>,
) -> Result<Box<dyn Query>> {
    let mut parser = QueryParser::for_index(
        index,
        vec![fields.content, fields.source_title, fields.heading_path],
    );
    parser.set_field_boost(fields.source_title, 2.0);
    parser.set_field_boost(fields.heading_path, 1.6);

    let parsed = parser.parse_query(query_text).or_else(|_| {
        let sanitized = sanitize_query_text(query_text);
        parser.parse_query(&sanitized)
    });
    let query = parsed.context("failed to parse BM25 query")?;

    let mut clauses: Vec<(Occur, Box<dyn Query>)> = vec![(Occur::Must, query)];
    if let Some(version) = doc_version.map(str::trim).filter(|v| !v.is_empty()) {
        let version_term = Term::from_field_text(fields.version_bucket, version);
        let version_query = TermQuery::new(version_term, IndexRecordOption::Basic);
        clauses.push((Occur::Must, Box::new(version_query)));
    }
    Ok(Box::new(BooleanQuery::new(clauses)))
}

fn sanitize_query_text(raw: &str) -> String {
    raw.chars()
        .map(|ch| match ch {
            ':' | '(' | ')' | '[' | ']' | '{' | '}' | '"' | '\\' | '^' | '~' | '*' | '?' | '!'
            | '+' | '-' => ' ',
            _ => ch,
        })
        .collect::<String>()
}

fn open_index(index_dir: &Path) -> Result<Index> {
    let index = Index::open_in_dir(index_dir).with_context(|| {
        format!(
            "failed to open BM25 index directory {}",
            index_dir.display()
        )
    })?;
    register_tokenizer(&index);
    Ok(index)
}

fn register_tokenizer(index: &Index) {
    let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(LowerCaser)
        .filter(RemoveLongFilter::limit(80))
        .build();
    index.tokenizers().register("aurora_text", analyzer);
}

fn resolve_fields(index: &Index) -> Result<IndexFields> {
    let schema = index.schema();
    Ok(IndexFields {
        chunk_id: schema
            .get_field("chunk_id")
            .context("BM25 schema missing field: chunk_id")?,
        version_bucket: schema
            .get_field("version_bucket")
            .context("BM25 schema missing field: version_bucket")?,
        source_title: schema
            .get_field("source_title")
            .context("BM25 schema missing field: source_title")?,
        heading_path: schema
            .get_field("heading_path")
            .context("BM25 schema missing field: heading_path")?,
        content: schema
            .get_field("content")
            .context("BM25 schema missing field: content")?,
        payload_json: schema
            .get_field("payload_json")
            .context("BM25 schema missing field: payload_json")?,
    })
}

fn build_schema() -> (Schema, IndexFields) {
    let mut builder = Schema::builder();

    let string_stored = STRING | STORED;
    let text_stored = TextOptions::default().set_stored().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("aurora_text")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    );

    let chunk_id = builder.add_text_field("chunk_id", string_stored.clone());
    let version_bucket = builder.add_text_field("version_bucket", string_stored);
    let source_title = builder.add_text_field("source_title", text_stored.clone());
    let heading_path = builder.add_text_field("heading_path", text_stored.clone());
    let content = builder.add_text_field("content", text_stored);
    let payload_json = builder.add_text_field("payload_json", STORED);

    let schema = builder.build();
    (
        schema,
        IndexFields {
            chunk_id,
            version_bucket,
            source_title,
            heading_path,
            content,
            payload_json,
        },
    )
}

fn is_index_fresh(meta_path: &Path, snapshot: &ChunksSnapshot) -> Result<bool> {
    if !meta_path.exists() {
        return Ok(false);
    }
    let file = File::open(meta_path)
        .with_context(|| format!("failed to open BM25 index meta {}", meta_path.display()))?;
    let meta: Bm25IndexMeta = serde_json::from_reader(file)
        .with_context(|| format!("failed to parse BM25 index meta {}", meta_path.display()))?;
    Ok(meta.build_version == BUILD_VERSION && meta.snapshot == *snapshot)
}

fn compute_chunks_snapshot(chunks_root: &Path) -> Result<ChunksSnapshot> {
    let manifest_path = chunks_root.join("manifest.json");
    let (manifest_sha1, manifest_mtime_ms) = if manifest_path.exists() {
        let mut file = File::open(&manifest_path)
            .with_context(|| format!("failed to open {}", manifest_path.display()))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .with_context(|| format!("failed to read {}", manifest_path.display()))?;
        let mut hasher = Sha1::new();
        hasher.update(&bytes);
        let hash = format!("{:x}", hasher.finalize());
        let mtime = fs::metadata(&manifest_path)
            .with_context(|| format!("failed to stat {}", manifest_path.display()))?
            .modified()
            .ok()
            .and_then(system_time_to_millis);
        (Some(hash), mtime)
    } else {
        (None, None)
    };

    let chunk_files_count = discover_chunk_jsonl_files(chunks_root)?.len();
    Ok(ChunksSnapshot {
        chunks_manifest_sha1: manifest_sha1,
        chunks_manifest_mtime_ms: manifest_mtime_ms,
        chunks_files_count: chunk_files_count,
    })
}

fn discover_chunk_jsonl_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let entries = fs::read_dir(root)
        .with_context(|| format!("failed to read chunks directory {}", root.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("failed to read entry in {}", root.display()))?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|v| v.to_str()) else {
            continue;
        };
        if name.starts_with(CHUNK_FILE_PREFIX) && name.ends_with(".jsonl") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn now_rfc3339() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

fn system_time_to_millis(t: SystemTime) -> Option<u128> {
    t.duration_since(UNIX_EPOCH).ok().map(|d| d.as_millis())
}
