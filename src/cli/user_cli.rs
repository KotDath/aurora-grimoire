use crate::cli::retrieval_bm25;
use crate::config::{
    AppConfig, DEFAULT_COLLECTION, DEFAULT_EMBED_MODEL, DEFAULT_OLLAMA_URL, DEFAULT_QDRANT_URL,
    DEFAULT_RERANK_MODEL, DEFAULT_RERANK_URL, DEFAULT_SEARCH_BM25_TOP_N,
    DEFAULT_SEARCH_BM25_WEIGHT, DEFAULT_SEARCH_DENSE_WEIGHT, DEFAULT_SEARCH_RERANK_ENABLED,
    DEFAULT_SEARCH_RERANK_FAIL_OPEN, DEFAULT_SEARCH_RERANK_TIMEOUT_MS, DEFAULT_SEARCH_RRF_K,
    DEFAULT_SEARCH_SCORE_THRESHOLD, DEFAULT_SEARCH_TOP_K, DEFAULT_SEARCH_TOP_N,
};
use anyhow::{Context, Result, anyhow};
use clap::{Args, Subcommand, ValueEnum};
use reqwest::{Method, StatusCode, blocking::Client};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    env,
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Debug, Args)]
pub struct CliArgs {
    #[command(subcommand)]
    pub command: CliCommand,
}

#[derive(Debug, Subcommand)]
pub enum CliCommand {
    /// Search the documentation vector index
    #[command(name = "search_docs", visible_alias = "search-docs")]
    SearchDocs(SearchDocsArgs),
}

#[derive(Debug, Args)]
pub struct SearchDocsArgs {
    /// Search query text
    pub query: String,

    /// Number of relevant chunks to return
    #[arg(long)]
    pub top_k: Option<usize>,

    /// Relevance threshold [0..1]
    #[arg(long)]
    pub score_threshold: Option<f32>,

    /// Return JSON instead of text
    #[arg(long, default_value_t = false)]
    pub json: bool,

    /// Include full chunk text in results
    #[arg(long, default_value_t = false)]
    pub with_content: bool,

    /// Return compact LLM-friendly context segments (full cleaned content + sources)
    #[arg(long, default_value_t = false)]
    pub with_context: bool,

    /// Qdrant base URL
    #[arg(long)]
    pub qdrant_url: Option<String>,

    /// Qdrant API key (or use QDRANT_API_KEY env)
    #[arg(long)]
    pub api_key: Option<String>,

    /// Qdrant collection name
    #[arg(long)]
    pub collection: Option<String>,

    /// Ollama base URL
    #[arg(long)]
    pub ollama_url: Option<String>,

    /// Embedding model name
    #[arg(long)]
    pub model: Option<String>,

    /// Filter results by documentation version bucket (e.g. 5.2.0, default)
    #[arg(long = "doc-version")]
    pub doc_version: Option<String>,

    /// Enable second-stage rerank
    #[arg(long, default_value_t = false)]
    pub rerank: bool,

    /// Disable second-stage rerank
    #[arg(long, default_value_t = false)]
    pub no_rerank: bool,

    /// Retrieval mode
    #[arg(long, value_enum)]
    pub retrieval_mode: Option<RetrievalMode>,

    /// Minimum confidence required to return sources
    #[arg(long, value_enum)]
    pub knowledge_threshold: Option<ConfidenceLevel>,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RetrievalMode {
    Hybrid,
    Dense,
    Bm25,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ConfidenceLevel {
    Low,
    Medium,
    High,
}

pub fn run(args: CliArgs) {
    match args.command {
        CliCommand::SearchDocs(search) => {
            if let Err(err) = run_search_docs(search) {
                eprintln!("[error] cli search_docs failed: {err:#}");
                std::process::exit(1);
            }
        }
    }
}

#[derive(Debug, Serialize)]
struct SearchResultOutput {
    id: String,
    score: f32,
    retrieval_score: f32,
    rerank_score: Option<f32>,
    retrieval_rank: usize,
    dense_rank: Option<usize>,
    bm25_rank: Option<usize>,
    rrf_score: Option<f32>,
    rerank_rank: Option<usize>,
    version_bucket: Option<String>,
    source_title: Option<String>,
    source_url: Option<String>,
    source_url_with_anchor: Option<String>,
    section_anchor: Option<String>,
    source_md: Option<String>,
    heading_path: Option<Vec<String>>,
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct SearchResponseOutput {
    query: String,
    retrieval_query: String,
    collection: String,
    retrieval_mode: RetrievalMode,
    fusion_method: Option<String>,
    top_k: usize,
    top_n: usize,
    bm25_top_n: usize,
    rrf_k: usize,
    dense_weight: f32,
    bm25_weight: f32,
    score_threshold: f32,
    doc_version: Option<String>,
    bm25_applied: bool,
    answer_confidence: String,
    advisory: Option<String>,
    knowledge_threshold: ConfidenceLevel,
    no_knowledge: bool,
    no_knowledge_message: Option<String>,
    rerank_enabled: bool,
    rerank_applied: bool,
    rerank_model: Option<String>,
    retrieval_latency_ms: u128,
    dense_latency_ms: Option<u128>,
    bm25_latency_ms: Option<u128>,
    rerank_latency_ms: Option<u128>,
    results: Vec<SearchResultOutput>,
}

#[derive(Debug, Serialize)]
struct ContextSegmentOutput {
    source_title: Option<String>,
    source_url: Option<String>,
    source_url_with_anchor: Option<String>,
    section_anchor: Option<String>,
    source_md: Option<String>,
    version_bucket: Option<String>,
    heading_path: Option<Vec<String>>,
    content: String,
}

#[derive(Debug, Serialize)]
struct SearchContextResponseOutput {
    query: String,
    doc_version: Option<String>,
    knowledge_threshold: ConfidenceLevel,
    no_knowledge: bool,
    no_knowledge_message: Option<String>,
    contexts: Vec<ContextSegmentOutput>,
}

#[derive(Debug, Clone)]
struct SearchCandidate {
    id: String,
    retrieval_score: f32,
    dense_rank: Option<usize>,
    bm25_rank: Option<usize>,
    rrf_score: Option<f32>,
    rerank_score: Option<f32>,
    retrieval_rank: usize,
    rerank_rank: Option<usize>,
    payload: Value,
}

#[derive(Debug, Clone, Copy)]
struct RerankScore {
    index: usize,
    relevance_score: f32,
}

#[derive(Debug, Clone)]
struct SearchRuntimeConfig {
    top_k: usize,
    score_threshold: f32,
    qdrant_url: String,
    collection: String,
    ollama_url: String,
    model: String,
    retrieval_mode: RetrievalMode,
    knowledge_threshold: ConfidenceLevel,
    top_n: usize,
    bm25_top_n: usize,
    rrf_k: usize,
    dense_weight: f32,
    bm25_weight: f32,
    rerank_url: String,
    rerank_model: String,
    rerank_timeout_ms: u64,
    rerank_fail_open: bool,
    rerank_default_enabled: bool,
}

fn run_search_docs(args: SearchDocsArgs) -> Result<()> {
    let app_cfg = AppConfig::load()?;
    let runtime = resolve_runtime_config(&args, &app_cfg);
    let query = args.query.clone();
    let doc_version = args.doc_version.clone();
    let qdrant_api_key = args
        .api_key
        .clone()
        .or_else(|| env::var("QDRANT_API_KEY").ok());
    let rerank_api_key = env::var("RERANK_API_KEY").ok();
    let client = Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .context("failed to build HTTP client")?;
    let rerank_client = Client::builder()
        .timeout(Duration::from_millis(runtime.rerank_timeout_ms.max(1)))
        .build()
        .context("failed to build rerank HTTP client")?;

    let top_k = runtime.top_k.max(1);
    let top_n = normalize_top_n(top_k, runtime.top_n);
    let bm25_top_n = normalize_top_n(top_n, runtime.bm25_top_n);
    let dense_weight = sanitize_channel_weight(runtime.dense_weight);
    let bm25_weight = sanitize_channel_weight(runtime.bm25_weight);
    let rerank_enabled =
        resolve_rerank_enabled(args.rerank, args.no_rerank, runtime.rerank_default_enabled);
    let retrieval_query = build_retrieval_query(&query);
    let retrieval_start = Instant::now();
    let mut dense_latency_ms = None;
    let mut bm25_latency_ms = None;
    let mut dense_error: Option<anyhow::Error> = None;
    let mut bm25_error: Option<anyhow::Error> = None;
    let mut dense_candidates: Vec<SearchCandidate> = Vec::new();
    let mut bm25_candidates: Vec<SearchCandidate> = Vec::new();

    if runtime.retrieval_mode != RetrievalMode::Bm25 {
        let dense_start = Instant::now();
        match retrieve_dense_candidates(
            &client,
            &runtime,
            qdrant_api_key.as_deref(),
            top_n,
            doc_version.as_deref(),
            &retrieval_query,
        ) {
            Ok(candidates) => dense_candidates = candidates,
            Err(err) => dense_error = Some(err),
        }
        dense_latency_ms = Some(dense_start.elapsed().as_millis());
    }

    if runtime.retrieval_mode != RetrievalMode::Dense {
        let bm25_start = Instant::now();
        match retrieve_bm25_candidates(
            &app_cfg,
            &args,
            bm25_top_n,
            doc_version.as_deref(),
            &retrieval_query,
        ) {
            Ok(candidates) => bm25_candidates = candidates,
            Err(err) => bm25_error = Some(err),
        }
        bm25_latency_ms = Some(bm25_start.elapsed().as_millis());
    }

    let mut bm25_applied = false;
    let mut candidates = match runtime.retrieval_mode {
        RetrievalMode::Dense => {
            if let Some(err) = dense_error {
                return Err(err.context("dense retrieval failed"));
            }
            dense_candidates
        }
        RetrievalMode::Bm25 => {
            if let Some(err) = bm25_error {
                return Err(err.context("bm25 retrieval failed"));
            }
            bm25_applied = true;
            bm25_candidates
        }
        RetrievalMode::Hybrid => {
            let dense_available = !dense_candidates.is_empty();
            let bm25_available = !bm25_candidates.is_empty();

            match (dense_available, bm25_available) {
                (true, true) => {
                    bm25_applied = true;
                    fuse_rrf(
                        dense_candidates,
                        bm25_candidates,
                        runtime.rrf_k,
                        dense_weight,
                        bm25_weight,
                        top_n,
                    )
                }
                (true, false) => {
                    if let Some(err) = bm25_error {
                        eprintln!(
                            "[warn] BM25 unavailable, fallback to dense retrieval: {}",
                            err
                        );
                    }
                    dense_candidates
                }
                (false, true) => {
                    if let Some(err) = dense_error {
                        eprintln!(
                            "[warn] dense retrieval unavailable, fallback to BM25: {}",
                            err
                        );
                    }
                    bm25_applied = true;
                    bm25_candidates
                }
                (false, false) => {
                    let mut reasons = Vec::new();
                    if let Some(err) = dense_error {
                        reasons.push(format!("dense: {err:#}"));
                    }
                    if let Some(err) = bm25_error {
                        reasons.push(format!("bm25: {err:#}"));
                    }
                    let details = if reasons.is_empty() {
                        "both channels returned no candidates".to_string()
                    } else {
                        reasons.join("; ")
                    };
                    return Err(anyhow!("hybrid retrieval failed: {details}"));
                }
            }
        }
    };

    for (idx, candidate) in candidates.iter_mut().enumerate() {
        candidate.retrieval_rank = idx + 1;
    }
    let retrieval_latency_ms = retrieval_start.elapsed().as_millis();

    let mut rerank_applied = false;
    let mut rerank_latency_ms = None;

    if rerank_enabled && !candidates.is_empty() {
        let retrieval_candidates = candidates.clone();
        let start = Instant::now();
        let rerank_result = rerank_candidates(
            &rerank_client,
            &runtime.rerank_url,
            rerank_api_key.as_deref(),
            &runtime.rerank_model,
            &query,
            candidates,
            top_n,
        );
        rerank_latency_ms = Some(start.elapsed().as_millis());

        match rerank_result {
            Ok(Some(reordered)) => {
                candidates = reordered;
                rerank_applied = true;
            }
            Ok(None) => {
                // No usable documents for rerank; keep retrieval order.
                candidates = retrieval_candidates;
            }
            Err(err) => {
                if runtime.rerank_fail_open {
                    // Keep retrieval order on rerank failure.
                    let _ = err;
                    candidates = retrieval_candidates;
                } else {
                    return Err(err.context("rerank failed"));
                }
            }
        }
    }

    let mut results = candidates
        .into_iter()
        .take(top_k)
        .map(|candidate| map_output_result(candidate, args.with_content || args.with_context))
        .collect::<Result<Vec<_>>>()?;
    let (answer_confidence, advisory) =
        evaluate_answer_quality(&query, doc_version.as_deref(), &results);
    let confidence_level = parse_confidence_level(&answer_confidence);
    let no_knowledge =
        confidence_rank(confidence_level) < confidence_rank(runtime.knowledge_threshold);
    let no_knowledge_message = if no_knowledge {
        Some(build_no_knowledge_message(doc_version.as_deref()))
    } else {
        None
    };
    if no_knowledge {
        results.clear();
    }

    let response = SearchResponseOutput {
        query,
        retrieval_query,
        collection: runtime.collection.clone(),
        retrieval_mode: runtime.retrieval_mode,
        fusion_method: if runtime.retrieval_mode == RetrievalMode::Hybrid {
            Some("rrf".to_string())
        } else {
            None
        },
        top_k,
        top_n,
        bm25_top_n,
        rrf_k: runtime.rrf_k,
        dense_weight,
        bm25_weight,
        score_threshold: runtime.score_threshold,
        doc_version,
        bm25_applied,
        answer_confidence,
        advisory,
        knowledge_threshold: runtime.knowledge_threshold,
        no_knowledge,
        no_knowledge_message,
        rerank_enabled,
        rerank_applied,
        rerank_model: if rerank_enabled {
            Some(runtime.rerank_model.clone())
        } else {
            None
        },
        retrieval_latency_ms,
        dense_latency_ms,
        bm25_latency_ms,
        rerank_latency_ms,
        results,
    };

    if args.with_context {
        let context_response = build_context_response(&response);
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&context_response)
                    .context("failed to serialize context JSON output")?
            );
            return Ok(());
        }

        if context_response.contexts.is_empty() {
            if let Some(message) = &context_response.no_knowledge_message {
                println!("{}", message);
            } else {
                println!("No context found");
            }
            return Ok(());
        }

        println!(
            "{} context segment(s) for query=\"{}\"",
            context_response.contexts.len(),
            context_response.query
        );
        for (idx, segment) in context_response.contexts.iter().enumerate() {
            println!(
                "{}. {}",
                idx + 1,
                segment.source_title.as_deref().unwrap_or("-")
            );
            if let Some(url) = &segment.source_url_with_anchor {
                if !url.is_empty() {
                    println!("   source: {}", url);
                }
            } else if let Some(url) = &segment.source_url {
                if !url.is_empty() {
                    println!("   source: {}", url);
                }
            }
            println!("   content: {}", segment.content.replace('\n', " "));
        }
        return Ok(());
    }

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&response).context("failed to serialize JSON output")?
        );
        return Ok(());
    }

    if response.results.is_empty() {
        if let Some(message) = &response.no_knowledge_message {
            println!("{}", message);
        } else {
            println!("No results found");
        }
        return Ok(());
    }

    println!(
        "{} results from {} (mode={}, retrieval={}ms, dense={}, bm25={}, rerank={} applied={}, confidence={})",
        response.results.len(),
        response.collection,
        retrieval_mode_name(response.retrieval_mode),
        response.retrieval_latency_ms,
        response
            .dense_latency_ms
            .map(|v| format!("{v}ms"))
            .unwrap_or_else(|| "-".to_string()),
        response
            .bm25_latency_ms
            .map(|v| format!("{v}ms"))
            .unwrap_or_else(|| "-".to_string()),
        response
            .rerank_latency_ms
            .map(|v| format!("{v}ms"))
            .unwrap_or_else(|| "-".to_string()),
        response.rerank_applied,
        response.answer_confidence
    );
    if let Some(advisory) = &response.advisory {
        println!("note: {}", advisory);
    }
    for (idx, result) in response.results.iter().enumerate() {
        let rerank_score = result
            .rerank_score
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{}. score={:.4} retrieval={:.4} rerank={} dense_rank={} bm25_rank={} version={} title={}",
            idx + 1,
            result.score,
            result.retrieval_score,
            rerank_score,
            result
                .dense_rank
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
            result
                .bm25_rank
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string()),
            result
                .version_bucket
                .as_deref()
                .filter(|v| !v.is_empty())
                .unwrap_or("-"),
            result
                .source_title
                .as_deref()
                .filter(|v| !v.is_empty())
                .unwrap_or("-")
        );
        if let Some(url) = &result.source_url_with_anchor {
            if !url.is_empty() {
                println!("   source: {}", url);
            }
        } else if let Some(url) = &result.source_url {
            if !url.is_empty() {
                println!("   source: {}", url);
            }
        }
        if let Some(content) = &result.content {
            println!("   content: {}", content.replace('\n', " "));
        }
    }
    Ok(())
}

fn resolve_runtime_config(args: &SearchDocsArgs, app_cfg: &AppConfig) -> SearchRuntimeConfig {
    let top_k = args
        .top_k
        .or(app_cfg.search.top_k)
        .unwrap_or(DEFAULT_SEARCH_TOP_K)
        .max(1);
    let score_threshold = args
        .score_threshold
        .or(app_cfg.search.score_threshold)
        .unwrap_or(DEFAULT_SEARCH_SCORE_THRESHOLD);
    let qdrant_url = args
        .qdrant_url
        .clone()
        .or_else(|| app_cfg.search.qdrant_url.clone())
        .or_else(|| app_cfg.deploy.qdrant_url.clone())
        .unwrap_or_else(|| DEFAULT_QDRANT_URL.to_string());
    let collection = args
        .collection
        .clone()
        .or_else(|| app_cfg.search.collection.clone())
        .or_else(|| app_cfg.deploy.collection.clone())
        .unwrap_or_else(|| DEFAULT_COLLECTION.to_string());
    let ollama_url = args
        .ollama_url
        .clone()
        .or_else(|| app_cfg.search.ollama_url.clone())
        .or_else(|| app_cfg.embed.ollama_url.clone())
        .unwrap_or_else(|| DEFAULT_OLLAMA_URL.to_string());
    let model = args
        .model
        .clone()
        .or_else(|| app_cfg.search.model.clone())
        .or_else(|| app_cfg.embed.model.clone())
        .unwrap_or_else(|| DEFAULT_EMBED_MODEL.to_string());
    let retrieval_mode = args
        .retrieval_mode
        .or_else(|| parse_retrieval_mode(app_cfg.search.retrieval_mode.as_deref()))
        .unwrap_or(RetrievalMode::Hybrid);
    let knowledge_threshold = args
        .knowledge_threshold
        .or_else(|| parse_knowledge_threshold(app_cfg.search.knowledge_threshold.as_deref()))
        .unwrap_or(ConfidenceLevel::Medium);
    let top_n = app_cfg
        .search
        .top_n
        .unwrap_or(DEFAULT_SEARCH_TOP_N)
        .max(top_k);
    let bm25_top_n = app_cfg
        .search
        .bm25_top_n
        .unwrap_or(DEFAULT_SEARCH_BM25_TOP_N)
        .max(top_n);
    let rrf_k = app_cfg.search.rrf_k.unwrap_or(DEFAULT_SEARCH_RRF_K).max(1);
    let dense_weight = app_cfg
        .search
        .dense_weight
        .unwrap_or(DEFAULT_SEARCH_DENSE_WEIGHT);
    let bm25_weight = app_cfg
        .search
        .bm25_weight
        .unwrap_or(DEFAULT_SEARCH_BM25_WEIGHT);
    let rerank_url = app_cfg
        .search
        .rerank_url
        .clone()
        .unwrap_or_else(|| DEFAULT_RERANK_URL.to_string());
    let rerank_model = app_cfg
        .search
        .rerank_model
        .clone()
        .unwrap_or_else(|| DEFAULT_RERANK_MODEL.to_string());
    let rerank_timeout_ms = app_cfg
        .search
        .rerank_timeout_ms
        .unwrap_or(DEFAULT_SEARCH_RERANK_TIMEOUT_MS)
        .max(1);
    let rerank_fail_open = app_cfg
        .search
        .rerank_fail_open
        .unwrap_or(DEFAULT_SEARCH_RERANK_FAIL_OPEN);
    let rerank_default_enabled = app_cfg
        .search
        .rerank_enabled
        .unwrap_or(DEFAULT_SEARCH_RERANK_ENABLED);

    SearchRuntimeConfig {
        top_k,
        score_threshold,
        qdrant_url,
        collection,
        ollama_url,
        model,
        retrieval_mode,
        knowledge_threshold,
        top_n,
        bm25_top_n,
        rrf_k,
        dense_weight,
        bm25_weight,
        rerank_url,
        rerank_model,
        rerank_timeout_ms,
        rerank_fail_open,
        rerank_default_enabled,
    }
}

fn parse_retrieval_mode(value: Option<&str>) -> Option<RetrievalMode> {
    match value?.trim().to_ascii_lowercase().as_str() {
        "hybrid" => Some(RetrievalMode::Hybrid),
        "dense" => Some(RetrievalMode::Dense),
        "bm25" => Some(RetrievalMode::Bm25),
        _ => None,
    }
}

fn parse_knowledge_threshold(value: Option<&str>) -> Option<ConfidenceLevel> {
    match value?.trim().to_ascii_lowercase().as_str() {
        "low" => Some(ConfidenceLevel::Low),
        "medium" => Some(ConfidenceLevel::Medium),
        "high" => Some(ConfidenceLevel::High),
        _ => None,
    }
}

fn retrieve_dense_candidates(
    client: &Client,
    runtime: &SearchRuntimeConfig,
    qdrant_api_key: Option<&str>,
    top_n: usize,
    doc_version: Option<&str>,
    retrieval_query: &str,
) -> Result<Vec<SearchCandidate>> {
    let query_embedding =
        embed_query(client, &runtime.ollama_url, &runtime.model, retrieval_query)?;
    let qdrant_points = search_qdrant(
        client,
        &runtime.qdrant_url,
        qdrant_api_key,
        &runtime.collection,
        &query_embedding,
        top_n,
        runtime.score_threshold,
        doc_version,
    )?;

    Ok(qdrant_points
        .into_iter()
        .enumerate()
        .map(|(idx, point)| SearchCandidate {
            id: point.id,
            retrieval_score: point.score,
            dense_rank: Some(idx + 1),
            bm25_rank: None,
            rrf_score: None,
            rerank_score: None,
            retrieval_rank: idx + 1,
            rerank_rank: None,
            payload: point.payload,
        })
        .collect::<Vec<_>>())
}

fn retrieve_bm25_candidates(
    app_cfg: &AppConfig,
    args: &SearchDocsArgs,
    top_n: usize,
    doc_version: Option<&str>,
    retrieval_query: &str,
) -> Result<Vec<SearchCandidate>> {
    let chunks_root = resolve_chunks_root(app_cfg)?;
    let bm25_root = resolve_bm25_root(app_cfg)?;
    let hits = retrieval_bm25::search_chunks(
        retrieval_query,
        &chunks_root,
        &bm25_root,
        doc_version,
        top_n,
    )?;
    let mut candidates = hits
        .into_iter()
        .map(|hit| SearchCandidate {
            id: hit.id,
            retrieval_score: hit.score,
            dense_rank: None,
            bm25_rank: Some(hit.rank),
            rrf_score: None,
            rerank_score: None,
            retrieval_rank: hit.rank,
            rerank_rank: None,
            payload: hit.payload,
        })
        .collect::<Vec<_>>();
    postprocess_bm25_candidates(&mut candidates, &args.query);
    Ok(candidates)
}

fn resolve_chunks_root(app_cfg: &AppConfig) -> Result<PathBuf> {
    if let Some(path) = &app_cfg.search.chunks_dir {
        return Ok(path.clone());
    }
    app_cfg.chunks_root()
}

fn resolve_bm25_root(app_cfg: &AppConfig) -> Result<PathBuf> {
    if let Some(path) = &app_cfg.search.bm25_data_dir {
        return Ok(path.clone());
    }
    app_cfg.bm25_root()
}

#[derive(Debug)]
struct QdrantPoint {
    id: String,
    score: f32,
    payload: Value,
}

fn map_output_result(candidate: SearchCandidate, with_content: bool) -> Result<SearchResultOutput> {
    let payload = candidate
        .payload
        .as_object()
        .ok_or_else(|| anyhow!("qdrant point payload is not a JSON object"))?;
    let raw_content = payload
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let content = normalize_content_for_context(raw_content);

    let content_value = if content.is_empty() {
        None
    } else if with_content {
        Some(content)
    } else {
        Some(shorten(&content, 280))
    };

    let heading_path = payload
        .get("heading_path")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        });

    Ok(SearchResultOutput {
        id: candidate.id,
        score: candidate.rerank_score.unwrap_or(candidate.retrieval_score),
        retrieval_score: candidate.retrieval_score,
        rerank_score: candidate.rerank_score,
        retrieval_rank: candidate.retrieval_rank,
        dense_rank: candidate.dense_rank,
        bm25_rank: candidate.bm25_rank,
        rrf_score: candidate.rrf_score,
        rerank_rank: candidate.rerank_rank,
        version_bucket: payload
            .get("version_bucket")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        source_title: payload
            .get("source_title")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        source_url: payload
            .get("source_url")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        source_url_with_anchor: payload
            .get("source_url_with_anchor")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        section_anchor: payload
            .get("section_anchor")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        source_md: payload
            .get("source_md")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        heading_path,
        content: content_value,
    })
}

fn build_context_response(response: &SearchResponseOutput) -> SearchContextResponseOutput {
    let contexts = response
        .results
        .iter()
        .filter_map(|result| {
            let content = result.content.as_deref().unwrap_or("").trim();
            if content.is_empty() {
                return None;
            }
            Some(ContextSegmentOutput {
                source_title: result.source_title.clone(),
                source_url: result.source_url.clone(),
                source_url_with_anchor: result.source_url_with_anchor.clone(),
                section_anchor: result.section_anchor.clone(),
                source_md: result.source_md.clone(),
                version_bucket: result.version_bucket.clone(),
                heading_path: result.heading_path.clone(),
                content: content.to_string(),
            })
        })
        .collect::<Vec<_>>();

    SearchContextResponseOutput {
        query: response.query.clone(),
        doc_version: response.doc_version.clone(),
        knowledge_threshold: response.knowledge_threshold,
        no_knowledge: response.no_knowledge,
        no_knowledge_message: response.no_knowledge_message.clone(),
        contexts,
    }
}

fn rerank_candidates(
    client: &Client,
    rerank_url: &str,
    api_key: Option<&str>,
    model: &str,
    query: &str,
    candidates: Vec<SearchCandidate>,
    top_n: usize,
) -> Result<Option<Vec<SearchCandidate>>> {
    let (documents, mapping) = build_rerank_documents(&candidates);
    if documents.is_empty() {
        return Ok(None);
    }

    let rerank_scores = call_rerank(client, rerank_url, api_key, model, query, &documents, top_n)?;
    if rerank_scores.is_empty() {
        return Ok(None);
    }

    let mut scored_candidates = Vec::new();
    for item in rerank_scores {
        if let Some(&candidate_idx) = mapping.get(item.index) {
            scored_candidates.push((candidate_idx, item.relevance_score));
        }
    }
    if scored_candidates.is_empty() {
        return Ok(None);
    }

    Ok(Some(apply_rerank_order(candidates, scored_candidates)))
}

fn build_rerank_documents(candidates: &[SearchCandidate]) -> (Vec<String>, Vec<usize>) {
    let mut documents = Vec::new();
    let mut mapping = Vec::new();

    for (idx, candidate) in candidates.iter().enumerate() {
        let Some(payload) = candidate.payload.as_object() else {
            continue;
        };
        let content = payload
            .get("content")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty());
        let Some(content_text) = content else {
            continue;
        };

        let mut parts = Vec::new();
        if let Some(title) = payload
            .get("source_title")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            parts.push(format!("Title: {title}"));
        }
        if let Some(version) = payload
            .get("version_bucket")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|v| !v.is_empty())
        {
            parts.push(format!("Version: {version}"));
        }
        if let Some(heading_path) = payload.get("heading_path").and_then(Value::as_array) {
            let path = heading_path
                .iter()
                .filter_map(Value::as_str)
                .filter(|v| !v.trim().is_empty())
                .collect::<Vec<_>>();
            if !path.is_empty() {
                parts.push(format!("Heading: {}", path.join(" > ")));
            }
        }
        parts.push(content_text.to_string());

        documents.push(parts.join("\n"));
        mapping.push(idx);
    }

    (documents, mapping)
}

fn call_rerank(
    client: &Client,
    rerank_url: &str,
    api_key: Option<&str>,
    model: &str,
    query: &str,
    documents: &[String],
    top_n: usize,
) -> Result<Vec<RerankScore>> {
    let url = format!("{}/v1/rerank", rerank_url.trim_end_matches('/'));
    let response = send_json_request(
        client,
        Method::POST,
        &url,
        api_key,
        Some(&json!({
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n.min(documents.len()),
            "return_documents": false
        })),
        true,
    )?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().unwrap_or_default();
        return Err(anyhow!(
            "rerank request failed: status={} body={}",
            status,
            body
        ));
    }

    let value: Value = response
        .json()
        .context("failed to parse rerank response JSON")?;
    parse_rerank_scores(&value)
}

fn parse_rerank_scores(value: &Value) -> Result<Vec<RerankScore>> {
    let results = value
        .get("results")
        .and_then(Value::as_array)
        .or_else(|| value.get("data").and_then(Value::as_array))
        .ok_or_else(|| anyhow!("rerank response missing results array"))?;

    let mut by_index: HashMap<usize, f32> = HashMap::new();
    for item in results {
        let Some(index) = item
            .get("index")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
        else {
            continue;
        };
        let score = item
            .get("relevance_score")
            .or_else(|| item.get("score"))
            .and_then(Value::as_f64)
            .map(|v| v as f32);
        let Some(score) = score else {
            continue;
        };
        by_index
            .entry(index)
            .and_modify(|prev| {
                if score > *prev {
                    *prev = score;
                }
            })
            .or_insert(score);
    }

    let mut out = by_index
        .into_iter()
        .map(|(index, relevance_score)| RerankScore {
            index,
            relevance_score,
        })
        .collect::<Vec<_>>();
    out.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
    Ok(out)
}

fn apply_rerank_order(
    candidates: Vec<SearchCandidate>,
    scored_candidates: Vec<(usize, f32)>,
) -> Vec<SearchCandidate> {
    let mut scored = scored_candidates;
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut used = HashSet::new();
    let mut ordered = Vec::with_capacity(candidates.len());
    let mut rank = 1usize;

    for (idx, score) in scored {
        if idx >= candidates.len() || used.contains(&idx) {
            continue;
        }
        let mut candidate = candidates[idx].clone();
        candidate.rerank_score = Some(score);
        candidate.rerank_rank = Some(rank);
        rank += 1;
        ordered.push(candidate);
        used.insert(idx);
    }

    for (idx, candidate) in candidates.into_iter().enumerate() {
        if used.contains(&idx) {
            continue;
        }
        ordered.push(candidate);
    }

    ordered
}

fn resolve_rerank_enabled(rerank: bool, no_rerank: bool, default_enabled: bool) -> bool {
    if no_rerank {
        return false;
    }
    if rerank {
        return true;
    }
    default_enabled
}

fn normalize_top_n(top_k: usize, top_n: usize) -> usize {
    top_n.max(top_k).max(1)
}

fn sanitize_channel_weight(value: f32) -> f32 {
    if !value.is_finite() || value <= 0.0 {
        return 1.0;
    }
    value
}

fn build_retrieval_query(raw: &str) -> String {
    let mut tokens = Vec::new();
    let lower = raw.to_lowercase();
    if !raw.trim().is_empty() {
        tokens.push(raw.trim().to_string());
    }

    let expansions = [
        ("рпм", "rpm"),
        ("rpm", "рпм"),
        ("эмулятор", "emulator"),
        ("emulator", "эмулятор"),
        ("скриншот", "screenshot"),
        ("screenshot", "скриншот"),
        ("видео", "video"),
        ("video", "видео"),
        ("подпис", "sign signing signature"),
        ("sign", "подписание подпись"),
        ("установ", "install deploy"),
        ("install", "установка установить"),
        ("запуск", "run launch"),
        ("запустить", "run launch"),
        ("launch", "запуск"),
        ("собрат", "build"),
        ("build", "сборка собрать"),
        ("терминал", "cli shell"),
        ("командн", "cli shell"),
        ("mb2", "mb2 build"),
    ];
    for (needle, addition) in expansions {
        if lower.contains(needle) {
            tokens.push(addition.to_string());
        }
    }

    let mut seen = HashSet::new();
    let mut uniq = Vec::new();
    for token in tokens {
        let t = token.trim();
        if t.is_empty() {
            continue;
        }
        if seen.insert(t.to_string()) {
            uniq.push(t.to_string());
        }
    }
    uniq.join(" ")
}

fn retrieval_mode_name(mode: RetrievalMode) -> &'static str {
    match mode {
        RetrievalMode::Hybrid => "hybrid",
        RetrievalMode::Dense => "dense",
        RetrievalMode::Bm25 => "bm25",
    }
}

fn fuse_rrf(
    dense: Vec<SearchCandidate>,
    bm25: Vec<SearchCandidate>,
    rrf_k: usize,
    dense_weight: f32,
    bm25_weight: f32,
    limit: usize,
) -> Vec<SearchCandidate> {
    #[derive(Debug)]
    struct FusionEntry {
        payload: Value,
        dense_rank: Option<usize>,
        bm25_rank: Option<usize>,
        score: f32,
    }

    let mut merged: HashMap<String, FusionEntry> = HashMap::new();
    let k = rrf_k.max(1) as f32;
    let dense_w = sanitize_channel_weight(dense_weight);
    let bm25_w = sanitize_channel_weight(bm25_weight);

    for candidate in dense {
        let rank = candidate
            .dense_rank
            .unwrap_or(candidate.retrieval_rank)
            .max(1);
        let entry = merged.entry(candidate.id).or_insert_with(|| FusionEntry {
            payload: candidate.payload,
            dense_rank: None,
            bm25_rank: None,
            score: 0.0,
        });
        entry.dense_rank = Some(rank);
        entry.score += dense_w * (1.0 / (k + rank as f32));
    }

    for candidate in bm25 {
        let rank = candidate
            .bm25_rank
            .unwrap_or(candidate.retrieval_rank)
            .max(1);
        let entry = merged.entry(candidate.id).or_insert_with(|| FusionEntry {
            payload: candidate.payload,
            dense_rank: None,
            bm25_rank: None,
            score: 0.0,
        });
        entry.bm25_rank = Some(rank);
        entry.score += bm25_w * (1.0 / (k + rank as f32));
    }

    let mut ranked = merged.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.score.total_cmp(&a.1.score));

    ranked
        .into_iter()
        .take(limit.max(1))
        .enumerate()
        .map(|(idx, (id, item))| SearchCandidate {
            id,
            retrieval_score: item.score,
            dense_rank: item.dense_rank,
            bm25_rank: item.bm25_rank,
            rrf_score: Some(item.score),
            rerank_score: None,
            retrieval_rank: idx + 1,
            rerank_rank: None,
            payload: item.payload,
        })
        .collect::<Vec<_>>()
}

fn postprocess_bm25_candidates(candidates: &mut [SearchCandidate], raw_query: &str) {
    if candidates.is_empty() {
        return;
    }
    let lower_query = raw_query.to_lowercase();
    let howto_intent = is_howto_query(&lower_query);

    for candidate in candidates.iter_mut() {
        let source_md = payload_str(&candidate.payload, "source_md").to_lowercase();
        let mut factor = 1.0f32;

        if source_md.contains("_changelog_") {
            factor *= 0.15;
        }
        if howto_intent && source_md.contains("platform__architecture") {
            factor *= 0.6;
        }
        if howto_intent && source_md.contains("__reference__") {
            factor *= 0.75;
        }
        if howto_intent
            && (source_md.contains("sdk__tools")
                || source_md.contains("sdk__app_development")
                || source_md.contains("software_development__guides"))
        {
            factor *= 1.15;
        }

        candidate.retrieval_score *= factor;
    }

    candidates.sort_by(|a, b| b.retrieval_score.total_cmp(&a.retrieval_score));
    for (idx, candidate) in candidates.iter_mut().enumerate() {
        candidate.bm25_rank = Some(idx + 1);
        candidate.retrieval_rank = idx + 1;
    }
}

fn evaluate_answer_quality(
    raw_query: &str,
    doc_version: Option<&str>,
    results: &[SearchResultOutput],
) -> (String, Option<String>) {
    if results.is_empty() {
        let advisory = doc_version.map(|v| {
            format!(
                "No direct answer found in docs version {v}. Try broader query or another version."
            )
        });
        return ("low".to_string(), advisory);
    }

    let profile = build_query_profile(raw_query);
    if profile.tokens.is_empty() {
        return ("medium".to_string(), None);
    }

    let mut max_coverage = 0.0f32;
    let mut action_supported = profile.required_action.is_none();
    let mut actionable_count = 0usize;

    for result in results.iter().take(5) {
        let rtokens = extract_canonical_tokens(&result_token_haystack(result));
        let coverage = token_coverage(&profile.tokens, &rtokens);
        if coverage > max_coverage {
            max_coverage = coverage;
        }
        if is_actionable_result_source(result) {
            actionable_count += 1;
        }
        if let Some(action) = &profile.required_action {
            if action
                .tokens
                .iter()
                .any(|token| token_present(&rtokens, token))
            {
                action_supported = true;
            }
        }
    }

    if !action_supported {
        let advisory = doc_version.map(|v| {
            format!(
                "No source with matching action intent found in docs version {v}; results may be indirect."
            )
        });
        return ("low".to_string(), advisory);
    }

    // Generic coverage gate: if query concepts are not co-located in any retrieved source, treat as low confidence.
    if max_coverage < 0.58 {
        let advisory = doc_version.map(|v| {
            format!(
                "No source with sufficient concept overlap found in docs version {v}; results may be indirect."
            )
        });
        return ("low".to_string(), advisory);
    }

    if max_coverage >= 0.9 && actionable_count >= 2 {
        return ("high".to_string(), None);
    }
    ("medium".to_string(), None)
}

fn parse_confidence_level(value: &str) -> ConfidenceLevel {
    match value {
        "high" => ConfidenceLevel::High,
        "medium" => ConfidenceLevel::Medium,
        _ => ConfidenceLevel::Low,
    }
}

fn confidence_rank(level: ConfidenceLevel) -> u8 {
    match level {
        ConfidenceLevel::Low => 0,
        ConfidenceLevel::Medium => 1,
        ConfidenceLevel::High => 2,
    }
}

fn build_no_knowledge_message(doc_version: Option<&str>) -> String {
    if let Some(version) = doc_version.map(str::trim).filter(|v| !v.is_empty()) {
        return format!(
            "No sufficiently relevant sources found for docs version {version}. No knowledge returned."
        );
    }
    "No sufficiently relevant sources found. No knowledge returned.".to_string()
}

fn is_howto_query(lower_query: &str) -> bool {
    let markers = [
        "как",
        "how",
        "установ",
        "install",
        "запуск",
        "запустить",
        "run",
        "launch",
        "собрат",
        "build",
        "подпис",
        "sign",
        "эмулятор",
        "emulator",
        "скриншот",
        "screenshot",
        "видео",
        "video",
    ];
    markers.iter().any(|m| lower_query.contains(m))
}

#[derive(Debug, Clone)]
struct QueryActionIntent {
    tokens: Vec<&'static str>,
}

#[derive(Debug, Clone)]
struct QueryProfile {
    tokens: Vec<String>,
    required_action: Option<QueryActionIntent>,
}

fn build_query_profile(query: &str) -> QueryProfile {
    let tokens = extract_canonical_tokens(query);
    let required_action = if tokens
        .iter()
        .any(|t| t == "sign" || t == "install" || t == "build" || t == "launch")
    {
        let mut action_tokens = Vec::new();
        if tokens.iter().any(|t| t == "sign") {
            action_tokens.extend(["sign"]);
        }
        if tokens.iter().any(|t| t == "install") {
            action_tokens.extend(["install"]);
        }
        if tokens.iter().any(|t| t == "build") {
            action_tokens.extend(["build"]);
        }
        if tokens.iter().any(|t| t == "launch") {
            action_tokens.extend(["launch"]);
        }
        Some(QueryActionIntent {
            tokens: action_tokens,
        })
    } else if tokens.iter().any(|t| t == "record" || t == "screenshot") {
        let mut action_tokens = Vec::new();
        if tokens.iter().any(|t| t == "record") {
            action_tokens.extend(["record"]);
        }
        if tokens.iter().any(|t| t == "screenshot") {
            action_tokens.extend(["screenshot"]);
        }
        Some(QueryActionIntent {
            tokens: action_tokens,
        })
    } else {
        None
    };
    QueryProfile {
        tokens,
        required_action,
    }
}

fn result_token_haystack(result: &SearchResultOutput) -> String {
    let mut parts = Vec::new();
    if let Some(title) = &result.source_title {
        parts.push(title.as_str());
    }
    if let Some(path) = &result.source_md {
        parts.push(path.as_str());
    }
    if let Some(content) = &result.content {
        parts.push(content.as_str());
    }
    parts.join(" ")
}

fn extract_canonical_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for raw in text.split(|ch: char| !ch.is_alphanumeric()) {
        let Some(token) = canonicalize_token(raw) else {
            continue;
        };
        if seen.insert(token.clone()) {
            out.push(token);
        }
    }
    out
}

fn canonicalize_token(raw: &str) -> Option<String> {
    let t = raw.trim().to_lowercase();
    if t.is_empty() {
        return None;
    }

    let mapped = if t.contains("эмулят") || t == "emulator" || t == "emulation" {
        "emulator"
    } else if t == "рпм" || t == "rpm" || t.starts_with("пакет") || t.starts_with("package")
    {
        "rpm"
    } else if t.starts_with("подпис")
        || t.starts_with("sign")
        || t.starts_with("signature")
        || t.starts_with("rpmsign")
    {
        "sign"
    } else if t.starts_with("устан")
        || t.starts_with("install")
        || t.starts_with("deploy")
        || t.starts_with("развер")
    {
        "install"
    } else if t.starts_with("собр") || t.starts_with("build") || t == "mb2" {
        "build"
    } else if t.starts_with("запус")
        || t.starts_with("launch")
        || t == "run"
        || t == "start"
        || t == "cli"
        || t == "terminal"
        || t.starts_with("терминал")
    {
        "launch"
    } else if t.starts_with("скрин") || t.starts_with("screenshot") || t.starts_with("screengrab")
    {
        "screenshot"
    } else if t.starts_with("запис") || t == "record" || t == "capture" {
        "record"
    } else if t.starts_with("видео") || t == "video" {
        "video"
    } else {
        let stop = [
            "как", "и", "в", "на", "из", "по", "для", "с", "или", "the", "a", "an", "to", "of",
        ];
        if stop.contains(&t.as_str()) {
            return None;
        }
        if t.chars().count() < 3 {
            return None;
        }
        return Some(t);
    };
    Some(mapped.to_string())
}

fn token_coverage(query_tokens: &[String], result_tokens: &[String]) -> f32 {
    if query_tokens.is_empty() {
        return 0.0;
    }
    let matched = query_tokens
        .iter()
        .filter(|qt| token_present(result_tokens, qt))
        .count();
    matched as f32 / query_tokens.len() as f32
}

fn token_present(tokens: &[String], needle: &str) -> bool {
    tokens.iter().any(|t| t == needle)
}

fn is_actionable_result_source(result: &SearchResultOutput) -> bool {
    let Some(source_md) = result.source_md.as_deref() else {
        return false;
    };
    let path = source_md.to_lowercase();
    path.contains("sdk__tools")
        || path.contains("sdk__app_development")
        || path.contains("software_development__guides")
}

fn payload_str(payload: &Value, key: &str) -> String {
    payload
        .as_object()
        .and_then(|obj| obj.get(key))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn search_qdrant(
    client: &Client,
    qdrant_url: &str,
    api_key: Option<&str>,
    collection: &str,
    vector: &[f32],
    limit: usize,
    score_threshold: f32,
    doc_version: Option<&str>,
) -> Result<Vec<QdrantPoint>> {
    let filter_value = build_version_filter(doc_version);
    let mut search_body = json!({
        "vector": vector,
        "limit": limit,
        "with_payload": true,
        "with_vector": false,
    });
    if score_threshold > 0.0 {
        search_body["score_threshold"] = json!(score_threshold);
    }
    if let Some(filter) = &filter_value {
        search_body["filter"] = filter.clone();
    }

    let search_url = format!(
        "{}/collections/{}/points/search",
        qdrant_url.trim_end_matches('/'),
        collection
    );
    let search_response = send_json_request(
        client,
        Method::POST,
        &search_url,
        api_key,
        Some(&search_body),
        false,
    )?;

    let value = if search_response.status() == StatusCode::NOT_FOUND {
        let mut query_body = json!({
            "query": vector,
            "limit": limit,
            "with_payload": true,
            "with_vector": false,
        });
        if score_threshold > 0.0 {
            query_body["score_threshold"] = json!(score_threshold);
        }
        if let Some(filter) = filter_value {
            query_body["filter"] = filter;
        }
        let query_url = format!(
            "{}/collections/{}/points/query",
            qdrant_url.trim_end_matches('/'),
            collection
        );
        let response = send_json_request(
            client,
            Method::POST,
            &query_url,
            api_key,
            Some(&query_body),
            false,
        )?;
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().unwrap_or_default();
            return Err(anyhow!(
                "Qdrant query request failed: status={} body={}",
                status,
                body
            ));
        }
        response
            .json::<Value>()
            .context("failed to parse Qdrant query response JSON")?
    } else if search_response.status().is_success() {
        search_response
            .json::<Value>()
            .context("failed to parse Qdrant search response JSON")?
    } else {
        let status = search_response.status();
        let body = search_response.text().unwrap_or_default();
        return Err(anyhow!(
            "Qdrant search request failed: status={} body={}",
            status,
            body
        ));
    };

    parse_qdrant_points(&value)
}

fn parse_qdrant_points(response: &Value) -> Result<Vec<QdrantPoint>> {
    let result_node = response
        .get("result")
        .ok_or_else(|| anyhow!("Qdrant response missing result field"))?;

    let points = if let Some(arr) = result_node.as_array() {
        arr.clone()
    } else if let Some(arr) = result_node.get("points").and_then(Value::as_array) {
        arr.clone()
    } else {
        return Err(anyhow!(
            "Qdrant response result is neither array nor object.points array"
        ));
    };

    let mut out = Vec::with_capacity(points.len());
    for point in points {
        let id_value = point
            .get("id")
            .ok_or_else(|| anyhow!("Qdrant point missing id"))?;
        let score = point
            .get("score")
            .and_then(Value::as_f64)
            .ok_or_else(|| anyhow!("Qdrant point missing numeric score"))?
            as f32;
        let payload = point.get("payload").cloned().unwrap_or(Value::Null);
        out.push(QdrantPoint {
            id: json_value_to_string(id_value),
            score,
            payload,
        });
    }
    Ok(out)
}

fn build_version_filter(doc_version: Option<&str>) -> Option<Value> {
    let version = doc_version.map(str::trim).filter(|v| !v.is_empty())?;
    Some(json!({
        "must": [{
            "key": "version_bucket",
            "match": {"value": version}
        }]
    }))
}

fn embed_query(client: &Client, ollama_url: &str, model: &str, query: &str) -> Result<Vec<f32>> {
    let base = ollama_url.trim_end_matches('/');
    let embed_url = format!("{base}/api/embed");
    let response = client
        .post(&embed_url)
        .json(&json!({
            "model": model,
            "input": [query],
        }))
        .send();

    if let Ok(resp) = response {
        if resp.status().is_success() {
            let value: Value = resp
                .json()
                .context("failed to parse ollama /api/embed response")?;
            if let Some(arr) = value.get("embeddings").and_then(Value::as_array) {
                if let Some(first) = arr.first() {
                    return parse_vector(first);
                }
            }
            if let Some(single) = value.get("embedding") {
                return parse_vector(single);
            }
        }
    }

    let old_url = format!("{base}/api/embeddings");
    let resp = client
        .post(&old_url)
        .json(&json!({
            "model": model,
            "prompt": query,
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
    parse_vector(vec_value)
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

fn send_json_request(
    client: &Client,
    method: Method,
    url: &str,
    api_key: Option<&str>,
    body: Option<&Value>,
    auth_bearer: bool,
) -> Result<reqwest::blocking::Response> {
    let mut request = client.request(method, url);
    if let Some(key) = api_key {
        request = if auth_bearer {
            request.header("authorization", format!("Bearer {key}"))
        } else {
            request.header("api-key", key)
        };
    }
    if let Some(payload) = body {
        request = request.json(payload);
    }
    request
        .send()
        .with_context(|| format!("request failed: {}", url))
}

fn json_value_to_string(value: &Value) -> String {
    if let Some(s) = value.as_str() {
        return s.to_string();
    }
    if let Some(n) = value.as_u64() {
        return n.to_string();
    }
    if let Some(n) = value.as_i64() {
        return n.to_string();
    }
    value.to_string()
}

fn shorten(text: &str, max_chars: usize) -> String {
    if max_chars <= 3 {
        return text.to_string();
    }
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_string();
    }
    let head: String = text.chars().take(max_chars - 3).collect();
    format!("{head}...")
}

fn normalize_content_for_context(text: &str) -> String {
    // Keep full semantics, but remove markdown noise and extra spacing for LLM context.
    let mut out_lines: Vec<String> = Vec::new();
    let mut lines = text.lines().peekable();

    while let Some(line) = lines.next() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            out_lines.push(String::new());
            continue;
        }

        // Convert markdown definition list:
        // `term`
        // : description
        if let Some(next_line) = lines.peek() {
            let next_trimmed = next_line.trim_start();
            if next_trimmed.starts_with(':') {
                let term = strip_wrapping_backticks(trimmed);
                let definition = next_trimmed.trim_start_matches(':').trim();
                let merged = if definition.is_empty() {
                    term.to_string()
                } else {
                    format!("{term} — {definition}")
                };
                out_lines.push(merged);
                let _ = lines.next();
                continue;
            }
        }

        out_lines.push(trimmed.to_string());
    }

    // Collapse repeated blank lines.
    let mut collapsed: Vec<String> = Vec::with_capacity(out_lines.len());
    let mut prev_blank = false;
    for line in out_lines {
        if line.is_empty() {
            if !prev_blank {
                collapsed.push(String::new());
                prev_blank = true;
            }
        } else {
            collapsed.push(line);
            prev_blank = false;
        }
    }

    collapsed.join("\n").trim().to_string()
}

fn strip_wrapping_backticks(value: &str) -> &str {
    let trimmed = value.trim();
    if trimmed.len() >= 2 && trimmed.starts_with('`') && trimmed.ends_with('`') {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str, score: f32, content: &str) -> SearchCandidate {
        SearchCandidate {
            id: id.to_string(),
            retrieval_score: score,
            dense_rank: None,
            bm25_rank: None,
            rrf_score: None,
            rerank_score: None,
            retrieval_rank: 1,
            rerank_rank: None,
            payload: json!({
                "content": content,
                "source_title": "Title",
                "heading_path": ["H1", "H2"],
                "version_bucket": "5.2.0"
            }),
        }
    }

    #[test]
    fn normalize_content_for_context_compacts_definition_list() {
        let src = "`application_id`\n: Unique app id\n\n`project_id`\n: Project id\n";
        let got = normalize_content_for_context(src);
        assert_eq!(
            got,
            "application_id — Unique app id\n\nproject_id — Project id"
        );
    }

    #[test]
    fn rerank_flag_resolution() {
        assert!(!resolve_rerank_enabled(false, false, false));
        assert!(resolve_rerank_enabled(false, false, true));
        assert!(resolve_rerank_enabled(true, false, false));
        assert!(!resolve_rerank_enabled(true, true, true));
    }

    #[test]
    fn top_n_is_never_below_top_k() {
        assert_eq!(normalize_top_n(5, 30), 30);
        assert_eq!(normalize_top_n(5, 3), 5);
        assert_eq!(normalize_top_n(1, 0), 1);
    }

    #[test]
    fn rerank_order_keeps_unscored_tail() {
        let mut a = candidate("a", 0.9, "A");
        a.retrieval_rank = 1;
        let mut b = candidate("b", 0.8, "B");
        b.retrieval_rank = 2;
        let mut c = candidate("c", 0.7, "C");
        c.retrieval_rank = 3;

        let out = apply_rerank_order(vec![a, b, c], vec![(2, 0.99), (0, 0.55)]);
        assert_eq!(out[0].id, "c");
        assert_eq!(out[0].rerank_rank, Some(1));
        assert_eq!(out[1].id, "a");
        assert_eq!(out[1].rerank_rank, Some(2));
        assert_eq!(out[2].id, "b");
        assert_eq!(out[2].rerank_rank, None);
    }

    #[test]
    fn rerank_documents_skip_empty_content() {
        let keep = candidate("keep", 0.9, "useful text");
        let mut skip = candidate("skip", 0.8, "   ");
        skip.payload["content"] = Value::String("".to_string());

        let (docs, mapping) = build_rerank_documents(&[keep, skip]);
        assert_eq!(docs.len(), 1);
        assert_eq!(mapping, vec![0usize]);
        assert!(docs[0].contains("useful text"));
        assert!(docs[0].contains("Version: 5.2.0"));
    }

    #[test]
    fn parse_rerank_scores_ignores_invalid_rows_and_sorts() {
        let response = json!({
            "results": [
                {"index": 2, "relevance_score": 0.41},
                {"index": 0, "relevance_score": 0.91},
                {"index": 1},
                {"relevance_score": 0.77},
                {"index": 2, "score": 0.67}
            ]
        });

        let scores = parse_rerank_scores(&response).expect("scores should parse");
        assert_eq!(scores.len(), 2);
        assert_eq!(scores[0].index, 0);
        assert!((scores[0].relevance_score - 0.91).abs() < 1e-6);
        assert_eq!(scores[1].index, 2);
        assert!((scores[1].relevance_score - 0.67).abs() < 1e-6);
    }

    #[test]
    fn rerank_flow_drops_out_of_range_indices_and_keeps_tail() {
        let mut a = candidate("a", 0.9, "A");
        a.retrieval_rank = 1;
        let mut b = candidate("b", 0.8, "B");
        b.retrieval_rank = 2;
        let mut c = candidate("c", 0.7, "C");
        c.retrieval_rank = 3;
        let candidates = vec![a, b, c];

        // index=5 is invalid and must be ignored safely.
        let rerank_scores = vec![
            RerankScore {
                index: 1,
                relevance_score: 0.95,
            },
            RerankScore {
                index: 5,
                relevance_score: 0.99,
            },
        ];
        let mapping = vec![0usize, 1usize, 2usize];

        let mut scored_candidates = Vec::new();
        for item in rerank_scores {
            if let Some(&candidate_idx) = mapping.get(item.index) {
                scored_candidates.push((candidate_idx, item.relevance_score));
            }
        }
        let out = apply_rerank_order(candidates, scored_candidates);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].id, "b");
        assert_eq!(out[0].rerank_rank, Some(1));
        assert_eq!(out[1].id, "a");
        assert_eq!(out[2].id, "c");
    }

    #[test]
    fn rrf_fusion_combines_dense_and_bm25_ranks() {
        let mut dense_a = candidate("a", 0.9, "A");
        dense_a.dense_rank = Some(1);
        dense_a.retrieval_rank = 1;
        let mut dense_b = candidate("b", 0.8, "B");
        dense_b.dense_rank = Some(2);
        dense_b.retrieval_rank = 2;

        let mut bm25_a = candidate("a", 11.0, "A");
        bm25_a.bm25_rank = Some(4);
        bm25_a.retrieval_rank = 4;
        let mut bm25_c = candidate("c", 12.0, "C");
        bm25_c.bm25_rank = Some(1);
        bm25_c.retrieval_rank = 1;

        let out = fuse_rrf(
            vec![dense_a, dense_b],
            vec![bm25_a, bm25_c],
            60,
            1.0,
            0.55,
            10,
        );
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].id, "a");
        assert_eq!(out[0].dense_rank, Some(1));
        assert_eq!(out[0].bm25_rank, Some(4));
        assert!(out[0].rrf_score.is_some());
    }

    #[test]
    fn retrieval_mode_value_enum_parses_cli_tokens() {
        assert_eq!(
            RetrievalMode::from_str("hybrid", true).unwrap(),
            RetrievalMode::Hybrid
        );
        assert_eq!(
            RetrievalMode::from_str("dense", true).unwrap(),
            RetrievalMode::Dense
        );
        assert_eq!(
            RetrievalMode::from_str("bm25", true).unwrap(),
            RetrievalMode::Bm25
        );
    }
}
