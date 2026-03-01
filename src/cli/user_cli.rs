use anyhow::{Context, Result, anyhow};
use clap::{Args, Subcommand};
use reqwest::{Method, StatusCode, blocking::Client};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    env,
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
    #[arg(long, default_value_t = 5)]
    pub top_k: usize,

    /// Relevance threshold [0..1]
    #[arg(long, default_value_t = 0.0)]
    pub score_threshold: f32,

    /// Return JSON instead of text
    #[arg(long, default_value_t = false)]
    pub json: bool,

    /// Include full chunk text in results
    #[arg(long, default_value_t = false)]
    pub with_content: bool,

    /// Qdrant base URL
    #[arg(long, default_value = "http://127.0.0.1:6333")]
    pub qdrant_url: String,

    /// Qdrant API key (or use QDRANT_API_KEY env)
    #[arg(long)]
    pub api_key: Option<String>,

    /// Qdrant collection name
    #[arg(long, default_value = "aurora_docs_qwen3_embedding_0_6b")]
    pub collection: String,

    /// Ollama base URL
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    pub ollama_url: String,

    /// Embedding model name
    #[arg(long, default_value = "qwen3-embedding:0.6b")]
    pub model: String,

    /// Filter results by documentation version bucket (e.g. 5.2.0, default)
    #[arg(long = "doc-version")]
    pub doc_version: Option<String>,

    /// Enable second-stage rerank
    #[arg(long, default_value_t = false)]
    pub rerank: bool,

    /// Disable second-stage rerank
    #[arg(long, default_value_t = false)]
    pub no_rerank: bool,

    /// Number of candidates retrieved from Qdrant before rerank
    #[arg(long, default_value_t = 30)]
    pub top_n: usize,

    /// Local rerank service base URL
    #[arg(long, default_value = "http://127.0.0.1:8081")]
    pub rerank_url: String,

    /// Rerank model id
    #[arg(long, default_value = "BAAI/bge-reranker-v2-m3")]
    pub rerank_model: String,

    /// Rerank API key (or use RERANK_API_KEY env)
    #[arg(long)]
    pub rerank_api_key: Option<String>,

    /// Rerank request timeout in milliseconds
    #[arg(long, default_value_t = 30000)]
    pub rerank_timeout_ms: u64,

    /// If rerank fails, fallback to retrieval order instead of hard-failing
    #[arg(long, default_value_t = true)]
    pub rerank_fail_open: bool,
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
    collection: String,
    top_k: usize,
    top_n: usize,
    score_threshold: f32,
    doc_version: Option<String>,
    rerank_enabled: bool,
    rerank_applied: bool,
    rerank_model: Option<String>,
    retrieval_latency_ms: u128,
    rerank_latency_ms: Option<u128>,
    results: Vec<SearchResultOutput>,
}

#[derive(Debug, Clone)]
struct SearchCandidate {
    id: String,
    retrieval_score: f32,
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

fn run_search_docs(args: SearchDocsArgs) -> Result<()> {
    let qdrant_api_key = args.api_key.or_else(|| env::var("QDRANT_API_KEY").ok());
    let rerank_api_key = args
        .rerank_api_key
        .or_else(|| env::var("RERANK_API_KEY").ok());
    let client = Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .context("failed to build HTTP client")?;
    let rerank_client = Client::builder()
        .timeout(Duration::from_millis(args.rerank_timeout_ms.max(1)))
        .build()
        .context("failed to build rerank HTTP client")?;

    let top_k = args.top_k.max(1);
    let top_n = normalize_top_n(top_k, args.top_n);
    let rerank_enabled = resolve_rerank_enabled(args.rerank, args.no_rerank);

    let query_embedding = embed_query(&client, &args.ollama_url, &args.model, &args.query)?;
    let retrieval_start = Instant::now();
    let qdrant_points = search_qdrant(
        &client,
        &args.qdrant_url,
        qdrant_api_key.as_deref(),
        &args.collection,
        &query_embedding,
        if rerank_enabled { top_n } else { top_k },
        args.score_threshold,
        args.doc_version.as_deref(),
    )?;
    let retrieval_latency_ms = retrieval_start.elapsed().as_millis();

    let mut candidates = qdrant_points
        .into_iter()
        .enumerate()
        .map(|(idx, point)| SearchCandidate {
            id: point.id,
            retrieval_score: point.score,
            rerank_score: None,
            retrieval_rank: idx + 1,
            rerank_rank: None,
            payload: point.payload,
        })
        .collect::<Vec<_>>();

    let mut rerank_applied = false;
    let mut rerank_latency_ms = None;

    if rerank_enabled && !candidates.is_empty() {
        let retrieval_candidates = candidates.clone();
        let start = Instant::now();
        let rerank_result = rerank_candidates(
            &rerank_client,
            &args.rerank_url,
            rerank_api_key.as_deref(),
            &args.rerank_model,
            &args.query,
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
                if args.rerank_fail_open {
                    // Keep retrieval order on rerank failure.
                    let _ = err;
                    candidates = retrieval_candidates;
                } else {
                    return Err(err.context("rerank failed"));
                }
            }
        }
    }

    let results = candidates
        .into_iter()
        .take(top_k)
        .map(|candidate| map_output_result(candidate, args.with_content))
        .collect::<Result<Vec<_>>>()?;

    let response = SearchResponseOutput {
        query: args.query,
        collection: args.collection,
        top_k,
        top_n,
        score_threshold: args.score_threshold,
        doc_version: args.doc_version,
        rerank_enabled,
        rerank_applied,
        rerank_model: if rerank_enabled {
            Some(args.rerank_model)
        } else {
            None
        },
        retrieval_latency_ms,
        rerank_latency_ms,
        results,
    };

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&response).context("failed to serialize JSON output")?
        );
        return Ok(());
    }

    if response.results.is_empty() {
        println!("No results found");
        return Ok(());
    }

    println!(
        "{} results from {} (retrieval={}ms, rerank={} applied={})",
        response.results.len(),
        response.collection,
        response.retrieval_latency_ms,
        response
            .rerank_latency_ms
            .map(|v| format!("{v}ms"))
            .unwrap_or_else(|| "-".to_string()),
        response.rerank_applied
    );
    for (idx, result) in response.results.iter().enumerate() {
        let rerank_score = result
            .rerank_score
            .map(|v| format!("{v:.4}"))
            .unwrap_or_else(|| "-".to_string());
        println!(
            "{}. score={:.4} retrieval={:.4} rerank={} version={} title={}",
            idx + 1,
            result.score,
            result.retrieval_score,
            rerank_score,
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
    let content = payload
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default();

    let content_value = if with_content {
        Some(content.to_string())
    } else if content.is_empty() {
        None
    } else {
        Some(shorten(content, 280))
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

fn resolve_rerank_enabled(rerank: bool, no_rerank: bool) -> bool {
    rerank && !no_rerank
}

fn normalize_top_n(top_k: usize, top_n: usize) -> usize {
    top_n.max(top_k).max(1)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: &str, score: f32, content: &str) -> SearchCandidate {
        SearchCandidate {
            id: id.to_string(),
            retrieval_score: score,
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
    fn rerank_flag_resolution() {
        assert!(!resolve_rerank_enabled(false, false));
        assert!(resolve_rerank_enabled(true, false));
        assert!(!resolve_rerank_enabled(true, true));
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
}
