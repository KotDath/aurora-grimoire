use super::RagFetchWebArgs;
use crate::config::AppConfig;
use anyhow::{Context, Result, anyhow};
use chrono::{SecondsFormat, Utc};
use reqwest::{
    blocking::Client,
    header::{ACCEPT, ACCEPT_LANGUAGE, HeaderMap, HeaderValue, USER_AGENT},
    redirect::Policy,
};
use scraper::{Html, Selector};
use serde::Serialize;
use serde_json::Value;
use sha1::{Digest, Sha1};
use std::{
    collections::{BTreeSet, HashSet, VecDeque},
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::Duration,
};
use url::Url;

const SEED_URL: &str = "https://developer.auroraos.ru/doc";
const MAX_DEPTH: usize = 10;
const TARGET_HOST: &str = "developer.auroraos.ru";
const TARGET_PATH_PREFIX: &str = "/doc";
const STATUS_EVERY: usize = 500;
const DEFAULT_USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36";

macro_rules! vprintln {
    ($verbose:expr, $($arg:tt)*) => {
        if $verbose {
            println!($($arg)*);
        }
    };
}

#[derive(Debug, Serialize)]
struct Manifest {
    seed_url: String,
    max_depth: usize,
    started_at: String,
    finished_at: String,
    visited_total: usize,
    saved_total: usize,
    errors_total: usize,
    records: Vec<ManifestRecord>,
}

#[derive(Debug, Serialize)]
struct ManifestRecord {
    worker: String,
    url: String,
    depth: usize,
    status: &'static str,
    http_status: Option<u16>,
    file_path: Option<String>,
    error: Option<String>,
    fetched_at: String,
}

impl ManifestRecord {
    fn error(
        worker: String,
        url: String,
        depth: usize,
        fetched_at: String,
        http_status: Option<u16>,
        error: String,
    ) -> Self {
        Self {
            worker,
            url,
            depth,
            status: "error",
            http_status,
            file_path: None,
            error: Some(error),
            fetched_at,
        }
    }
}

#[derive(Debug)]
struct WorkerReport {
    visited_total: usize,
    saved_total: usize,
    records: Vec<ManifestRecord>,
}

#[derive(Clone, Debug)]
enum CrawlScope {
    DefaultUnversioned,
    Version(String),
}

impl CrawlScope {
    fn label(&self) -> String {
        match self {
            Self::DefaultUnversioned => "default".to_string(),
            Self::Version(version) => format!("version:{version}"),
        }
    }

    fn seed_url(&self) -> Result<Url> {
        match self {
            Self::DefaultUnversioned => {
                Url::parse(SEED_URL).context("failed to parse default seed URL")
            }
            Self::Version(version) => Url::parse(&format!("{SEED_URL}/{version}"))
                .with_context(|| format!("failed to parse version seed URL for {version}")),
        }
    }

    fn matches(&self, url: &Url) -> bool {
        if !is_in_scope(url) {
            return false;
        }

        match self {
            Self::DefaultUnversioned => version_segment_from_doc_path(url.path()).is_none(),
            Self::Version(version) => {
                version_segment_from_doc_path(url.path()) == Some(version.as_str())
            }
        }
    }
}

pub fn run(args: RagFetchWebArgs) -> Result<()> {
    let verbose = args.verbose;
    let output_root = resolve_output_root()?;
    fs::create_dir_all(&output_root).with_context(|| {
        format!(
            "failed to create output directory: {}",
            output_root.display()
        )
    })?;

    vprintln!(verbose, "[fetch-web] seed: {SEED_URL}");
    vprintln!(verbose, "[fetch-web] depth limit: {MAX_DEPTH}");
    vprintln!(verbose, "[fetch-web] output: {}", output_root.display());

    let discovery_client = build_http_client()?;
    let versions = discover_versions(&discovery_client, verbose);
    vprintln!(
        verbose,
        "[fetch-web] discovered {} version workers (+ default worker)",
        versions.len()
    );

    let mut scopes = Vec::with_capacity(1 + versions.len());
    scopes.push(CrawlScope::DefaultUnversioned);
    scopes.extend(versions.into_iter().map(CrawlScope::Version));

    let started_at = now_rfc3339();
    let global_saved = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::with_capacity(scopes.len());

    for scope in scopes {
        let scope_label = scope.label();
        let output_root_clone = output_root.clone();
        let global_saved_clone = Arc::clone(&global_saved);
        vprintln!(verbose, "[fetch-web] worker start: {scope_label}");

        handles.push(thread::spawn(move || {
            crawl_scope(scope, output_root_clone, global_saved_clone, verbose)
        }));
    }

    let mut records: Vec<ManifestRecord> = Vec::new();
    let mut visited_total = 0usize;
    let mut saved_total = 0usize;

    for handle in handles {
        let report = handle
            .join()
            .map_err(|err| anyhow!("fetch-web worker thread panicked: {err:?}"))??;
        visited_total += report.visited_total;
        saved_total += report.saved_total;
        records.extend(report.records);
    }

    let finished_at = now_rfc3339();
    let errors_total = records
        .iter()
        .filter(|record| record.status == "error")
        .count();

    let manifest = Manifest {
        seed_url: SEED_URL.to_string(),
        max_depth: MAX_DEPTH,
        started_at,
        finished_at,
        visited_total,
        saved_total,
        errors_total,
        records,
    };

    let manifest_path = output_root.join("manifest.json");
    let manifest_json =
        serde_json::to_vec_pretty(&manifest).context("failed to serialize manifest")?;
    fs::write(&manifest_path, manifest_json)
        .with_context(|| format!("failed to write manifest: {}", manifest_path.display()))?;

    vprintln!(
        verbose,
        "[fetch-web] finished: visited={}, saved={}, errors={}",
        manifest.visited_total,
        manifest.saved_total,
        manifest.errors_total
    );
    vprintln!(verbose, "[fetch-web] manifest: {}", manifest_path.display());
    if manifest.errors_total > 0 {
        vprintln!(
            verbose,
            "[fetch-web][warning] crawl completed with {} failed pages",
            manifest.errors_total
        );
    }
    println!(
        "{} documents downloaded and stored in {}",
        manifest.saved_total,
        output_root.display()
    );

    Ok(())
}

fn crawl_scope(
    scope: CrawlScope,
    output_root: PathBuf,
    global_saved: Arc<AtomicUsize>,
    verbose: bool,
) -> Result<WorkerReport> {
    let worker_label = scope.label();
    let seed_url = scope.seed_url()?;
    let client = build_http_client()?;
    let mut queue = VecDeque::from([(seed_url, 0usize)]);
    let mut visited: HashSet<String> = HashSet::new();
    let mut records: Vec<ManifestRecord> = Vec::new();
    let mut saved_total = 0usize;

    while let Some((raw_url, depth)) = queue.pop_front() {
        if depth > MAX_DEPTH {
            continue;
        }

        let Some(url) = canonicalize_url(&raw_url) else {
            continue;
        };
        if !scope.matches(&url) {
            continue;
        }

        let request_key = url.as_str().to_string();
        if !visited.insert(request_key.clone()) {
            continue;
        }

        let fetched_at = now_rfc3339();
        let response = match client.get(url.clone()).send() {
            Ok(response) => response,
            Err(err) => {
                records.push(ManifestRecord::error(
                    worker_label.clone(),
                    request_key,
                    depth,
                    fetched_at,
                    None,
                    format!("request failed: {err}"),
                ));
                continue;
            }
        };

        let http_status = response.status().as_u16();
        if !response.status().is_success() {
            records.push(ManifestRecord::error(
                worker_label.clone(),
                request_key,
                depth,
                fetched_at,
                Some(http_status),
                format!("http status {http_status}"),
            ));
            continue;
        }

        let final_url = match canonicalize_url(response.url()) {
            Some(final_url) => final_url,
            None => {
                records.push(ManifestRecord::error(
                    worker_label.clone(),
                    request_key,
                    depth,
                    fetched_at,
                    Some(http_status),
                    "invalid redirect target URL".to_string(),
                ));
                continue;
            }
        };

        if !scope.matches(&final_url) {
            records.push(ManifestRecord::error(
                worker_label.clone(),
                request_key,
                depth,
                fetched_at,
                Some(http_status),
                format!("redirected out of worker scope: {}", final_url.as_str()),
            ));
            continue;
        }

        visited.insert(final_url.as_str().to_string());

        let body = match response.text() {
            Ok(body) => body,
            Err(err) => {
                records.push(ManifestRecord::error(
                    worker_label.clone(),
                    final_url.as_str().to_string(),
                    depth,
                    fetched_at,
                    Some(http_status),
                    format!("failed to read response body: {err}"),
                ));
                continue;
            }
        };

        let file_path = build_html_file_path(&output_root, &final_url);
        if let Some(parent) = file_path.parent()
            && let Err(err) = fs::create_dir_all(parent)
        {
            records.push(ManifestRecord::error(
                worker_label.clone(),
                final_url.as_str().to_string(),
                depth,
                fetched_at,
                Some(http_status),
                format!("failed to create parent directory: {err}"),
            ));
            continue;
        }
        if let Err(err) = fs::write(&file_path, &body) {
            records.push(ManifestRecord::error(
                worker_label.clone(),
                final_url.as_str().to_string(),
                depth,
                fetched_at,
                Some(http_status),
                format!("failed to write html file: {err}"),
            ));
            continue;
        }

        let file_path_rel = relative_path_string(&output_root, &file_path);
        records.push(ManifestRecord {
            worker: worker_label.clone(),
            url: final_url.as_str().to_string(),
            depth,
            status: "ok",
            http_status: Some(http_status),
            file_path: Some(file_path_rel),
            error: None,
            fetched_at,
        });

        saved_total += 1;
        let global_saved_value = global_saved.fetch_add(1, Ordering::Relaxed) + 1;
        maybe_print_status(global_saved_value, &final_url, verbose);

        if depth < MAX_DEPTH {
            for link in extract_links(&body, &final_url) {
                if scope.matches(&link) {
                    queue.push_back((link, depth + 1));
                }
            }
        }
    }

    Ok(WorkerReport {
        visited_total: visited.len(),
        saved_total,
        records,
    })
}

fn discover_versions(client: &Client, verbose: bool) -> Vec<String> {
    let Ok(seed_url) = Url::parse(SEED_URL) else {
        return Vec::new();
    };

    let response = match client.get(seed_url.clone()).send() {
        Ok(response) => response,
        Err(err) => {
            vprintln!(
                verbose,
                "[fetch-web][warning] version discovery request failed: {err}"
            );
            return Vec::new();
        }
    };

    if !response.status().is_success() {
        vprintln!(
            verbose,
            "[fetch-web][warning] version discovery returned HTTP {}",
            response.status().as_u16()
        );
        return Vec::new();
    }

    let body = match response.text() {
        Ok(body) => body,
        Err(err) => {
            vprintln!(
                verbose,
                "[fetch-web][warning] failed to read discovery response body: {err}"
            );
            return Vec::new();
        }
    };

    let mut versions: BTreeSet<String> = BTreeSet::new();
    versions.extend(extract_versions_from_next_data(&body));
    versions.extend(extract_versions_from_links(&body, &seed_url));
    versions.into_iter().collect()
}

fn extract_versions_from_next_data(html: &str) -> Vec<String> {
    let document = Html::parse_document(html);
    let selector =
        Selector::parse(r#"script#__NEXT_DATA__"#).expect("__NEXT_DATA__ selector must be valid");
    let Some(node) = document.select(&selector).next() else {
        return Vec::new();
    };
    let next_data_text = node.text().collect::<String>();
    let Ok(value) = serde_json::from_str::<Value>(&next_data_text) else {
        return Vec::new();
    };
    let Some(versions) = value
        .pointer("/props/pageProps/initialState/version/availableVersions")
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };

    versions
        .iter()
        .filter_map(Value::as_str)
        .filter(|item| is_version_segment(item))
        .map(ToString::to_string)
        .collect()
}

fn extract_versions_from_links(html: &str, base_url: &Url) -> Vec<String> {
    let document = Html::parse_document(html);
    let selector = Selector::parse("a[href]").expect("a[href] selector must be valid");
    let mut versions: BTreeSet<String> = BTreeSet::new();

    for node in document.select(&selector) {
        let Some(raw_href) = node.value().attr("href") else {
            continue;
        };
        let Ok(joined_url) = base_url.join(raw_href) else {
            continue;
        };
        let Some(canonical_url) = canonicalize_url(&joined_url) else {
            continue;
        };
        if !is_in_scope(&canonical_url) {
            continue;
        }
        if let Some(version) = version_segment_from_doc_path(canonical_url.path()) {
            versions.insert(version.to_string());
        }
    }

    versions.into_iter().collect()
}

fn build_http_client() -> Result<Client> {
    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static(DEFAULT_USER_AGENT));
    headers.insert(
        ACCEPT,
        HeaderValue::from_static("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
    );
    headers.insert(
        ACCEPT_LANGUAGE,
        HeaderValue::from_static("ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7"),
    );

    Client::builder()
        .default_headers(headers)
        .timeout(Duration::from_secs(30))
        .redirect(Policy::limited(10))
        .build()
        .context("failed to build HTTP client")
}

fn resolve_output_root() -> Result<PathBuf> {
    let cfg = AppConfig::load()?;
    let data_root = cfg.data_root()?;
    Ok(data_root.join("html_data"))
}

#[cfg(test)]
fn build_output_root_from_home(home_dir: &Path) -> PathBuf {
    home_dir.join(".aurora-grimoire").join("html_data")
}

fn extract_links(html: &str, base_url: &Url) -> Vec<Url> {
    let document = Html::parse_document(html);
    let selector = Selector::parse("a[href]").expect("a[href] selector must be valid");
    let mut unique_links = HashSet::new();
    let mut output = Vec::new();

    for node in document.select(&selector) {
        let Some(raw_href) = node.value().attr("href") else {
            continue;
        };
        let href = raw_href.trim();
        if href.is_empty()
            || href.starts_with('#')
            || href.starts_with("javascript:")
            || href.starts_with("mailto:")
            || href.starts_with("tel:")
        {
            continue;
        }

        let Ok(joined_url) = base_url.join(href) else {
            continue;
        };
        let Some(canonical_url) = canonicalize_url(&joined_url) else {
            continue;
        };
        if !is_in_scope(&canonical_url) {
            continue;
        }

        let key = canonical_url.as_str().to_string();
        if unique_links.insert(key) {
            output.push(canonical_url);
        }
    }

    output
}

fn canonicalize_url(url: &Url) -> Option<Url> {
    if !matches!(url.scheme(), "http" | "https") {
        return None;
    }

    let mut canonical = url.clone();
    canonical.set_fragment(None);
    if canonical.query() == Some("") {
        canonical.set_query(None);
    }

    let normalized_path = normalize_path(canonical.path());
    canonical.set_path(&normalized_path);

    if matches!(
        (canonical.scheme(), canonical.port()),
        ("http", Some(80)) | ("https", Some(443))
    ) {
        let _ = canonical.set_port(None);
    }

    Some(canonical)
}

fn normalize_path(path: &str) -> String {
    if path.is_empty() {
        return "/".to_string();
    }
    let trimmed = path.trim_end_matches('/');
    if trimmed.is_empty() {
        "/".to_string()
    } else {
        trimmed.to_string()
    }
}

fn is_in_scope(url: &Url) -> bool {
    let Some(host) = url.host_str() else {
        return false;
    };
    if !host.eq_ignore_ascii_case(TARGET_HOST) {
        return false;
    }

    let normalized_path = normalize_path(url.path());
    normalized_path == TARGET_PATH_PREFIX || normalized_path.starts_with("/doc/")
}

fn version_segment_from_doc_path(path: &str) -> Option<&str> {
    let trimmed = path.trim_end_matches('/');
    let remainder = trimmed.strip_prefix("/doc/")?;
    if remainder.is_empty() {
        return None;
    }
    let first_segment = remainder.split('/').next()?;
    if is_version_segment(first_segment) {
        Some(first_segment)
    } else {
        None
    }
}

fn is_version_segment(segment: &str) -> bool {
    if !segment.contains('.') {
        return false;
    }
    segment
        .split('.')
        .all(|part| !part.is_empty() && part.chars().all(|ch| ch.is_ascii_digit()))
}

fn build_html_file_path(output_root: &Path, url: &Url) -> PathBuf {
    let mut file_path = output_root.to_path_buf();
    let normalized_path = normalize_path(url.path());
    let stripped_path = normalized_path.trim_start_matches('/');

    if stripped_path.is_empty() {
        file_path.push("root");
    } else {
        for segment in stripped_path.split('/') {
            file_path.push(sanitize_path_segment(segment));
        }
    }

    if let Some(query) = url.query() {
        file_path.push(format!("__q_{}", short_hash(query)));
    }

    file_path.push("index.html");
    file_path
}

fn sanitize_path_segment(segment: &str) -> String {
    let mut sanitized = String::with_capacity(segment.len());
    for ch in segment.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            sanitized.push(ch);
        } else {
            sanitized.push('_');
        }
    }

    if sanitized.is_empty() {
        "_".to_string()
    } else {
        sanitized
    }
}

fn short_hash(value: &str) -> String {
    let mut hasher = Sha1::new();
    hasher.update(value.as_bytes());
    let hash = format!("{:x}", hasher.finalize());
    hash[..10].to_string()
}

fn relative_path_string(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn now_rfc3339() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true)
}

fn maybe_print_status(global_saved: usize, url: &Url, verbose: bool) {
    if !verbose || global_saved == 0 || global_saved % STATUS_EVERY != 0 {
        return;
    }
    println!(
        "[fetch-web][status] saved={} last={}",
        global_saved,
        url.as_str()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalize_url_removes_fragment_and_trailing_slash() {
        let raw =
            Url::parse("https://developer.auroraos.ru/doc/sdk/#anchor").expect("url should parse");
        let canonical = canonicalize_url(&raw).expect("url should be canonicalized");
        assert_eq!(canonical.as_str(), "https://developer.auroraos.ru/doc/sdk");
    }

    #[test]
    fn scope_filter_only_accepts_doc_paths_on_target_host() {
        let in_scope = Url::parse("https://developer.auroraos.ru/doc/platform").expect("parse");
        let wrong_host = Url::parse("https://example.com/doc/platform").expect("parse");
        let wrong_path = Url::parse("https://developer.auroraos.ru/qa").expect("parse");

        assert!(is_in_scope(&in_scope));
        assert!(!is_in_scope(&wrong_host));
        assert!(!is_in_scope(&wrong_path));
    }

    #[test]
    fn default_scope_excludes_versioned_paths() {
        let default_scope = CrawlScope::DefaultUnversioned;
        let default_page = Url::parse("https://developer.auroraos.ru/doc/sdk").expect("parse");
        let versioned_page =
            Url::parse("https://developer.auroraos.ru/doc/5.2.1/sdk").expect("parse");

        assert!(default_scope.matches(&default_page));
        assert!(!default_scope.matches(&versioned_page));
    }

    #[test]
    fn version_scope_matches_only_its_version() {
        let scope = CrawlScope::Version("5.2.1".to_string());
        let own_page = Url::parse("https://developer.auroraos.ru/doc/5.2.1/sdk").expect("parse");
        let other_version =
            Url::parse("https://developer.auroraos.ru/doc/5.2.0/sdk").expect("parse");
        let unversioned = Url::parse("https://developer.auroraos.ru/doc/sdk").expect("parse");

        assert!(scope.matches(&own_page));
        assert!(!scope.matches(&other_version));
        assert!(!scope.matches(&unversioned));
    }

    #[test]
    fn html_file_path_uses_tree_layout() {
        let root = PathBuf::from("tmp/html_data");
        let url = Url::parse("https://developer.auroraos.ru/doc/sdk/tools").expect("parse");
        let path = build_html_file_path(&root, &url);
        assert_eq!(
            relative_path_string(&root, &path),
            "doc/sdk/tools/index.html"
        );
    }

    #[test]
    fn html_file_path_with_query_adds_hash_suffix() {
        let root = PathBuf::from("tmp/html_data");
        let url_a = Url::parse("https://developer.auroraos.ru/doc/sdk?view=brief").expect("parse");
        let url_b = Url::parse("https://developer.auroraos.ru/doc/sdk?view=full").expect("parse");
        let path_a = build_html_file_path(&root, &url_a);
        let path_b = build_html_file_path(&root, &url_b);

        assert_ne!(
            relative_path_string(&root, &path_a),
            relative_path_string(&root, &path_b)
        );
    }

    #[test]
    fn output_root_is_joined_from_home_path() {
        let home = PathBuf::from("home-dir");
        let root = build_output_root_from_home(&home);
        assert!(root.ends_with(Path::new(".aurora-grimoire").join("html_data")));
    }

    #[test]
    fn version_segment_detection_works() {
        assert_eq!(
            version_segment_from_doc_path("/doc/5.2.1/sdk"),
            Some("5.2.1")
        );
        assert_eq!(version_segment_from_doc_path("/doc/sdk"), None);
        assert_eq!(version_segment_from_doc_path("/doc"), None);
    }
}
