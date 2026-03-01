use super::{RagDeployArgs, RagEmbedArgs, RagTestE2eArgs, deploy, embed};
use crate::config::{AppConfig, DEFAULT_COLLECTION, DEFAULT_QDRANT_URL, DEFAULT_TEST_E2E_QUERY};
use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use std::process::Command;

pub fn run(args: RagTestE2eArgs) -> Result<()> {
    let cfg = AppConfig::load()?;
    let qdrant_url = args
        .qdrant_url
        .clone()
        .or_else(|| cfg.deploy.qdrant_url.clone())
        .unwrap_or_else(|| DEFAULT_QDRANT_URL.to_string());
    let collection = args
        .collection
        .clone()
        .or_else(|| cfg.deploy.collection.clone())
        .unwrap_or_else(|| DEFAULT_COLLECTION.to_string());
    let query = args
        .query
        .clone()
        .unwrap_or_else(|| DEFAULT_TEST_E2E_QUERY.to_string());
    let top_k = args.top_k.unwrap_or(5).max(1);

    if !args.skip_embed {
        embed::run(RagEmbedArgs {
            verbose: args.verbose,
            ollama_url: None,
            model: None,
            batch_size: None,
            workers: None,
            input: None,
            output: None,
            resume: false,
        })
        .context("embed step failed in e2e")?;
    }

    if !args.skip_deploy {
        deploy::run(RagDeployArgs {
            verbose: args.verbose,
            url: Some(qdrant_url.clone()),
            api_key: None,
            collection: Some(collection.clone()),
            input: None,
            batch_size: None,
            recreate: args.recreate,
            from_bundle: false,
        })
        .context("deploy step failed in e2e")?;
    }

    let exe = std::env::current_exe().context("failed to resolve current executable path")?;
    let mut cmd = Command::new(exe);
    cmd.arg("cli")
        .arg("search_docs")
        .arg(&query)
        .arg("--top-k")
        .arg(top_k.to_string())
        .arg("--qdrant-url")
        .arg(&qdrant_url)
        .arg("--collection")
        .arg(&collection)
        .arg("--json");

    if let Some(version) = &args.doc_version {
        cmd.arg("--doc-version").arg(version);
    }
    if args.rerank {
        cmd.arg("--rerank");
    } else {
        cmd.arg("--no-rerank");
    }

    let output = cmd
        .output()
        .context("failed to execute cli search_docs in e2e")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("search_docs step failed in e2e: {}", stderr.trim()));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let value: Value = serde_json::from_str(&stdout)
        .context("search_docs in e2e did not return valid JSON output")?;
    let result_count = value
        .get("results")
        .and_then(Value::as_array)
        .map(|arr| arr.len())
        .unwrap_or(0usize);

    println!(
        "E2E completed successfully: results={} rerank={}",
        result_count, args.rerank
    );
    Ok(())
}
