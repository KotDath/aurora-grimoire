use super::{RagDeployArgs, RagEmbedArgs, RagTestE2eArgs, deploy, embed};
use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use std::process::Command;

pub fn run(args: RagTestE2eArgs) -> Result<()> {
    if !args.skip_embed {
        embed::run(RagEmbedArgs {
            verbose: args.verbose,
            ollama_url: "http://127.0.0.1:11434".to_string(),
            model: "qwen3-embedding:0.6b".to_string(),
            batch_size: 16,
            workers: 2,
            input: None,
            output: None,
            resume: false,
        })
        .context("embed step failed in e2e")?;
    }

    if !args.skip_deploy {
        deploy::run(RagDeployArgs {
            verbose: args.verbose,
            url: args.qdrant_url.clone(),
            api_key: None,
            collection: args.collection.clone(),
            input: None,
            batch_size: 256,
            recreate: args.recreate,
            from_bundle: false,
        })
        .context("deploy step failed in e2e")?;
    }

    let exe = std::env::current_exe().context("failed to resolve current executable path")?;
    let mut cmd = Command::new(exe);
    cmd.arg("cli")
        .arg("search_docs")
        .arg(&args.query)
        .arg("--top-k")
        .arg(args.top_k.max(1).to_string())
        .arg("--qdrant-url")
        .arg(&args.qdrant_url)
        .arg("--collection")
        .arg(&args.collection)
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
