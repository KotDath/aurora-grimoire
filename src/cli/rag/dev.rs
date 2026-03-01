use super::{
    RagDevArgs, RagDevCommand, RagDevDownArgs, RagDevLogsArgs, RagDevStatusArgs, RagDevUpArgs,
};
use anyhow::{Context, Result, anyhow};
use reqwest::blocking::Client;
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
    thread,
    time::{Duration, Instant},
};

const PROJECT_NAME: &str = "aurora_grimoire_dev";
const DEFAULT_RERANK_MODEL: &str = "BAAI/bge-reranker-v2-m3";

macro_rules! vprintln {
    ($verbose:expr, $($arg:tt)*) => {
        if $verbose {
            println!($($arg)*);
        }
    };
}

pub fn run(args: RagDevArgs) -> Result<()> {
    match args.command {
        RagDevCommand::Up(up) => run_up(up),
        RagDevCommand::Down(down) => run_down(down),
        RagDevCommand::Status(status) => run_status(status),
        RagDevCommand::Logs(logs) => run_logs(logs),
    }
}

fn run_up(args: RagDevUpArgs) -> Result<()> {
    let compose_files = compose_file_paths(args.gpu)?;
    assert_docker_ready()?;
    let mut services = vec!["qdrant".to_string(), "ollama".to_string()];
    if args.with_rerank {
        services.push("rerank".to_string());
    }

    let mut up_args = vec![
        "compose".to_string(),
        "-p".to_string(),
        PROJECT_NAME.to_string(),
    ];
    append_compose_files(&mut up_args, &compose_files);
    up_args.push("up".to_string());
    up_args.push("-d".to_string());
    if args.build {
        up_args.push("--build".to_string());
    }
    up_args.extend(services);
    if args.verbose {
        // Stream docker build/start logs live when verbose is enabled.
        run_cmd_stream("docker", &up_args, args.verbose)
            .context("failed to start Docker compose services")?;
    } else {
        run_cmd("docker", &up_args, args.verbose)
            .context("failed to start Docker compose services")?;
    }

    wait_for_http_ready(
        "Qdrant",
        "http://127.0.0.1:6333/collections",
        args.wait_timeout_sec,
        args.verbose,
    )?;
    wait_for_http_ready(
        "Ollama",
        "http://127.0.0.1:11434/api/tags",
        args.wait_timeout_sec,
        args.verbose,
    )?;
    if args.with_rerank {
        wait_for_http_ready(
            "Rerank",
            "http://127.0.0.1:8081/health",
            args.wait_timeout_sec,
            args.verbose,
        )?;
    }

    if !args.skip_model_pull {
        let mut pull_args = vec![
            "compose".to_string(),
            "-p".to_string(),
            PROJECT_NAME.to_string(),
        ];
        append_compose_files(&mut pull_args, &compose_files);
        pull_args.extend([
            "exec".to_string(),
            "-T".to_string(),
            "ollama".to_string(),
            "ollama".to_string(),
            "pull".to_string(),
            args.model.clone(),
        ]);
        run_cmd("docker", &pull_args, args.verbose).with_context(|| {
            format!(
                "failed to pull embedding model '{}' into ollama container",
                args.model
            )
        })?;
    }

    if args.with_rerank {
        println!(
            "RAG dev stack is up (qdrant=6333, ollama=11434, rerank=8081, rerank_model={})",
            env::var("RERANK_MODEL").unwrap_or_else(|_| DEFAULT_RERANK_MODEL.to_string())
        );
    } else {
        println!("RAG dev stack is up (qdrant=6333, ollama=11434, rerank=off)");
    }
    Ok(())
}

fn run_down(args: RagDevDownArgs) -> Result<()> {
    let compose_file = compose_file_path()?;
    assert_docker_ready()?;

    let mut down_args = vec![
        "compose".to_string(),
        "-p".to_string(),
        PROJECT_NAME.to_string(),
        "-f".to_string(),
        compose_file.to_string_lossy().to_string(),
        "down".to_string(),
    ];
    if args.volumes {
        down_args.push("-v".to_string());
    }
    run_cmd("docker", &down_args, args.verbose)
        .context("failed to stop Docker compose services")?;
    println!("RAG dev stack is down");
    Ok(())
}

fn run_status(args: RagDevStatusArgs) -> Result<()> {
    let compose_file = compose_file_path()?;
    assert_docker_ready()?;

    let ps_args = vec![
        "compose".to_string(),
        "-p".to_string(),
        PROJECT_NAME.to_string(),
        "-f".to_string(),
        compose_file.to_string_lossy().to_string(),
        "ps".to_string(),
    ];
    run_cmd("docker", &ps_args, args.verbose).context("failed to inspect Docker compose status")?;

    let qdrant_ok = is_http_ready("http://127.0.0.1:6333/collections");
    let ollama_ok = is_http_ready("http://127.0.0.1:11434/api/tags");

    if args.with_rerank {
        let rerank_ok = is_http_ready("http://127.0.0.1:8081/health");
        println!(
            "health: qdrant={} ollama={} rerank={}",
            if qdrant_ok { "ok" } else { "down" },
            if ollama_ok { "ok" } else { "down" },
            if rerank_ok { "ok" } else { "down" }
        );
    } else {
        println!(
            "health: qdrant={} ollama={}",
            if qdrant_ok { "ok" } else { "down" },
            if ollama_ok { "ok" } else { "down" }
        );
    }
    Ok(())
}

fn run_logs(args: RagDevLogsArgs) -> Result<()> {
    let compose_file = compose_file_path()?;
    assert_docker_ready()?;

    let mut logs_args = vec![
        "compose".to_string(),
        "-p".to_string(),
        PROJECT_NAME.to_string(),
        "-f".to_string(),
        compose_file.to_string_lossy().to_string(),
        "logs".to_string(),
        "--tail".to_string(),
        args.tail.to_string(),
    ];
    if args.follow {
        logs_args.push("--follow".to_string());
    }
    if args.services.is_empty() {
        logs_args.push("qdrant".to_string());
        logs_args.push("ollama".to_string());
    } else {
        logs_args.extend(args.services);
    }

    run_cmd_stream("docker", &logs_args, args.verbose)
        .context("failed to read Docker compose logs")?;
    Ok(())
}

fn compose_file_path() -> Result<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("ops/docker-compose.rag-dev.yml");
    if !path.exists() {
        return Err(anyhow!("compose file does not exist: {}", path.display()));
    }
    Ok(path)
}

fn compose_file_paths(with_gpu: bool) -> Result<Vec<PathBuf>> {
    let mut paths = vec![compose_file_path()?];
    if with_gpu {
        let gpu_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("ops/docker-compose.rag-dev.gpu.yml");
        if !gpu_path.exists() {
            return Err(anyhow!(
                "GPU compose override does not exist: {}",
                gpu_path.display()
            ));
        }
        paths.push(gpu_path);
    }
    Ok(paths)
}

fn append_compose_files(args: &mut Vec<String>, compose_files: &[PathBuf]) {
    for file in compose_files {
        args.push("-f".to_string());
        args.push(file.to_string_lossy().to_string());
    }
}

fn assert_docker_ready() -> Result<()> {
    let output = Command::new("docker")
        .arg("compose")
        .arg("version")
        .output()
        .context("failed to execute 'docker compose version'")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "docker compose is not available or not running: {}",
            stderr.trim()
        ));
    }
    Ok(())
}

fn run_cmd(bin: &str, args: &[String], verbose: bool) -> Result<()> {
    vprintln!(verbose, "[dev] running: {} {}", bin, args.join(" "));
    let output = Command::new(bin)
        .args(args)
        .output()
        .with_context(|| format!("failed to execute '{} {}'", bin, args.join(" ")))?;
    if output.status.success() {
        if verbose {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if !stdout.trim().is_empty() {
                println!("{}", stdout.trim());
            }
        }
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stderr_text = enhance_device_error(stderr.trim());
    Err(anyhow!(
        "command failed: {} {}: {}",
        bin,
        args.join(" "),
        stderr_text
    ))
}

fn run_cmd_stream(bin: &str, args: &[String], verbose: bool) -> Result<()> {
    vprintln!(verbose, "[dev] running: {} {}", bin, args.join(" "));
    let status = Command::new(bin)
        .args(args)
        .status()
        .with_context(|| format!("failed to execute '{} {}'", bin, args.join(" ")))?;
    if status.success() {
        return Ok(());
    }
    Err(anyhow!(
        "command failed: {} {}: exit status {}",
        bin,
        args.join(" "),
        status
    ))
}

fn enhance_device_error(stderr: &str) -> String {
    if stderr.contains("custom device \"/dev/dri\": no such file or directory") {
        return format!(
            "{}\nhint: current Docker daemon cannot see /dev/dri. If you use Docker Desktop context, switch to host daemon and run with `rag dev up --gpu` there.",
            stderr
        );
    }
    stderr.to_string()
}

fn wait_for_http_ready(name: &str, url: &str, timeout_sec: u64, verbose: bool) -> Result<()> {
    let client = Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .context("failed to build HTTP client for readiness checks")?;
    let timeout = Duration::from_secs(timeout_sec.max(1));
    let started = Instant::now();
    loop {
        if let Ok(resp) = client.get(url).send()
            && resp.status().is_success()
        {
            vprintln!(verbose, "[dev] {} is ready at {}", name, url);
            return Ok(());
        }
        if started.elapsed() >= timeout {
            return Err(anyhow!(
                "{} did not become ready within {}s ({})",
                name,
                timeout_sec,
                url
            ));
        }
        thread::sleep(Duration::from_secs(2));
    }
}

fn is_http_ready(url: &str) -> bool {
    let Ok(client) = Client::builder().timeout(Duration::from_secs(3)).build() else {
        return false;
    };
    let Ok(resp) = client.get(url).send() else {
        return false;
    };
    resp.status().is_success()
}
