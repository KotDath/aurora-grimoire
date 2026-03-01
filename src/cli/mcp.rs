use crate::config::{
    AppConfig, DEFAULT_MCP_HOST, DEFAULT_MCP_PORT, DEFAULT_SEARCH_KNOWLEDGE_THRESHOLD,
    DEFAULT_SEARCH_RETRIEVAL_MODE,
};
use anyhow::{Context, Result, anyhow};
use clap::{Args, Subcommand};
use rmcp::{
    ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolRequestParams, ServerCapabilities, ServerInfo},
    object, schemars, tool, tool_handler, tool_router,
    transport::{
        ConfigureCommandExt, TokioChildProcess, stdio,
        streamable_http_server::{
            StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
        },
    },
};
use std::{
    path::PathBuf,
    process::{Command as StdCommand, Stdio},
};
use tokio_util::sync::CancellationToken;

const DEFAULT_MCP_SMOKE_QUERY: &str = "как собрать проект через mb2";

#[derive(Debug, Clone, Copy)]
enum McpTransport {
    Stdio,
    Http,
}

#[derive(Debug, Args)]
pub struct McpArgs {
    #[command(subcommand)]
    pub command: McpCommand,
}

#[derive(Debug, Subcommand)]
pub enum McpCommand {
    /// Start the MCP server
    Start(McpStartArgs),
    /// Run local MCP smoke test against this binary
    Smoke(McpSmokeArgs),
}

#[derive(Debug, Args)]
pub struct McpStartArgs {
    /// Use HTTP transport
    #[arg(long, default_value_t = false, conflicts_with = "stdio")]
    pub http: bool,

    /// Use stdio transport (default)
    #[arg(long, default_value_t = false, conflicts_with = "http")]
    pub stdio: bool,

    /// Host for HTTP mode
    #[arg(long)]
    pub host: Option<String>,

    /// Port for HTTP mode
    #[arg(long)]
    pub port: Option<u16>,
}

#[derive(Debug, Args)]
pub struct McpSmokeArgs {
    /// Search query for smoke call
    #[arg(long)]
    pub query: Option<String>,

    /// Optional docs version filter
    #[arg(long = "doc-version")]
    pub doc_version: Option<String>,

    /// Retrieval mode passed to search_docs
    #[arg(long)]
    pub retrieval_mode: Option<String>,

    /// Knowledge threshold passed to search_docs
    #[arg(long)]
    pub knowledge_threshold: Option<String>,

    /// top-k passed to search_docs
    #[arg(long)]
    pub top_k: Option<usize>,

    /// Print raw JSON payload returned by tool
    #[arg(long, default_value_t = false)]
    pub raw: bool,
}

impl McpStartArgs {
    fn transport(&self) -> McpTransport {
        if self.http {
            McpTransport::Http
        } else {
            McpTransport::Stdio
        }
    }
}

#[derive(Debug, Clone)]
struct SearchDocsMcpServer {
    binary_path: PathBuf,
    tool_router: ToolRouter<Self>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct SearchDocsToolInput {
    #[schemars(description = "Search query text")]
    query: String,
    #[schemars(description = "Documentation version filter, e.g. 5.2.0")]
    doc_version: Option<String>,
    #[schemars(description = "Number of chunks to return")]
    top_k: Option<usize>,
    #[schemars(description = "Retrieval mode: hybrid|dense|bm25")]
    retrieval_mode: Option<String>,
    #[schemars(description = "Confidence gate: low|medium|high")]
    knowledge_threshold: Option<String>,
    #[schemars(description = "Enable rerank stage")]
    rerank: Option<bool>,
    #[schemars(description = "Include full content in each result")]
    with_content: Option<bool>,
}

#[tool_router]
impl SearchDocsMcpServer {
    fn new(binary_path: PathBuf) -> Self {
        Self {
            binary_path,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "search_docs",
        description = "Search Aurora documentation and return JSON results with source links"
    )]
    async fn search_docs(
        &self,
        Parameters(input): Parameters<SearchDocsToolInput>,
    ) -> Result<String, String> {
        self.run_cli_search(input).map_err(|e| e.to_string())
    }

    fn run_cli_search(&self, input: SearchDocsToolInput) -> Result<String> {
        let mut cmd = StdCommand::new(&self.binary_path);
        cmd.arg("cli")
            .arg("search_docs")
            .arg(input.query)
            .arg("--json")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(doc_version) = input.doc_version.as_deref().map(str::trim) {
            if !doc_version.is_empty() {
                cmd.arg("--doc-version").arg(doc_version);
            }
        }
        if let Some(top_k) = input.top_k {
            cmd.arg("--top-k").arg(top_k.to_string());
        }
        if let Some(mode) = input.retrieval_mode.as_deref().map(str::trim) {
            if !mode.is_empty() {
                cmd.arg("--retrieval-mode").arg(mode);
            }
        }
        if let Some(threshold) = input.knowledge_threshold.as_deref().map(str::trim) {
            if !threshold.is_empty() {
                cmd.arg("--knowledge-threshold").arg(threshold);
            }
        }
        if input.with_content.unwrap_or(false) {
            cmd.arg("--with-content");
        }
        match input.rerank {
            Some(true) => {
                cmd.arg("--rerank");
            }
            Some(false) => {
                cmd.arg("--no-rerank");
            }
            None => {}
        }

        let output = cmd
            .output()
            .with_context(|| format!("failed to execute {}", self.binary_path.display()))?;
        let stdout = String::from_utf8(output.stdout).context("search_docs stdout is not UTF-8")?;
        let stderr = String::from_utf8(output.stderr).unwrap_or_default();

        if !output.status.success() {
            let details = if !stderr.trim().is_empty() {
                stderr.trim().to_string()
            } else {
                stdout.trim().to_string()
            };
            return Err(anyhow!(
                "search_docs command failed (status={}): {}",
                output.status,
                details
            ));
        }

        serde_json::from_str::<serde_json::Value>(&stdout)
            .context("search_docs did not return valid JSON")?;
        Ok(stdout)
    }
}

#[tool_handler]
impl ServerHandler for SearchDocsMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "Use tool search_docs to query Aurora documentation and get source-linked segments."
                    .into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

pub fn run(args: McpArgs) {
    match args.command {
        McpCommand::Start(start) => {
            if let Err(err) = run_start(start) {
                eprintln!("[error] mcp start failed: {err:#}");
                std::process::exit(1);
            }
        }
        McpCommand::Smoke(smoke) => {
            if let Err(err) = run_smoke(smoke) {
                eprintln!("[error] mcp smoke failed: {err:#}");
                std::process::exit(1);
            }
        }
    }
}

fn run_start(start: McpStartArgs) -> Result<()> {
    let cfg = AppConfig::load()?;
    let transport = start.transport();
    let host = start
        .host
        .clone()
        .or_else(|| cfg.mcp.host.clone())
        .unwrap_or_else(|| DEFAULT_MCP_HOST.to_string());
    let port = start.port.or(cfg.mcp.port).unwrap_or(DEFAULT_MCP_PORT);
    let binary_path =
        std::env::current_exe().context("failed to resolve current executable path")?;
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to create Tokio runtime for MCP server")?;

    rt.block_on(async move {
        match transport {
            McpTransport::Stdio => run_stdio(binary_path).await,
            McpTransport::Http => run_http(binary_path, host, port).await,
        }
    })
}

async fn run_stdio(binary_path: PathBuf) -> Result<()> {
    eprintln!(
        "[mcp] starting stdio server with tool search_docs (binary={})",
        binary_path.display()
    );
    let service = SearchDocsMcpServer::new(binary_path)
        .serve(stdio())
        .await
        .context("failed to serve MCP over stdio")?;
    service.waiting().await.context("stdio MCP server failed")?;
    Ok(())
}

async fn run_http(binary_path: PathBuf, host: String, port: u16) -> Result<()> {
    let addr = format!("{host}:{port}");
    eprintln!(
        "[mcp] starting streamable HTTP server at http://{}/mcp with tool search_docs",
        addr
    );

    let token = CancellationToken::new();
    let service = StreamableHttpService::new(
        move || Ok(SearchDocsMcpServer::new(binary_path.clone())),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig {
            cancellation_token: token.child_token(),
            ..Default::default()
        },
    );
    let router = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("failed to bind MCP HTTP address {addr}"))?;

    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            let _ = tokio::signal::ctrl_c().await;
            token.cancel();
        })
        .await
        .context("MCP HTTP server exited with error")?;
    Ok(())
}

fn run_smoke(args: McpSmokeArgs) -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to create Tokio runtime for MCP smoke test")?;
    rt.block_on(async move { run_smoke_async(args).await })
}

async fn run_smoke_async(args: McpSmokeArgs) -> Result<()> {
    let cfg = AppConfig::load()?;
    let query = args
        .query
        .clone()
        .unwrap_or_else(|| DEFAULT_MCP_SMOKE_QUERY.to_string());
    let retrieval_mode = args
        .retrieval_mode
        .clone()
        .or_else(|| cfg.search.retrieval_mode.clone())
        .unwrap_or_else(|| DEFAULT_SEARCH_RETRIEVAL_MODE.to_string());
    let knowledge_threshold = args
        .knowledge_threshold
        .clone()
        .or_else(|| cfg.search.knowledge_threshold.clone())
        .unwrap_or_else(|| DEFAULT_SEARCH_KNOWLEDGE_THRESHOLD.to_string());
    let top_k = args.top_k.or(cfg.search.top_k).unwrap_or(5).max(1);

    let binary_path =
        std::env::current_exe().context("failed to resolve current executable path")?;
    eprintln!(
        "[mcp-smoke] spawning stdio server from {}",
        binary_path.display()
    );

    let client = ()
        .serve(
            TokioChildProcess::new(tokio::process::Command::new(&binary_path).configure(|cmd| {
                cmd.arg("mcp").arg("start").arg("--stdio");
            }))
            .map_err(rmcp::RmcpError::transport_creation::<TokioChildProcess>)
            .context("failed to create MCP child-process transport")?,
        )
        .await
        .context("failed to initialize MCP client against stdio server")?;

    if let Some(info) = client.peer_info() {
        eprintln!(
            "[mcp-smoke] connected to server: name={} version={}",
            info.server_info.name, info.server_info.version
        );
    } else {
        eprintln!("[mcp-smoke] connected to server");
    }

    let tools = client
        .list_all_tools()
        .await
        .context("failed to list MCP tools")?;
    let has_search_docs = tools.iter().any(|t| t.name.as_ref() == "search_docs");
    if !has_search_docs {
        let _ = client.cancel().await;
        return Err(anyhow!(
            "tool 'search_docs' is not exposed by MCP server (tools={})",
            tools.len()
        ));
    }
    eprintln!(
        "[mcp-smoke] tools listed: {} (search_docs found)",
        tools.len()
    );

    let mut arguments = object!({
        "query": query,
        "top_k": top_k,
        "retrieval_mode": retrieval_mode,
        "knowledge_threshold": knowledge_threshold
    });
    if let Some(doc_version) = args.doc_version.clone() {
        arguments.insert(
            "doc_version".to_string(),
            serde_json::Value::String(doc_version),
        );
    }

    let tool_result = client
        .call_tool(CallToolRequestParams {
            meta: None,
            name: "search_docs".into(),
            arguments: Some(arguments),
            task: None,
        })
        .await
        .context("failed to call MCP tool 'search_docs'")?;
    client
        .cancel()
        .await
        .context("failed to shutdown MCP client after smoke test")?;

    let payload_text = tool_result
        .content
        .iter()
        .find_map(|c| c.raw.as_text().map(|t| t.text.clone()))
        .ok_or_else(|| anyhow!("search_docs tool result does not contain text content"))?;
    let payload_json: serde_json::Value = serde_json::from_str(&payload_text)
        .context("search_docs tool returned non-JSON text payload")?;

    let results_len = payload_json
        .get("results")
        .and_then(serde_json::Value::as_array)
        .map(|v| v.len())
        .unwrap_or(0);
    let no_knowledge = payload_json
        .get("no_knowledge")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    if args.raw {
        println!(
            "{}",
            serde_json::to_string_pretty(&payload_json).context("failed to pretty-print JSON")?
        );
    } else {
        println!(
            "[mcp-smoke] ok: search_docs call succeeded, results={}, no_knowledge={}",
            results_len, no_knowledge
        );
    }
    Ok(())
}
