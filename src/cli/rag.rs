mod bundle;
mod chunk_md;
mod deploy;
mod dev;
mod embed;
mod fetch_web;
mod struct_md;
mod test_e2e;

use clap::{Args, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Args)]
pub struct RagArgs {
    #[command(subcommand)]
    pub command: RagCommand,
}

#[derive(Debug, Subcommand)]
pub enum RagCommand {
    /// Recursively fetch HTML documentation
    #[command(name = "fetch-web", visible_alias = "fetch_web")]
    FetchWeb(RagFetchWebArgs),

    /// Convert HTML into structured Markdown files
    #[command(name = "struct")]
    Struct(RagStructArgs),

    /// Split Markdown files into JSON chunks
    Chunk(RagChunkArgs),

    /// Compute embeddings for chunks
    Embed(RagEmbedArgs),

    /// Write vectors to a vector store
    Deploy(RagDeployArgs),

    /// Export/import portable chunk/vector bundles
    Bundle(RagBundleArgs),

    /// Manage local Docker-based RAG dev stack
    Dev(RagDevArgs),

    /// Run end-to-end smoke test (embed -> deploy -> search)
    #[command(name = "test-e2e")]
    TestE2e(RagTestE2eArgs),

    /// Clear RAG artifacts/index
    Clear(RagClearArgs),
}

#[derive(Debug, Args)]
pub struct RagFetchWebArgs {
    /// Verbose crawler logs
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,
}

#[derive(Debug, Args)]
pub struct RagStructArgs {
    /// Verbose converter logs
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,
}

#[derive(Debug, Args)]
pub struct RagChunkArgs {
    /// Verbose chunker logs
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,
}

#[derive(Debug, Args)]
pub struct RagEmbedArgs {
    /// Verbose embedding logs
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,

    /// Ollama base URL
    #[arg(long, default_value = "http://127.0.0.1:11434")]
    pub ollama_url: String,

    /// Embedding model name
    #[arg(long, default_value = "qwen3-embedding:0.6b")]
    pub model: String,

    /// Number of chunks per embedding batch
    #[arg(long, default_value_t = 16)]
    pub batch_size: usize,

    /// Parallel embedding workers
    #[arg(long, default_value_t = 2)]
    pub workers: usize,

    /// Chunk source directory
    #[arg(long)]
    pub input: Option<PathBuf>,

    /// Output vectors directory
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Resume and skip chunks that already have vectors
    #[arg(long, default_value_t = false)]
    pub resume: bool,
}

#[derive(Debug, Args)]
pub struct RagDeployArgs {
    /// Verbose deploy logs
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,

    /// Qdrant base URL
    #[arg(long, default_value = "http://127.0.0.1:6333")]
    pub url: String,

    /// Qdrant API key
    #[arg(long)]
    pub api_key: Option<String>,

    /// Collection/index in the vector store
    #[arg(long, default_value = "aurora_docs_qwen3_embedding_0_6b")]
    pub collection: String,

    /// Input vectors directory or bundle file
    #[arg(long)]
    pub input: Option<PathBuf>,

    /// Number of points per upsert batch
    #[arg(long, default_value_t = 256)]
    pub batch_size: usize,

    /// Recreate collection before upload
    #[arg(long, default_value_t = false)]
    pub recreate: bool,

    /// Treat input as bundle and read vectors from it
    #[arg(long, default_value_t = false)]
    pub from_bundle: bool,
}

#[derive(Debug, Args)]
pub struct RagBundleArgs {
    #[command(subcommand)]
    pub command: RagBundleCommand,
}

#[derive(Debug, Subcommand)]
pub enum RagBundleCommand {
    /// Create bundle from chunk/vector artifacts
    Create(RagBundleCreateArgs),
    /// Print bundle manifest
    Inspect(RagBundleInspectArgs),
    /// Extract bundle contents
    Extract(RagBundleExtractArgs),
}

#[derive(Debug, Args)]
pub struct RagBundleCreateArgs {
    /// Verbose bundle logs
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,

    /// Bundle type: chunks, vectors or dual
    #[arg(long = "type", default_value = "dual")]
    pub bundle_type: String,

    /// Input chunks directory
    #[arg(long)]
    pub input_chunks: Option<PathBuf>,

    /// Input vectors directory
    #[arg(long)]
    pub input_vectors: Option<PathBuf>,

    /// Output bundle file (.tar.zst)
    #[arg(long)]
    pub out: PathBuf,
}

#[derive(Debug, Args)]
pub struct RagBundleInspectArgs {
    /// Bundle file path
    #[arg(long)]
    pub file: PathBuf,
}

#[derive(Debug, Args)]
pub struct RagBundleExtractArgs {
    /// Verbose extract logs
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,

    /// Bundle file path
    #[arg(long)]
    pub file: PathBuf,

    /// Extraction root directory
    #[arg(long)]
    pub out: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub struct RagDevArgs {
    #[command(subcommand)]
    pub command: RagDevCommand,
}

#[derive(Debug, Subcommand)]
pub enum RagDevCommand {
    /// Start local qdrant + ollama stack
    Up(RagDevUpArgs),
    /// Stop local stack
    Down(RagDevDownArgs),
    /// Show current status and health
    Status(RagDevStatusArgs),
    /// Show docker compose logs
    Logs(RagDevLogsArgs),
}

#[derive(Debug, Args)]
pub struct RagDevUpArgs {
    /// Verbose logs
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,

    /// Build images before startup
    #[arg(long, default_value_t = false)]
    pub build: bool,

    /// Enable Ollama GPU passthrough via /dev/dri (requires host Docker daemon access)
    #[arg(long, default_value_t = false)]
    pub gpu: bool,

    /// Also start rerank service (disabled by default)
    #[arg(long, default_value_t = false)]
    pub with_rerank: bool,

    /// Embedding model to pull into Ollama
    #[arg(long, default_value = "qwen3-embedding:0.6b")]
    pub model: String,

    /// Skip model pull in Ollama
    #[arg(long, default_value_t = false)]
    pub skip_model_pull: bool,

    /// Wait timeout in seconds for services to become healthy
    #[arg(long, default_value_t = 240)]
    pub wait_timeout_sec: u64,
}

#[derive(Debug, Args)]
pub struct RagDevDownArgs {
    /// Verbose logs
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,

    /// Remove docker volumes
    #[arg(long, default_value_t = false)]
    pub volumes: bool,
}

#[derive(Debug, Args)]
pub struct RagDevStatusArgs {
    /// Verbose logs
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,

    /// Also check rerank health
    #[arg(long, default_value_t = false)]
    pub with_rerank: bool,
}

#[derive(Debug, Args)]
pub struct RagDevLogsArgs {
    /// Verbose logs
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,

    /// Follow log output
    #[arg(short = 'f', long, default_value_t = false)]
    pub follow: bool,

    /// Number of lines to show from the end of logs
    #[arg(long, default_value_t = 200)]
    pub tail: usize,

    /// Service names (qdrant, ollama, rerank). If omitted, show stack defaults.
    #[arg(long = "service")]
    pub services: Vec<String>,
}

#[derive(Debug, Args)]
pub struct RagTestE2eArgs {
    /// Verbose logs
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,

    /// Search query used for smoke test
    #[arg(long, default_value = "how to work with dbus in aurora os")]
    pub query: String,

    /// Optional docs version filter for search_docs
    #[arg(long = "doc-version")]
    pub doc_version: Option<String>,

    /// Number of results
    #[arg(long, default_value_t = 5)]
    pub top_k: usize,

    /// Enable rerank during smoke query
    #[arg(long, default_value_t = false)]
    pub rerank: bool,

    /// Qdrant base URL
    #[arg(long, default_value = "http://127.0.0.1:6333")]
    pub qdrant_url: String,

    /// Collection name
    #[arg(long, default_value = "aurora_docs_qwen3_embedding_0_6b")]
    pub collection: String,

    /// Skip embed step
    #[arg(long, default_value_t = false)]
    pub skip_embed: bool,

    /// Skip deploy step
    #[arg(long, default_value_t = false)]
    pub skip_deploy: bool,

    /// Recreate collection on deploy
    #[arg(long, default_value_t = true)]
    pub recreate: bool,
}

#[derive(Debug, Args)]
pub struct RagClearArgs {
    /// Clear all artifacts and index
    #[arg(long, default_value_t = false)]
    pub all: bool,

    /// Clear raw HTML
    #[arg(long, default_value_t = false)]
    pub html: bool,

    /// Clear structured Markdown
    #[arg(long, default_value_t = false)]
    pub md: bool,

    /// Clear JSON chunks
    #[arg(long, default_value_t = false)]
    pub chunks: bool,

    /// Clear vector store index
    #[arg(long, default_value_t = false)]
    pub index: bool,
}

pub fn run(args: RagArgs) {
    match args.command {
        RagCommand::FetchWeb(fetch) => {
            if let Err(err) = fetch_web::run(fetch) {
                eprintln!("[error] rag fetch-web failed: {err:#}");
                std::process::exit(1);
            }
        }
        RagCommand::Struct(st) => {
            if let Err(err) = struct_md::run(st) {
                eprintln!("[error] rag struct failed: {err:#}");
                std::process::exit(1);
            }
        }
        RagCommand::Chunk(chunk) => {
            if let Err(err) = chunk_md::run(chunk) {
                eprintln!("[error] rag chunk failed: {err:#}");
                std::process::exit(1);
            }
        }
        RagCommand::Embed(embed_args) => {
            if let Err(err) = embed::run(embed_args) {
                eprintln!("[error] rag embed failed: {err:#}");
                std::process::exit(1);
            }
        }
        RagCommand::Deploy(deploy) => {
            if let Err(err) = deploy::run(deploy) {
                eprintln!("[error] rag deploy failed: {err:#}");
                std::process::exit(1);
            }
        }
        RagCommand::Bundle(bundle_args) => {
            if let Err(err) = bundle::run(bundle_args) {
                eprintln!("[error] rag bundle failed: {err:#}");
                std::process::exit(1);
            }
        }
        RagCommand::Dev(dev_args) => {
            if let Err(err) = dev::run(dev_args) {
                eprintln!("[error] rag dev failed: {err:#}");
                std::process::exit(1);
            }
        }
        RagCommand::TestE2e(test_args) => {
            if let Err(err) = test_e2e::run(test_args) {
                eprintln!("[error] rag test-e2e failed: {err:#}");
                std::process::exit(1);
            }
        }
        RagCommand::Clear(clear) => {
            println!("[stub] rag clear: {clear:#?}");
        }
    }
}
