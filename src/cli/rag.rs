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

    /// Write chunks to a vector store
    Deploy(RagDeployArgs),

    /// Clear RAG artifacts/index
    Clear(RagClearArgs),
}

#[derive(Debug, Args)]
pub struct RagFetchWebArgs {
    /// Documentation start URL
    #[arg(long, value_name = "URL")]
    pub start_url: Option<String>,

    /// Directory for raw HTML output
    #[arg(long, default_value = "data/raw/html")]
    pub output_dir: PathBuf,

    /// Maximum link traversal depth
    #[arg(long, default_value_t = 8)]
    pub max_depth: usize,

    /// Allow crawling external domains
    #[arg(long, default_value_t = false)]
    pub follow_external: bool,

    /// Overwrite existing files
    #[arg(long, default_value_t = false)]
    pub force: bool,
}

#[derive(Debug, Args)]
pub struct RagStructArgs {
    /// Input directory with raw HTML
    #[arg(long, default_value = "data/raw/html")]
    pub input_dir: PathBuf,

    /// Output directory for structured Markdown
    #[arg(long, default_value = "data/structured/md")]
    pub output_dir: PathBuf,

    /// Overwrite existing Markdown files
    #[arg(long, default_value_t = false)]
    pub force: bool,
}

#[derive(Debug, Args)]
pub struct RagChunkArgs {
    /// Input directory with structured Markdown
    #[arg(long, default_value = "data/structured/md")]
    pub input_dir: PathBuf,

    /// Output JSON file with chunks
    #[arg(long, default_value = "data/chunks/chunks.json")]
    pub output_file: PathBuf,

    /// Maximum chunk size in characters
    #[arg(long, default_value_t = 1200)]
    pub chunk_size: usize,

    /// Chunk overlap in characters
    #[arg(long, default_value_t = 200)]
    pub chunk_overlap: usize,

    /// Minimum chunk size
    #[arg(long, default_value_t = 50)]
    pub min_chars: usize,
}

#[derive(Debug, Args)]
pub struct RagDeployArgs {
    /// Chunk source (JSON)
    #[arg(long, default_value = "data/chunks/chunks.json")]
    pub input_file: PathBuf,

    /// Vector store provider name
    #[arg(long, default_value = "local")]
    pub provider: String,

    /// Collection/index in the vector store
    #[arg(long, default_value = "aurora_docs")]
    pub collection: String,

    /// Reset collection before upload
    #[arg(long, default_value_t = false)]
    pub reset_collection: bool,

    /// Validate input without writing to the store
    #[arg(long, default_value_t = false)]
    pub dry_run: bool,
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
            println!("[stub] rag fetch-web: {fetch:#?}");
        }
        RagCommand::Struct(st) => {
            println!("[stub] rag struct: {st:#?}");
        }
        RagCommand::Chunk(chunk) => {
            println!("[stub] rag chunk: {chunk:#?}");
        }
        RagCommand::Deploy(deploy) => {
            println!("[stub] rag deploy: {deploy:#?}");
        }
        RagCommand::Clear(clear) => {
            println!("[stub] rag clear: {clear:#?}");
        }
    }
}
