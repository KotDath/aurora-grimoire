mod chunk_md;
mod fetch_web;
mod struct_md;

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
        RagCommand::Deploy(deploy) => {
            println!("[stub] rag deploy: {deploy:#?}");
        }
        RagCommand::Clear(clear) => {
            println!("[stub] rag clear: {clear:#?}");
        }
    }
}
