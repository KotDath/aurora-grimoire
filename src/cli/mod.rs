pub mod mcp;
pub mod rag;
pub mod user_cli;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "aurora-grimoire",
    version,
    about = "CLI/MCP tool for AuroraOS documentation.",
    arg_required_else_help = true
)]
pub struct App {
    #[command(subcommand)]
    pub command: RootCommand,
}

#[derive(Debug, Subcommand)]
pub enum RootCommand {
    /// MCP server and transport
    Mcp(mcp::McpArgs),
    /// RAG pipeline (fetch -> struct -> chunk -> deploy)
    Rag(rag::RagArgs),
    /// User CLI operations
    Cli(user_cli::CliArgs),
}

pub fn run(app: App) {
    match app.command {
        RootCommand::Mcp(args) => mcp::run(args),
        RootCommand::Rag(args) => rag::run(args),
        RootCommand::Cli(args) => user_cli::run(args),
    }
}
