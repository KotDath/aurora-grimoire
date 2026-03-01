pub mod agents;
pub mod mcp;
pub mod rag;
pub mod retrieval_bm25;
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
    /// Agent integration helpers (skills/commands installers)
    Agents(agents::AgentsArgs),
    /// MCP server and transport
    Mcp(mcp::McpArgs),
    /// RAG pipeline (fetch -> struct -> chunk -> deploy)
    Rag(rag::RagArgs),
    /// User CLI operations
    Cli(user_cli::CliArgs),
}

pub fn run(app: App) {
    let needs_config = match &app.command {
        RootCommand::Rag(args) => !matches!(args.command, rag::RagCommand::Config(_)),
        RootCommand::Agents(_) => false,
        RootCommand::Mcp(_) | RootCommand::Cli(_) => true,
    };

    if needs_config && let Err(err) = crate::config::AppConfig::load_required() {
        eprintln!("[error] {}", err);
        std::process::exit(1);
    }

    match app.command {
        RootCommand::Agents(args) => agents::run(args),
        RootCommand::Mcp(args) => mcp::run(args),
        RootCommand::Rag(args) => rag::run(args),
        RootCommand::Cli(args) => user_cli::run(args),
    }
}
