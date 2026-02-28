use clap::{Args, Subcommand};

#[derive(Debug, Args)]
pub struct CliArgs {
    #[command(subcommand)]
    pub command: CliCommand,
}

#[derive(Debug, Subcommand)]
pub enum CliCommand {
    /// Search the documentation vector index
    #[command(name = "search_docs", visible_alias = "search-docs")]
    SearchDocs(SearchDocsArgs),
}

#[derive(Debug, Args)]
pub struct SearchDocsArgs {
    /// Search query text
    pub query: String,

    /// Number of relevant chunks to return
    #[arg(long, default_value_t = 5)]
    pub top_k: usize,

    /// Relevance threshold [0..1]
    #[arg(long, default_value_t = 0.0)]
    pub score_threshold: f32,

    /// Return JSON instead of text
    #[arg(long, default_value_t = false)]
    pub json: bool,

    /// Include full chunk text in results
    #[arg(long, default_value_t = false)]
    pub with_content: bool,
}

pub fn run(args: CliArgs) {
    match args.command {
        CliCommand::SearchDocs(search) => {
            println!("[stub] cli search_docs: {search:#?}");
        }
    }
}
