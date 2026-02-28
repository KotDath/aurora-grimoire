use clap::{Args, Subcommand};

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
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port for HTTP mode
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
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

pub fn run(args: McpArgs) {
    match args.command {
        McpCommand::Start(start) => {
            let transport = start.transport();
            println!(
                "[stub] mcp start: transport={transport:?}, host={}, port={}",
                start.host, start.port
            );
        }
    }
}
