use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Subcommand, ValueEnum};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Args)]
pub struct AgentsArgs {
    #[command(subcommand)]
    pub command: AgentsCommand,
}

#[derive(Debug, Subcommand)]
pub enum AgentsCommand {
    /// Install Aurora docs helper skill/command for supported agent runtimes
    Install(AgentsInstallArgs),
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum AgentRuntime {
    Claude,
    Opencode,
    Codex,
    All,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum InstallScope {
    Global,
    Local,
}

#[derive(Debug, Args)]
pub struct AgentsInstallArgs {
    /// Target runtime: claude, opencode, codex, or all
    #[arg(long, value_enum, default_value_t = AgentRuntime::All)]
    pub runtime: AgentRuntime,

    /// Install scope: global (~/.config) or local (./.<runtime>)
    #[arg(long, value_enum, default_value_t = InstallScope::Local)]
    pub scope: InstallScope,

    /// Override runtime config directory (only when --runtime is not all)
    #[arg(long)]
    pub config_dir: Option<PathBuf>,

    /// Overwrite existing files
    #[arg(long, default_value_t = false)]
    pub force: bool,

    /// Verbose logs
    #[arg(short, long, default_value_t = false)]
    pub verbose: bool,
}

const CLAUDE_COMMAND_CONTENT: &str = r#"---
name: aurora:search-docs
description: Search Aurora OS docs via aurora-grimoire MCP and answer with source links.
argument-hint: "<question> [--doc-version X.Y.Z]"
---
<objective>
Answer Aurora OS documentation questions with verifiable source links.
</objective>

<process>
1. Parse $ARGUMENTS:
   - Extract optional `--doc-version X.Y.Z`.
   - Remaining text is query.
2. If query is empty, ask user to provide question text.
3. Call MCP tool `search_docs` with:
   - query: parsed question
   - doc_version: parsed version (if provided)
   - top_k: 8
   - knowledge_threshold: "medium"
4. If tool returns `no_knowledge=true`, state that documentation does not contain reliable answer.
5. Otherwise produce answer in user language and include a `Sources:` list with returned URLs.
</process>
"#;

const OPENCODE_COMMAND_CONTENT: &str = r#"---
description: Search Aurora OS docs via aurora-grimoire MCP and answer with sources.
---
Use MCP tool `search_docs` to answer the user question.

Input:
- Arguments are user query text.
- Optional: `--doc-version X.Y.Z`.

Execution:
1. Parse optional `--doc-version`.
2. Call `search_docs` with `query`, optional `doc_version`, `top_k=8`, `knowledge_threshold="medium"`.
3. If `no_knowledge=true`, report that reliable information was not found in docs.
4. Otherwise answer in user language and append `Sources:` with URLs from tool response.
"#;

const CODEX_SKILL_CONTENT: &str = r#"---
name: aurora-search-docs
description: Search Aurora OS docs through aurora-grimoire MCP tool and answer with source links.
---

# Aurora Docs Search

Invoke with: `$aurora-search-docs <question> [--doc-version X.Y.Z]`

Use this skill when the user asks documentation questions about Aurora OS.

## Workflow
1. Parse user text:
- required: question text
- optional: `--doc-version X.Y.Z`
2. Call MCP tool `search_docs` with:
- `query`: question text
- `doc_version`: optional parsed version
- `top_k`: `8`
- `knowledge_threshold`: `medium`
3. If response has `no_knowledge=true`, return a concise "not found in current docs" answer.
4. Else produce an answer in the user's language.
5. Always include a `Sources:` block with returned URLs.

## Notes
- Do not invent facts outside retrieved sources.
- Prefer `source_url_with_anchor` when available.
"#;

pub fn run(args: AgentsArgs) {
    match args.command {
        AgentsCommand::Install(install) => {
            if let Err(err) = run_install(install) {
                eprintln!("[error] agents install failed: {err:#}");
                std::process::exit(1);
            }
        }
    }
}

fn run_install(args: AgentsInstallArgs) -> Result<()> {
    if args.runtime == AgentRuntime::All && args.config_dir.is_some() {
        bail!("--config-dir can only be used with a single runtime");
    }

    let runtimes = match args.runtime {
        AgentRuntime::All => vec![
            AgentRuntime::Claude,
            AgentRuntime::Opencode,
            AgentRuntime::Codex,
        ],
        one => vec![one],
    };

    let mut installed = 0usize;
    let mut skipped = 0usize;
    let mut touched_paths: Vec<PathBuf> = Vec::new();

    for runtime in runtimes {
        let base = if let Some(custom) = &args.config_dir {
            custom.clone()
        } else {
            default_base_dir(runtime, args.scope)?
        };

        if args.verbose {
            eprintln!(
                "[agents] runtime={} scope={} base={}",
                runtime_name(runtime),
                scope_name(args.scope),
                base.display()
            );
        }

        let (path, content) = runtime_target(runtime, &base);
        let outcome = write_if_needed(&path, content, args.force)?;
        match outcome {
            WriteOutcome::Installed => {
                installed += 1;
                touched_paths.push(path.clone());
                if args.verbose {
                    eprintln!("[agents][ok] installed {}", path.display());
                }
            }
            WriteOutcome::Skipped => {
                skipped += 1;
                if args.verbose {
                    eprintln!(
                        "[agents][skip] {} already exists (use --force to overwrite)",
                        path.display()
                    );
                }
            }
        }
    }

    if installed == 0 && skipped > 0 {
        println!("No files installed: all target files already exist (use --force).");
    } else {
        println!("Installed {} agent integration file(s).", installed);
    }

    if args.verbose && !touched_paths.is_empty() {
        eprintln!("[agents] written paths:");
        for path in touched_paths {
            eprintln!("  - {}", path.display());
        }
    }

    Ok(())
}

enum WriteOutcome {
    Installed,
    Skipped,
}

fn write_if_needed(path: &Path, content: &str, force: bool) -> Result<WriteOutcome> {
    if path.exists() && !force {
        return Ok(WriteOutcome::Skipped);
    }
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("target path has no parent: {}", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create directory {}", parent.display()))?;
    fs::write(path, content).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(WriteOutcome::Installed)
}

fn runtime_target(runtime: AgentRuntime, base: &Path) -> (PathBuf, &'static str) {
    match runtime {
        AgentRuntime::Claude => (
            base.join("commands").join("aurora").join("search-docs.md"),
            CLAUDE_COMMAND_CONTENT,
        ),
        AgentRuntime::Opencode => (
            base.join("command").join("aurora-search-docs.md"),
            OPENCODE_COMMAND_CONTENT,
        ),
        AgentRuntime::Codex => (
            base.join("skills")
                .join("aurora-search-docs")
                .join("SKILL.md"),
            CODEX_SKILL_CONTENT,
        ),
        AgentRuntime::All => unreachable!("all runtime is expanded earlier"),
    }
}

fn default_base_dir(runtime: AgentRuntime, scope: InstallScope) -> Result<PathBuf> {
    match scope {
        InstallScope::Local => {
            let cwd = env::current_dir().context("failed to resolve current directory")?;
            let dir = match runtime {
                AgentRuntime::Claude => ".claude",
                AgentRuntime::Opencode => ".opencode",
                AgentRuntime::Codex => ".codex",
                AgentRuntime::All => return Err(anyhow!("runtime=all is not valid here")),
            };
            Ok(cwd.join(dir))
        }
        InstallScope::Global => match runtime {
            AgentRuntime::Claude => {
                if let Ok(path) = env::var("CLAUDE_CONFIG_DIR") {
                    return Ok(expand_tilde(path));
                }
                Ok(home_dir()?.join(".claude"))
            }
            AgentRuntime::Opencode => {
                if let Ok(path) = env::var("OPENCODE_CONFIG_DIR") {
                    return Ok(expand_tilde(path));
                }
                if let Ok(path) = env::var("OPENCODE_CONFIG") {
                    return Ok(expand_tilde(path)
                        .parent()
                        .ok_or_else(|| anyhow!("invalid OPENCODE_CONFIG path"))?
                        .to_path_buf());
                }
                if let Ok(path) = env::var("XDG_CONFIG_HOME") {
                    return Ok(expand_tilde(path).join("opencode"));
                }
                Ok(home_dir()?.join(".config").join("opencode"))
            }
            AgentRuntime::Codex => {
                if let Ok(path) = env::var("CODEX_HOME") {
                    return Ok(expand_tilde(path));
                }
                Ok(home_dir()?.join(".codex"))
            }
            AgentRuntime::All => Err(anyhow!("runtime=all is not valid here")),
        },
    }
}

fn home_dir() -> Result<PathBuf> {
    dirs::home_dir().ok_or_else(|| anyhow!("failed to resolve home directory"))
}

fn expand_tilde(raw: String) -> PathBuf {
    if let Some(stripped) = raw.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(stripped);
        }
    }
    PathBuf::from(raw)
}

fn runtime_name(runtime: AgentRuntime) -> &'static str {
    match runtime {
        AgentRuntime::Claude => "claude",
        AgentRuntime::Opencode => "opencode",
        AgentRuntime::Codex => "codex",
        AgentRuntime::All => "all",
    }
}

fn scope_name(scope: InstallScope) -> &'static str {
    match scope {
        InstallScope::Global => "global",
        InstallScope::Local => "local",
    }
}
