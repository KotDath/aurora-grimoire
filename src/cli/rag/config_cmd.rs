use super::{RagConfigArgs, RagConfigCommand, RagConfigInitArgs};
use crate::config::resolve_config_path;
use anyhow::{Context, Result, anyhow};
use std::fs;

pub fn run(args: RagConfigArgs) -> Result<()> {
    match args.command {
        RagConfigCommand::Init(init) => run_init(init),
    }
}

fn run_init(args: RagConfigInitArgs) -> Result<()> {
    let config_path = resolve_config_path()?;
    if config_path.exists() && !args.force {
        return Err(anyhow!(
            "config already exists at {}. Use --force to overwrite.",
            config_path.display()
        ));
    }

    if let Some(parent) = config_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "failed to create parent directories for {}",
                config_path.display()
            )
        })?;
    }

    let template = include_str!("../../../config.example.toml");
    fs::write(&config_path, template).with_context(|| {
        format!(
            "failed to write config template to {}",
            config_path.display()
        )
    })?;

    println!("Config initialized at {}", config_path.display());
    Ok(())
}
