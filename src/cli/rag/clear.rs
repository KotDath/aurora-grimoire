use super::RagClearArgs;
use crate::config::AppConfig;
use anyhow::{Context, Result, anyhow};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

pub fn run(args: RagClearArgs) -> Result<()> {
    let root = AppConfig::load()?.data_root()?;

    let targets = collect_targets(&root, &args);
    if targets.is_empty() {
        return Err(anyhow!(
            "no targets selected; use one of --html, --md, --chunks, --index or --all"
        ));
    }

    let mut removed = Vec::new();
    let mut skipped = Vec::new();
    for target in &targets {
        if remove_target(target)? {
            removed.push(short_label(&root, target));
        } else {
            skipped.push(short_label(&root, target));
        }
    }

    if removed.is_empty() {
        println!("[clear] nothing to remove");
    } else {
        println!(
            "[clear] removed {} target(s): {}",
            removed.len(),
            removed.join(", ")
        );
    }
    if !skipped.is_empty() {
        println!("[clear] skipped missing target(s): {}", skipped.join(", "));
    }
    println!("[clear] root: {}", root.display());
    Ok(())
}

fn collect_targets(root: &Path, args: &RagClearArgs) -> Vec<PathBuf> {
    let mut out = BTreeSet::new();

    if args.all || args.html {
        out.insert(root.join("html_data"));
    }
    if args.all || args.md {
        out.insert(root.join("md_data"));
    }
    if args.all || args.chunks {
        out.insert(root.join("chunks"));
    }
    if args.all || args.index {
        out.insert(root.join("vectors_data"));
        out.insert(root.join("bm25_data"));
    }
    if args.all {
        out.insert(root.join("bundles"));
    }

    out.into_iter().collect::<Vec<_>>()
}

fn remove_target(path: &Path) -> Result<bool> {
    if !path.exists() {
        return Ok(false);
    }
    if path.is_dir() {
        fs::remove_dir_all(path)
            .with_context(|| format!("failed to remove directory {}", path.display()))?;
    } else {
        fs::remove_file(path)
            .with_context(|| format!("failed to remove file {}", path.display()))?;
    }
    Ok(true)
}

fn short_label(root: &Path, target: &Path) -> String {
    target
        .strip_prefix(root)
        .ok()
        .and_then(|p| p.to_str())
        .map(ToString::to_string)
        .unwrap_or_else(|| target.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(all: bool, html: bool, md: bool, chunks: bool, index: bool) -> RagClearArgs {
        RagClearArgs {
            all,
            html,
            md,
            chunks,
            index,
        }
    }

    #[test]
    fn collect_targets_for_index_includes_vectors_and_bm25() {
        let root = PathBuf::from("/tmp/aurora-grimoire-test");
        let out = collect_targets(&root, &args(false, false, false, false, true));
        assert!(out.contains(&root.join("vectors_data")));
        assert!(out.contains(&root.join("bm25_data")));
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn collect_targets_all_includes_known_directories() {
        let root = PathBuf::from("/tmp/aurora-grimoire-test");
        let out = collect_targets(&root, &args(true, false, false, false, false));
        let expected = [
            root.join("html_data"),
            root.join("md_data"),
            root.join("chunks"),
            root.join("vectors_data"),
            root.join("bm25_data"),
            root.join("bundles"),
        ];
        for path in expected {
            assert!(out.contains(&path));
        }
    }
}
