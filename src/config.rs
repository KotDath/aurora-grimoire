use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use std::{
    env, fs,
    path::{Path, PathBuf},
};

pub const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434";
pub const DEFAULT_QDRANT_URL: &str = "http://127.0.0.1:6333";
pub const DEFAULT_COLLECTION: &str = "aurora_docs_qwen3_embedding_0_6b";
pub const DEFAULT_EMBED_MODEL: &str = "qwen3-embedding:0.6b";
pub const DEFAULT_RERANK_URL: &str = "http://127.0.0.1:8081";
pub const DEFAULT_RERANK_MODEL: &str = "BAAI/bge-reranker-v2-m3";
pub const DEFAULT_MCP_HOST: &str = "127.0.0.1";
pub const DEFAULT_MCP_PORT: u16 = 8080;
pub const DEFAULT_EMBED_BATCH_SIZE: usize = 16;
pub const DEFAULT_EMBED_WORKERS: usize = 2;
pub const DEFAULT_DEPLOY_BATCH_SIZE: usize = 256;
pub const DEFAULT_DEV_WAIT_TIMEOUT_SEC: u64 = 240;
pub const DEFAULT_SEARCH_TOP_K: usize = 10;
pub const DEFAULT_SEARCH_SCORE_THRESHOLD: f32 = 0.0;
pub const DEFAULT_SEARCH_TOP_N: usize = 60;
pub const DEFAULT_SEARCH_BM25_TOP_N: usize = 300;
pub const DEFAULT_SEARCH_RRF_K: usize = 60;
pub const DEFAULT_SEARCH_DENSE_WEIGHT: f32 = 1.0;
pub const DEFAULT_SEARCH_BM25_WEIGHT: f32 = 0.55;
pub const DEFAULT_SEARCH_RERANK_TIMEOUT_MS: u64 = 30_000;
pub const DEFAULT_SEARCH_RERANK_FAIL_OPEN: bool = true;
pub const DEFAULT_SEARCH_RETRIEVAL_MODE: &str = "hybrid";
pub const DEFAULT_SEARCH_KNOWLEDGE_THRESHOLD: &str = "medium";
pub const DEFAULT_SEARCH_RERANK_ENABLED: bool = false;
pub const DEFAULT_TEST_E2E_QUERY: &str = "how to work with dbus in aurora os";

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct AppConfig {
    pub storage: StorageConfig,
    pub embed: EmbedConfig,
    pub deploy: DeployConfig,
    pub search: SearchConfig,
    pub dev: DevConfig,
    pub mcp: McpConfig,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct StorageConfig {
    pub root_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct EmbedConfig {
    pub ollama_url: Option<String>,
    pub model: Option<String>,
    pub batch_size: Option<usize>,
    pub workers: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct DeployConfig {
    pub qdrant_url: Option<String>,
    pub collection: Option<String>,
    pub batch_size: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct SearchConfig {
    pub top_k: Option<usize>,
    pub score_threshold: Option<f32>,
    pub retrieval_mode: Option<String>,
    pub knowledge_threshold: Option<String>,
    pub top_n: Option<usize>,
    pub bm25_top_n: Option<usize>,
    pub rrf_k: Option<usize>,
    pub dense_weight: Option<f32>,
    pub bm25_weight: Option<f32>,
    pub qdrant_url: Option<String>,
    pub collection: Option<String>,
    pub ollama_url: Option<String>,
    pub model: Option<String>,
    pub rerank_enabled: Option<bool>,
    pub rerank_url: Option<String>,
    pub rerank_model: Option<String>,
    pub rerank_timeout_ms: Option<u64>,
    pub rerank_fail_open: Option<bool>,
    pub bm25_data_dir: Option<PathBuf>,
    pub chunks_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct DevConfig {
    pub model: Option<String>,
    pub wait_timeout_sec: Option<u64>,
    pub with_rerank: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct McpConfig {
    pub host: Option<String>,
    pub port: Option<u16>,
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        let path = resolve_config_path()?;
        Self::load_from_path(&path)
    }

    pub fn load_required() -> Result<Self> {
        let path = resolve_config_path()?;
        if !path.exists() {
            return Err(anyhow!(
                "config file is required but not found: {}. Run `aurora-grimoire rag config init` first.",
                path.display()
            ));
        }
        Self::load_from_path(&path)
    }

    pub fn load_from_path(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read config file {}", path.display()))?;
        toml::from_str::<Self>(&raw)
            .with_context(|| format!("failed to parse TOML config {}", path.display()))
    }

    pub fn data_root(&self) -> Result<PathBuf> {
        let home = dirs::home_dir()
            .ok_or_else(|| anyhow!("failed to resolve home directory for data root"))?;

        match self.storage.root_dir.as_deref() {
            Some(raw) => Ok(resolve_user_path(raw, &home)),
            None => Ok(home.join(".aurora-grimoire")),
        }
    }

    pub fn chunks_root(&self) -> Result<PathBuf> {
        Ok(self.data_root()?.join("chunks"))
    }

    pub fn bm25_root(&self) -> Result<PathBuf> {
        Ok(self.data_root()?.join("bm25_data"))
    }
}

pub fn resolve_config_path() -> Result<PathBuf> {
    if let Ok(raw) = env::var("AURORA_GRIMOIRE_CONFIG") {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(anyhow!(
                "AURORA_GRIMOIRE_CONFIG is set but empty; either unset it or provide a valid path"
            ));
        }
        let home = dirs::home_dir()
            .ok_or_else(|| anyhow!("failed to resolve home directory for config path"))?;
        return Ok(resolve_user_path(Path::new(trimmed), &home));
    }

    let home = dirs::home_dir()
        .ok_or_else(|| anyhow!("failed to resolve home directory for config path"))?;
    Ok(home.join(".aurora-grimoire").join("config.toml"))
}

fn resolve_user_path(path: &Path, home: &Path) -> PathBuf {
    let text = path.to_string_lossy();
    if text == "~" {
        return home.to_path_buf();
    }
    if let Some(stripped) = text.strip_prefix("~/") {
        return home.join(stripped);
    }
    if path.is_absolute() {
        return path.to_path_buf();
    }
    home.join(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_user_path_supports_tilde() {
        let home = PathBuf::from("/tmp/home");
        assert_eq!(
            resolve_user_path(Path::new("~/data"), &home),
            home.join("data")
        );
        assert_eq!(resolve_user_path(Path::new("~"), &home), home);
    }

    #[test]
    fn resolve_user_path_supports_relative() {
        let home = PathBuf::from("/tmp/home");
        assert_eq!(
            resolve_user_path(Path::new("nested/path"), &home),
            home.join("nested/path")
        );
    }

    #[test]
    fn load_from_path_missing_returns_default() {
        let missing = PathBuf::from("/tmp/aurora-grimoire-nonexistent-config.toml");
        let cfg = AppConfig::load_from_path(&missing).expect("load should not fail");
        assert!(cfg.storage.root_dir.is_none());
    }
}
