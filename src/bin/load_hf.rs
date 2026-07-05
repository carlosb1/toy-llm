// src/models/hub.rs
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use hf_hub::api::sync::{Api, ApiBuilder};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "hf-cli")]
#[command(about = "Small Hugging Face CLI in Rust")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Search/list models from Hugging Face Hub
    List {
        /// Search query, for example: qwen, llama, bert
        query: String,

        /// Number of results
        #[arg(short, long, default_value_t = 10)]
        limit: usize,
    },

    /// Download model files from Hugging Face Hub
    Download {
        /// Model repo id, for example: Qwen/Qwen2.5-0.5B-Instruct
        repo_id: String,
    },
}

/// Resultado de una búsqueda en el Hub.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelSearchResult {
    #[serde(rename = "id")]
    pub repo_id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
    #[serde(rename = "pipeline_tag", default)]
    pub pipeline_tag: Option<String>,
}

use std::fmt;

impl fmt::Display for ModelSearchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<45} ⭳ {:>10}  ♥ {:>6}  [{}]",
            self.repo_id,
            self.downloads,
            self.likes,
            self.pipeline_tag.as_deref().unwrap_or("-"),
        )
    }
}

/// Archivos que un modelo necesita para inferencia.
#[derive(Debug, Clone)]
pub struct DownloadedModel {
    pub repo_id: String,
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub weights_paths: Vec<PathBuf>,
}

impl fmt::Display for DownloadedModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "repo:      {}", self.repo_id)?;
        writeln!(f, "config:    {}", self.config_path.display())?;
        writeln!(f, "tokenizer: {}", self.tokenizer_path.display())?;
        write!(f, "weights:   {} file(s)", self.weights_paths.len())?;
        for path in &self.weights_paths {
            write!(f, "\n  - {}", path.display())?;
        }
        Ok(())
    }
}

/// Cliente para buscar y descargar modelos de HuggingFace Hub.
pub struct HubClient {
    api: Api,
    token: Option<String>,
}

impl HubClient {
    /// Crea un cliente. `cache_dir` es dónde se guardan los pesos.
    pub fn new(cache_dir: impl Into<PathBuf>) -> Result<Self> {
        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir.into())
            .build()
            .context("failed to build hf-hub API client")?;
        Ok(Self { api, token: None })
    }

    /// Añade un token para repos privados / rate limits altos.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Busca modelos por texto. Usa la API HTTP del Hub.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<ModelSearchResult>> {
        let url = format!(
            "https://huggingface.co/api/models?search={}&limit={}&full=false",
            urlencoding::encode(query),
            limit
        );

        let mut req = ureq::get(&url);
        if let Some(token) = &self.token {
            req = req.header("Authorization", &format!("Bearer {token}"));
        }

        let results: Vec<ModelSearchResult> = req
            .call()
            .context("hub search request failed")?
            .into_body()
            .read_json()
            .context("failed to parse hub search response")?;

        Ok(results)
    }

    /// Descarga config + tokenizer + pesos de un repo.
    pub fn download(&self, repo_id: &str) -> Result<DownloadedModel> {
        let repo = self.api.model(repo_id.to_string());

        let config_path = repo
            .get("config.json")
            .with_context(|| format!("no config.json in {repo_id}"))?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .with_context(|| format!("no tokenizer.json in {repo_id}"))?;

        // Los pesos pueden estar en un solo archivo o en shards.
        let weights_paths = self.resolve_weights(&repo, repo_id)?;

        Ok(DownloadedModel {
            repo_id: repo_id.to_string(),
            config_path,
            tokenizer_path,
            weights_paths,
        })
    }

    /// Maneja pesos únicos o sharded (model.safetensors.index.json).
    fn resolve_weights(
        &self,
        repo: &hf_hub::api::sync::ApiRepo,
        repo_id: &str,
    ) -> Result<Vec<PathBuf>> {
        // Caso 1: archivo único.
        if let Ok(single) = repo.get("model.safetensors") {
            return Ok(vec![single]);
        }

        // Caso 2: sharded → leer el índice y bajar cada shard.
        let index_path = repo
            .get("model.safetensors.index.json")
            .with_context(|| format!("no weights found in {repo_id}"))?;

        let index: SafetensorsIndex = serde_json::from_slice(
            &std::fs::read(&index_path).context("cannot read weights index")?,
        )
        .context("invalid safetensors index")?;

        // Nombres de shard únicos.
        let mut shards: Vec<String> = index.weight_map.into_values().collect();
        shards.sort();
        shards.dedup();

        shards
            .iter()
            .map(|shard| {
                repo.get(shard)
                    .with_context(|| format!("failed to download shard {shard}"))
            })
            .collect()
    }
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: std::collections::HashMap<String, String>,
}

fn main() {
    let args = Cli::parse();

    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| std::env::current_dir().unwrap())
        .join("assets/hf");

    let client = HubClient::new(cache_dir).expect("failed to create HubClient");

    match args.command {
        Commands::List { query, limit } => {
            let results = client.search(&query, limit).expect("search failed");
            for model in results {
                println!("{}", model);
            }
        }
        Commands::Download { repo_id } => {
            let model = client.download(&repo_id).expect("download failed");
            println!("Downloaded model:\n{}", model);
        }
    }
}
