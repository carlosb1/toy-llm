use crate::models::llama::cacheconfig::CacheConfig;
use crate::models::llama::llama::{check_context_length, Llama};
use crate::models::llama::llamaconfig::LlamaConfig;
use crate::models::llama::pretrained::Pretrained;
#[allow(unused_imports)]
use crate::models::llama::pretrained::{self, ModelMeta};
#[cfg(feature = "llama3")]
use crate::tokenizer::Tiktoken;
use crate::tokenizer::Tokenizer;
use burn::prelude::{Backend, Device};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Llama3_2_3B,
    Llama3_1_8B,
}
impl ModelKind {
    pub fn id(&self) -> &'static str {
        match self {
            Self::Llama3_2_3B => "llama3_2_3b_instruct",
            Self::Llama3_1_8B => "llama3_1_8b_instruct",
        }
    }

    pub fn manifest_path(&self) -> &'static str {
        let current_dir = std::env::current_dir().unwrap();
        println!("Current dir: {}", current_dir.display());
        match self {
            Self::Llama3_2_3B => "assets/Llama-3.2-3B-Instruct.toml",
            Self::Llama3_1_8B => "assets/Llama-3.1-8B-Instruct.toml",
        }
    }
    pub fn load<B: Backend>(
        &self,
        max_seq_len: usize,
        device: &B::Device,
    ) -> anyhow::Result<(Llama<B, Tiktoken>, CacheConfig)> {
        load_llama_from_manifest(self.manifest_path(), max_seq_len, device)
    }
}

pub fn load_llama_from_manifest<B, T>(
    manifest_path: &str,
    max_seq_len: usize,
    device: &Device<B>,
) -> anyhow::Result<(Llama<B, T>, CacheConfig)>
where
    B: Backend,
    T: Tokenizer,
{
    use burn::record::{HalfPrecisionSettings, NamedMpkFileRecorder};

    let manifest = ModelManifest::from_toml_file(manifest_path)
        .map_err(|err| err.to_string())
        .map_err(|err| {
            anyhow::anyhow!("Failed to load model manifest from {manifest_path}: {err}")
        })?;

    check_context_length(max_seq_len, manifest.max_context_len);
    let files = resolve_and_download_model_files(&manifest)
        .map_err(|err| anyhow::anyhow!("Failed to resolve model files from manifest: {err}"))?;
    let llama_config = manifest
        .config
        .clone()
        .with_max_seq_len(max_seq_len)
        .with_tokenizer(files.tokenizer_path.to_string_lossy().to_string());
    let cache_config = CacheConfig::from(llama_config.clone());
    let mut llama = llama_config
        .init::<B, T>(device)
        .map_err(|err| anyhow::anyhow!("Failed to initialize Llama model.\nError: {err}"))?;
    let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();

    llama = llama
        .load(files.checkpoint_path.to_str().unwrap(), &recorder)
        .map_err(|err| format!("Failed to load pre-trained Llama model.\nError: {err}"))
        .map_err(|err| {
            anyhow::anyhow!(
                "Failed to load pre-trained Llama model from {}: {err}",
                files.checkpoint_path.display()
            )
        })?;
    Ok((llama, cache_config))
}

pub fn resolve_and_download_model_files(
    manifest: &ModelManifest,
) -> Result<ResolvedModelFiles, String> {
    if let Some(pretrained_manifest) = &manifest.pretrained {
        let pretrained = Pretrained {
            name: pretrained_manifest.name.clone(),
            model: pretrained_manifest.model.clone(),
            tokenizer: pretrained_manifest.tokenizer.clone(),
        };

        let checkpoint_path = pretrained
            .download_weights()
            .map_err(|err| format!("Could not download weights.\nError: {err}"))?;

        let tokenizer_path = pretrained
            .download_tokenizer()
            .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

        return Ok(ResolvedModelFiles {
            checkpoint_path,
            tokenizer_path,
        });
    }

    if let Some(local) = &manifest.local {
        return Ok(ResolvedModelFiles {
            checkpoint_path: PathBuf::from(&local.model),
            tokenizer_path: PathBuf::from(&local.tokenizer),
        });
    }

    Err("Model manifest must define either [pretrained] or [local]".to_string())
}

#[derive(Debug, Clone)]
pub struct ResolvedModelFiles {
    pub checkpoint_path: PathBuf,
    pub tokenizer_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelManifest {
    pub id: String,
    pub architecture: ModelArchitecture,
    pub tokenizer_kind: TokenizerKind,
    pub checkpoint_format: CheckpointFormat,
    pub weight_layout: WeightLayout,
    pub max_context_len: usize,
    pub pretrained: Option<PretrainedManifest>,
    pub local: Option<LocalManifest>,
    pub config: LlamaConfig, //TODO fix this to be generic over model config
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PretrainedManifest {
    pub name: String,
    pub model: String,
    pub tokenizer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalManifest {
    pub model: String,
    pub tokenizer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelSource {
    Pretrained { model: String, tokenizer: String },
    Local { model: String, tokenizer: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSource {
    pub uri: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerSource {
    pub uri: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelArchitecture {
    Llama,
    // Qwen2,
    // Mistral,
    // Gemma,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerKind {
    Tiktoken,
    SentencePiece,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointFormat {
    Safetensors,
    PytorchBin,
    BurnMpk,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightLayout {
    LlamaHf,
    TinyLlamaHf,
    BurnNative,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelConfig {
    Llama(LlamaConfig),
}

impl ModelManifest {
    pub fn from_toml_file(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let manifest: Self = toml::from_str(&text)?;
        Ok(manifest)
    }
}

#[derive(Debug, Clone)]
pub struct LlamaModelFile {
    pub id: String,
    pub checkpoint_path: String,
    pub tokenizer_kind: TokenizerKind,
    pub checkpoint_format: CheckpointFormat,
    pub weight_layout: WeightLayout,
    pub config: LlamaConfig,
}
