use serde::Deserialize;
use std::path::Path;

/// Qwen3 model configuration matching HuggingFace `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    // MoE fields (None for dense models)
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub decoder_sparse_step: Option<usize>,
    #[serde(default)]
    pub norm_topk_prob: Option<bool>,
    #[serde(default)]
    pub mlp_only_layers: Option<Vec<usize>>,
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: u32,
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_head_dim() -> usize {
    128
}

fn default_bos_token_id() -> u32 {
    151643
}

fn default_eos_token_id() -> u32 {
    151645
}

impl Qwen3Config {
    /// Load config from a `config.json` file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&contents)?;
        Ok(config)
    }

    /// Returns true if this is a Mixture of Experts model.
    pub fn is_moe(&self) -> bool {
        self.num_experts.unwrap_or(0) > 1
    }

    /// Returns whether a given layer index uses MoE (vs dense FFN).
    /// MoE layers are determined by `decoder_sparse_step` (every N-th layer is MoE)
    /// and `mlp_only_layers` (explicit dense-only overrides).
    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        if !self.is_moe() {
            return false;
        }
        // Check if this layer is in the dense-only override list
        if let Some(ref dense_layers) = self.mlp_only_layers {
            if dense_layers.contains(&layer_idx) {
                return false;
            }
        }
        // decoder_sparse_step=1 means every layer is MoE
        let step = self.decoder_sparse_step.unwrap_or(1);
        if step == 0 {
            return false;
        }
        layer_idx.is_multiple_of(step)
    }

    /// Qwen3-0.6B configuration.
    pub fn qwen3_0_6b() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            intermediate_size: 3072,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            head_dim: 128,
            tie_word_embeddings: true,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            decoder_sparse_step: None,
            norm_topk_prob: None,
            mlp_only_layers: None,
            bos_token_id: 151643,
            eos_token_id: 151645,
        }
    }

    /// Qwen3-1.7B configuration.
    pub fn qwen3_1_7b() -> Self {
        Self {
            hidden_size: 1536,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            intermediate_size: 4608,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            head_dim: 128,
            tie_word_embeddings: true,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            decoder_sparse_step: None,
            norm_topk_prob: None,
            mlp_only_layers: None,
            bos_token_id: 151643,
            eos_token_id: 151645,
        }
    }

    /// Qwen3-4B configuration.
    pub fn qwen3_4b() -> Self {
        Self {
            hidden_size: 2560,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 9728,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            head_dim: 128,
            tie_word_embeddings: true,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            decoder_sparse_step: None,
            norm_topk_prob: None,
            mlp_only_layers: None,
            bos_token_id: 151643,
            eos_token_id: 151645,
        }
    }

    /// Qwen3-8B configuration.
    pub fn qwen3_8b() -> Self {
        Self {
            hidden_size: 4096,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            intermediate_size: 12288,
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            head_dim: 128,
            tie_word_embeddings: false,
            num_experts: None,
            num_experts_per_tok: None,
            moe_intermediate_size: None,
            decoder_sparse_step: None,
            norm_topk_prob: None,
            mlp_only_layers: None,
            bos_token_id: 151643,
            eos_token_id: 151645,
        }
    }

    /// Qwen3-30B-A3B (MoE) configuration.
    pub fn qwen3_30b_a3b() -> Self {
        Self {
            hidden_size: 2048,
            num_hidden_layers: 48,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            intermediate_size: 768, // dense intermediate (not used for MoE layers)
            vocab_size: 151936,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            head_dim: 128,
            tie_word_embeddings: true,
            num_experts: Some(128),
            num_experts_per_tok: Some(8),
            moe_intermediate_size: Some(768),
            decoder_sparse_step: Some(1),
            norm_topk_prob: Some(true),
            mlp_only_layers: Some(vec![]),
            bos_token_id: 151643,
            eos_token_id: 151645,
        }
    }

    /// Qwen3-235B-A22B (MoE) configuration.
    pub fn qwen3_235b_a22b() -> Self {
        Self {
            hidden_size: 4096,
            num_hidden_layers: 94,
            num_attention_heads: 64,
            num_key_value_heads: 4,
            intermediate_size: 1536, // dense intermediate (not used for MoE layers)
            vocab_size: 152064,
            max_position_embeddings: 40960,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            head_dim: 128,
            tie_word_embeddings: false,
            num_experts: Some(128),
            num_experts_per_tok: Some(8),
            moe_intermediate_size: Some(1536),
            decoder_sparse_step: Some(1),
            norm_topk_prob: Some(true),
            mlp_only_layers: Some(vec![]),
            bos_token_id: 151643,
            eos_token_id: 151645,
        }
    }
}
