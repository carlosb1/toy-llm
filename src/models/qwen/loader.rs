use crate::models::qwen::gguf;
use crate::models::qwen::model::{
    apply_quantization, detect_gguf_quantization, load_gguf_weights, QuantizationMode, Qwen3,
};
use crate::models::qwen::qwenconfig::Qwen3Config;
use crate::models::qwen::transformer::{AttentionKvCache, MoeConfig, RotaryEmbedding, Transformer};
use burn::prelude::{Backend, Device};
use std::path::Path;

/// Load a Qwen3 model from a directory containing `config.json` and safetensors weight files.
pub fn from_pretrained<B: Backend>(
    model_dir: impl AsRef<Path>,
    max_seq_len: usize,
    quantization: QuantizationMode,
    device: &Device<B>,
) -> Result<Qwen3<B>, Box<dyn std::error::Error>> {
    let model_dir = model_dir.as_ref();

    // Load config
    let config = Qwen3Config::from_file(model_dir.join("config.json"))?;

    // Build MoE config if this is a MoE model
    let moe_config = if config.is_moe() {
        Some(MoeConfig {
            num_experts: config.num_experts.unwrap(),
            num_experts_per_tok: config.num_experts_per_tok.unwrap(),
            moe_intermediate_size: config.moe_intermediate_size.unwrap(),
            norm_topk_prob: config.norm_topk_prob.unwrap_or(true),
            mlp_only_layers: config.mlp_only_layers.clone().unwrap_or_default(),
            decoder_sparse_step: config.decoder_sparse_step.unwrap_or(1),
        })
    } else {
        None
    };

    // Create model
    let transformer = Transformer::new(
        config.vocab_size,
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.intermediate_size,
        config.rms_norm_eps,
        config.tie_word_embeddings,
        moe_config.as_ref(),
        device,
    );

    // Load weights from safetensors
    let transformer = crate::models::qwen::model::load_safetensors_weights(
        transformer,
        model_dir,
        &config,
        device,
    )?;

    // Quantize weights if requested (Auto means no quantization for SafeTensors)
    let effective = match quantization {
        QuantizationMode::Auto => QuantizationMode::None,
        other => other,
    };
    let transformer = apply_quantization(transformer, effective);

    let rope = RotaryEmbedding::new(config.head_dim, max_seq_len, config.rope_theta, device);

    let eos_token_id = config.eos_token_id;
    Ok(Qwen3 {
        transformer,
        rope,
        config,
        eos_token_id,
        max_seq_len,
        device: device.clone(),
    })
}

/// Load a Qwen3 model from a GGUF file.
///
/// Only requires a single `.gguf` file; config is extracted from GGUF metadata.
/// Supported GGUF quantization types: F32, F16, BF16, Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K.
pub fn from_gguf<B: Backend>(
    gguf_path: impl AsRef<Path>,
    max_seq_len: usize,
    quantization: QuantizationMode,
    device: &Device<B>,
) -> Result<Qwen3<B>, Box<dyn std::error::Error>> {
    let (gguf_file, mut file) = gguf::GgufFile::open(gguf_path)?;

    // Extract config from GGUF metadata
    let config = gguf::extract_config(&gguf_file)?;
    eprintln!(
        "GGUF config: hidden={}, layers={}, heads={}, kv_heads={}, vocab={}{}",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.vocab_size,
        if config.is_moe() {
            format!(", MoE experts={}", config.num_experts.unwrap_or(0))
        } else {
            String::new()
        }
    );

    // Build MoE config if needed
    let moe_config = if config.is_moe() {
        Some(MoeConfig {
            num_experts: config.num_experts.unwrap(),
            num_experts_per_tok: config.num_experts_per_tok.unwrap(),
            moe_intermediate_size: config.moe_intermediate_size.unwrap(),
            norm_topk_prob: config.norm_topk_prob.unwrap_or(true),
            mlp_only_layers: config.mlp_only_layers.clone().unwrap_or_default(),
            decoder_sparse_step: config.decoder_sparse_step.unwrap_or(1),
        })
    } else {
        None
    };

    // Create model skeleton
    let transformer = Transformer::new(
        config.vocab_size,
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.intermediate_size,
        config.rms_norm_eps,
        config.tie_word_embeddings,
        moe_config.as_ref(),
        device,
    );

    // Resolve quantization mode: auto-detect from GGUF when Auto
    let detected = detect_gguf_quantization(&gguf_file);
    let resolved_quant = match quantization {
        QuantizationMode::Auto => detected,
        other => other,
    };

    // Build QuantScheme for per-tensor quantized loading
    let quant_scheme = {
        use burn::tensor::quantization::{QuantLevel, QuantScheme, QuantValue};
        match resolved_quant {
            QuantizationMode::Auto | QuantizationMode::None => None,
            QuantizationMode::Int8 => {
                eprintln!("Loading GGUF with per-tensor INT8 quantization...");
                Some(
                    QuantScheme::default()
                        .with_value(QuantValue::Q8S)
                        .with_level(QuantLevel::block([32])),
                )
            }
            QuantizationMode::Int4 => {
                eprintln!("Loading GGUF with per-tensor INT4 quantization...");
                Some(
                    QuantScheme::default()
                        .with_value(QuantValue::Q4S)
                        .with_level(QuantLevel::block([32])),
                )
            }
        }
    };

    // Load weights from GGUF (per-tensor quantization applied during loading if scheme is set)
    let per_tensor_quantized = quant_scheme.is_some();
    let transformer = load_gguf_weights(
        transformer,
        &gguf_file,
        &mut file,
        &config,
        quant_scheme,
        device,
    )?;

    // Only apply whole-model quantization if per-tensor wasn't already done
    let transformer = if per_tensor_quantized {
        transformer
    } else {
        apply_quantization(transformer, resolved_quant)
    };

    let rope = RotaryEmbedding::new(config.head_dim, max_seq_len, config.rope_theta, device);

    let eos_token_id = config.eos_token_id;
    Ok(Qwen3 {
        transformer,
        rope,
        config,
        eos_token_id,
        max_seq_len,
        device: device.clone(),
    })
}

pub fn initialize_cache<B: Backend>(
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    max_seq_len: usize,
    head_dim: usize,
    device: &Device<B>,
) -> Vec<AttentionKvCache<B>> {
    (0..num_hidden_layers)
        .map(|_| AttentionKvCache::new(1, num_key_value_heads, max_seq_len, head_dim, device))
        .collect()
}
