use burn::module::Param;
use burn::prelude::{Backend};
use burn::Tensor;
use burn::prelude::*;
use burn_store::{ModuleSnapshot, SafetensorsStore};



#[derive(Debug, Clone)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
}

impl Qwen3Config {
    pub fn from_your_weights() -> Self {
        Self {
            vocab_size: 151_936,
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
        }
    }

    pub fn q_proj_out(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_proj_out(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

use burn::module::Module;
use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct Qwen3ForCausalLm<B: Backend> {
    pub model: Qwen3Model<B>,
    pub lm_head: Linear<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3Model<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<Qwen3DecoderLayer<B>>,
    pub norm: RmsNorm<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3DecoderLayer<B: Backend> {
    pub self_attn: Qwen3Attention<B>,
    pub mlp: Qwen3Mlp<B>,
    pub input_layernorm: RmsNorm<B>,
    pub post_attention_layernorm: RmsNorm<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3Attention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub q_norm: RmsNorm<B>,
    pub k_norm: RmsNorm<B>,
}

#[derive(Module, Debug)]
pub struct Qwen3Mlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> Qwen3ForCausalLm<B> {
    pub fn new(config: &Qwen3Config, device: &B::Device) -> Self {
        Self {
            model: Qwen3Model::new(config, device),
            lm_head: LinearConfig::new(config.hidden_size, config.vocab_size)
                .with_bias(false)
                .init(device),
        }
    }
}

impl<B: Backend> Qwen3Model<B> {
    pub fn new(config: &Qwen3Config, device: &B::Device) -> Self {
        Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size, config.hidden_size)
                .init(device),

            layers: (0..config.num_hidden_layers)
                .map(|_| Qwen3DecoderLayer::new(config, device))
                .collect(),

            norm: RmsNorm::new(config.hidden_size, config.rms_norm_eps, device),
        }
    }
}

impl<B: Backend> Qwen3DecoderLayer<B> {
    pub fn new(config: &Qwen3Config, device: &B::Device) -> Self {
        Self {
            self_attn: Qwen3Attention::new(config, device),
            mlp: Qwen3Mlp::new(config, device),
            input_layernorm: RmsNorm::new(config.hidden_size, config.rms_norm_eps, device),
            post_attention_layernorm: RmsNorm::new(config.hidden_size, config.rms_norm_eps, device),
        }
    }
}

impl<B: Backend> Qwen3Attention<B> {
    pub fn new(config: &Qwen3Config, device: &B::Device) -> Self {
        Self {
            q_proj: LinearConfig::new(config.hidden_size, config.q_proj_out())
                .with_bias(false)
                .init(device),

            k_proj: LinearConfig::new(config.hidden_size, config.kv_proj_out())
                .with_bias(false)
                .init(device),

            v_proj: LinearConfig::new(config.hidden_size, config.kv_proj_out())
                .with_bias(false)
                .init(device),

            o_proj: LinearConfig::new(config.q_proj_out(), config.hidden_size)
                .with_bias(false)
                .init(device),

            q_norm: RmsNorm::new(config.head_dim, config.rms_norm_eps, device),
            k_norm: RmsNorm::new(config.head_dim, config.rms_norm_eps, device),
        }
    }
}

impl<B: Backend> Qwen3Mlp<B> {
    pub fn new(config: &Qwen3Config, device: &B::Device) -> Self {
        Self {
            gate_proj: LinearConfig::new(config.hidden_size, config.intermediate_size)
                .with_bias(false)
                .init(device),

            up_proj: LinearConfig::new(config.hidden_size, config.intermediate_size)
                .with_bias(false)
                .init(device),

            down_proj: LinearConfig::new(config.intermediate_size, config.hidden_size)
                .with_bias(false)
                .init(device),
        }
    }
}



#[derive(Module, Debug)]
pub struct RmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    pub eps: f64,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(size: usize, eps: f64, device: &B::Device) -> Self {
        let weight = Tensor::<B, 1>::ones([size], device);

        Self {
            weight: Param::from_tensor(weight),
            eps,
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let variance = x.clone().powf_scalar(2.0).mean_dim(D - 1);
        let x = x / (variance + self.eps).sqrt();

        x * self.weight.val().unsqueeze()
    }
}


pub fn load_qwen3_from_safetensors<B: Backend>(
    mut model: Qwen3ForCausalLm<B>,
    safetensors_path: impl AsRef<std::path::Path>,
) -> anyhow::Result<Qwen3ForCausalLm<B>> {
    let mut store = SafetensorsStore::from_file(safetensors_path.as_ref().to_path_buf())
        .allow_partial(false);

    model.load_from(&mut store)?;

    Ok(model)
}
