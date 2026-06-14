use burn::record::{RecorderError};
use burn::{
    config::Config,
    nn::{RotaryEncodingConfig},
    tensor::{
        activation::softmax, backend::Backend, Device, ElementConversion, Int, Shape, Tensor,
        TensorData,
    },
};
#[cfg(feature = "import")]
use burn_store::{
    KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, PytorchStore, SafetensorsStore,
};
use crate::models::llama::{check_context_length, Llama};
#[cfg(feature = "pretrained")]
#[allow(unused_imports)]
use crate::models::pretrained::{self, ModelMeta};
#[cfg(feature = "tiny")]
use crate::tokenizer::SentiencePieceTokenizer;
#[cfg(feature = "llama3")]
use crate::tokenizer::Tiktoken;
use crate::{
    sampling::Sampler,
    tokenizer::Tokenizer,
    transformer::{KeyValueCache, Transformer, TransformerConfig},
};
use crate::models::cacheconfig::CacheConfig;
use crate::models::pretrained::Pretrained;

#[derive(Config, Debug)]
pub struct LlamaConfig {
    /// The size of the model.
    #[config(default = "4096")]
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of transformer blocks.
    #[config(default = "32")]
    pub num_hidden_layers: usize,
    /// The number of attention heads.
    #[config(default = "32")]
    pub num_attention_heads: usize,
    /// The number of key-value heads.
    pub num_key_value_heads: Option<usize>,
    /// The vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon
    #[config(default = "1e-5")]
    pub norm_eps: f64,
    /// Rotary positional encoding (RoPE).
    #[config(default = "RopeConfig::new(10000.0)")]
    pub rope: RopeConfig,
    /// Maximum sequence length for input text.
    #[config(default = "128")]
    pub max_seq_len: usize,
    /// Maximum batch size (used for key-value cache).
    #[config(default = "1")]
    pub max_batch_size: usize,
    /// The tokenizer path.
    pub tokenizer: String,
}

/// Rotary positional encoding (RoPE)
#[derive(Config, Debug)]
pub struct RopeConfig {
    pub theta: f32,
    #[config(default = "None")]
    pub scaled: Option<RopeFrequencyScaling>,
}

/// RoPE frequency scaling.
#[derive(Config, Debug)]
pub struct RopeFrequencyScaling {
    #[config(default = "8.")]
    pub scale_factor: f32,
    #[config(default = "1.")]
    pub low_freq_factor: f32,
    #[config(default = "4.")]
    pub high_freq_factor: f32,
    #[config(default = "8192.")]
    pub old_context_len: f32,
}

pub fn load_llama3_2_3b<B: Backend>(
    checkpoint: &str,
    tokenizer_path: &str,
    max_seq_len: usize,
    device: &Device<B>,
) -> Result<(Llama<B, Tiktoken>, CacheConfig), String> {
    use burn::record::{HalfPrecisionSettings, NamedMpkFileRecorder};

    let llama_config = LlamaConfig::llama3_2_3b(tokenizer_path)
        .with_max_seq_len(max_seq_len);

    let mut llama = llama_config.init::<B, Tiktoken>(device)?;
    let cache_config = CacheConfig::from(llama_config);
    let recorder = NamedMpkFileRecorder::<HalfPrecisionSettings>::new();
    llama = llama
        .load(checkpoint, &recorder)
        .map_err(|err| format!("Failed to load pre-trained Llama model.\nError: {err}"))?;

    Ok((llama, cache_config))
}


pub fn my_pretrained_llama323b_instruct() -> Pretrained {
    Pretrained {
        name: "Llama-3.2-3B-Instruct",
        model: "https://huggingface.co/tracel-ai/llama-3.2-3b-instruct-burn/resolve/main/model.mpk?download=true",
        tokenizer: "https://huggingface.co/tracel-ai/llama-3.2-3b-instruct-burn/resolve/main/tokenizer.model?download=true",
    }
}
pub fn llama3_2_3b_pretrained_tiktoken<B: Backend>(
    max_seq_len: usize,
    device: &Device<B>,
) -> Result<(Llama<B, Tiktoken>, CacheConfig), String> {
    // Llama-3.2 models support context length up to 128K tokens.
    check_context_length(max_seq_len, 128 * 1024);

    // Download checkpoint and tokenizer
    let model = my_pretrained_llama323b_instruct();
    let checkpoint = model
        .download_weights()
        .map_err(|err| format!("Could not download weights.\nError: {err}"))?;
    let tokenizer = model
        .download_tokenizer()
        .map_err(|err| format!("Could not download tokenizer.\nError: {err}"))?;

    load_llama3_2_3b(
        checkpoint.to_str().unwrap(),
        tokenizer.to_str().unwrap(),
        max_seq_len,
        device,
    )
}


impl LlamaConfig {
    /// Llama-3.2-3B configuration.
    pub fn llama3_2_3b(tokenizer_path: &str) -> Self {
        // hidden_size = 8192; vocab_size = 128256
        Self::new(8192, 128256, tokenizer_path.to_string())
            .with_d_model(3072)
            .with_num_hidden_layers(28)
            .with_num_attention_heads(24)
            .with_num_key_value_heads(Some(8))
            .with_rope(
                RopeConfig::new(500000.0)
                    .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.))),
            )
    }

    /// Llama-3.2-1B configuration.
    pub fn llama3_2_1b(tokenizer_path: &str) -> Self {
        // hidden_size = 8192; vocab_size = 128256
        Self::new(8192, 128256, tokenizer_path.to_string())
            .with_d_model(2048)
            .with_num_hidden_layers(16)
            .with_num_key_value_heads(Some(8))
            .with_rope(
                RopeConfig::new(500000.0)
                    .with_scaled(Some(RopeFrequencyScaling::new().with_scale_factor(32.))),
            )
    }

    /// Llama-3.1-8B configuration.
    pub fn llama3_1_8b(tokenizer_path: &str) -> Self {
        // hidden_size = 14336; vocab_size = 128256
        Self::new(14336, 128256, tokenizer_path.to_string())
            .with_num_key_value_heads(Some(8))
            .with_rope(RopeConfig::new(500000.0).with_scaled(Some(RopeFrequencyScaling::new())))
    }

    /// Llama-3-8B configuration.
    pub fn llama3_8b(tokenizer_path: &str) -> Self {
        // hidden_size = 14336; vocab_size = 128256
        Self::new(14336, 128256, tokenizer_path.to_string())
            .with_num_key_value_heads(Some(8))
            .with_rope(RopeConfig::new(500000.0))
    }

    /// TinyLlama-1.1B Chat v1.0 configuration.
    pub fn tiny_llama(tokenizer_path: &str) -> Self {
        // hidden_size = 5632; vocab_size = 32000
        Self::new(5632, 32000, tokenizer_path.to_string())
            .with_d_model(2048)
            .with_num_hidden_layers(22)
            .with_num_key_value_heads(Some(4))
            .with_rope(RopeConfig::new(10000.0))
    }
    pub fn generate_cache_configuration(self) -> CacheConfig {
        let num_key_value_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        CacheConfig::new(
            self.num_attention_heads,
            self.num_hidden_layers,
            self.max_batch_size,
            self.max_seq_len,
            self.d_model,
            Some(num_key_value_heads),
        )
    }


    /// Initialize a new [Llama] module.
    pub fn init<B: Backend, T: Tokenizer>(
        &self,
        device: &Device<B>,
    ) -> Result<Llama<B, T>, String> {
        let tokenizer = T::new(&self.tokenizer)?;
        let num_key_value_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        let model = TransformerConfig::new(
            self.vocab_size,
            self.num_hidden_layers,
            self.d_model,
            self.hidden_size,
            self.num_attention_heads,
            num_key_value_heads,
        )
            .with_max_seq_len(self.max_seq_len)
            .with_norm_eps(self.norm_eps)
            .init(device);


        let rope = RotaryEncodingConfig::new(
            self.max_seq_len * 2,
            self.d_model / self.num_attention_heads,
        )
            .with_theta(self.rope.theta);

        let rope = if let Some(scaling) = &self.rope.scaled {
            let freq_scaling_fn = move |x| scaling.freq_scaling_by_parts(x);
            rope.init_with_frequency_scaling(freq_scaling_fn, device)
        } else {
            rope.init(device)
        };

        Ok(Llama {
            tokenizer,
            model,
            rope,
            device: device.clone(),
        })
    }
}

impl RopeFrequencyScaling {
    /// Applies frequency scaling by parts following Llama 3.1's scheme.
    ///
    /// Adapted from: <https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py#L45>
    pub fn freq_scaling_by_parts<B: Backend>(&self, freqs: Tensor<B, 1>) -> Tensor<B, 1> {
        let low_freq_wavelen = self.old_context_len / self.low_freq_factor;
        let high_freq_wavelen = self.old_context_len / self.high_freq_factor;

        let wavelen = freqs.clone().recip().mul_scalar(2. * core::f32::consts::PI);

        // if wavelen >= high_freq_wavelen
        let cond = wavelen.clone().greater_equal_elem(high_freq_wavelen);
        let smooth = wavelen
            .clone()
            .recip()
            .mul_scalar(self.old_context_len)
            .sub_scalar(self.low_freq_factor)
            .div_scalar(self.high_freq_factor - self.low_freq_factor);
        // (1 - smooth) * freq / scale_factor + smooth * freq
        let new_freqs = smooth
            .clone()
            .neg()
            .add_scalar(1.)
            .mul(freqs.clone().div_scalar(self.scale_factor))
            .add(smooth.clone().mul(freqs.clone()));
        let new_freqs = freqs.clone().mask_where(cond, new_freqs);

        // if wavelen > low_freq_wavelen
        let cond = wavelen.clone().greater_elem(low_freq_wavelen);
        let new_freqs = new_freqs.mask_where(cond, freqs.clone().div_scalar(self.scale_factor));

        // if wavelen < high_freq_wavelen
        let cond = wavelen.lower_elem(high_freq_wavelen);

        new_freqs.mask_where(cond, freqs)
    }
}
