use burn::prelude::Backend;
use crate::models::cacheconfig::CacheConfig;
use crate::models::llama::Llama;
use crate::models::llamaconfig::llama3_2_3b_pretrained_tiktoken;
use crate::sampling::Sampler;
use crate::tokenizer::Tiktoken;

#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub sampler: Sampler,
    pub temperature: f64,
    pub sample_len: usize,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            sampler: Sampler::Argmax,
            temperature: 0.6,
            sample_len: 65,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        }
    }
}

pub struct BurnEngineLlama<B: Backend> {
    pub llama: Llama<B, Tiktoken>, // TODO decoupled
    pub sampler: Sampler,
    pub temperature: f64,
    pub sample_len: usize,
    pub cache_config: CacheConfig
}

impl<B: Backend> BurnEngineLlama<B> {
    pub fn new(
        llama: Llama<B, Tiktoken>,
        generation_config: GenerationConfig,
        cache_config: CacheConfig
    ) -> Self {
        Self {
            llama,
            sampler: generation_config.sampler,
            temperature: generation_config.temperature,
            sample_len: generation_config.sample_len,
            cache_config
        }
    }

    pub fn load_with_device_tiktoken(device: &B::Device) -> anyhow::Result<Self> {
        let temperature = 0.6;
        let max_seq_len = 128;
        let sample_len = 65;
        let sampler = Sampler::Argmax;

        let (llama, cache_config) = llama3_2_3b_pretrained_tiktoken::<B>(max_seq_len, device).map_err(|err| anyhow::anyhow!("Failed to load Llama model: {}", err))?;

        let generation_config = GenerationConfig {
            sampler,
            temperature,
            sample_len,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        };
        let engine = Self::new(
            llama,
            generation_config,
            cache_config
        );
        Ok(engine)
    }
}