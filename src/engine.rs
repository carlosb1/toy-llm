use crate::models::llama::cacheconfig::CacheConfig;
use crate::models::llama::loader::ModelKind;
use crate::models::llama::model::Llama;
use crate::models::llama::sampling::Sampler;
#[cfg(feature = "llama3")]
use crate::tokenizer::Tiktoken;
use burn::prelude::Backend;

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
#[cfg(feature = "llama3")]
pub struct BurnEngineLlama<B: Backend> {
    pub llama: Llama<B, Tiktoken>, // TODO decoupled
    pub default_generation_config: GenerationConfig,
    pub cache_config: CacheConfig,
}
#[cfg(feature = "llama3")]
impl<B: Backend> BurnEngineLlama<B> {
    pub fn new(
        llama: Llama<B, Tiktoken>,
        default_generation_config: GenerationConfig,
        cache_config: CacheConfig,
    ) -> Self {
        Self {
            llama,
            default_generation_config,
            //  sampler: generation_config.sampler,
            //  temperature: generation_config.temperature,
            //  sample_len: generation_config.sample_len,
            cache_config,
        }
    }

    pub fn load_with_device_tiktoken(device: &B::Device) -> anyhow::Result<Self> {
        let temperature = 0.6;
        let max_seq_len = 128;
        let sample_len = 65;
        let sampler = Sampler::Argmax;
        // TODO, It hardcod which model to choose
        let (llama, cache_config) = ModelKind::Llama3_2_3B
            .load(max_seq_len, device)
            .map_err(|err| anyhow::anyhow!("Failed to load Llama model: {}", err))?;

        let generation_config = GenerationConfig {
            sampler,
            temperature,
            sample_len,
            top_p: None,
            top_k: None,
            repetition_penalty: None,
        };
        let engine = Self::new(llama, generation_config, cache_config);
        Ok(engine)
    }
}
