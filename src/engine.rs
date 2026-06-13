use burn::prelude::Backend;
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
}

impl<B: Backend> BurnEngineLlama<B> {
    pub fn new(
        llama: Llama<B, Tiktoken>,
        generated: GenerationConfig,
    ) -> Self {
        Self {
            llama,
            sampler: generated.sampler,
            temperature: generated.temperature,
            sample_len: generated.sample_len,
        }
    }

    pub fn load_with_device_tiktoken(device: &B::Device) -> anyhow::Result<Self> {
        let temperature = 0.6;
        let max_seq_len = 128;
        let sample_len = 65;
        let sampler = Sampler::Argmax;

        let llama = llama3_2_3b_pretrained_tiktoken::<B>(max_seq_len, device).map_err(|err| anyhow::anyhow!("Failed to load Llama model: {}", err))?;

        Ok(Self::new(
            llama,
            GenerationConfig {
                sampler,
                temperature,
                sample_len,
                top_p: None,
                top_k: None,
                repetition_penalty: None,
            }
        ))
    }
}