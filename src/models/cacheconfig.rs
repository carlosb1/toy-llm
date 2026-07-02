use burn::prelude::{Backend, Device};
use crate::models::llama::llamaconfig::LlamaConfig;
use crate::models::transformer::KeyValueCache;

#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub max_batch_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
    pub num_key_value_heads: usize,
}

impl CacheConfig {
    pub fn new(
        num_attention_heads: usize,
        num_hidden_layers: usize,
        max_batch_size: usize,
        max_seq_len: usize,
        d_model: usize,
        num_key_value_heads: Option<usize>,
    ) -> Self {
        let num_key_value_heads = num_key_value_heads.unwrap_or(num_attention_heads);

        Self {
            num_attention_heads,
            num_hidden_layers,
            max_batch_size,
            max_seq_len,
            d_model,
            num_key_value_heads,
        }
    }

    pub fn init_cache<B: Backend>(&self, device: &Device<B>) -> Vec<KeyValueCache<B>> {
        let cache = (0..self.num_hidden_layers)
            .map(|_| {
                KeyValueCache::new(
                    self.max_batch_size,
                    self.num_key_value_heads,
                    self.max_seq_len,
                    self.d_model / self.num_attention_heads,
                    device,
                )
            })
            .collect::<Vec<_>>();
        cache

    }
}

impl From<LlamaConfig> for CacheConfig {
    fn from(config: LlamaConfig) -> Self {
        let num_attention_heads = config.num_attention_heads;
        let num_hidden_layers = config.num_hidden_layers;
        let max_batch_size = config.max_batch_size;
        let max_seq_len = config.max_seq_len;
        let d_model = config.d_model;

        let num_key_value_heads = config
            .num_key_value_heads
            .unwrap_or(num_attention_heads);

        Self {
            num_attention_heads,
            num_hidden_layers,
            max_batch_size,
            max_seq_len,
            d_model,
            num_key_value_heads,
        }
    }
}

