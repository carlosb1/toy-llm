use crate::backend::selected;
use crate::models::llama::engine::BurnEngineLlama;
use burn::prelude::Backend;

pub const DEFAULT_MODEL: &str = "default";
#[derive(Clone)]
pub struct ModelResolverService {}

impl ModelResolverService {
    pub fn new() -> Self {
        Self {}
    }
    pub async fn get(
        &self,
        _model_name: String,
    ) -> anyhow::Result<BurnEngineLlama<selected::Backend>> {
        let device = selected::device();
        BurnEngineLlama::load_with_device_tiktoken(&device)
    }
}

pub async fn init_default_llama_engine<B>(
    device: B::Device,
) -> anyhow::Result<(String, BurnEngineLlama<B>)>
where
    B: Backend + Send + 'static,
    B::Device: Send + 'static,
    BurnEngineLlama<B>: Send + 'static,
{
    Ok((
        DEFAULT_MODEL.to_string(),
        tokio::task::spawn_blocking(move || {
            BurnEngineLlama::<B>::load_with_device_tiktoken(&device)
        })
        .await
        .expect("Failed to spawn blocking task")
        .expect("Failed to load default model"),
    ))
}
