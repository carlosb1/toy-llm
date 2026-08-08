use crate::backend::selected;
use crate::engine::BurnEngineLlama;

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
