use crate::models;
use crate::models::llama;
use crate::models::pretrained::ModelMeta;
use std::collections::HashMap;
use strum::IntoEnumIterator;

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
}
#[derive(Debug, Clone)]
pub struct RegistryService {
    pub models: HashMap<String, ModelInfo>,
}

impl RegistryService {
    pub fn new() -> Self {
        let mut llama_models: Vec<String> = models::llama::pretrained::Llama::iter()
            .map(|model| llama::pretrained::Llama::pretrained(&model).name)
            .collect::<Vec<String>>()
            .into();
        let mut qwen_models: Vec<String> = models::qwen::pretrained::Qwen::iter()
            .map(|model| models::qwen::pretrained::Qwen::pretrained(&model).name)
            .collect::<Vec<String>>()
            .into();

        llama_models.append(&mut qwen_models);

        RegistryService {
            models: llama_models
                .into_iter()
                .map(|name| (name.clone(), ModelInfo { name }))
                .collect(),
        }
    }

    pub fn list_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect::<Vec<String>>()
    }
}
