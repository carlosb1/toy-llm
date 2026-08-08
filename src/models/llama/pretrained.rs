use crate::models::pretrained::{ModelMeta, Pretrained};
use strum_macros::EnumIter;

/// Llama pre-trained weights.
#[derive(EnumIter, Debug, PartialEq)]
pub enum Llama {
    /// Llama-3-8B.
    Llama3,
    /// Llama-3-8B-Instruct.
    Llama3Instruct,
    /// Llama-3.1-8B-Instruct.
    Llama31Instruct,
    /// Llama-3.2-3B-Instruct.
    Llama323bInstruct,
    /// Llama-3.2-1B-Instruct.
    Llama321bInstruct,
    /// TinyLlama-1.1B Chat v1.0.
    TinyLlama,
}

impl ModelMeta for Llama {
    fn pretrained(&self) -> Pretrained {
        match self {
            Self::Llama3 => Pretrained {
                name: "Llama-3-8B".to_string(),
                model: "https://huggingface.co/tracel-ai/llama-3-8b-burn/resolve/main/model.mpk?download=true".to_string(),
                tokenizer: "https://huggingface.co/tracel-ai/llama-3-8b-burn/resolve/main/tokenizer.model?download=true".to_string(),
                tokenizer_config: None,
            },
            Self::Llama3Instruct => Pretrained {
                name: "Llama-3-8B-Instruct".to_string(),
                model: "https://huggingface.co/tracel-ai/llama-3-8b-instruct-burn/resolve/main/model.mpk?download=true".to_string(),
                tokenizer: "https://huggingface.co/tracel-ai/llama-3-8b-instruct-burn/resolve/main/tokenizer.model?download=true".to_string(),
                tokenizer_config: None,
            },
            Self::Llama31Instruct => Pretrained {
                name: "Llama-3.1-8B-Instruct".to_string(),
                model: "https://huggingface.co/tracel-ai/llama-3.1-8b-instruct-burn/resolve/main/model.mpk?download=true".to_string(),
                tokenizer: "https://huggingface.co/tracel-ai/llama-3.1-8b-instruct-burn/resolve/main/tokenizer.model?download=true".to_string(),
                tokenizer_config: Some(String::from("assets/Llama-3.1-8B-Instruct.tokenizer-config.json")),
            },
            Self::Llama323bInstruct => Pretrained {
                name: "Llama-3.2-3B-Instruct".to_string(),
                model: "https://huggingface.co/tracel-ai/llama-3.2-3b-instruct-burn/resolve/main/model.mpk?download=true".to_string(),
                tokenizer: "https://huggingface.co/tracel-ai/llama-3.2-3b-instruct-burn/resolve/main/tokenizer.model?download=true".to_string(),
                tokenizer_config: Some(String::from("assets/Llama-3.2-3B-Instruct.tokenizer-config.json")),
            },
            Self::Llama321bInstruct => Pretrained {
                name: "Llama-3.2-1B-Instruct".to_string(),
                model: "https://huggingface.co/tracel-ai/llama-3.2-1b-instruct-burn/resolve/main/model.mpk?download=true".to_string(),
                tokenizer: "https://huggingface.co/tracel-ai/llama-3.2-1b-instruct-burn/resolve/main/tokenizer.model?download=true".to_string(),
                tokenizer_config: Some(String::from("assets/Llama-3.2-1B-Instruct.tokenizer-config.json")),

            },
            Self::TinyLlama => Pretrained {
                name: "TinyLlama-1.1B".to_string(),
                model: "https://huggingface.co/tracel-ai/tiny-llama-1.1b-burn/resolve/main/model.mpk?download=true".to_string(),
                tokenizer: "https://huggingface.co/tracel-ai/tiny-llama-1.1b-burn/resolve/main/tokenizer.json?download=true".to_string(),
                tokenizer_config: None,
            },
        }
    }
}
