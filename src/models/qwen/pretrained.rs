use crate::models::pretrained::{ModelMeta, Pretrained};
use strum_macros::EnumIter;

/// Llama pre-trained weights.
#[derive(EnumIter, Debug, PartialEq)]
pub enum Qwen {
    /// Qwen-3-8B.
    Qwen3,
    /// Qwen-3-8B-Instruct.
    Qwen3Instruct,
}

impl ModelMeta for crate::models::qwen::pretrained::Qwen {
    fn pretrained(&self) -> Pretrained {
        match self {
            Self::Qwen3 => Pretrained {
                name: "Qwen-3-8B".to_string(),
                model: "".to_string(),
                tokenizer: "".to_string(),
                tokenizer_config: None,
            },
            Self::Qwen3Instruct => Pretrained {
                name: "Qwen-3-8B-Instruct".to_string(),
                model: "".to_string(),
                tokenizer: "".to_string(),
                tokenizer_config: None,
            },
        }
    }
}
