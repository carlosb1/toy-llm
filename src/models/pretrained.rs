/// Pre-trained model metadata.
pub struct Pretrained {
    pub name: String,
    pub model: String,
    pub tokenizer: String,
}

#[cfg(feature = "pretrained")]
#[cfg_attr(docsrs, doc(cfg(feature = "pretrained")))]
mod downloader {
    use super::*;
    use burn::data::network::downloader;
    use std::fs::{create_dir_all, File};
    use std::io::Write;
    use std::path::PathBuf;

    impl Pretrained {
        /// Download the file to the local cache directory.
        fn download(&self, url: &str) -> Result<PathBuf, std::io::Error> {
            // Model cache directory
            let model_dir = dirs::home_dir()
                .expect("Should be able to get home directory")
                .join(".cache")
                .join("llama-burn")
                .join(self.name.as_str());

            if !model_dir.exists() {
                create_dir_all(&model_dir)?;
            }

            let file_base_name = url
                .rsplit_once('/')
                .unwrap()
                .1
                .replace("?download=true", "");
            let file_name = model_dir.join(&file_base_name);
            if !file_name.exists() {
                // Download file content
                let bytes = downloader::download_file_as_bytes(url, &file_base_name);

                // Write content to file
                let mut output_file = File::create(&file_name)?;
                output_file.write_all(&bytes)?; // write_all is not OS limited (files over 2GB)
            }

            Ok(file_name)
        }

        /// Download the pre-trained model weights to the local cache directory.
        pub fn download_weights(&self) -> Result<PathBuf, std::io::Error> {
            self.download(self.model.as_str())
        }

        /// Download the tokenizer to the local cache directory.
        pub fn download_tokenizer(&self) -> Result<PathBuf, std::io::Error> {
            self.download(self.tokenizer.as_str())
        }
    }
}

pub trait ModelMeta {
    fn pretrained(&self) -> Pretrained;
}

/// Llama pre-trained weights.
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
            },
            Self::Llama3Instruct => Pretrained {
                name: "Llama-3-8B-Instruct".to_string(),
                model: "https://huggingface.co/tracel-ai/llama-3-8b-instruct-burn/resolve/main/model.mpk?download=true".to_string(),
                tokenizer: "https://huggingface.co/tracel-ai/llama-3-8b-instruct-burn/resolve/main/tokenizer.model?download=true".to_string(),
            },
            Self::Llama31Instruct => Pretrained {
                name: "Llama-3.1-8B-Instruct".to_string(),
                model: "https://huggingface.co/tracel-ai/llama-3.1-8b-instruct-burn/resolve/main/model.mpk?download=true".to_string(),
                tokenizer: "https://huggingface.co/tracel-ai/llama-3.1-8b-instruct-burn/resolve/main/tokenizer.model?download=true".to_string(),
            },
            Self::Llama323bInstruct => Pretrained {
                name: "Llama-3.2-3B-Instruct".to_string(),
                model: "https://huggingface.co/tracel-ai/llama-3.2-3b-instruct-burn/resolve/main/model.mpk?download=true".to_string(),
                tokenizer: "https://huggingface.co/tracel-ai/llama-3.2-3b-instruct-burn/resolve/main/tokenizer.model?download=true".to_string(),
            },
            Self::Llama321bInstruct => Pretrained {
                name: "Llama-3.2-1B-Instruct".to_string(),
                model: "https://huggingface.co/tracel-ai/llama-3.2-1b-instruct-burn/resolve/main/model.mpk?download=true".to_string(),
                tokenizer: "https://huggingface.co/tracel-ai/llama-3.2-1b-instruct-burn/resolve/main/tokenizer.model?download=true".to_string(),
            },
            Self::TinyLlama => Pretrained {
                name: "TinyLlama-1.1B".to_string(),
                model: "https://huggingface.co/tracel-ai/tiny-llama-1.1b-burn/resolve/main/model.mpk?download=true".to_string(),
                tokenizer: "https://huggingface.co/tracel-ai/tiny-llama-1.1b-burn/resolve/main/tokenizer.json?download=true".to_string(),
            },
        }
    }
}
