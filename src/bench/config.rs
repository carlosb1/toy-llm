use crate::api::cli::{ApiType, BenchServeArgs};

use async_openai::types::chat::{
    ChatCompletionRequestUserMessageArgs, ChatCompletionRequestUserMessageContent,
    CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
};
use async_openai::types::completions::{CreateCompletionRequest, CreateCompletionRequestArgs};
use serde::Serialize;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
pub enum BenchmarkRequest {
    Completion(CreateCompletionRequest),
    Chat(CreateChatCompletionRequest),
}

#[derive(Debug)]
pub struct BenchmarkConfig {
    pub endpoint_url: String,
    pub api: ApiType,
    pub model: String,
    pub requests: usize,
    pub load_mode: LoadMode,
    pub max_concurrency: usize,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub warmup: usize,
    pub timeout: Duration,
    pub stream: bool,
    pub output: Option<PathBuf>,
}

#[derive(Debug)]
pub enum LoadMode {
    Rate { requests_per_second: f64 },
    Concurrency { concurrent_requests: usize },
}
impl BenchmarkConfig {
    pub fn request(&self) -> Result<BenchmarkRequest, async_openai::error::OpenAIError> {
        match self.api {
            ApiType::Completions => {
                let request = CreateCompletionRequestArgs::default()
                    .model(&self.model)
                    .prompt(&self.prompt)
                    .max_tokens(self.max_tokens)
                    .temperature(self.temperature)
                    .stream(self.stream)
                    .build()?;

                Ok(BenchmarkRequest::Completion(request))
            }

            ApiType::Chat => {
                let message = ChatCompletionRequestUserMessageArgs::default()
                    .content(ChatCompletionRequestUserMessageContent::Text(
                        self.prompt.clone(),
                    ))
                    .build()?
                    .into();

                let request = CreateChatCompletionRequestArgs::default()
                    .model(&self.model)
                    .messages(vec![message])
                    .max_tokens(self.max_tokens)
                    .temperature(self.temperature)
                    .stream(self.stream)
                    .build()?;

                Ok(BenchmarkRequest::Chat(request))
            }
        }
    }
}

impl TryFrom<BenchServeArgs> for BenchmarkConfig {
    type Error = anyhow::Error;

    fn try_from(args: BenchServeArgs) -> Result<Self, Self::Error> {
        anyhow::ensure!(args.requests > 0, "--requests must be greater than zero");

        anyhow::ensure!(
            args.max_tokens > 0,
            "--max-tokens must be greater than zero"
        );

        anyhow::ensure!(
            args.max_concurrency > 0,
            "--max-concurrency must be greater than zero"
        );

        anyhow::ensure!(args.temperature >= 0.0, "--temperature cannot be negative");

        let load_mode = match (args.rate, args.concurrency) {
            (Some(rate), None) => {
                anyhow::ensure!(
                    rate > 0.0 && rate.is_finite(),
                    "--rate must be a finite number greater than zero"
                );

                LoadMode::Rate {
                    requests_per_second: rate,
                }
            }

            (None, Some(concurrency)) => {
                anyhow::ensure!(concurrency > 0, "--concurrency must be greater than zero");

                LoadMode::Concurrency {
                    concurrent_requests: concurrency,
                }
            }

            // Si no se indica nada, ejecuta una petición cada vez.
            (None, None) => LoadMode::Concurrency {
                concurrent_requests: 1,
            },

            // Clap ya impide esta combinación mediante conflicts_with.
            (Some(_), Some(_)) => {
                anyhow::bail!("--rate and --concurrency cannot be used together");
            }
        };

        let base_url = args.url.trim_end_matches('/');
        let endpoint_url = format!("{}{}", base_url, args.api.endpoint());

        Ok(Self {
            endpoint_url,
            api: args.api,
            model: args.model,
            requests: args.requests,
            load_mode,
            max_concurrency: args.max_concurrency,
            prompt: args.prompt,
            max_tokens: args.max_tokens,
            temperature: args.temperature,
            warmup: args.warmup,
            timeout: Duration::from_secs(args.timeout),
            stream: args.stream,
            output: args.output,
        })
    }
}
