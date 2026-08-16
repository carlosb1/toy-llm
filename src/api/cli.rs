use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "toy-llm")]
#[command(version)]
#[command(about = "A toy LLM inference server")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Start the inference server.
    Serve(ServeArgs),

    /// Run benchmarks.
    Bench(BenchArgs),

    Metrics(MetricsArgs),

    Config(ConfigArgs),
}

#[derive(Debug, Args)]
pub struct ConfigArgs {
    #[arg(short, long, action = clap::ArgAction::SetTrue)]
    pub models: bool,
}

#[derive(Debug, Args)]
pub struct MetricsArgs {
    #[arg(short, long, action = clap::ArgAction::SetTrue)]
    pub reset: bool,

    #[arg(long, default_value = "http://127.0.0.1:3000")]
    pub url: String,
}

#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Model identifier or local model path.
    pub model: String,

    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    #[arg(long, default_value_t = 3000)]
    pub port: u16,
}

#[derive(Debug, Args)]
pub struct BenchArgs {
    #[command(subcommand)]
    pub command: BenchCommand,
}

#[derive(Debug, Subcommand)]
pub enum BenchCommand {
    /// Benchmark a running OpenAI-compatible server.
    Serve(BenchServeArgs),
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ApiType {
    /// POST /v1/completions
    Completions,

    /// POST /v1/chat/completions
    Chat,
}

impl ApiType {
    pub fn endpoint(self) -> &'static str {
        match self {
            Self::Completions => "/v1/completions",
            Self::Chat => "/v1/chat/completions",
        }
    }
}

#[derive(Debug, Args)]
pub struct BenchServeArgs {
    /// OpenAI-compatible API to benchmark.
    #[arg(long, value_enum, default_value_t = ApiType::Chat)]
    pub api: ApiType,

    /// Base URL of the running server.
    #[arg(long, default_value = "http://127.0.0.1:3000")]
    pub url: String,

    /// Model name included in each request.
    #[arg(long)]
    pub model: String,

    /// Total number of measured requests.
    #[arg(long, default_value_t = 100)]
    pub requests: usize,

    /// Requests sent per second.
    ///
    /// Cannot be combined with --concurrency.
    #[arg(long, conflicts_with = "concurrency")]
    pub rate: Option<f64>,

    /// Number of requests kept active.
    ///
    /// Cannot be combined with --rate.
    #[arg(long, conflicts_with = "rate")]
    pub concurrency: Option<usize>,

    /// Maximum number of simultaneously active requests in rate mode.
    #[arg(long, default_value_t = 64)]
    pub max_concurrency: usize,

    /// Prompt used for every request.
    #[arg(long, default_value = "Explain the Rust ownership system.")]
    pub prompt: String,

    /// Maximum number of generated tokens per request.
    #[arg(long, default_value_t = 32)]
    pub max_tokens: u32,

    /// Sampling temperature.
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,

    /// Number of initial requests excluded from the metrics.
    #[arg(long, default_value_t = 5)]
    pub warmup: usize,

    /// Timeout for each request, in seconds.
    #[arg(long, default_value_t = 60)]
    pub timeout: u64,

    /// Enable or disable streaming.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub stream: bool,

    /// Optional JSON results file.
    #[arg(long)]
    pub output: Option<PathBuf>,
}
