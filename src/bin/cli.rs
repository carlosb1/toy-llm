use clap::{command, Parser};
use reqwest::{Client, Error, Response};
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use toy_llm::api::cli::{BenchCommand, BenchServeArgs, Cli, Command, MetricsArgs, ServeArgs};
use toy_llm::api::http::metrics::{MetricsResultHttpResponse, ResetMetricsHttpResponse};
use toy_llm::app::build_app;
use toy_llm::backend::selected;
use toy_llm::bench::config::{BenchmarkConfig, BenchmarkRequest, LoadMode};

const WARMUP_CONCURRENCY: usize = 4;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=info,axum=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    let cli = Cli::parse();

    match cli.command {
        Command::Serve(args) => run_server(args).await,

        Command::Bench(args) => match args.command {
            BenchCommand::Serve(args) => run_benchmark(args).await,
        },
        Command::Metrics(args) => run_metrics(args).await,
        Command::Config(args) => run_config(args).await,
    }
}

async fn run_config(args: toy_llm::api::cli::ConfigArgs) -> anyhow::Result<()> {
    if args.models {
        let models = toy_llm::models::registry_service::RegistryService::new().list_models();
        println!("Available models:");
        for model in models {
            println!("- {}", model);
        }
    }

    Ok(())
}

async fn run_metrics(args: MetricsArgs) -> anyhow::Result<()> {
    let client = Client::new();
    let url_metrics = format!("{:}/metrics", args.url);
    if args.reset {
        let response = client
            .delete(url_metrics.clone())
            .send()
            .await?
            .error_for_status()?
            .json::<ResetMetricsHttpResponse>()
            .await?;
        if response.reset {
            println!("Metrics reset successfully.\n");
        }
    }
    if let Ok(metrics) = client
        .get(url_metrics)
        .send()
        .await?
        .error_for_status()?
        .json::<MetricsResultHttpResponse>()
        .await
    {
        println!("{:?}", metrics.snapshot);
    }

    Ok(())
}

async fn run_server(args: ServeArgs) -> anyhow::Result<()> {
    let address = format!("{}:{}", args.host, args.port);
    tracing::info!("Starting model {} on {}", args.model, address);
    tracing::info!(
        "starting inference service for device={:?}",
        selected::device()
    );
    let app = build_app(args.model).await;
    let listener = tokio::net::TcpListener::bind(address).await?;

    tracing::info!("listening on {:?}", listener.local_addr()?);

    axum::serve(listener, app).await?;

    Ok(())
}

async fn run_benchmark(args: BenchServeArgs) -> anyhow::Result<()> {
    let config = BenchmarkConfig::try_from(args)?;
    tracing::info!("Benchmarking {}", config.endpoint_url);
    tracing::info!("Requests: {}", config.requests);

    match config.load_mode {
        LoadMode::Rate {
            requests_per_second,
        } => {
            println!("Request rate: {requests_per_second:.2} req/s");
        }
        LoadMode::Concurrency {
            concurrent_requests,
        } => {
            println!("Concurrency: {concurrent_requests}");
        }
    }

    let warmup_size = config.warmup;

    let warmup_results = run_bench_warmup(warmup_size, &config).await;
    // check if it had fails

    let results = execute_bench(&config).await;

    Ok(())
}

async fn run_bench_warmup(warmup_reqs: usize, config: &BenchmarkConfig) -> Vec<PostRequestResult> {
    let client = Client::new();
    let prompt = config.request().unwrap();
    let req_results = run_requests(
        &client,
        warmup_reqs,
        WARMUP_CONCURRENCY,
        &config.endpoint_url,
        &prompt,
    )
    .await;
    req_results
}

async fn execute_bench(config: &BenchmarkConfig) -> Vec<PostRequestResult> {
    let client = Client::new();
    let prompt = config.request().unwrap();
    let req_results = run_requests(
        &client,
        config.requests,
        WARMUP_CONCURRENCY,
        &config.endpoint_url,
        &prompt,
    )
    .await;
    req_results
}
#[derive(Debug)]
struct PostRequestResult {
    latency: Duration,
    status: u16,
    success: bool,
}

async fn run_requests(
    client: &Client,
    total_requests: usize,
    concurrency: usize,
    url: &str,
    prompt: &BenchmarkRequest,
) -> Vec<PostRequestResult> {
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut tasks = Vec::with_capacity(total_requests);

    for request_number in 0..total_requests {
        let client = client.clone();
        let url = url.to_owned();
        let prompt = prompt.clone();
        let semaphore = Arc::clone(&semaphore);
        tasks.push(tokio::spawn(async move {
            let permit = semaphore.acquire_owned().await.expect("semaphore closed");

            let result = post_request(&client, url, request_number, &prompt).await;

            drop(permit);

            result
        }));
    }

    let mut results = Vec::with_capacity(total_requests);

    for task in tasks {
        match task.await {
            Ok(result) => results.push(result),
            Err(error) => {
                eprintln!("request task failed: {error}");
            }
        }
    }

    results
}

async fn post_request<T: ?Sized + Serialize>(
    client: &Client,
    url: String,
    request_number: usize,
    body: &T,
) -> PostRequestResult {
    let started = Instant::now();
    let response = client.post(url).json(&body).send().await;
    let latency = started.elapsed();

    match response {
        Ok(response) => {
            let status = response.status();

            let body_result = response.bytes().await;

            PostRequestResult {
                latency,
                status: status.as_u16(),
                success: status.is_success() && body_result.is_ok(),
            }
        }

        Err(error) => {
            eprintln!("request {request_number} failed: {error}");

            PostRequestResult {
                latency,
                status: 0,
                success: false,
            }
        }
    }
}
