use burn::prelude::Backend;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use toy_llm::engine::BurnEngineLlama;
use toy_llm::http;
use toy_llm::http::{AppState, TokenizerHandle};
use toy_llm::models::llama::llama::{InferenceRequest, RequestState};
use toy_llm::worker::burn_worker;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

// backend definition
use toy_llm::backend::selected;

pub enum Message<B: Backend> {
    Request(RequestState<B>),
    Stop,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=info,axum=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!(
        "starting inference service for device={:?}",
        selected::device()
    );

    let (tx, rx) = mpsc::channel::<InferenceRequest<selected::Backend>>(128);

    let engine = tokio::task::spawn_blocking(|| {
        println!("Loading model...");
        let device = selected::device();
        BurnEngineLlama::load_with_device_tiktoken(&device).expect("Failed to load model")
    })
    .await
    .expect("Failed to spawn blocking task");

    let state = AppState {
        tx,
        tokenizer_handler: TokenizerHandle {
            tokenizer: engine.llama.tokenizer.clone(),
            device: selected::device(),
        },
    };
    let cache_config = engine.cache_config.clone();

    let engine = Arc::new(Mutex::new(engine));

    tokio::spawn(burn_worker::<selected::Backend>(rx, engine, cache_config));

    let app = http::router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();

    println!("listening on http://127.0.0.1:3000");

    axum::serve(listener, app).await.unwrap();
}
