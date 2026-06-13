use std::sync::Arc;
use burn::prelude::{Backend};
use llama_burn::models::llama::{InferenceRequest, RequestState};
use tokio::sync::{mpsc, Mutex, RwLock};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::{util::SubscriberInitExt};
use llama_burn::http::{AppState, TokenizerHandle};
use llama_burn::{http};
use llama_burn::engine::BurnEngineLlama;
use llama_burn::worker::burn_worker;

// backend definition
use llama_burn::backend::selected;

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

    tracing::info!("starting inference service for device={:?}", selected::device());

    let (tx, rx) = mpsc::channel::<InferenceRequest<selected::Backend>>(128);

    let engine = tokio::task::spawn_blocking(|| {
        println!("Loading model...");
        let device = selected::device();
        BurnEngineLlama::load_with_device_tiktoken(&device).expect("Failed to load model")
    }).await.expect("Failed to spawn blocking task");

    let state = AppState {
        tx,
        tokenizer_handler: TokenizerHandle {
            tokenizer: engine.llama.tokenizer.clone(),
            device: selected::device(),
        }
    };

    tokio::spawn(burn_worker::<selected::Backend>(rx, Arc::new(Mutex::new(engine))));

    let app = http::router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();

    println!("listening on http://127.0.0.1:3000");

    axum::serve(listener, app).await.unwrap();
}
