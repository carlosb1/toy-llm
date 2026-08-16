use crate::api;
use crate::api::http::AppState;
use crate::api::openai;
use crate::backend::selected;
use crate::engine::BurnEngineLlama;
use crate::models::llama::model::{InferenceRequest, RequestState};
use crate::models::registry_service::RegistryService;
use crate::models::resolver_service::ModelResolverService;
use crate::profiler::MetricsRegistry;
use crate::prompt::load_chat_template;
use crate::tokenizer::custom_tokenizer::TokenizerHandle;
use crate::tokenizer::Tokenizer;
use crate::worker::burn_worker;
use axum::Router;
use burn::prelude::Backend;
use hf_chat_template::ChatTemplate;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use tokio::sync::{mpsc, Mutex};

pub enum Message<B: Backend> {
    Request(RequestState<B>),
    Stop,
}

pub fn generate_state<B: Backend, T: Tokenizer>(
    tx: Sender<InferenceRequest<B>>,
    chat_template: ChatTemplate,
    tokenizer_handler: Arc<TokenizerHandle<B, T>>,
) -> AppState<B, T> {
    let state = AppState {
        tx,
        tokenizer_handler,
        registry: RegistryService::new(),
        resolver: ModelResolverService::new(),
        prompter: Arc::new(api::http::PromptProcessor::new(chat_template)),
        metrics: Arc::new(MetricsRegistry::default()),
    };

    state
}

pub async fn build_app(_model: String) -> Router {
    let (tx, rx) = mpsc::channel::<InferenceRequest<selected::Backend>>(128);

    // set up an engine
    let engine = tokio::task::spawn_blocking(|| {
        let device = selected::device();
        BurnEngineLlama::load_with_device_tiktoken(&device).expect("Failed to load model")
    })
    .await
    .expect("Failed to spawn blocking task");

    // set up a chat prompt template
    let chat_template = load_chat_template(None).expect("Failed to load chat templates");

    let cache_config = engine.cache_config.clone();

    // Set up a tokenizer
    let tokenizer_handler = Arc::new(TokenizerHandle {
        tokenizer: engine.llama.tokenizer.clone(),
        device: selected::device(),
    });

    let state = generate_state(tx, chat_template, tokenizer_handler);

    let engine = Arc::new(Mutex::new(engine));

    tokio::spawn(burn_worker::<selected::Backend>(rx, engine, cache_config));

    let app = Router::new()
        .merge(api::http::routes())
        .merge(openai::openai::routes())
        .merge(api::metrics::routes())
        .with_state(state);
    app
}
