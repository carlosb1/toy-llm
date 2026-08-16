use crate::api::http;
use crate::api::http::generate::AppState;
use crate::api::http::openai;
use crate::backend::selected;
use crate::models::llama::model::{InferenceRequest, RequestState};
use crate::models::registry_service::RegistryService;
use crate::models::resolver_service::init_default_llama_engine;
use crate::profiler::MetricsRegistry;
use crate::prompt::load_chat_template;
use crate::tokenizer::custom_tokenizer::TokenizerHandle;
use crate::tokenizer::Tokenizer;
use crate::worker::burn_worker;
use axum::Router;
use burn::prelude::Backend;
use hf_chat_template::ChatTemplate;
use std::collections::HashMap;
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
        prompter: Arc::new(http::generate::PromptProcessor::new(chat_template)),
        metrics: Arc::new(MetricsRegistry::default()),
    };

    state
}

pub async fn build_app(_model: String) -> Router {
    let llama_model_registries = Arc::new(tokio::sync::RwLock::new(HashMap::new()));

    let (tx, rx) = mpsc::channel::<InferenceRequest<selected::Backend>>(128);

    let device = selected::device();
    // set up an engine
    let (default_name, engine) = init_default_llama_engine::<selected::Backend>(device)
        .await
        .expect("Failed to initialize engine");
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
    llama_model_registries
        .write()
        .await
        .insert(default_name, engine.clone());
    tokio::spawn(burn_worker::<selected::Backend>(
        rx,
        engine,
        llama_model_registries,
        cache_config,
    ));

    let app = Router::new()
        .merge(http::generate::routes())
        .merge(openai::openai::routes())
        .merge(http::metrics::routes())
        .merge(http::models::routes())
        .with_state(state);
    app
}
