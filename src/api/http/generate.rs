use crate::api::utils::create_token_tensors;
use crate::models::config::GenerationConfig;
use crate::models::llama::model::{InferenceRequest, TokenTensor};
use crate::models::llama::sampling::Sampler;
use crate::models::registry_service::RegistryService;
use crate::models::resolver_service::{ModelResolverService, DEFAULT_MODEL};
use crate::profiler::{GenerationProfiler, MetricsRegistry};
pub use crate::prompt::PromptProcessor;
use crate::tokenizer::custom_tokenizer::TokenizerHandle;
use crate::tokenizer::Tokenizer;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use burn::prelude::{Backend, Device, Int, Shape, TensorData};
use burn::Tensor;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use uuid::Uuid;

#[derive(Deserialize, Debug)]
pub struct GenerateHttpRequest {
    prompt: String,
    sample_len: usize,
    temperature: f64,
}

#[derive(Serialize)]
pub struct GenerateHttpResponse {
    text: String,
    tokens: usize,
    time: f64,
}

// ADD THIS - Custom error type for Axum
#[derive(Debug)]
pub enum AppError {
    Internal(String),
    BadRequest(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
        };

        (status, Json(json!({ "error": message }))).into_response()
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        AppError::Internal(err.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateState {
    id: Uuid,
    status: String,
    generated: String,
}

impl GenerateState {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            status: "pending".to_string(),
            generated: "".to_string(),
        }
    }
    pub fn update(&mut self, new_text: &str) {
        self.status = "generated".to_string();
        self.generated.push_str(new_text);
    }
}

#[derive(Clone)]
pub struct AppState<B: Backend, T: Tokenizer> {
    pub tx: mpsc::Sender<InferenceRequest<B>>,
    pub tokenizer_handler: Arc<TokenizerHandle<B, T>>,
    pub registry: RegistryService,
    pub prompter: Arc<PromptProcessor>,
    pub metrics: Arc<MetricsRegistry>,
}
async fn generate_handler<B: Backend, T: Tokenizer>(
    State(state): State<AppState<B, T>>,
    Json(req): Json<GenerateHttpRequest>,
) -> Result<Json<GenerateHttpResponse>, AppError>
where
    B: Backend + Send + Sync + 'static,
    T: Tokenizer + Clone + Send + Sync + 'static,
{
    tracing::info!(
        sample_len = req.sample_len,
        temperature = req.temperature,
        prompt_chars = req.prompt.len(),
        "generation request received"
    );

    let generation_config = GenerationConfig {
        sampler: Sampler::Argmax,
        temperature: req.temperature,
        max_new_tokens: req.sample_len,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
    };

    let mut profiler = GenerationProfiler::new();
    /* setting up memory */
    let start_time = Instant::now();
    let token_tensors = create_token_tensors(&state, req.prompt.clone(), req.sample_len);
    profiler.set_input_tokens(token_tensors.prompt_len);

    let elapsed_time = start_time.elapsed();
    println!("Time elapsed: {:?}", elapsed_time);
    let (response_tx, response_rx) = oneshot::channel();

    tracing::debug!("Right now we are only supporting our default model");
    let model_name = DEFAULT_MODEL;

    let inference_req = InferenceRequest::from_tensors(
        model_name,
        token_tensors,
        Some(generation_config),
        response_tx,
        Some(profiler),
    );

    tracing::info!("sending inference request to worker");

    state
        .tx
        .send(inference_req)
        .await
        .map_err(|_| AppError::BadRequest("inference worker is down".to_string()))?;

    // waiting for a response
    let output = response_rx
        .await
        .map_err(|_| AppError::BadRequest("inference worker dropped response".to_string()))?
        .map_err(|e| AppError::BadRequest(e.to_string()))?;

    if let Some(profiler) = output.profiler {
        let metrics = profiler.metrics();
        tracing::info!(
            prompt_tokens = metrics.prompt_tokens,
            generated_tokens = metrics.generated_tokens,
            queue_ms = metrics
                .queue_duration
                .map(|duration| duration.as_secs_f64() * 1_000.0),
            prefill_ms = metrics
                .prefill_duration
                .map(|duration| duration.as_secs_f64() * 1_000.0),
            ttft_ms = metrics
                .time_to_first_token
                .map(|duration| duration.as_secs_f64() * 1_000.0),
            decode_tps = metrics.decode_tokens_per_second,
            "completion generation completed"
        );

        tracing::info!(
            "Metrics: {:?}",
            serde_json::to_string(&metrics)
                .unwrap_or_else(|_| "Failed to serialize metrics".to_string())
        );
        state.metrics.record_success(&metrics);
    }

    Ok(Json(GenerateHttpResponse {
        text: output.text,
        tokens: output.tokens,
        time: output.time,
    }))
}

pub fn routes<B, T>() -> Router<AppState<B, T>>
where
    B: Backend + Send + Sync + 'static,
    T: Tokenizer + Clone + Send + Sync + 'static,
    AppState<B, T>: Clone + Send + Sync + 'static,
{
    Router::new().route("/generate", post(generate_handler::<B, T>))
}
