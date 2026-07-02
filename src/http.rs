use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use burn::prelude::{Backend, Device, Int, Shape, TensorData};
use burn::Tensor;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use uuid::Uuid;
use crate::models::llama::llama::{InferenceRequest, TokenTensor};
use crate::tokenizer::Tokenizer;




#[derive(Deserialize)]
#[derive(Debug)]
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
    generated: String
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
pub struct TokenizerHandle<B: Backend, T: Tokenizer>  {
    /// The tokenizer.
    pub tokenizer: T,
    pub device: Device<B>,
}

impl<B: Backend, T: Tokenizer>  TokenizerHandle<B, T> {
    pub fn tokenize(&self, text: &str) -> Tensor<B, 1, Int> {
        let bos = !cfg!(feature = "tiny"); // TinyLlama Chat doesn't prepend BOS token with the chat format
        let tokens = self.tokenizer.encode(text, bos, false);

        let shape = Shape::new([tokens.len()]);
        Tensor::<B, 1, Int>::from_data(TensorData::new(tokens, shape), &self.device)
    }
}


#[derive(Clone)]
pub struct AppState<B: Backend, T: Tokenizer> {
    pub tx: mpsc::Sender<InferenceRequest<B>>,
    pub tokenizer_handler: TokenizerHandle<B,T>,

}


pub fn create_token_tensors<B: Backend, T: Tokenizer>(state: &AppState<B, T>, req: &GenerateHttpRequest) -> TokenTensor<B> {
    let input_tokens = state.tokenizer_handler.tokenize(req.prompt.as_str());
    let prompt_len = input_tokens.dims()[0];

    tracing::info!(
        prompt_tokens = prompt_len,
        total_tokens = prompt_len + req.sample_len,
        "prompt tokenized"
    );

    let mut tokens = Tensor::<B, 1, Int>::empty([prompt_len + req.sample_len], &state.tokenizer_handler.device);
    tokens = tokens.slice_assign([0..prompt_len], input_tokens);

    tracing::info!(
        "input tensor prepared with shape {:?}",
        tokens.shape()
    );
    let input_pos = Tensor::<B, 1, Int>::arange(0..prompt_len as i64, &state.tokenizer_handler.device);
    let stop_tokens = Tensor::from_ints(state.tokenizer_handler.tokenizer.stop_ids().as_slice(), &state.tokenizer_handler.device);
    let token_tensors = TokenTensor{prompt_len, tokens, input_pos, stop_tokens};
    token_tensors
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

    /* setting up memory */
    let start_time = Instant::now();
    let token_tensors = create_token_tensors(&state, &req);
    let elapsed_time = start_time.elapsed();
    println!("Time elapsed: {:?}", elapsed_time);
    let (response_tx, response_rx) = oneshot::channel();
    let inference_req = InferenceRequest::from_tensors(token_tensors, req.sample_len, req.temperature, response_tx);


    tracing::info!(
        "sending inference request to worker"
    );

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


    Ok(Json(GenerateHttpResponse {
        text: output.text,
        tokens: output.tokens,
        time: output.time,
    }))
}

pub fn router<B, T>(state: AppState<B, T>) -> axum::Router
where
    B: Backend + Send + Sync + 'static,
    T: Tokenizer + Clone + Send + Sync + 'static,
    AppState<B, T>: Clone + Send + Sync + 'static,
{
    axum::Router::new()
        .route("/generate", post(generate_handler::<B, T>))
        .with_state(state)
}