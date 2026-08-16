use crate::api::http::{AppError, AppState};
use crate::tokenizer::Tokenizer;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::api::openai::conversors::{
    asst_content_to_string, dev_content_to_string, sys_content_to_string, to_hf_msg,
    tool_content_to_string, user_content_to_string,
};
use crate::api::utils::create_token_tensors;
use crate::engine::GenerationConfig;
use crate::models::llama::model::InferenceRequest;
use crate::models::llama::sampling::Sampler;
use crate::models::registry_service::ModelInfo;
use crate::profiler::GenerationProfiler;
use async_openai::types::chat::{
    ChatChoice, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestAssistantMessageContentPart, ChatCompletionRequestDeveloperMessageContent,
    ChatCompletionRequestDeveloperMessageContentPart, ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestSystemMessageContentPart,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestToolMessageContentPart,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionResponseMessage, Choice, CompletionFinishReason, FinishReason, Prompt, Role,
};
use async_openai::types::{
    chat::{CreateChatCompletionRequest, CreateChatCompletionResponse},
    completions::{CreateCompletionRequest, CreateCompletionResponse},
    models::{ListModelResponse, Model},
};
use axum::http::StatusCode;
use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use burn::prelude::Backend;
use hf_chat_template::Message;
use tokio::sync::oneshot;
use tokio::time::Instant;
use uuid::Uuid;

pub fn to_model(model_info: &ModelInfo) -> Model {
    Model {
        id: model_info.name.clone(),
        object: "model".to_string(),
        created: 0,
        owned_by: "user".to_string(),
    }
}

pub async fn list_models<B, T>(State(state): State<AppState<B, T>>) -> Json<ListModelResponse>
where
    B: Backend,
    T: Tokenizer,
{
    let data: Vec<Model> = state
        .registry
        .models
        .iter()
        .map(|(_, model_info)| to_model(model_info))
        .collect();

    let response = ListModelResponse {
        object: "list".to_string(),
        data,
    };
    Json(response)
}

pub async fn retrieve_model<B, T>(
    State(state): State<AppState<B, T>>,
    Path(search_model): Path<String>,
) -> Result<Json<Model>, StatusCode>
where
    B: Backend,
    T: Tokenizer,
{
    let model = state
        .registry
        .models
        .get(&search_model)
        .ok_or(StatusCode::NOT_FOUND)?;
    Ok(Json(to_model(model)))
}

pub async fn create_chat_completion<B, T>(
    State(state): State<AppState<B, T>>,
    Json(request): Json<CreateChatCompletionRequest>,
) -> Result<Json<CreateChatCompletionResponse>, StatusCode>
where
    B: Backend,
    T: Tokenizer,
{
    let model_name = request.model.clone();
    if request.stream.unwrap_or(false) {
        return Err(StatusCode::NOT_FOUND);
    }

    let max_new_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(12) as usize;

    let mut messages = Vec::new();
    for msg in request.messages {
        let hf_message = to_hf_msg(&msg);
        messages.push(Message::new(hf_message.role, hf_message.content));
    }

    let prompt = state
        .prompter
        .encode_chat(&messages)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let generation_config = GenerationConfig {
        sampler: Sampler::Argmax,
        temperature: request.temperature.unwrap_or(1.0) as f64,
        max_new_tokens,
        top_p: request.top_p.map(|value| value as f64),
        top_k: None,
        repetition_penalty: None,
    };
    /* generation profiler */
    let mut profiler = GenerationProfiler::new();
    /* setting up memory */
    let start_time = Instant::now();
    let token_tensors = create_token_tensors(&state, prompt, max_new_tokens);
    profiler.set_input_tokens(token_tensors.prompt_len);
    let elapsed_time = start_time.elapsed();
    println!("Time elapsed: {:?}", elapsed_time);

    let (response_tx, response_rx) = oneshot::channel();
    let inference_req = InferenceRequest::from_tensors(
        token_tensors,
        Some(generation_config),
        response_tx,
        Some(profiler),
    );

    tracing::info!("sending inference request to worker");

    state.tx.send(inference_req).await.map_err(|e| {
        tracing::error!("failed to send inference request to worker {:?}", e);
        StatusCode::BAD_REQUEST
    })?;

    // waiting for a response
    let output = response_rx
        .await
        .map_err(|e| {
            tracing::error!("failed to receive response from worker {:?}", e);
            StatusCode::BAD_REQUEST
        })?
        .map_err(|e| {
            tracing::error!("failed to generate output from worker {:?}", e);
            StatusCode::BAD_REQUEST
        })?;

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

    let id = Uuid::new_v4();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            tracing::error!("system clock error: {error:?}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    let choices = vec![ChatChoice {
        index: 0,

        message: ChatCompletionResponseMessage {
            role: Role::Assistant,
            content: Some(output.text),
            refusal: None,
            tool_calls: None,
            annotations: None,
            function_call: None,
            audio: None,
        },
        finish_reason: Some(FinishReason::Stop), // TODO hardcoded
        logprobs: None,
    }];

    //state.resolver.get(model_name.clone()).await;
    let response = CreateChatCompletionResponse {
        id: id.to_string(),
        object: "chat.completion".to_string(),
        created: created.as_secs() as u32,
        model: model_name,
        service_tier: None,
        choices,
        usage: None,
        system_fingerprint: None,
    };
    Ok(Json(response))
}

pub async fn create_completion<B, T>(
    State(state): State<AppState<B, T>>,
    Json(request): Json<CreateCompletionRequest>,
) -> Result<Json<CreateCompletionResponse>, StatusCode>
where
    B: Backend,
    T: Tokenizer,
{
    if request.stream.unwrap_or(false) {
        return Err(StatusCode::NOT_IMPLEMENTED);
    }

    // Tu worker solo genera una completion por petición.
    if request.n.unwrap_or(1) != 1 {
        tracing::warn!("multiple completions are not supported yet");
        return Err(StatusCode::NOT_IMPLEMENTED);
    }

    if request.best_of.unwrap_or(1) != 1 {
        tracing::warn!("best_of is not supported yet");
        return Err(StatusCode::NOT_IMPLEMENTED);
    }

    let model_name = request.model.clone();

    //righ not max tokens small. like 12 for better testing we can add more
    let max_new_tokens = request.max_tokens.unwrap_or(12) as usize;

    let generation_config = GenerationConfig {
        sampler: Sampler::Argmax,
        temperature: request.temperature.unwrap_or(1.0) as f64,
        max_new_tokens,
        top_p: request.top_p.map(f64::from),
        top_k: None,
        repetition_penalty: None,
    };

    let prompt_tokens = match request.prompt {
        Prompt::String(prompt) => prompt,
        Prompt::IntegerArray(token_ids) => {
            tracing::warn!("batch completion prompts are not supported yet");
            return Err(StatusCode::NOT_IMPLEMENTED);
        }
        Prompt::StringArray(_) | Prompt::ArrayOfIntegerArray(_) => {
            tracing::warn!("batch completion prompts are not supported yet");
            return Err(StatusCode::NOT_IMPLEMENTED);
        }
    };

    let mut profiler = GenerationProfiler::new();
    let token_tensors = create_token_tensors(&state, prompt_tokens, max_new_tokens);
    profiler.set_input_tokens(token_tensors.prompt_len);
    let (response_tx, response_rx) = oneshot::channel();
    let inference_request = InferenceRequest::from_tensors(
        token_tensors,
        Some(generation_config),
        response_tx,
        Some(profiler),
    );
    tracing::info!(
        model = %model_name,
        max_new_tokens,
        "sending completion request to inference worker"
    );

    state.tx.send(inference_request).await.map_err(|error| {
        tracing::error!("failed to send completion request: {error:?}");
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    let output = response_rx
        .await
        .map_err(|error| {
            tracing::error!("inference worker dropped response channel: {error:?}");

            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .map_err(|error| {
            tracing::error!("completion generation failed: {error:?}");

            StatusCode::INTERNAL_SERVER_ERROR
        })?;

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
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            tracing::error!("system clock error: {error:?}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?
        .as_secs();

    let created = u32::try_from(created).map_err(|error| {
        tracing::error!("completion timestamp does not fit in u32: {error:?}");

        StatusCode::INTERNAL_SERVER_ERROR
    })?;
    let finish_reason = CompletionFinishReason::Stop;
    let response = CreateCompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4().simple()),

        object: "text_completion".to_string(),

        created,

        model: model_name,

        choices: vec![Choice {
            text: output.text,
            index: 0,
            logprobs: None,
            finish_reason: Some(finish_reason),
        }],

        usage: None,
        system_fingerprint: None,
    };

    Ok(Json(response))
}

pub fn routes<B, T>() -> Router<AppState<B, T>>
where
    B: Backend + Send + Sync + 'static,
    T: Tokenizer + Clone + Send + Sync + 'static,
    AppState<B, T>: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/models", get(list_models::<B, T>))
        .route("/models/{model}", get(retrieve_model::<B, T>))
        .route("/chat/completions", post(create_chat_completion::<B, T>))
        .route("/completions", post(create_completion::<B, T>))
}
