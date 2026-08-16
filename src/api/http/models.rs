use crate::api::http::generate::AppState;
use crate::models::registry_service::ModelInfo;
use crate::tokenizer::Tokenizer;
use async_openai::types::models::{ListModelResponse, Model};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::routing::get;
use axum::{Json, Router};
use burn::prelude::Backend;

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

pub fn routes<B, T>() -> Router<AppState<B, T>>
where
    B: Backend + Send + Sync + 'static,
    T: Tokenizer + Clone + Send + Sync + 'static,
    AppState<B, T>: Clone + Send + Sync + 'static,
{
    Router::new()
        .route("/models", get(list_models::<B, T>))
        .route("/models/{model}", get(retrieve_model::<B, T>))
}
