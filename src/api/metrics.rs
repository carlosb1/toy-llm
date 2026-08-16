use axum::{extract::State, routing::get, Json, Router};
use burn::prelude::Backend;
use serde::{Deserialize, Serialize};

use crate::profiler::MetricsSnapshot;
use crate::{
    api::http::{AppError, AppState},
    tokenizer::Tokenizer,
};

#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsResultHttpResponse {
    pub snapshot: MetricsSnapshot,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ResetMetricsHttpResponse {
    pub reset: bool,
}

async fn get_metrics_handler<B, T>(
    State(state): State<AppState<B, T>>,
) -> Result<Json<MetricsResultHttpResponse>, AppError>
where
    B: Backend + Send + Sync + 'static,
    T: Tokenizer + Clone + Send + Sync + 'static,
{
    Ok(Json(MetricsResultHttpResponse {
        snapshot: state.metrics.snapshot(),
    }))
}

async fn reset_metrics_handler<B, T>(
    State(state): State<AppState<B, T>>,
) -> Result<Json<ResetMetricsHttpResponse>, AppError>
where
    B: Backend + Send + Sync + 'static,
    T: Tokenizer + Clone + Send + Sync + 'static,
{
    state.metrics.reset();

    Ok(Json(ResetMetricsHttpResponse { reset: true }))
}

pub fn routes<B, T>() -> Router<AppState<B, T>>
where
    B: Backend + Send + Sync + 'static,
    T: Tokenizer + Clone + Send + Sync + 'static,
    AppState<B, T>: Clone + Send + Sync + 'static,
{
    Router::new().route(
        "/metrics",
        get(get_metrics_handler::<B, T>).delete(reset_metrics_handler::<B, T>),
    )
}
