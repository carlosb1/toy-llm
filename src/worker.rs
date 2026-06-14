use std::sync::Arc;
use anyhow::anyhow;
use burn::prelude::Backend;
use tokio::sync::{mpsc, Mutex};
use crate::engine::BurnEngineLlama;
use crate::models::llama::{InferenceRequest, RequestState};
use crate::transformer::KeyValueCache;

pub async fn burn_worker<B: Backend>(mut rx: mpsc::Receiver<InferenceRequest<B>>, engine: Arc<Mutex<BurnEngineLlama<B>>>, cache: Arc<Mutex<Vec<KeyValueCache<B>>>>) {
    while let Some(req) = rx.recv().await {
        let temperature = req.temperature;
        let response_tx = req.response_tx;
        let prompt_len = req.prompt_len;
        let tokens = req.tokens;
        let stop_tokens = req.stop_tokens;
        let input_pos = req.input_pos;
        let sample_len = req.sample_len;


        let mut state = RequestState {
            prompt_len,
            tokens,
            stop_tokens,
            input_pos,
            sample_len,
            num_generated_tokens: 0,
            is_finished: false,
        };

        let result = {
            let guard = &mut *engine.lock().await;
            let llama = &mut guard.llama;
            let sampler = &mut guard.sampler;
            let mut  cache = cache.lock().await;

            llama.generate_from_tokens(
                &mut state,
                sampler,
                temperature,
                &mut cache
            )
        };

        let output = match result {
            Ok(output) => output,
            Err(e) => {
                tracing::error!("generation error: {}", e);
                let _ = response_tx.send(Err(anyhow!("generation error: {}", e)));
                continue;
            }
        };

        tracing::info!(
            generated_tokens = state.num_generated_tokens,
            total_tokens = state.prompt_len + state.num_generated_tokens,
            "generation completed"
        );

        let _ = response_tx.send(Ok(output));
    }
}