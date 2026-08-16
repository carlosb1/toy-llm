use crate::models::llama::cacheconfig::CacheConfig;
use crate::models::llama::engine::BurnEngineLlama;
use crate::models::llama::model::{InferenceRequest, RequestState};
use crate::profile;
use anyhow::anyhow;
use burn::prelude::Backend;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};

pub async fn burn_worker<B: Backend>(
    mut rx: mpsc::Receiver<InferenceRequest<B>>,
    engine: Arc<Mutex<BurnEngineLlama<B>>>,
    model_registries: Arc<RwLock<HashMap<String, Arc<Mutex<BurnEngineLlama<B>>>>>>,
    cache_config: CacheConfig,
) {
    while let Some(mut req) = rx.recv().await {
        profile!(req.profiler, worker_started);

        let response_tx = req.response_tx;
        let prompt_len = req.prompt_len;
        let tokens = req.tokens;
        let stop_tokens = req.stop_tokens;
        let input_pos = req.input_pos;

        let result = {
            let engine = model_registries
                .read()
                .await
                .get(&req.model_name)
                .map(|engine| engine.clone())
                .unwrap_or_else(|| {
                    tracing::info!("model {} not found, using default engine", req.model_name);
                    engine.clone()
                });
            let guard = &mut *engine.lock().await;
            let default_gen_config = guard.default_generation_config.clone();
            let mut gen_config = req.generation_config.unwrap_or(default_gen_config);
            let mut state = RequestState {
                prompt_len,
                tokens,
                stop_tokens,
                input_pos,
                max_new_tokens: gen_config.max_new_tokens,
                num_generated_tokens: 0,
                is_finished: false,
            };

            let llama = &mut guard.llama;
            let mut cache = cache_config.init_cache(&llama.device);

            let generated = llama.generate_from_tokens(
                &mut state,
                &mut gen_config.sampler,
                gen_config.temperature,
                &mut cache,
                &mut req.profiler,
            );
            tracing::info!(
                generated_tokens = state.num_generated_tokens,
                total_tokens = state.prompt_len + state.num_generated_tokens,
                "generation completed"
            );
            generated
        };

        profile!(req.profiler, worker_finished);

        let mut output = match result {
            Ok(output) => output,
            Err(e) => {
                tracing::error!("generation error: {}", e);
                let _ = response_tx.send(Err(anyhow!("generation error: {}", e)));
                continue;
            }
        };

        if req.profiler.is_some() {
            output.profiler = req.profiler;
        }

        let _ = response_tx.send(Ok(output));
    }
}
