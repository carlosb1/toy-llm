use async_openai::types::chat::FinishReason;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub struct GenerationMetrics {
    pub queue_time: Duration,
    pub ttft: Duration,
    pub generation_time: Duration,
    pub total_time: Duration,

    pub prompt_tokens: usize,
    pub generated_tokens: usize,

    pub finish_reason: FinishReason,
}

#[derive(Debug, Default)]
pub struct MetricsRegistry {
    total_requests: AtomicU64,
    successful_requests: AtomicU64,
    failed_requests: AtomicU64,
    active_requests: AtomicU64,
    prompt_tokens: AtomicU64,
    generated_tokens: AtomicU64,
}

impl MetricsRegistry {
    pub fn request_started(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.active_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_success(&self, metrics: &GenerationMetrics) {
        self.successful_requests.fetch_add(1, Ordering::Relaxed);

        self.active_requests.fetch_sub(1, Ordering::Relaxed);

        self.prompt_tokens
            .fetch_add(metrics.prompt_tokens as u64, Ordering::Relaxed);

        self.generated_tokens
            .fetch_add(metrics.generated_tokens as u64, Ordering::Relaxed);
    }

    pub fn record_failure(&self) {
        self.failed_requests.fetch_add(1, Ordering::Relaxed);

        self.active_requests.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),

            successful_requests: self.successful_requests.load(Ordering::Relaxed),

            failed_requests: self.failed_requests.load(Ordering::Relaxed),

            active_requests: self.active_requests.load(Ordering::Relaxed),

            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),

            generated_tokens: self.generated_tokens.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct MetricsSnapshot {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub active_requests: u64,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
}

pub struct GenerationProfiler {
    request_started_at: Instant,
    request_finished_at: Option<Instant>,

    queued_at: Option<Instant>,
    worker_started_at: Option<Instant>,

    prefill_started_at: Option<Instant>,
    prefill_finished_at: Option<Instant>,
    first_token_at: Option<Instant>,
    last_token_at: Option<Instant>,
    worker_finished_at: Option<Instant>,

    input_tokens: usize,
    output_tokens: usize,
    decode_started_at: Option<Instant>,
    decode_finished_at: Option<Instant>,
}

impl GenerationProfiler {
    pub fn new() -> Self {
        Self {
            request_started_at: Instant::now(),
            request_finished_at: None,

            queued_at: None,
            worker_started_at: None,

            prefill_started_at: None,
            prefill_finished_at: None,
            first_token_at: None,
            last_token_at: None,
            worker_finished_at: None,

            input_tokens: 0,
            output_tokens: 0,

            decode_started_at: None,
            decode_finished_at: None,
        }
    }

    pub fn set_input_tokens(&mut self, input_tokens: usize) {
        self.input_tokens = input_tokens;
    }

    pub fn queued(&mut self) {
        self.queued_at = Some(Instant::now());
    }

    pub fn worker_started(&mut self) {
        self.worker_started_at = Some(Instant::now());
    }

    pub fn prefill_started(&mut self) {
        self.prefill_started_at = Some(Instant::now());
    }

    pub fn prefill_finished(&mut self) {
        self.prefill_finished_at = Some(Instant::now());
    }

    /// Marca cuándo el motor ha producido su primera decisión.
    ///
    /// Puede ser un token normal o un token de parada.
    pub fn first_token_sampled(&mut self) {
        self.first_token_at.get_or_insert_with(Instant::now);
    }

    /// Contabiliza un token incorporado a la salida.
    pub fn output_token_generated(&mut self) {
        let now = Instant::now();

        self.first_token_at.get_or_insert(now);
        self.last_token_at = Some(now);
        self.output_tokens += 1;
    }

    pub fn worker_finished(&mut self) {
        self.worker_finished_at = Some(Instant::now());
    }

    pub fn request_finished(&mut self) {
        self.request_finished_at = Some(Instant::now());
    }

    pub fn decode_started(&mut self) {
        self.decode_started_at = Some(Instant::now());
    }

    pub fn decode_finished(&mut self) {
        self.decode_finished_at = Some(Instant::now());
    }

    pub fn decode_duration(&self) -> Option<Duration> {
        Some(
            self.decode_finished_at?
                .duration_since(self.decode_started_at?),
        )
    }

    pub fn decode_tokens(&self) -> usize {
        self.output_tokens.saturating_sub(1)
    }

    pub fn decode_tokens_per_second(&self) -> Option<f64> {
        let tokens = self.decode_tokens();
        let seconds = self.decode_duration()?.as_secs_f64();

        if tokens == 0 || seconds == 0.0 {
            return None;
        }

        Some(tokens as f64 / seconds)
    }
}

#[macro_export]
macro_rules! profile {
    ($profiler:expr, $method:ident $(, $arg:expr)* $(,)?) => {
        if let Some(profiler) = ($profiler).as_mut() {
            profiler.$method($($arg),*);
        }
    };
}
