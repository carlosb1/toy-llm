use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GenerationMetrics {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,

    pub preprocessing_duration: Option<Duration>,
    pub queue_duration: Option<Duration>,
    pub prefill_duration: Option<Duration>,
    pub time_to_first_token: Option<Duration>,
    pub decode_duration: Option<Duration>,
    pub worker_duration: Option<Duration>,
    pub request_duration: Option<Duration>,

    pub prefill_tokens_per_second: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub engine_time_to_first_token: Option<Duration>,
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

    #[deprecated]
    pub fn record_request(
        &self,
        successful: bool,
        input_tokens: u64,
        output_tokens: u64,
        latency_ms: u64,
    ) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        if successful {
            self.successful_requests.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_requests.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn reset(&self) {
        self.total_requests.store(0, Ordering::Relaxed);
        self.successful_requests.store(0, Ordering::Relaxed);
        self.failed_requests.store(0, Ordering::Relaxed);
        self.active_requests.store(0, Ordering::Relaxed);
        self.prompt_tokens.store(0, Ordering::Relaxed);
        self.generated_tokens.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct MetricsSnapshot {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub active_requests: u64,
    pub prompt_tokens: u64,
    pub generated_tokens: u64,
}

use std::fmt;

impl fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let success_rate = if self.total_requests > 0 {
            self.successful_requests as f64 / self.total_requests as f64 * 100.0
        } else {
            0.0
        };

        let failure_rate = if self.total_requests > 0 {
            self.failed_requests as f64 / self.total_requests as f64 * 100.0
        } else {
            0.0
        };

        let total_tokens = self.prompt_tokens + self.generated_tokens;

        writeln!(f, "Metrics report")?;
        writeln!(f, "==============")?;
        writeln!(f, "Requests")?;
        writeln!(f, "  Total:      {}", self.total_requests)?;
        writeln!(
            f,
            "  Successful: {} ({success_rate:.2}%)",
            self.successful_requests
        )?;
        writeln!(
            f,
            "  Failed:     {} ({failure_rate:.2}%)",
            self.failed_requests
        )?;
        writeln!(f, "  Active:     {}", self.active_requests)?;
        writeln!(f)?;
        writeln!(f, "Tokens")?;
        writeln!(f, "  Prompt:     {}", self.prompt_tokens)?;
        writeln!(f, "  Generated:  {}", self.generated_tokens)?;
        writeln!(f, "  Total:      {total_tokens}")?;

        Ok(())
    }
}

#[derive(Debug)]
pub struct GenerationProfiler {
    request_started_at: Instant,
    request_finished_at: Option<Instant>,

    queued_at: Option<Instant>,
    worker_started_at: Option<Instant>,

    prefill_started_at: Option<Instant>,
    prefill_finished_at: Option<Instant>,
    first_token_at: Option<Instant>,

    decode_started_at: Option<Instant>,
    decode_finished_at: Option<Instant>,

    worker_finished_at: Option<Instant>,

    input_tokens: usize,
    output_tokens: usize,
}

impl Default for GenerationProfiler {
    fn default() -> Self {
        Self::new()
    }
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

            decode_started_at: None,
            decode_finished_at: None,

            worker_finished_at: None,

            input_tokens: 0,
            output_tokens: 0,
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

    /// Registra la primera decisión producida por el modelo.
    ///
    /// Puede ser un token de salida o un token de parada.
    pub fn first_token_sampled(&mut self) {
        self.first_token_at.get_or_insert_with(Instant::now);
    }

    /// Registra un token añadido realmente a la salida.
    pub fn output_token_generated(&mut self) {
        self.output_tokens += 1;
    }

    pub fn decode_started(&mut self) {
        self.decode_started_at = Some(Instant::now());
    }

    pub fn decode_finished(&mut self) {
        self.decode_finished_at = Some(Instant::now());
    }

    pub fn worker_finished(&mut self) {
        self.worker_finished_at = Some(Instant::now());
    }

    pub fn request_finished(&mut self) {
        self.request_finished_at = Some(Instant::now());
    }

    pub fn preprocessing_duration(&self) -> Option<Duration> {
        self.queued_at
            .map(|queued| queued.duration_since(self.request_started_at))
    }

    pub fn queue_duration(&self) -> Option<Duration> {
        duration_between(self.queued_at, self.worker_started_at)
    }

    pub fn prefill_duration(&self) -> Option<Duration> {
        duration_between(self.prefill_started_at, self.prefill_finished_at)
    }

    /// TTFT desde que se creó el profiler.
    pub fn time_to_first_token(&self) -> Option<Duration> {
        self.first_token_at
            .map(|first| first.duration_since(self.request_started_at))
    }

    /// TTFT interno del motor, sin preprocesamiento ni cola.
    pub fn engine_time_to_first_token(&self) -> Option<Duration> {
        duration_between(self.worker_started_at, self.first_token_at)
    }

    pub fn decode_duration(&self) -> Option<Duration> {
        duration_between(self.decode_started_at, self.decode_finished_at)
    }

    pub fn worker_duration(&self) -> Option<Duration> {
        duration_between(self.worker_started_at, self.worker_finished_at)
    }

    pub fn request_duration(&self) -> Option<Duration> {
        self.request_finished_at
            .map(|finished| finished.duration_since(self.request_started_at))
    }

    pub fn prefill_tokens_per_second(&self) -> Option<f64> {
        rate(self.input_tokens, self.prefill_duration()?)
    }

    pub fn decode_tokens(&self) -> usize {
        self.output_tokens.saturating_sub(1)
    }

    pub fn decode_tokens_per_second(&self) -> Option<f64> {
        rate(self.decode_tokens(), self.decode_duration()?)
    }

    pub fn metrics(&self) -> GenerationMetrics {
        GenerationMetrics {
            prompt_tokens: self.input_tokens,
            generated_tokens: self.output_tokens,

            preprocessing_duration: self.preprocessing_duration(),
            queue_duration: self.queue_duration(),
            prefill_duration: self.prefill_duration(),
            time_to_first_token: self.time_to_first_token(),
            engine_time_to_first_token: self.engine_time_to_first_token(),
            decode_duration: self.decode_duration(),
            worker_duration: self.worker_duration(),
            request_duration: self.request_duration(),

            prefill_tokens_per_second: self.prefill_tokens_per_second(),
            decode_tokens_per_second: self.decode_tokens_per_second(),
        }
    }
}

fn duration_between(start: Option<Instant>, finish: Option<Instant>) -> Option<Duration> {
    match (start, finish) {
        (Some(start), Some(finish)) => finish.checked_duration_since(start),
        _ => None,
    }
}

fn rate(tokens: usize, duration: Duration) -> Option<f64> {
    let seconds = duration.as_secs_f64();

    if tokens == 0 || seconds == 0.0 {
        return None;
    }

    Some(tokens as f64 / seconds)
}

#[macro_export]
macro_rules! profile {
    ($profiler:expr, $method:ident $(, $arg:expr)* $(,)?) => {
        if let Some(profiler) = ($profiler).as_mut() {
            profiler.$method($($arg),*);
        }
    };
}
