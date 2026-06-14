use std::collections::HashMap;
use std::time::Instant;

use burn::module::Module;
use burn::record::{FileRecorder, RecorderError};
use burn::tensor::cast::ToElement;
use burn::{
    config::Config,
    nn::{RotaryEncoding, RotaryEncodingConfig},
    tensor::{
        activation::softmax, backend::Backend, Device, ElementConversion, Int, Shape, Tensor,
        TensorData,
    },
};

use crate::{
    sampling::Sampler,
    tokenizer::Tokenizer,
    transformer::{KeyValueCache, Transformer, TransformerConfig},
};
#[cfg(feature = "import")]
use burn_store::{
    KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, PytorchStore, SafetensorsStore,
};
use tokio::sync::oneshot;

/// Generated text sample output.
pub struct GenerationOutput {
    /// The generated text.
    pub text: String,
    /// The number of generated tokens.
    pub tokens: usize,
    /// The time it took to produce the output tokens (generation + decoding).
    pub time: f64,
}

#[cfg(feature = "pretrained")]
#[allow(unused_imports)]
use crate::models::pretrained::{self, ModelMeta};

pub struct TokenTensor<B: Backend> {
    pub prompt_len: usize,
    pub tokens: Tensor<B, 1, Int>,
    pub input_pos: Tensor<B, 1, Int>,
    pub stop_tokens: Tensor<B, 1, Int>,
}


pub struct InferenceRequest<B: Backend> {
    pub prompt_len: usize,
    pub tokens: Tensor<B, 1, Int>,
    pub stop_tokens: Tensor<B, 1, Int>,
    pub input_pos: Tensor<B, 1, Int>,
    pub temperature: f64,
    pub sample_len: usize,
    pub response_tx: oneshot::Sender<anyhow::Result<GenerationOutput>>,
}

impl<B: Backend> InferenceRequest<B> {
    pub fn from_tensors(
        tensors: TokenTensor<B>,
        sample_len: usize,
        temperature: f64,
        response_tx: oneshot::Sender<anyhow::Result<GenerationOutput>>,
    ) -> Self {
        Self {
            prompt_len: tensors.prompt_len,
            tokens: tensors.tokens,
            stop_tokens: tensors.stop_tokens,
            input_pos: tensors.input_pos,
            temperature,
            sample_len,
            response_tx,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RequestState<B: Backend> {
    pub prompt_len: usize,
    pub tokens: Tensor<B, 1, Int>,
    pub stop_tokens: Tensor<B, 1, Int>,
    pub input_pos: Tensor<B, 1, Int>,
    pub sample_len: usize,
    // status
    pub num_generated_tokens: usize,
    pub is_finished: bool,
}

/// Meta Llama large language model and tokenizer.
pub struct Llama<B: Backend, T: Tokenizer> {
    /// The tokenizer.
    pub tokenizer: T,
    /// Llama decoder-only transformer.
    pub model: Transformer<B>,
    /// Key-value cache for each transformer block.
  //  pub cache: Vec<KeyValueCache<B>>,
    /// Rotary positional encoding (RoPE).
    pub rope: RotaryEncoding<B>,
    pub device: Device<B>,
}

impl<B: Backend, T: Tokenizer> Llama<B, T> {
    pub fn prepare_request(&self, prompt: &str, sample_len: usize) -> RequestState<B> {
        let input_tokens = self.tokenize(prompt);
        let prompt_len = input_tokens.dims()[0];

        let mut tokens = Tensor::<B, 1, Int>::empty([prompt_len + sample_len], &self.device);
        tokens = tokens.slice_assign([0..prompt_len], input_tokens);

        let input_pos = Tensor::<B, 1, Int>::arange(0..prompt_len as i64, &self.device);

        let stop_tokens = Tensor::from_ints(self.tokenizer.stop_ids().as_slice(), &self.device);

        RequestState {
            prompt_len,
            tokens,
            stop_tokens,
            num_generated_tokens: 0,
            input_pos,
            sample_len,
            is_finished: false,
        }
    }

    fn prefill(&mut self, state: &mut RequestState<B>, cache: &mut Vec<KeyValueCache<B>>) -> Tensor<B, 2> {
        let x = state
            .tokens
            .clone()
            .select(0, state.input_pos.clone())
            .reshape([1, -1]);

        let logits = self.model.forward(x, cache, &self.rope);

        let [batch_size, seq_len, _vocab_size] = logits.dims();

        let next_token_logits = logits
            .slice([0..batch_size, seq_len - 1..seq_len])
            .squeeze_dim(1);

        let t = state.input_pos.dims()[0];
        state.input_pos = state.input_pos.clone().slice([t - 1..t]) + 1;

        next_token_logits
    }
    fn decode_step(&mut self, state: &mut RequestState<B>, cache: &mut Vec<KeyValueCache<B>>) -> Tensor<B, 2> {
        let x = state
            .tokens
            .clone()
            .select(0, state.input_pos.clone())
            .reshape([1, -1]);

        let logits = self.model.forward(x, cache, &self.rope);

        let [batch_size, seq_len, _vocab_size] = logits.dims();

        let next_token_logits = logits
            .slice([0..batch_size, seq_len - 1..seq_len])
            .squeeze_dim(1);

        let t = state.input_pos.dims()[0];
        state.input_pos = state.input_pos.clone().slice([t - 1..t]) + 1;

        next_token_logits
    }

    fn sample_next_token(
        &self,
        next_token_logits: Tensor<B, 2>,
        temperature: f64,
        sampler: &mut Sampler,
    ) -> Tensor<B, 1, Int> {
        let mut logits = next_token_logits;

        if temperature > 0.0 {
            logits = temperature_scaled_softmax(logits, temperature);
        }

        sampler.sample(logits).squeeze_dim(0)
    }

    fn should_stop(&self, next_token: Tensor<B, 1, Int>, stop_tokens: &Tensor<B, 1, Int>) -> bool {
        stop_tokens
            .clone()
            .equal(next_token)
            .any()
            .into_scalar()
            .to_bool()
    }

    fn finalize_output(&self, state: RequestState<B>, elapsed: f64) -> GenerationOutput {
        let tokens = state.tokens.into_data().as_slice::<B::IntElem>().unwrap()
            [state.prompt_len..state.prompt_len + state.num_generated_tokens]
            .iter()
            .map(|t| t.elem::<u32>())
            .collect::<Vec<_>>();

        let generated = self.tokenizer.decode(tokens);

        GenerationOutput {
            text: generated,
            tokens: state.num_generated_tokens,
            time: elapsed,
        }
    }
    fn append_token(&self, state: &mut RequestState<B>, next_token: Tensor<B, 1, Int>) {
        let pos = state.prompt_len + state.num_generated_tokens;
        state.tokens = state
            .tokens
            .clone()
            .slice_assign([pos..pos + 1], next_token);
        state.num_generated_tokens += 1;
    }

    #[allow(clippy::single_range_in_vec_init)]
    pub fn generate(
        &mut self,
        prompt: &str,
        sample_len: usize,
        temperature: f64,
        sampler: &mut Sampler,
        cache: &mut Vec<KeyValueCache<B>>,
    ) -> GenerationOutput {
        let mut state = self.prepare_request(prompt, sample_len);
        let stop_tokens = Tensor::from_ints(self.tokenizer.stop_ids().as_slice(), &self.device);

        let now = Instant::now();

        // PREFILL
        let logits = self.prefill(&mut state, cache);
        let next_token = self.sample_next_token(logits, temperature, sampler);

        if !self.should_stop(next_token.clone(), &stop_tokens) {
            self.append_token(&mut state, next_token);
        } else {
            state.is_finished = true;
        }

        // DECODE
        while !state.is_finished && state.num_generated_tokens < sample_len {
            let logits = self.decode_step(&mut state, cache);
            let next_token = self.sample_next_token(logits, temperature, sampler);

            if self.should_stop(next_token.clone(), &stop_tokens) {
                state.is_finished = true;
                break;
            }

            self.append_token(&mut state, next_token);
        }

        let elapsed = now.elapsed().as_secs_f64();
        self.finalize_output(state, elapsed)
    }

    /// Encode a string into a tensor of tokens.
    pub fn tokenize(&self, text: &str) -> Tensor<B, 1, Int> {
        let bos = !cfg!(feature = "tiny"); // TinyLlama Chat doesn't prepend BOS token with the chat format
        let tokens = self.tokenizer.encode(text, bos, false);

        let shape = Shape::new([tokens.len()]);
        Tensor::<B, 1, Int>::from_data(TensorData::new(tokens, shape), &self.device)
    }

    /// Save Llama model to file using the specified recorder.
    pub fn save<R: FileRecorder<B>>(
        self,
        file_path: &str,
        recorder: &R,
    ) -> Result<(), RecorderError> {
        println!("Saving record...");
        let now = Instant::now();
        self.model.save_file(file_path, recorder)?;
        let elapsed = now.elapsed().as_secs();
        println!("Saved in {}s", elapsed);

        Ok(())
    }

    /// Load Llama model from file using the specified recorder.
    pub fn load<R: FileRecorder<B>>(
        mut self,
        file_path: &str,
        recorder: &R,
    ) -> Result<Self, RecorderError> {
        println!("Loading record...");
        let now = Instant::now();
        self.model = self.model.load_file(file_path, recorder, &self.device)?;
        let elapsed = now.elapsed().as_secs();
        println!("Loaded in {}s", elapsed);

        Ok(self)
    }

    /// Reset the model state (used between generations)
    /*
    pub fn reset(&mut self,  cache: &mut Vec<KeyValueCache<B>>) {
        cache.iter_mut().for_each(|cache| cache.reset());
    }
*/
    pub fn generate_from_tokens(
        &mut self,
        state: &mut RequestState<B>,
        sampler: &mut Sampler,
        temperature: f64,
        cache: &mut Vec<KeyValueCache<B>>
    ) -> anyhow::Result<GenerationOutput> {
        let now = Instant::now();

        // PREFILL
        tracing::info!(
            "Starting prefill generation with prompt length: {}",
            state.prompt_len
        );
        let logits = self.prefill(state, cache);
        let next_token = self.sample_next_token(logits, temperature, sampler);

        if !self.should_stop(next_token.clone(), &state.stop_tokens) {
            self.append_token(state, next_token);
        } else {
            state.is_finished = true;
        }

        // DECODE
        tracing::info!(
            "Prefilled in {}s starting decoding",
            now.elapsed().as_secs()
        );
        while !state.is_finished && state.num_generated_tokens < state.sample_len {
            println!(
                "Generated {} tokens so far...\r",
                state.num_generated_tokens
            );
            println!(
                "Decoding step with input position: {:?} and stop tokens: {:?}\r",
                state.input_pos, state.stop_tokens
            );
            let logits = self.decode_step(state, cache);
            let next_token = self.sample_next_token(logits, temperature, sampler);

            println!("Sampled next token: {:?}\r", next_token);
            if self.should_stop(next_token.clone(), &state.stop_tokens) {
                state.is_finished = true;
                break;
            }

            self.append_token(state, next_token);
        }

        let elapsed = now.elapsed().as_secs_f64();
        let generation_output = self.finalize_output(state.clone(), elapsed);

        tracing::info!("Final generation output");
        Ok(generation_output)
    }
}

/// Check that the requested context length is within the model's supported maximum.

pub fn check_context_length(max_seq_len: usize, max_context_len: usize) {
    if max_seq_len > max_context_len {
        eprintln!(
            "Warning: max_seq_len ({}) exceeds the model's maximum context length ({})",
            max_seq_len, max_context_len
        );
    }
}

pub(crate) fn temperature_scaled_softmax<B: Backend>(
    logits: Tensor<B, 2>,
    temperature: f64,
) -> Tensor<B, 2> {
    softmax(logits / temperature, 1)
}
