use crate::api::http::generate::{AppState, GenerateHttpRequest};
use crate::models::llama::model::TokenTensor;
use crate::tokenizer::Tokenizer;
use burn::prelude::{Backend, Int};
use burn::Tensor;

pub fn create_token_tensors<B: Backend, T: Tokenizer>(
    state: &AppState<B, T>,
    prompt: String,
    sample_len: usize,
) -> TokenTensor<B> {
    let input_tokens = state.tokenizer_handler.tokenize(prompt.as_str());
    let prompt_len = input_tokens.dims()[0];

    tracing::info!(
        prompt_tokens = prompt_len,
        total_tokens = prompt_len + sample_len,
        "prompt tokenized"
    );

    let mut tokens =
        Tensor::<B, 1, Int>::empty([prompt_len + sample_len], &state.tokenizer_handler.device);
    tokens = tokens.slice_assign([0..prompt_len], input_tokens);

    tracing::info!("input tensor prepared with shape {:?}", tokens.shape());
    let input_pos =
        Tensor::<B, 1, Int>::arange(0..prompt_len as i64, &state.tokenizer_handler.device);
    let stop_tokens = Tensor::from_ints(
        state.tokenizer_handler.tokenizer.stop_ids().as_slice(),
        &state.tokenizer_handler.device,
    );
    let token_tensors = TokenTensor {
        prompt_len,
        tokens,
        input_pos,
        stop_tokens,
    };
    token_tensors
}
