use crate::tokenizer::Tokenizer;
use burn::prelude::{Backend, Device, Int, Shape, TensorData};
use burn::Tensor;

#[derive(Clone)]
pub struct TokenizerHandle<B: Backend, T: Tokenizer> {
    /// The tokenizer.
    pub tokenizer: T,
    pub device: Device<B>,
}

impl<B: Backend, T: Tokenizer> TokenizerHandle<B, T> {
    pub fn tokenize(&self, text: &str) -> Tensor<B, 1, Int> {
        let bos = !cfg!(feature = "tiny"); // TinyLlama Chat doesn't prepend BOS token with the chat format
        let tokens = self.tokenizer.encode(text, bos, false);

        let shape = Shape::new([tokens.len()]);
        Tensor::<B, 1, Int>::from_data(TensorData::new(tokens, shape), &self.device)
    }
}
