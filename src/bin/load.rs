use llama_burn::backend::selected;
use llama_burn::models::cacheconfig::CacheConfig;
use llama_burn::models::loader;
use llama_burn::tokenizer::Tiktoken;
use llama_burn::models::llama::Llama;

pub fn main()   {
    println!("loading inference service");
    let device = selected::device();
    let res: (Llama<selected::Backend,Tiktoken>, CacheConfig) = loader::ModelKind::Llama3_2_3B.load(2048, &device).expect("Failed to load model");

    println!("loaded inference service ");
}