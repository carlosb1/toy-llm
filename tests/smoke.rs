use burn::backend::ndarray::NdArrayDevice;
use burn::backend::NdArray;
use toy_llm::models::qwen::loader::initialize_cache;
use toy_llm::models::qwen::model::QuantizationMode;
use toy_llm::models::qwen::sampling::Sampler;
use toy_llm::models::qwen::tokenizer::Qwen3Tokenizer;

fn model_dir() -> std::path::PathBuf {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.join("models").join("Qwen3-0.6B-GGUF")
}

fn gguf_path() -> std::path::PathBuf {
    model_dir().join("Qwen3-0.6B-Q8_0.gguf")
}

fn tokenizer_path() -> std::path::PathBuf {
    model_dir().join("tokenizer.json")
}

type B = NdArray;
#[test]
#[ignore]
fn smoke_gguf_generate() {
    tracing::subscriber::set_global_default(
        tracing_subscriber::FmtSubscriber::builder()
            .with_env_filter("info")
            .finish(),
    )
    .expect("Failed to set global default subscriber");

    let gguf = gguf_path();
    let tok = tokenizer_path();
    if !gguf.exists() || !tok.exists() {
        eprintln!(
            "Skipping smoke test: model files not found at {}",
            model_dir().display()
        );
        return;
    }

    let device = NdArrayDevice::Cpu;
    let tokenizer = Qwen3Tokenizer::new(&tok).expect("Failed to load tokenizer");

    eprintln!("Loading model (f32, no quantization)...");
    let mut model =
        toy_llm::models::qwen::loader::from_gguf::<B>(&gguf, 2048, QuantizationMode::None, &device)
            .expect("Failed to load GGUF model");

    let prompt = tokenizer.apply_chat_template(
        "You are a helpful assistant.",
        "What is the capital of France?",
    );

    let caches = initialize_cache(
        model.config.num_hidden_layers,
        model.config.num_key_value_heads,
        model.max_seq_len,
        model.config.head_dim,
        &device.clone(),
    );

    let mut sampler = Sampler::Argmax;
    let result = model
        .generate(caches, &tokenizer, &prompt, 32, 0.0, &mut sampler)
        .expect("Generation failed");

    eprintln!("Generated text: {:?}", result.text);
    eprintln!("Tokens: {}, Time: {:.2}s", result.tokens, result.time);

    tracing::info!("generated text: {:?}", result.text);
    tracing::info!("Tokens: {}, Time: {:.2}s", result.tokens, result.time);

    assert!(
        !result.text.is_empty(),
        "Generated text should not be empty"
    );
    assert!(result.tokens > 0, "Should generate at least one token");

    // Deterministic argmax output for regression detection.
    // The model produces a <think> block first, so we check for a known prefix.
    assert!(
        result.text.starts_with("<think>"),
        "Expected output to start with '<think>', got: {:?}",
        &result.text[..result.text.len().min(80)]
    );
}
