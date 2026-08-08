use toy_llm::app::build_app;

#[tokio::test]
//#[ignore]
async fn chat_completion_e2e() {
    tracing::subscriber::set_global_default(
        tracing_subscriber::FmtSubscriber::builder()
            .with_env_filter("info")
            .finish(),
    )
    .expect("Failed to set global default subscriber");

    tracing::info!("starting main loop");
    let app = build_app().await;
    tracing::info!("Setting up {:?}", app);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let client = reqwest::Client::new();

    let req = serde_json::json!({
        "model": "llama",
        "messages": [
            {
                "role": "user",
                "content": "Say hello in one word."
            }
        ],
        "max_tokens": 10,
        "temperature": 0.0
    });

    tracing::info!("Sending request to http://{addr}/chat/completions");
    let response = client
        .post(format!("http://{addr}/chat/completions"))
        .json(&req)
        .send()
        .await
        .unwrap();

    //    assert_eq!(response.status(), reqwest::StatusCode::OK);

    let body: serde_json::Value = response.json().await.unwrap();

    tracing::info!("{}", serde_json::to_string_pretty(&body).unwrap());

    assert!(body["choices"]
        .as_array()
        .is_some_and(|choices| !choices.is_empty()));

    assert!(body["choices"][0]["message"]["content"]
        .as_str()
        .is_some_and(|s| !s.is_empty()));
}
