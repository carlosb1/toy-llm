use std::time::Duration;

const URL: &str = "http://127.0.0.1:8000/v1/completions";
const MODEL: &str = "qwen3-0.6b";

const TOTAL_REQUESTS: usize = 100;
const CONCURRENCY: usize = 4;
const WARMUP_REQUESTS: usize = 5;
const MAX_TOKENS: usize = 64;

#[derive(Debug)]
struct RequestResult {
    latency: Duration,
    status: u16,
    success: bool,
}
#[tokio::main]
async fn main() {
    println!("Benchmarking with {} requests", TOTAL_REQUESTS);
}

/*
#[tokio::main]
async fn main() {
    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .expect("failed to create HTTP client");

    println!(
        "Warmup: {} requests, concurrency: {}",
        WARMUP_REQUESTS, CONCURRENCY
    );

    run_requests(&client, WARMUP_REQUESTS, CONCURRENCY).await;

    println!(
        "Benchmark: {} requests, concurrency: {}",
        TOTAL_REQUESTS, CONCURRENCY
    );

    let benchmark_started = Instant::now();

    let results = run_requests(&client, TOTAL_REQUESTS, CONCURRENCY).await;

    let benchmark_elapsed = benchmark_started.elapsed();

    print_report(results, benchmark_elapsed);
}

async fn run_requests(
    client: &Client,
    total_requests: usize,
    concurrency: usize,
) -> Vec<RequestResult> {
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut tasks = Vec::with_capacity(total_requests);

    for request_number in 0..total_requests {
        let client = client.clone();
        let semaphore = Arc::clone(&semaphore);

        tasks.push(tokio::spawn(async move {
            let permit = semaphore.acquire_owned().await.expect("semaphore closed");

            let result = send_request(&client, request_number).await;

            drop(permit);

            result
        }));
    }

    let mut results = Vec::with_capacity(total_requests);

    for task in tasks {
        match task.await {
            Ok(result) => results.push(result),
            Err(error) => {
                eprintln!("request task failed: {error}");
            }
        }
    }

    results
}

async fn send_request(client: &Client, request_number: usize) -> RequestResult {
    let body = json!({
        "model": MODEL,
        "prompt": format!(
            "Request {request_number}: explain briefly what a binary search tree is."
        ),
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": false
    });

    let started = Instant::now();

    let response = client.post(URL).json(&body).send().await;

    let latency = started.elapsed();

    match response {
        Ok(response) => {
            let status = response.status();

            // Consumimos el body para medir la petición completa.
            let body_result = response.bytes().await;

            RequestResult {
                latency,
                status: status.as_u16(),
                success: status.is_success() && body_result.is_ok(),
            }
        }

        Err(error) => {
            eprintln!("request {request_number} failed: {error}");

            RequestResult {
                latency,
                status: 0,
                success: false,
            }
        }
    }
}

fn print_report(mut results: Vec<RequestResult>, total_elapsed: Duration) {
    if results.is_empty() {
        println!("No requests completed");
        return;
    }

    results.sort_by_key(|result| result.latency);

    let successful = results.iter().filter(|result| result.success).count();

    let failed = results.len() - successful;

    let latencies: Vec<Duration> = results.iter().map(|result| result.latency).collect();

    let min = latencies[0];
    let max = latencies[latencies.len() - 1];

    let average_seconds =
        latencies.iter().map(Duration::as_secs_f64).sum::<f64>() / latencies.len() as f64;

    let p50 = percentile(&latencies, 0.50);
    let p95 = percentile(&latencies, 0.95);
    let p99 = percentile(&latencies, 0.99);

    let requests_per_second = successful as f64 / total_elapsed.as_secs_f64();

    println!();
    println!("Results");
    println!("-------");
    println!("Requests:       {}", results.len());
    println!("Successful:     {successful}");
    println!("Failed:         {failed}");
    println!("Total time:     {:.3} s", total_elapsed.as_secs_f64());
    println!("Requests/sec:   {requests_per_second:.2}");
    println!("Min latency:    {:.3} ms", duration_ms(min));
    println!("Average:        {:.3} ms", average_seconds * 1_000.0);
    println!("p50 latency:    {:.3} ms", duration_ms(p50));
    println!("p95 latency:    {:.3} ms", duration_ms(p95));
    println!("p99 latency:    {:.3} ms", duration_ms(p99));
    println!("Max latency:    {:.3} ms", duration_ms(max));

    let failed_statuses: Vec<u16> = results
        .iter()
        .filter(|result| !result.success)
        .map(|result| result.status)
        .collect();

    if !failed_statuses.is_empty() {
        println!("Failed statuses: {failed_statuses:?}");
    }
}

fn percentile(sorted_values: &[Duration], percentile: f64) -> Duration {
    if sorted_values.is_empty() {
        return Duration::ZERO;
    }

    let index = (percentile * (sorted_values.len() - 1) as f64).round() as usize;

    sorted_values[index]
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

 */
