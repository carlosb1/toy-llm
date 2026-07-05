/*
pub fn main()   {
    println!("loading inference service");
    let device = selected::device();
    let res: (Llama<selected::Backend,Tiktoken>, CacheConfig) = loader::ModelKind::Llama3_2_3B.load(2048, &device).expect("Failed to load model");

    println!("loaded inference service ");
}*/

// Inspect safetensors weight file structure

use burn::store::{ModuleStore, SafetensorsStore};
use clap::Parser;
use std::path::{Path, PathBuf};

mod load_hf;

#[derive(Parser, Debug)]
#[command(name = "load")]
#[command(about = "Inspect a model folder: find config.json + safetensors")]
struct Args {
    /// Folder containing config.json and the weights (.safetensors)
    #[arg(short = 'p', long, default_value = "assets/qwen3")]
    path: PathBuf,

    /// How many tensor keys to print
    #[arg(long, default_value_t = 30)]
    limit: usize,

    /// Force: continue even if config.json is missing / no confirmation prompt
    #[arg(short = 'f', long)]
    force: bool,
}

/// What we discovered inside a model folder.
#[derive(Debug, Default)]
struct ModelDirScan {
    config: Option<PathBuf>,
    weights: Vec<PathBuf>,
    weights_index: Option<PathBuf>, // model.safetensors.index.json (sharded)
}

/// Look inside `dir` and find config.json + all *.safetensors files.
fn scan_model_dir(dir: &Path) -> std::io::Result<ModelDirScan> {
    let mut scan = ModelDirScan::default();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        // Skip subdirectories; we only care about files here.
        if !path.is_file() {
            continue;
        }

        // File name as &str (skip if not valid UTF-8).
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };

        match name {
            "config.json" => scan.config = Some(path.clone()),
            "model.safetensors.index.json" => scan.weights_index = Some(path.clone()),
            _ => {
                // Match by extension for weight shards.
                if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
                    scan.weights.push(path.clone());
                }
            }
        }
    }

    // Deterministic order (read_dir order is OS-dependent).
    scan.weights.sort();

    Ok(scan)
}

fn main() {
    let args = Args::parse(); // your clap struct with `path: PathBuf`

    if !args.path.is_dir() {
        eprintln!("error: '{}' is not a directory", args.path.display());
        std::process::exit(1);
    }

    let scan = match scan_model_dir(&args.path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error reading '{}': {e}", args.path.display());
            std::process::exit(1);
        }
    };

    // Report findings.
    match &scan.config {
        Some(p) => eprintln!("config.json:  found -> {}", p.display()),
        None => eprintln!("config.json:  MISSING"),
    }

    if scan.weights.is_empty() {
        eprintln!("safetensors:  MISSING");
    } else {
        eprintln!("safetensors:  {} file(s)", scan.weights.len());
        for w in &scan.weights {
            eprintln!("   - {}", w.display());
        }
    }

    if let Some(idx) = &scan.weights_index {
        eprintln!("shard index:  {}", idx.display());
    }

    // Decide whether to proceed.
    if scan.weights.is_empty() {
        eprintln!("no weights to inspect, aborting");
        std::process::exit(1);
    }

    // Inspect the first shard (or loop over all of them).
    let weights_path = &scan.weights[0];
    eprintln!("\nInspecting: {}", weights_path.display());

    let mut store = SafetensorsStore::from_file(weights_path);
    // ... your existing store.keys() loop ...
}
