use burn_store::{ModuleStore, SafetensorsStore};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "check")]
#[command(about = "Inspect a .safetensors file: list tensor keys, shapes, dtypes")]
struct Args {
    /// Path to the .safetensors file to inspect
    #[arg(default_value = "assets/qwen3/model.safetensors")]
    path: PathBuf,

    /// How many keys to print
    #[arg(long, default_value_t = 30)]
    limit: usize,

    /// Print all keys (ignores --limit)
    #[arg(long)]
    all: bool,
}
fn main() {
    let args = Args::parse();
    let path = args.path;
    eprintln!("Inspecting: {:?}", path);

    let mut store = SafetensorsStore::from_file(&path);

    match store.keys() {
        Ok(keys) => {
            println!("Total keys: {}", keys.len());
            println!();
            let take = if args.all { keys.len() } else { args.limit };
            // Print first 30 keys
            for key in keys.iter().take(take) {
                if let Ok(Some(snapshot)) = store.get_snapshot(key) {
                    if let Ok(data) = snapshot.to_data() {
                        println!("{}: {:?} dtype={:?}", key, data.shape, data.dtype);
                    } else {
                        println!("{}: (could not read data)", key);
                    }
                } else {
                    println!("{}: (no snapshot)", key);
                }
            }

            if keys.len() > 30 {
                println!("... ({} more)", keys.len() - 30);
            }
        }
        Err(e) => {
            eprintln!("Error getting keys: {:?}", e);
        }
    }
}
