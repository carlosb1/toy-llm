# Runtime and Backend Architecture

```text id="rhxuh7"
CLIENT (CLI / server / bindings)
  |
  v
+------------------------------+
| Frontend                     | <- main/CLI, llama-server, Python bindings, etc.
+------------------------------+
  |
  v
+------------------------------+
| Tokenizer                    | <- BPE/SPM depending on the model, chat templates
+------------------------------+
  |
  v
+------------------------------+
| Model Loader (GGUF/GGML)     | <- reads weights, metadata, tensor layout
| + Quantization info          |
+------------------------------+
  |
  v
+------------------------------+
| Graph Builder (ggml)         | <- builds the operation graph: matmul, rmsnorm...
+------------------------------+
  |
  v
+------------------------------+
| Inference Runtime            | <- executes the graph step by step: prefill/decode
|  - KV Cache                  |    stores K/V per layer
|  - Threading                 |    splits CPU work
+------------------------------+
  |
  v
+------------------------------+
| Backend Compute              | <- CPU AVX/NEON, OpenBLAS
| (ggml backends)              |    CUDA, Metal, Vulkan, etc.
+------------------------------+
  |
  v
+------------------------------+
| Sampler                      | <- top-k, top-p, temperature, mirostat, penalties
+------------------------------+
  |
  v
+------------------------------+
| Detokenizer / Streaming      | <- prints tokens or responds over HTTP
+------------------------------+
```

## What each module does

### Frontend

The frontend is the system entry point.

It can be:

* A classic CLI.
* `llama-server`.
* Language bindings, such as Python bindings.
* External integrations.

Its main responsibilities are:

* Receiving the prompt or HTTP request.
* Loading the configuration.
* Initializing the model.
* Coordinating generation.
* Returning the output to the user or client.

Examples:

```text id="3s1b34"
main / CLI
llama-server
Python bindings
other bindings
```

---

### Tokenizer

The tokenizer converts text into tokens and tokens back into text.

Its main responsibilities are:

* Converting the prompt into token IDs.
* Applying the format expected by the model.
* Handling chat templates.
* Adapting to the tokenizer type used by the model, for example BPE or SentencePiece.

Conceptual example:

```text id="efnt91"
"Hello, how are you?" -> [token_1, token_2, token_3, ...]
```

For chat models, this layer may also apply a template like:

```text id="w7qpae"
<|system|>
You are a helpful assistant.
<|user|>
Explain what llama.cpp is.
<|assistant|>
```

---

### Model Loader

The `Model Loader` loads the model from disk.

In `llama.cpp`, this is usually done from formats such as:

* GGUF.
* GGML, in older versions.

Its main responsibilities are:

* Reading the model tensors.
* Reading the metadata.
* Detecting the quantization type.
* Preparing the memory layout.
* Mapping weights into memory, for example using `mmap` when applicable.

The metadata may include information such as:

```text id="34cg7i"
n_layers
n_heads
hidden_size
rope settings
vocab size
context length
quantization type
```

It also loads quantized weights, for example:

```text id="o7tp6r"
Q4
Q5
Q8
FP16
```

---

### Graph Builder

The `Graph Builder` builds the computation graph that represents the model forward pass.

In `llama.cpp`, this part is based on `ggml`.

The graph includes operations such as:

* RMSNorm or LayerNorm.
* MatMuls.
* RoPE.
* Attention.
* MLP.
* Q/K/V/O projections.
* Activations.
* Logit operations.

Conceptually, instead of executing each operation directly in isolation, a graph of operations is built and then executed by the runtime.

Simplified example:

```text id="9sykvx"
input tokens
  -> embeddings
  -> transformer block 1
  -> transformer block 2
  -> ...
  -> transformer block N
  -> logits
```

---

### Inference Runtime

The `Inference Runtime` executes the graph during inference.

Generation is usually split into two main phases:

#### Prefill

Processes the full initial prompt.

```text id="a2kaq1"
full prompt -> initial KV cache -> logits for the last token
```

#### Decode

Generates tokens one by one.

```text id="k2vmk6"
last token -> use KV cache -> new logits -> sample next token
```

The runtime also maintains the `KV Cache`.

The `KV Cache` stores the attention keys and values per layer and per token, avoiding recomputation of the full context at every generation step.

Conceptually:

```text id="jgy6ma"
KV Cache
  layer 0:
    K previous tokens
    V previous tokens
  layer 1:
    K previous tokens
    V previous tokens
  ...
```

It also manages CPU parallelism through threading, splitting work across multiple threads when possible.

---

### Backend Compute

The backend compute layer is where the mathematical operations are actually executed.

In `llama.cpp`, this is done through the `ggml` backends.

It may use:

* Optimized CPU execution.
* SIMD instructions such as AVX, AVX2, AVX512, or NEON.
* OpenBLAS.
* CUDA.
* Metal.
* Vulkan.
* Other supported backends.

Its main responsibilities are:

* Executing kernels.
* Deciding where each operation runs.
* Managing partial or full GPU offloading.
* Optimizing matmuls and attention operations.
* Coordinating memory between CPU and GPU when applicable.

Conceptual example:

```text id="7vknms"
matmul -> CPU AVX
matmul -> CUDA
rmsnorm -> Metal
attention -> Vulkan
```

---

### Sampler

The sampler transforms the model logits into the selected next token.

Conceptually, this is similar to what happens in other engines such as vLLM.

Input:

```text id="cwcrqn"
logits
```

Output:

```text id="7ljj6y"
next_token
```

It can apply strategies such as:

* Greedy decoding.
* Temperature.
* Top-k.
* Top-p / nucleus sampling.
* Mirostat.
* Repetition penalty.
* Frequency penalty.
* Presence penalty.

Simplified example:

```text id="73ahjt"
logits -> apply temperature -> filter with top-k/top-p -> choose token
```

---

### Detokenizer / Streaming

The detokenizer converts generated tokens back into text.

Its main responsibilities are:

* Converting token IDs into text.
* Handling partial characters.
* Emitting output progressively.
* Sending chunks over HTTP in streaming mode.
* Printing tokens to the console in CLI mode.

Example:

```text id="e4y1w8"
[token_10, token_42, token_91] -> "Hello world"
```

In streaming mode, the flow would look like this:

```text id="5gm4le"
token -> partial text -> send chunk -> next token -> next chunk
```

---

## Key difference compared to vLLM

The main difference is the design goal.

### vLLM

vLLM is mainly designed for serving many concurrent requests.

It stands out because of:

* Continuous batching.
* PagedAttention.
* Efficient KV cache management in reusable blocks.
* High throughput in multi-user scenarios.
* Large-scale concurrent serving.

Its architecture is designed to receive many simultaneous requests and dynamically batch them to make better use of the GPU.

Conceptually:

```text id="1d9if1"
request A
request B
request C
   |
   v
continuous batching
   |
   v
efficient GPU execution
```

### llama.cpp

`llama.cpp` shines more in portability, local efficiency, and lightweight deployments.

It stands out because it:

* Runs very well on CPU.
* Supports practical quantization.
* Runs on modest machines.
* Is portable.
* Supports Metal, CUDA, Vulkan, and other backends.
* Is easy to use locally.
* Provides `llama-server` for exposing inference over HTTP.

Although it can also serve models through `llama-server`, its historical core was not originally designed as a large-scale multi-tenant serving system like vLLM.

---

## Quick summary

```text id="w9kbcc"
llama.cpp:
  better for local inference, portability, quantization, and lightweight deployments.

vLLM:
  better for concurrent serving, high throughput, and multi-user workloads.
```

In simple terms:

```text id="hmh7a5"
llama.cpp = portable and efficient runtime for running models in many environments.

vLLM = serving engine optimized for many concurrent requests on GPU.
```
