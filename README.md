# Kestrel

LLM inference engine in C++

## Features
- Loads GPT-2 weights from custom binary format
- BPE tokenizer (compatible with GPT-2)
- KV Cache
- Threading & loop unrolling
- NEON SIMD
- Quantization
- Speculative Decoding

## Building

The model files `gpt2_weights.kestrel` and `gpt2_tokenizer.kestrel` must be in the root of the repository.
```bash
make
./bin/main

make bench # benchmarking script
```
