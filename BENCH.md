## Run 1

### Settings
Prompt: "Hello world"
20 new tokens

### Optimizations
Compiler optimizations: "-O3 -mcpu=native"

### Timings
Generation: ~5 tok/s

## Run 2

### Settings
Prompt: "Hello world"
20 new tokens

### Optimizations
Compiler optimizations: "-O3 -mcpu=native"
KV Cache

### Timings
Generation: 16.3 tok/s


## Run 3
### Settings
Prompt: "Hello world"
20 new tokens

### Optimizations
Compiler optimizations: "-O3 -mcpu=native"
KV Cache
Multiple accumulators

### Timings 
Generation: 30.7 tok/s

## Run 4
### Settings
Prompt: "Hello world"
20 new tokens

### Optimizations
Compiler optimizations: "-O3 -mcpu=native"
KV Cache
Multiple accumulators
Threads (4 threads)

### Timings
Generation: 34.4 tok/s

## Run 5
### Settings
Prompt: "Hello world"
20 new tokens

### Optimizations
Compiler optimizations: "-O3 -mcpu=native"
KV Cache
Multiple accumulators
Threads (8 threads)
Thread pool

### Timings
Generation: 55 tok/s

## Run 6
### Settings
Prompt: "Hello world"
20 new tokens

### Optimizations
Compiler optimizations: "-O3 -mcpu=native"
KV Cache
Multiple accumulators
Threads (8 threads)
Thread pool
NEON SIMD

### Timings
Generation: ~50 tok/s

## Run 7
### Settings
Prompt: "Hello world"
20 new tokens

### Optimizations
Compiler optimizations: "-O3 -mcpu=native"
KV Cache
Multiple accumulators
Threads (8 threads)
Thread pool
NEON SIMD
INT weight-only quantization (per-row)

### Timings
Generation: ~88.9 tok/s

## Run 8
### Settings
Prompt: "Hello world"
20 new tokens

### Optimizations
Compiler optimizations: "-O3 -mcpu=native"
KV Cache
Multiple accumulators
Threads (8 threads)
Thread pool
NEON SIMD
INT weight-only quantization (per-row)
Batched Prefill

### Timings
Generation: ~90.51 tok/s
Brought a 384-token prompt from 5.48s to 1.27s (4.3x)

