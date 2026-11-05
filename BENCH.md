## Run 1

### Settings
Prompt: "Hello world"
20 new tokens

### Optimizations
Compiler optimizations: "-O3 -mcpu=native"

### Timings
Average: 3.99s   Fastest: 3.85s

## Run 2

### Settings
Prompt: "Hello world"
20 new tokens

### Optimizations
Compiler optimizations: "-O3 -mcpu=native"
KV Cache

### Timings
Average: 1.76s   Fastest: 1.61s


## Run 3
### Settings
Prompt: "Hello world"
20 new tokens

### Optimizations
Compiler optimizations: "-O3 -mcpu=native"
KV Cache
Multiple accumulators

### Timings 
Average: 1.00s   Fastest: 0.92s

## Run 4
### Settings
Prompt: "Hello world"
20 new tokens

### Optimizations
Compiler optimizations: "-O3 -mcpu=native"
KV Cache
Multiple accumulators
Threads (8 threads)

### Timings
Average: 0.80s   Fastest: 0.72s

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
Average: 0.75s   Fastest: 0.65s

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
Average: 0.86s   Fastest: 0.73s
