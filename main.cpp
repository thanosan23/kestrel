#include "kestrel/config.hpp"
#include "kestrel/gpt2.hpp"
#include "kestrel/io.hpp"
#include "kestrel/kv_cache.hpp"
#include "kestrel/spec_decode.hpp"
#include "kestrel/tokenizer.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

constexpr int END_OF_TEXT = 50256;

static bool config_by_name(const std::string &name, GPT2Config &out) {
  if (name == "small")
    out = gpt::small;
  else if (name == "medium")
    out = gpt::medium;
  else if (name == "large")
    out = gpt::large;
  else if (name == "xl")
    out = gpt::xl;
  else
    return false;
  return true;
}

static GPT2 load_model(const std::string &path, const std::string &size) {
  GPT2Config config;
  if (!config_by_name(size, config)) {
    std::cerr << "unknown model size: " << size
              << " (use small, medium, large or xl)\n";
    std::exit(1);
  }
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "could not open " << path << '\n';
    std::exit(1);
  }
  return GPT2(IO::read_weights(file), config);
}

static const char *usage =
    "Usage: ./bin/main [options]\n"
    " -p TEXT              prompt\n"
    " -n N                 number of tokens to generate\n"
    " --temp T             temperature (0 = greedy)\n"
    " --top-k K            sample from K most likely tokens\n"
    " --seed S             random seed\n"
    " --weights FILE       model weights (default ./gpt2_weights.kestrel)\n"
    " --model SIZE         small, medium, large or xl (default small)\n"
    " --draft-weights FILE enables speculative decoding with this draft model\n"
    " --draft-model SIZE   draft model size (default small)\n"
    " --spec-k K           draft tokens per round (default 4)\n"
    " --dump FILE          save the prompt's token ids and logits\n"
    " --help               prints usage\n";

int main(int argc, char **argv) {
  using Clock = std::chrono::steady_clock;
  auto seconds_since = [](Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
  };

  std::string prompt = "Hello world";
  int n_new_tokens = 20;
  float temperature = 0.8f;
  int top_k = 40;
  unsigned seed = std::random_device{}();
  std::string dump_path;
  std::string weights_path = "./gpt2_weights.kestrel";
  std::string model_size = "small";
  std::string draft_path;
  std::string draft_size = "small";
  int spec_k = 4;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value after " << arg << '\n';
        std::exit(1);
      }
      return argv[++i];
    };
    if (arg == "-p" || arg == "--prompt")
      prompt = value();
    else if (arg == "-n")
      n_new_tokens = std::stoi(value());
    else if (arg == "--temp")
      temperature = std::stof(value());
    else if (arg == "--top-k")
      top_k = std::stoi(value());
    else if (arg == "--seed")
      seed = std::stoul(value());
    else if (arg == "--dump")
      dump_path = value();
    else if (arg == "--weights")
      weights_path = value();
    else if (arg == "--model")
      model_size = value();
    else if (arg == "--draft-weights")
      draft_path = value();
    else if (arg == "--draft-model")
      draft_size = value();
    else if (arg == "--spec-k")
      spec_k = std::stoi(value());
    else if (arg == "-h" || arg == "--help") {
      std::cout << usage;
      return 0;
    } else {
      std::cerr << "unknown option: " << arg << '\n';
      return 1;
    }
  }
  const bool speculative = !draft_path.empty();
  if (speculative && spec_k < 1) {
    std::cerr << "--spec-k must be at least 1\n";
    return 1;
  }

  auto t0 = Clock::now();
  GPT2 model = load_model(weights_path, model_size);
  double load_s = seconds_since(t0);

  std::optional<GPT2> draft_model;
  if (speculative) {
    t0 = Clock::now();
    draft_model.emplace(load_model(draft_path, draft_size));
    load_s += seconds_since(t0);
  }

  Tokenizer tok;
  if (!tok.load("./gpt2_tokenizer.kestrel")) {
    std::cerr << "could not load ./gpt2_tokenizer.kestrel\n";
    return 1;
  }

  std::vector<int> tokens = tok.encode(prompt);
  if (tokens.empty()) {
    std::cerr << "prompt is empty\n";
    return 1;
  }
  if ((int)tokens.size() >= GPT2Config::n_ctx) {
    std::cerr << "prompt is too long\n";
    return 1;
  }
  const int n_prompt = tokens.size();
  std::cout << prompt << std::flush;

  KVCache cache(model.config());
  auto t1 = Clock::now();
  std::vector<float> logits = model.forward(tokens, cache);
  double prompt_s = seconds_since(t1);

  if (!dump_path.empty()) {
    std::ofstream dump(dump_path, std::ios::binary);
    const int32_t n = tokens.size();
    dump.write(reinterpret_cast<const char *>(&n), sizeof(n));
    dump.write(reinterpret_cast<const char *>(tokens.data()), sizeof(int) * n);
    dump.write(reinterpret_cast<const char *>(logits.data()),
               logits.size() * sizeof(float));
    if (!dump) {
      std::cerr << "could not write " << dump_path << '\n';
      return 1;
    }
  }

  std::mt19937 rng(seed);
  auto t2 = Clock::now();
  SpecStats stats;

  auto emit = [&](int id) {
    if (id == END_OF_TEXT || (int)tokens.size() >= GPT2Config::n_ctx ||
        (int)tokens.size() - n_prompt >= n_new_tokens) {
      return false;
    }
    tokens.push_back(id);
    std::cout << tok.decode({id}) << std::flush;
    return true;
  };

  int last = sample(to_probs(logits.data(), temperature, top_k), rng);
  if (!speculative) {
    while (emit(last)) {
      logits = model.forward({last}, cache);
      last = sample(to_probs(logits.data(), temperature, top_k), rng);
    }
  } else {
    const GPT2 &draft = *draft_model;
    KVCache dcache(draft.config());
    draft.forward(tokens, dcache);

    bool running = emit(last);
    while (running) {
      const int remaining = n_new_tokens - ((int)tokens.size() - n_prompt);
      const int room = GPT2Config::n_ctx - cache.len - 1;
      const int k = std::min({spec_k, remaining, room});
      if (k < 1)
        break;
      std::vector<int> fresh = spec_round(draft, dcache, model, cache, last, k,
                                          temperature, top_k, rng, stats);
      for (int id : fresh) {
        if (!(running = emit(id)))
          break;
      }
      last = fresh.back();
    }
  }
  double gen_s = seconds_since(t2);
  const int generated = tokens.size() - n_prompt;

  std::cout << '\n';
  std::cerr << "\nload: " << load_s << "s\n"
            << "prompt: " << prompt_s << "s (" << n_prompt << " tokens)\n"
            << "generation: " << generated / gen_s << " tokens/sec\n";
  if (speculative && stats.rounds > 0) {
    std::cerr << "speculative: k=" << spec_k << ", acceptance "
              << 100.0 * stats.accepted / stats.drafted << "%, "
              << (double)(stats.accepted + stats.rounds) / stats.rounds
              << " tokens per target pass\n";
  }
  return 0;
}
