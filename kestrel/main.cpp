#include "config.hpp"
#include "gpt2.hpp"
#include "io.hpp"
#include "kv_cache.hpp"
#include "tokenizer.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

static int argmax(const std::vector<float> &v) {
  return std::max_element(v.begin(), v.end()) - v.begin();
}

static int sample(const std::vector<float> &logits, float temperature,
                  int top_k, std::mt19937 &rng) {
  if (temperature <= 0.0f)
    return argmax(logits);

  std::vector<std::pair<float, int>> candidates;
  candidates.reserve(logits.size());
  for (int i = 0; i < (int)logits.size(); i++) {
    candidates.push_back({logits[i], i});
  }

  int k =
      (top_k > 0 && top_k < (int)candidates.size() ? top_k
                                                   : (int)candidates.size());
  std::partial_sort(
      candidates.begin(), candidates.begin() + k, candidates.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });
  candidates.resize(k);

  const float max_logit = candidates[0].first;
  std::vector<float> probs(k);
  for (int i = 0; i < k; i++) {
    probs[i] = std::exp((candidates[i].first - max_logit) / temperature);
  }

  std::discrete_distribution<int> pick(probs.begin(), probs.end());
  return candidates[pick(rng)].second;
}

static const char *usage =
    "Usage: ./bin/main [options]\n"
    " -p TEXT      prompt\n"
    " -n N         number of tokens to generate\n"
    " --temp T     temperature\n"
    " --top-k K    sample from K most likely tokens\n"
    " --dump FILE  save the prompt's token ids and logits"
    " --help       prints usage\n";

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
    else if (arg == "--seed") {
      seed = std::stoul(value());
    } else if (arg == "--dump") {
      dump_path = value();

    } else if (arg == "-h" || arg == "--help") {
      std::cout << usage;
      return 0;
    } else {
      std::cerr << "unknown option: " << arg << '\n';
      return 1;
    }
  }

  std::ifstream weight_file("./gpt2_weights.kestrel");
  if (!weight_file) {
    std::cerr << "could not open ./gpt2_weights.kestrel\n";
    return 1;
  }

  auto t0 = Clock::now();
  GPT2 model(IO::read_weights(weight_file), gpt::small);
  double load_s = seconds_since(t0);

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
  std::cout << prompt << std::flush;

  KVCache cache(model.config());
  auto t1 = Clock::now();
  std::vector<float> logits;
  for (int pos = 0; pos < (int)tokens.size(); pos++) {
    logits = model.step(tokens[pos], pos, cache);
  }
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
  constexpr int END_OF_TEXT = 50256;
  auto t2 = Clock::now();
  int generated = 0;
  for (int i = 0; i < n_new_tokens; i++) {
    int next = sample(logits, temperature, top_k, rng);
    if (next == END_OF_TEXT)
      break;
    if ((int)tokens.size() >= GPT2Config::n_ctx)
      break;
    tokens.push_back(next);
    std::cout << tok.decode({next}) << std::flush;
    logits = model.step(next, tokens.size() - 1, cache);
    generated++;
  }
  double gen_s = seconds_since(t2);

  double tok_per_s = generated / gen_s;

  std::cout << '\n';

  std::cerr << "\nload: " << load_s << "s\n"
            << "prompt: " << prompt_s << "s (" << tokens.size() - generated
            << " tokens)\n"
            << "generation: " << tok_per_s << " tokens/sec\n";

  return 0;
}
