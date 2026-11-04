#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

struct Tokenizer {
  struct Merge {
    int rank;
    int merged_id;
  };
  std::vector<std::string> id_to_bytes;
  std::unordered_map<uint64_t, Merge> merges;
  std::array<int, 256> byte_to_id{};

  bool load(const std::string &path);
  std::vector<int> encode(const std::string &text) const;
  std::string decode(const std::vector<int> &ids) const;

private:
  std::vector<int> bpe(const std::string &chunk) const;
};
