#include "tokenizer.hpp"
#include "io.hpp"

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <ios>
#include <string>
#include <vector>

static uint64_t pair_key(int a, int b) {
  return (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
}

bool Tokenizer::load(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return false;
  char magic[4];
  f.read(magic, 4);
  if (std::memcmp(magic, "KTOK", 4) != 0)
    return false;
  if (IO::read_u32(f) != 1)
    return false;

  id_to_bytes.resize(IO::read_u32(f));
  for (auto &tok : id_to_bytes) {
    tok.resize(IO::read_u32(f));
    f.read(tok.data(), tok.size());
  }

  uint32_t n_merges = IO::read_u32(f);
  merges.reserve(n_merges);
  for (uint32_t r = 0; r < n_merges; r++) {
    int a = IO::read_u32(f), b = IO::read_u32(f), m = IO::read_u32(f);
    merges[pair_key(a, b)] = {int(r), m};
  }

  for (auto &id : byte_to_id)
    id = IO::read_u32(f);

  return bool(f);
}

std::vector<int> Tokenizer::bpe(const std::string &chunk) const {
  std::vector<int> ids;
  for (unsigned char c : chunk)
    ids.push_back(byte_to_id[c]);
  while (ids.size() > 1) {
    int best_rank = INT_MAX, best_id = -1;
    size_t best_i = 0;
    for (size_t i = 0; i + 1 < ids.size(); i++) {
      auto it = merges.find(pair_key(ids[i], ids[i + 1]));
      if (it != merges.end() && it->second.rank < best_rank) {
        best_rank = it->second.rank;
        best_id = it->second.merged_id;
        best_i = i;
      }
    }
    if (best_id < 0)
      break;

    int a = ids[best_i], b = ids[best_i + 1];
    std::vector<int> out;
    for (size_t i = 0; i < ids.size();) {
      if (i + 1 < ids.size() && ids[i] == a && ids[i + 1] == b) {
        out.push_back(best_id);
        i += 2;
      } else {
        out.push_back(ids[i++]);
      }
    }
    ids.swap(out);
  }
  return ids;
}

static bool is_space(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
         c == '\f';
}

static int char_class(unsigned char c) {
  // 0 for space, 1 for letter, 2 for digits, 3 for other
  if (is_space(c))
    return 0;
  if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80)
    return 1;
  if (c >= '0' && c <= '9')
    return 2;
  return 3;
}

static std::vector<std::string> pretokenize(const std::string &s) {
  std::vector<std::string> out;
  size_t i = 0, n = s.size();
  while (i < n) {
    if (s[i] == '\'') {
      static const char *cs[] = {"re", "ve", "ll", "s", "t", "m", "d"};
      bool hit = false;
      for (const char *c : cs) {
        size_t len = std::strlen(c);
        if (s.compare(i + 1, len, c) == 0) {
          out.push_back(s.substr(i, len + 1));
          i += len + 1;
          hit = true;
          break;
        }
      }
      if (hit)
        continue;
    }

    size_t j = i;
    if (s[j] == ' ' && j + 1 < n && !is_space(s[j + 1]))
      j++;
    int cls = char_class(s[j]);
    if (cls != 0) {
      size_t e = j + 1;
      while (e < n && char_class(s[e]) == cls)
        e++;
      out.push_back(s.substr(i, e - i));
      i = e;
      continue;
    }

    size_t e = i;
    while (e < n && is_space(s[e]))
      e++;
    if (e < n && e - i > 1)
      e--;
    out.push_back(s.substr(i, e - i));
    i = e;
  }
  return out;
}

std::vector<int> Tokenizer::encode(const std::string &text) const {
  std::vector<int> result;
  for (const auto &chunk : pretokenize(text)) {
    auto ids = bpe(chunk);
    result.insert(result.end(), ids.begin(), ids.end());
  }
  return result;
}

std::string Tokenizer::decode(const std::vector<int> &ids) const {
  std::string out;
  for (int id : ids)
    out += id_to_bytes[id];
  return out;
}
