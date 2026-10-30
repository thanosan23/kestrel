#include "io.hpp"
#include "tensor.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <string>

int main() {
  std::ifstream f("gpt2_weights.kestrel");
  if (!f) {
    std::cerr << "can't open file" << "\n";
    return 1;
  }
  std::map<std::string, Tensor> weights = IO::read_weights(f);
  std::cout << weights.size() << '\n';
}
