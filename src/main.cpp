#include <algorithm>
#include <complex>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include <algorithm.cuh>

int main() {
  // 1. Read the file
  std::ifstream input("test.json");
  std::stringstream ss;
  ss << input.rdbuf();
  // 2. Filter
  auto res = ss.str();
  filtering::filter(res);
}
