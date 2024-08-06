#include <cstdint>
#include <utility>
#include <vector>
#include <numeric>
#include <climits>
#include <algorithm>
#include <execution>

namespace mymorton {
using mc_t = std::pair<uint32_t, uint64_t>;

inline mc_t get_morton_code_3D(uint32_t X, uint32_t Y, uint32_t Z) {
  mc_t code(0, 0);

  // 21 * 3 = 63
  for (int i = 0; i < 21; ++i) {
    code.second = code.second | ((uint64_t) X >> i & 1) << (3 * i + 0);
    code.second = code.second | ((uint64_t) Y >> i & 1) << (3 * i + 1);
    code.second = code.second | ((uint64_t) Z >> i & 1) << (3 * i + 2);
  }

  // X for the MSB of the 64-bit number
  code.second = code.second | ((uint64_t) X >> 21 & 1) << 63;

  // YZ for the starting LSB of the 32-bit number
  code.first = code.first | ((uint64_t) Y >> 21 & 1) << 0;
  code.first = code.first | ((uint64_t) Z >> 21 & 1) << 1;

  // fill in the remaining ones in the 32-bit number
  for (int i = 22; i < 32; ++i) {
    code.first = code.first | ((uint64_t) X >> i & 1) << (3 * (i - 21) + 2);
    code.first = code.first | ((uint64_t) Y >> i & 1) << (3 * (i - 21) + 0);
    code.first = code.first | ((uint64_t) Z >> i & 1) << (3 * (i - 21) + 1);
  }

  return code;
}

inline std::vector<unsigned int> get_morton_order(std::vector<int32_t> &X, std::vector<int32_t> &Y, std::vector<int32_t> &Z) {
  size_t num_points = X.size();
  std::vector<mc_t> morton_codes(num_points);

  std::vector<unsigned int> indices(num_points);
  std::iota(indices.begin(), indices.end(), 0);
  std::for_each(std::execution::par_unseq, indices.cbegin(), indices.cend(), [&](const int &index) {
    uint32_t shifted_x = (uint32_t) ((int64_t) X[index] - (int64_t) INT_MIN);
    uint32_t shifted_y = (uint32_t) ((int64_t) Y[index] - (int64_t) INT_MIN);
    uint32_t shifted_z = (uint32_t) ((int64_t) Z[index] - (int64_t) INT_MIN);
    morton_codes[index] = get_morton_code_3D(shifted_x, shifted_y, shifted_z);
  });

  std::vector<unsigned int> order(num_points);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(std::execution::par_unseq, order.begin(), order.end(), [&morton_codes](const int64_t &a, const int64_t &b) {
    return morton_codes[a] < morton_codes[b];
  });
  return order;
}
}
