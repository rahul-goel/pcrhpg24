#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iostream>
#include <bitset>

using namespace std;

template<typename T>
struct Huffman {
  struct HuffmanNode {
    T value;
    unsigned int freq;
    HuffmanNode *left;
    HuffmanNode *right;

    HuffmanNode(unsigned int freq) {
      this->value = 0;
      this->freq = freq;
      this->left = nullptr;
      this->right = nullptr;
    }

    HuffmanNode(T value, unsigned int freq) {
      this->value = value;
      this->freq = freq;
      this->left = nullptr;
      this->right = nullptr;
    }
  };

  // my data
  unordered_map<T,unsigned int> frequencies;
  HuffmanNode *head;
  unordered_map<T,vector<bool>> dictionary;
  unsigned int num_nodes = 0;

  // methods
  void calculate_frequencies(vector<T> &data) {
    frequencies.clear();
    for (T &item : data) {
      if (frequencies.find(item) == frequencies.end()) {
        frequencies[item] = 1;
      } else {
        frequencies[item] += 1;
      }
    }
  }

  static bool sort_cmp(const HuffmanNode *a, const HuffmanNode *b) {
    return a->freq > b->freq;
  }

  HuffmanNode *merge(HuffmanNode *a, HuffmanNode *b) {
    HuffmanNode *parent = new HuffmanNode(a->freq + b->freq);
    ++num_nodes;
    parent->value = -1;
    parent->left = b;
    parent->right = a;
    return parent;
  }

  void generate_huffman_tree() {
    vector<HuffmanNode*> all_nodes;
    for (auto &[item, freq] : frequencies) {
      all_nodes.push_back(new HuffmanNode(item, freq));
      ++num_nodes;
    }
    sort(all_nodes.begin(), all_nodes.end(), sort_cmp);

    while (all_nodes.size() > 1) {
      HuffmanNode *a, *b;
      a = all_nodes.back();
      all_nodes.pop_back();
      b = all_nodes.back();
      all_nodes.pop_back();

      HuffmanNode *parent = merge(a, b);
      all_nodes.push_back(parent);
      sort(all_nodes.begin(), all_nodes.end(), sort_cmp);
    }

    head = all_nodes.back();
  }

  vector<pair<T,unsigned int>> get_sorted_frequencies() {
    vector<pair<T, unsigned int>> ret;
    for (auto &[key, val] : frequencies) ret.push_back({key, val});
    sort(ret.begin(), ret.end(), [] (const pair<T, unsigned int> &a, const pair<T, unsigned int> &b) {
      return a.second > b.second; // should return false for equality
    });
    return ret;
  }

  void create_dictionary(HuffmanNode *cur, vector<bool> &cur_code, unordered_map<T,vector<bool>> &dict) {
    bool is_leaf = true;
    if (cur->left) {
      is_leaf = false;
      cur_code.push_back(0);
      create_dictionary(cur->left, cur_code, dict);
      cur_code.pop_back();
    }
    if (cur->right) {
      is_leaf = false;
      cur_code.push_back(1);
      create_dictionary(cur->right, cur_code, dict);
      cur_code.pop_back();
    }
    if (is_leaf) {
      vector<bool> cur_code_copy = cur_code;
      dict[cur->value] = cur_code_copy;
    }
  }

  void create_dictionary() {
    vector<bool> cur_code;
    dictionary.clear();
    HuffmanNode *cur = head;
    create_dictionary(cur, cur_code, dictionary);
  }

  template<typename udtype>
  unordered_map<T, pair<udtype,int>> get_collapsed_dictionary() {
    int max_cw_size = get_max_codeword_size();
    unordered_map<T, pair<udtype,int>> collapsed_dictionary;
    for (auto &[symbol, cw_vector] : dictionary) {
      udtype cw = 0;
      for (bool bit : cw_vector) cw = (cw << 1) | bit;
      collapsed_dictionary[symbol] = {cw, cw_vector.size()};
    }
    return collapsed_dictionary;
  }

  vector<bool> compress(vector<T> &data) {
    vector<bool> bitstream;
    for (T &item : data) {
      auto &vec = dictionary[item];
      bitstream.insert(bitstream.end(), vec.begin(), vec.end());
    }
    return bitstream;
  }

  vector<T> decompress(vector<bool> &bitstream) {
    vector<T> data;
    HuffmanNode *cur = head;
    for (bool bit : bitstream) {
      bool is_child = !cur->left and !cur->right;
      if (is_child) {
        data.push_back(cur->value);
        cur = head;
      }

      if (!bit) {
        cur = cur->left;
      } else if (bit) {
        cur = cur->right;
      }
    }
    data.push_back(cur->value);
    return data;
  }

  template<typename udtype, typename Iterator>
  vector<udtype> compress_udtype_subarray_fast(Iterator begin, Iterator end, unordered_map<T, pair<udtype,int>> &collapsed_dictionary) {
    vector<udtype> vec;                                         // final vector that stores the values in the specified data type
    const int udtype_num_bits = 8 * sizeof(udtype);             // number of bits in the data type
    udtype chunk = 0;                                           // element that reprsents the current value to be pushed into the final vector
    int chunk_rem_bits = udtype_num_bits;                       // number of unoccupied bits (from the LSB side) in the cur_value

    udtype mask;
    for (Iterator it = begin; it != end; ++it) {
      pair<udtype,int> p = collapsed_dictionary[*it];
      udtype cw = p.first;
      int cw_tot_bits = p.second;
      int cw_rem_bits = cw_tot_bits;

      while (cw_rem_bits) {
        int min_bits = min(chunk_rem_bits, cw_rem_bits);
        mask = (((udtype) 1 << cw_rem_bits) - 1) - (((udtype) 1 << (cw_rem_bits - min_bits)) - 1);
        chunk = chunk | (((mask & cw) >> (cw_rem_bits - min_bits)) << (chunk_rem_bits - min_bits));

        cw_rem_bits -= min_bits;
        chunk_rem_bits -= min_bits;

        if (chunk_rem_bits == 0) {
          vec.push_back(chunk);
          chunk = 0;
          chunk_rem_bits = udtype_num_bits;
        }
      }
    }

    if (chunk_rem_bits < udtype_num_bits) {
      vec.push_back(chunk);
    }

    return vec;
  }

  template<typename udtype>
  vector<udtype> compress_udtype(vector<T> &data) {
    // final vector that stores the values in the specified data type
    vector<udtype> vec;

    // number of bits in the data type
    const int udtype_num_bits = 8 * sizeof(udtype);

    // element that reprsents the current value to be pushed into the final vector
    udtype cur_value = 0;
    // number of unoccupied bits (from the LSB side) in the cur_value
    int unoccupied_bits = udtype_num_bits;

    for (T &item : data) {
      auto huffman_code = dictionary[item];
      reverse(huffman_code.begin(), huffman_code.end());

      while (huffman_code.size()) {
        // if out of bits
        if (unoccupied_bits == 0) {
          vec.push_back(cur_value);
          cur_value = 0;
          unoccupied_bits = udtype_num_bits;
        }

        cur_value = (cur_value << 1) | (udtype) huffman_code.back();
        unoccupied_bits -= 1;
        huffman_code.pop_back();
      }
    }
    // push in the final value
    cur_value = cur_value << unoccupied_bits;
    vec.push_back(cur_value);

    return vec;
  }

  /*
   * Compress using Markus' idea. If there is a value that is non-frequent. Then
   * just store it separately.
   * If first bit is 0 -> it means that there is a separate storage
   * If first bit is 1 -> then do Huffman stuff.
   */
  template<typename udtype, typename delta_dtype>
  pair<vector<udtype>,vector<delta_dtype>> compress_udtype_markus_idea(vector<T> &data) {
    // final vector that stores the values in the specified data type
    vector<udtype> vec;

    // number of bits in the data type
    const int udtype_num_bits = 8 * sizeof(udtype);

    // element that reprsents the current value to be pushed into the final vector
    udtype cur_value = 0;
    // number of unoccupied bits (from the LSB side) in the cur_value
    int unoccupied_bits = udtype_num_bits;

    vector<delta_dtype> separate_data;
    int64_t delta_dtype_max = (1ll << (sizeof(delta_dtype) * 8 - 1)) - 1, delta_dtype_min = -(1ll << (sizeof(delta_dtype) * 8 - 1));

    for (T &item : data) {
      if (dictionary.find(item) == dictionary.end()) {
        // put a 0
        if (unoccupied_bits == 0) {
          vec.push_back(cur_value);
          cur_value = 0;
          unoccupied_bits = udtype_num_bits;
        }
        cur_value = (cur_value << 1) | (udtype) 0;
        unoccupied_bits -= 1;

        // assert that the value fits within delta_dtype
        assert(delta_dtype_min <= (int64_t) item);
        assert((int64_t) item <= delta_dtype_max);
        separate_data.push_back(item);
      } else {
        // put a 1 at the beginning of the huffman code
        auto huffman_code = dictionary[item];
        reverse(huffman_code.begin(), huffman_code.end());
        huffman_code.push_back(1);

        while (huffman_code.size()) {
          // if out of bits
          if (unoccupied_bits == 0) {
            vec.push_back(cur_value);
            cur_value = 0;
            unoccupied_bits = udtype_num_bits;
          }

          cur_value = (cur_value << 1) | (udtype) huffman_code.back();
          unoccupied_bits -= 1;
          huffman_code.pop_back();
        }
      }
    }
    // push in the final value
    cur_value = cur_value << unoccupied_bits;
    vec.push_back(cur_value);

    return {vec, separate_data};
  }

  template<typename udtype>
  vector<T> decompress_udtype(vector<udtype> &bitstream, unsigned int num_elements) {
    // final data vector
    vector<T> data;
    // number of bits in the data type
    const int udtype_num_bits = 8 * sizeof(udtype);

    HuffmanNode *cur = head;

    for (udtype &word : bitstream) {
      for (int b = udtype_num_bits - 1; b >= 0; --b) {
        bool is_child = !cur->left and !cur->right;
        if (is_child) {
          data.push_back(cur->value);
          if (data.size() == num_elements) break;
          cur = head;
        }

        bool bit = (word >> b) & (udtype) 1;
        if (!bit) {
          cur = cur->left;
        } else if (bit) {
          cur = cur->right;
        }
      }
      if (data.size() == num_elements) break;
    }

    // last bit read finishes the last value
    if (data.size() < num_elements) {
      data.push_back(cur->value);
    }

    return data;
  }

  /*
   * Compress using Markus' idea. If there is a value that is non-frequent. Then
   * just store it separately.
   * If first bit is 0 -> it means that there is a separate storage
   * If first bit is 1 -> then do Huffman stuff.
   */
  template<typename udtype, typename delta_dtype>
  vector<T> decompress_udtype_markus_idea(vector<udtype> &bitstream, vector<delta_dtype> &separate_data, unsigned int num_elements) {
    // final data vector
    vector<T> data;
    // number of bits in the data type
    const int udtype_num_bits = 8 * sizeof(udtype);

    HuffmanNode *cur = head;

    int sep_ptr = 0;
    bool new_code = true;

    for (udtype &word : bitstream) {
      for (int b = udtype_num_bits - 1; b >= 0; --b) {
        bool is_child = !cur->left and !cur->right;
        if (is_child) {
          new_code = true;
          data.push_back(cur->value);
          if (data.size() == num_elements) break;
          cur = head;
        }

        bool bit = (word >> b) & (udtype) 1;
        if (new_code and bit == 0) {
          // separate data case
          data.push_back(separate_data[sep_ptr++]);
          new_code = true;
          if (data.size() == num_elements) break;
        } else if (new_code) {
          // huffman decoding next bit onwards
          new_code = false;
        } else {
          // huffman traversal
          if (!bit) {
            cur = cur->left;
            new_code = false;
          } else if (bit) {
            cur = cur->right;
            new_code = false;
          }
        }
      }
      if (data.size() == num_elements) break;
    }

    // last bit read finishes the last value
    if (data.size() < num_elements) {
      data.push_back(cur->value);
    }

    return data;
  }

  /*
   * Compress using Markus' idea. If there is a value that is non-frequent. Then
   * just store it separately.
   * If first bit is 0 -> it means that there is a separate storage
   * If first bit is 1 -> then do Huffman stuff.
   */
  template<typename udtype, typename delta_dtype>
  vector<T> decompress_udtype_markus_idea_using_table(vector<pair<T,int>> &table, vector<udtype> &bitstream, vector<delta_dtype> &separate_data, unsigned int num_elements) {
    // final data vector
    vector<T> data;
    // number of bits in the data type
    const int udtype_num_bits = 8 * sizeof(udtype);

    int table_size = table.size();
    int max_cw_size = -1;
    while (table_size) max_cw_size++, table_size >>= 1;

    udtype window = 0;
    udtype mask = ~((udtype) 0) << (udtype_num_bits - max_cw_size); // 1111 0000 0000 0000
    int shift = udtype_num_bits - max_cw_size;
    udtype single_mask = ((udtype) 1) << max_cw_size - 1;
    int single_shift = udtype_num_bits - 1;

    int read_bits = udtype_num_bits;
    int bitstream_ptr = 0;
    int sep_ptr = 0;
    bool new_code = true;

    while (data.size() < num_elements) {
      int used_bits = 0;
      if (new_code) {
        if ((window & single_mask) >> single_shift) {
          // hufmann
        } else {
          // separate data
          data.push_back(separate_data[sep_ptr++]);
          new_code = true;
        }
        used_bits = 1;
      } else {
        udtype idx = (window & mask) >> shift;
        auto &[val, cw_size] = table[idx];
        data.push_back(val);
        new_code = true;
        used_bits = cw_size;
      }

      // discard first "used_bits" and load next
      read_bits += used_bits;
      window <<= used_bits;
    }

    HuffmanNode *cur = head;


    for (udtype &word : bitstream) {
      for (int b = udtype_num_bits - 1; b >= 0; --b) {
        bool is_child = !cur->left and !cur->right;
        if (is_child) {
          new_code = true;
          data.push_back(cur->value);
          if (data.size() == num_elements) break;
          cur = head;
        }

        bool bit = (word >> b) & (udtype) 1;
        if (new_code and bit == 0) {
          // separate data case
          data.push_back(separate_data[sep_ptr++]);
          new_code = true;
          if (data.size() == num_elements) break;
        } else if (new_code) {
          // huffman decoding next bit onwards
          new_code = false;
        } else {
          // huffman traversal
          if (!bit) {
            cur = cur->left;
            new_code = false;
          } else if (bit) {
            cur = cur->right;
            new_code = false;
          }
        }
      }
      if (data.size() == num_elements) break;
    }

    // last bit read finishes the last value
    if (data.size() < num_elements) {
      data.push_back(cur->value);
    }

    return data;
  }

  template<typename udtype>
  vector<bool> get_bools_from_bitstream(vector<udtype> &bitstream) {
    vector<bool> data;
    const int udtype_num_bits = 8 * sizeof(udtype);
    for (udtype &word : bitstream) {
      for (int b = udtype_num_bits - 1; b >= 0; --b) {
        bool bit = (word >> b) & (udtype) 1;
        data.push_back(bit);
      }
    }
    return data;
  }

  int get_max_codeword_size() {
    int ret = 0;
    for (auto &[val, cw] : dictionary) {
      ret = max(ret, (int) cw.size());
    }
    return ret;
  }

  vector<pair<T,int>> get_gpu_huffman_table() {
    int max_cw_size = get_max_codeword_size();

    vector<pair<T,int>> table(1 << max_cw_size, {-1, -1});
    for (auto [val, cw] : dictionary) {
      int num_bits = cw.size();
      int rem_bits = max_cw_size - num_bits;
      int cw_as_number = 0;
      while (cw.size()) {
        cw_as_number <<= 1;
        cw_as_number = cw_as_number | cw.back();
        cw.pop_back();
      }
      cw_as_number = cw_as_number << rem_bits;
      table[cw_as_number] = {val, num_bits};
    }

    for (int i = 1; i < table.size(); ++i) {
      if (table[i].second == -1) {
        table[i] = table[i - 1];
      }
    }

    return table;
  }

  void print_gpu_huffman_table() {
    auto table = get_gpu_huffman_table();

    for (int i = 0; i < table.size(); ++i) {
      cout << bitset<16>(i) << ": " << table[i].first << " " << table[i].second << endl;
    }
  }
};

