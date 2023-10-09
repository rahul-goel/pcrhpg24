#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>

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
};

