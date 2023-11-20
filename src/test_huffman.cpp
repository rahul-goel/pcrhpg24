#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <future>
#include <functional>

#include "huffman.h"

using namespace std;

static std::mutex m;

template<typename T, typename udtype>
void test_huffman(string T_str, string udtype_str, int num_points) {
  vector<T> data;
  for (int i = 0; i < num_points; ++i) {
    data.push_back(rand() % 10000);
  }

  // Huffman<T> hfmn;
  // hfmn.calculate_frequencies(data);
  // hfmn.constraint_table_size(4096);
  // hfmn.generate_huffman_tree();
  // hfmn.create_dictionary();
  // hfmn.clear_huffman_tree();

  // auto collapsed_dict = hfmn.template get_collapsed_dictionary<udtype>();
  // vector<pair<T,int>> decoder_table = hfmn.get_gpu_huffman_table();

  // vector<udtype> encoded = hfmn.template compress_udtype_subarray_fast<udtype, typename vector<T>::iterator>(data.begin(), data.end(), collapsed_dict);
  // vector<T> decoded(data.size());
  // hfmn.template decompress_udtype_subarray_fast<udtype, typename vector<T>::iterator>(decoded.begin(), decoded.end(), encoded, decoder_table);

  int table_size = 4096;
  Huffman<T> hfmn;
  hfmn.calculate_frequencies(data);
  hfmn.generate_huffman_tree();
  auto dict = hfmn.template create_dictionary_pjn<udtype>(table_size);
  auto table = hfmn.template get_gpu_huffman_table_pjn<udtype>(dict, table_size);
  hfmn.clear_huffman_tree();

  // auto encoded = hfmn.template compress_udtype_subarray_fast_pjn_idea<udtype, it_type, int32_t>(data.begin(), data.end(), dict);
  auto encoded = hfmn.template compress_udtype_subarray_fast_pjn_idea<
    udtype, typename vector<T>::iterator, T>(data.begin(), data.end(),
                                             dict, table_size);

  vector<T> decoded(data.size());
  hfmn.template decompress_udtype_subarray_fast_pjn_idea<
      udtype, typename vector<T>::iterator, T>(
      decoded.begin(), decoded.end(), encoded.first, encoded.second, table);

  bool equal = (decoded == data);
  {
    std::lock_guard<std::mutex> lock(m);
    cout << "Data type: " << T_str << endl;
    cout << "Encoding type: " << udtype_str << endl;
    cout << "Are they equal???? : " << equal << endl;
  }
  assert(equal);
}

// lets do multithreaded huffman on the cpu
template<typename T, typename udtype>
void multithreaded_huffman(string T_str, string udtype_str, int num_points) {
  vector<T> data;
  for (int i = 0; i < num_points; ++i) {
    data.push_back(rand() % 10);
  }
  cout << "Created array" << endl;

  Huffman<T> hfmn;
  hfmn.calculate_frequencies(data);
  hfmn.constraint_table_size(4096);
  hfmn.generate_huffman_tree();
  hfmn.create_dictionary();
  hfmn.clear_huffman_tree();

  auto collapsed_dict = hfmn.template get_collapsed_dictionary<udtype>();
  vector<pair<T,int>> decoder_table = hfmn.get_gpu_huffman_table();
  cout << "Created huffman" << endl;

  const int num_threads = 4;
  int done = 0;
  vector<vector<udtype>> encoded_values;
  vector<future<vector<udtype>>> e_futures;
  vector<vector<T>> decoded_values;
  vector<future<void>> d_futures;
  for (int tid = 0; tid < num_threads; ++tid) {
    int this_thread_num_points = num_points / num_threads + (tid < num_points % num_threads);
    decoded_values.push_back(vector<T>(this_thread_num_points));
  }

  // encode
  for (int tid = 0; tid < num_threads; ++tid) {
    int this_thread_num_points = num_points / num_threads + (tid < num_points % num_threads);
    auto start = data.begin() + done;
    auto end = data.begin() + done + this_thread_num_points;
    done += this_thread_num_points;

    // pass everything by reference but the iterators by value since they go out of scope
    future<vector<udtype>> f =
        async(launch::async, [&collapsed_dict, start, end, &hfmn] {
          return hfmn.template compress_udtype_subarray_fast<
              udtype, typename vector<T>::iterator>(start, end, collapsed_dict);
        });
    e_futures.push_back(std::move(f));
  }

  for (int tid = 0; tid < num_threads; ++tid) {
    encoded_values.push_back(e_futures[tid].get());
  }
  cout << "encoded" << endl;

  // decode
  done = 0;
  for (int tid = 0; tid < num_threads; ++tid) {
    int this_thread_num_points = num_points / num_threads + (tid < num_points % num_threads);

    auto &encoded = encoded_values[tid];
    auto start = decoded_values[tid].begin();
    auto end = decoded_values[tid].end();
    // pass everything by reference but the iterators by value since they go out of scope
    future<void> f =
        async(launch::async, [&decoder_table, &encoded, start, end, &hfmn] {
          return hfmn.template decompress_udtype_subarray_fast<
              udtype, typename vector<T>::iterator>(start, end, encoded,
                                                    decoder_table);
        });
    d_futures.push_back(std::move(f));
  }
  for (int tid = 0; tid < num_threads; ++tid) {
    d_futures[tid].get();
  }

  vector<T> all_decoded_data;
  for (int tid = 0; tid < num_threads; ++tid) all_decoded_data.insert(all_decoded_data.end(), decoded_values[tid].begin(), decoded_values[tid].end());
  cout << "decoded" << endl;

  assert(all_decoded_data == data);
  cout << "tested" << endl;
}

int main(int argc, char *argv[]) {
  assert(argc == 2);
  int num_points = atoi(argv[1]);
  assert(num_points > 0);

  // multithreaded_huffman<int32_t, uint32_t>("int32_t", "uint32_t", num_points);

  vector<thread> threads;
  threads.push_back(std::move(thread(test_huffman<char, uint32_t>, "char", "uint32_t", num_points)));
  threads.push_back(std::move(thread(test_huffman<unsigned char, uint32_t>, "unsigned char", "uint32_t", num_points)));
  threads.push_back(std::move(thread(test_huffman<short, uint32_t>, "short", "uint32_t", num_points)));
  threads.push_back(std::move(thread(test_huffman<int8_t, uint32_t>, "int8_t", "uint32_t", num_points)));
  threads.push_back(std::move(thread(test_huffman<int16_t, uint32_t>, "int16_t", "uint32_t", num_points)));
  threads.push_back(std::move(thread(test_huffman<int32_t, uint32_t>, "int32_t", "uint32_t", num_points)));
  threads.push_back(std::move(thread(test_huffman<int64_t, uint32_t>, "int64_t", "uint32_t", num_points)));
  threads.push_back(std::move(thread(test_huffman<uint8_t, uint32_t>, "uint8_t", "uint32_t", num_points)));
  threads.push_back(std::move(thread(test_huffman<uint16_t, uint32_t>, "uint16_t", "uint32_t", num_points)));
  threads.push_back(std::move(thread(test_huffman<uint32_t, uint32_t>, "uint32_t", "uint32_t", num_points)));
  threads.push_back(std::move(thread(test_huffman<uint64_t, uint32_t>, "uint64_t", "uint32_t", num_points)));

  for (thread &t : threads) t.join();
  return 0;
}
