#include "morton3D_LUTs.h"
#include "unsuck.hpp"
#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>
#include <iostream>
#include <limits>
#include <cassert>
#include <thread>
#include <bitset>
#include <map>
#include <set>
#include <unordered_set>
#include <queue>
#include <chrono>
#include "morton.h"
#include "mymorton.h"
#include "huffman.h"

#define TARGET_START_X    0
#define TARGET_START_Y    TARGET_START_X + 4
#define TARGET_START_Z    TARGET_START_Y + 4
#define TARGET_START_RGBA TARGET_START_Z + 4
#define TARGET_SZ         4 * 4

#define NUM_THREADS 1

string LASFILE = "/home/rg/lidar_data/morro_bay.las";
// string LASFILE = "/home/rg/lidar_data/tree.las";
// string LASFILE = "/home/rg/lidar_data/points.las";
// const int64_t BATCH_SIZE = NUM_THREADS * 200;
const int64_t BATCH_SIZE = 1e7;

using namespace std;
using glm::dvec3;
using glm::ivec3;

struct LasPoints{
	shared_ptr<Buffer> buffer;
	int64_t numPoints;
  dvec3 c_scale;
  dvec3 c_offset;
};

struct LasLoader{
	LasLoader(){}

	static LasPoints loadSync(string file, int64_t firstPoint, int64_t wantedPoints){

		auto headerBuffer = new Buffer(2 * 1024);
		readBinaryFile(file, 0, headerBuffer ->size, headerBuffer ->data);

		uint64_t offsetToPointData = headerBuffer ->get<uint32_t>(96);
		int format = headerBuffer ->get<uint8_t>(104);
		int recordLength = headerBuffer ->get<uint16_t>(105);

		int versionMajor = headerBuffer ->get<uint8_t>(24);
		int versionMinor = headerBuffer ->get<uint8_t>(25);

		int64_t numPoints = 0;
		if(versionMajor == 1 && versionMinor <= 3){
			numPoints = headerBuffer ->get<uint32_t>(107);
		}else{
			numPoints = headerBuffer ->get<int64_t>(247);
		}

		dvec3 c_scale = {
			headerBuffer->get<double>(131),
			headerBuffer->get<double>(139),
			headerBuffer->get<double>(147)
		};

		dvec3 c_offset = {
			headerBuffer->get<double>(155),
			headerBuffer->get<double>(163),
			headerBuffer->get<double>(171)
		};

		int64_t batchSize_points = std::min(numPoints - firstPoint, wantedPoints);

		int64_t byteOffset = offsetToPointData + recordLength * firstPoint;
		int64_t byteSize = batchSize_points * recordLength;

		auto rawBuffer = make_shared<Buffer>(byteSize);
		readBinaryFile(file, byteOffset, byteSize, rawBuffer->data);

		// transform to XYZRGBA
		auto targetBuffer = make_shared<Buffer>(32 * batchSize_points);

		for(int i = 0; i < batchSize_points; i++){
			int64_t offset = i * recordLength;

			int32_t X = rawBuffer->get<int32_t>(offset + 0);
			int32_t Y = rawBuffer->get<int32_t>(offset + 4);
			int32_t Z = rawBuffer->get<int32_t>(offset + 8);
			uint16_t R = rawBuffer->get<uint16_t>(offset + 28);
			uint16_t G = rawBuffer->get<uint16_t>(offset + 30);
			uint16_t B = rawBuffer->get<uint16_t>(offset + 32);

			R = R > 255 ? R / 255 : R;
			G = G > 255 ? G / 255 : G;
			B = B > 255 ? B / 255 : B;
			uint32_t color = (R << 0) | (G << 8) | (B << 16);

			targetBuffer->set<int32_t>(X, TARGET_SZ * i + TARGET_START_X); // 4 bytes
			targetBuffer->set<int32_t>(Y, TARGET_SZ * i + TARGET_START_Y); // 4 bytes
			targetBuffer->set<int32_t>(Z, TARGET_SZ * i + TARGET_START_Z); // 4 bytes
			targetBuffer->set<uint32_t>(color, TARGET_SZ * i + TARGET_START_RGBA); // 4 bytes
		}

		LasPoints laspoints;
		laspoints.buffer = targetBuffer;
		laspoints.numPoints = batchSize_points;
    laspoints.c_scale = c_scale;
    laspoints.c_offset = c_offset;

		return laspoints;
	}

	static void load(string file, int64_t firstPoint, int64_t wantedPoints, function<void(shared_ptr<Buffer>, int64_t numLoaded)> callback){

		thread t([=](){
			auto laspoints = LasLoader::loadSync(file, firstPoint, wantedPoints);

			callback(laspoints.buffer, laspoints.numPoints);
		});

		t.detach();
	}

}; 

template<typename T>
vector<T> get_delta_values(vector<T> &x) {
  assert(x.size());

  vector<T> delta_values;
  T prev_val;

  prev_val = x[0];
  for (T &val : x) {
    delta_values.push_back(val - prev_val);
    prev_val = val;
  }

  return delta_values;
}

template<typename T>
vector<T> get_all_delta_values(vector<T> &x, vector<T> &y, vector<T> &z) {
  assert(x.size()); assert(y.size()); assert(z.size());

  vector<T> all_delta_values;
  auto delta_x = get_delta_values<T>(x);
  auto delta_y = get_delta_values<T>(y);
  auto delta_z = get_delta_values<T>(z);

  all_delta_values.insert(all_delta_values.end(), delta_x.begin(), delta_x.end());
  all_delta_values.insert(all_delta_values.end(), delta_y.begin(), delta_y.end());
  all_delta_values.insert(all_delta_values.end(), delta_z.begin(), delta_z.end());

  return all_delta_values;
}

vector<unsigned int> get_morton_order(vector<int32_t> &actual_x, vector<int32_t> &actual_y, vector<int32_t> &actual_z) {
  int num_points = actual_x.size();

  vector<uint64_t> point_to_morton(num_points);

  for (int i = 0; i < num_points; ++i) {
    uint32_t shifted_x = (uint32_t) ((int64_t) actual_x[i] - (int64_t) INT_MIN);
    uint32_t shifted_y = (uint32_t) ((int64_t) actual_y[i] - (int64_t) INT_MIN);
    uint32_t shifted_z = (uint32_t) ((int64_t) actual_z[i] - (int64_t) INT_MIN);
    // shifted_x = shifted_x >> (32 - 21);
    // shifted_y = shifted_y >> (32 - 21);
    // shifted_z = shifted_z >> (32 - 21);

    point_to_morton[i] = libmorton::morton3D_64_encode(shifted_x, shifted_y, shifted_z);
  }

  vector<unsigned int> order(num_points);
  iota(order.begin(), order.end(), 0);
  // stable_sort(order.begin(), order.end(), [&point_to_morton] (const int64_t &a, const int64_t &b) {
  //   return point_to_morton[a] < point_to_morton[b];
  // });
  stable_sort(order.begin(), order.end(), [&point_to_morton] (const unsigned int &a, const unsigned int &b) {
    return point_to_morton[a] < point_to_morton[b];
  });

  return order;
}

vector<pair<int,int>> split_batch_indices_into_threads(pair<int,int> batch) {
  vector<pair<int,int>> threads(NUM_THREADS);
  int per_thread_min = batch.second / NUM_THREADS;
  int gets_more = batch.second - NUM_THREADS * per_thread_min;

  // first "gets_more" get "per_thread_min + 1". remaining get "per_thread_min"
  int done = 0;
  for (int i = 0; i < NUM_THREADS; ++i) {
    if (i < gets_more) {
      threads[i] = {done, per_thread_min + 1};
      done += per_thread_min + 1;
    } else {
      threads[i] = {done, per_thread_min};
      done += per_thread_min;
    }
  }
  return threads;
}

void old_frequency_calculation(vector<int32_t> &actual_x, vector<int32_t> &actual_y, vector<int32_t> &actual_z) {
  cout << endl << "Old Frequency Calculation Method: " << endl;

  auto all_delta_values = get_all_delta_values<int32_t>(actual_x, actual_y, actual_z);
  Huffman<int32_t> hfmn;
  hfmn.calculate_frequencies(all_delta_values);
  auto sorted_frequencies = hfmn.get_sorted_frequencies();

  int64_t sum = 0;
  for (int i = 0; i < 100; ++i) {
    sum += sorted_frequencies[i].second;
  }
  
  int64_t num_points = actual_x.size();
  cout << "Coverage of all values: " << (double) sum / (double) (num_points * 3) << endl;

  hfmn.generate_huffman_tree();
  hfmn.create_dictionary();

  auto encoded = hfmn.compress_udtype<uint32_t>(all_delta_values);
  auto decoded = hfmn.decompress_udtype<uint32_t>(encoded, all_delta_values.size());

  assert(decoded == all_delta_values);
  cout << "Is data equal after decompression?: " << (bool) (decoded == all_delta_values) << endl;

  // how many points have all coordinates lying in their respective top-100 values?
  set<int32_t> good_values;
  for (int i = 0; i < 100; ++i) {
    good_values.insert(sorted_frequencies[i].first);
  }

  auto delta_x = get_delta_values<int32_t>(actual_x);
  auto delta_y = get_delta_values<int32_t>(actual_y);
  auto delta_z = get_delta_values<int32_t>(actual_z);

  int64_t good_points = 0;
  for (int i = 0; i < num_points; ++i) {
    bool good = true;
    if (good_values.find(delta_x[i]) == good_values.end()) good = false;
    if (good_values.find(delta_y[i]) == good_values.end()) good = false;
    if (good_values.find(delta_z[i]) == good_values.end()) good = false;
    good_points += good;
  }
  cout << "Points with all coordinates falling in top-100: " << (double) good_points / (double) num_points << endl;

  // create chains in the old frequency method
  vector<pair<int,int>> batches;
  batches.push_back({0, 1});
  for (int i = 1; i < num_points; ++i) {
    bool good = true;
    if (good_values.find(delta_x[i]) == good_values.end()) good = false;
    if (good_values.find(delta_y[i]) == good_values.end()) good = false;
    if (good_values.find(delta_z[i]) == good_values.end()) good = false;
    if (good)
      batches.back().second += 1;
    else
      batches.push_back({i, 1});
  }
  sort(batches.begin(), batches.end(), [] (const pair<int,int> &a, const pair<int,int> &b) {
    return a.second > b.second;
  });

  int64_t batches_sum = 0;
  for (auto &batch : batches) {
    batches_sum += batch.second;
    cout << batch.first << " " << batch.second << " " << (double) batches_sum / (double) num_points << endl;
  }
}

void simulation(vector<int32_t> &actual_x, vector<int32_t> &actual_y, vector<int32_t> &actual_z) {
  cout << "Simulation" << endl;
  int num_points = actual_x.size();

  // calculate delta values separately
  auto delta_x = get_delta_values<int32_t>(actual_x);
  auto delta_y = get_delta_values<int32_t>(actual_y);
  auto delta_z = get_delta_values<int32_t>(actual_z);

  // calculate common huffman and the best frequencies
  Huffman<int32_t> hfmn;
  auto all_delta_values = get_all_delta_values<int32_t>(actual_x, actual_y, actual_z);
  hfmn.calculate_frequencies(all_delta_values);
  all_delta_values.clear();
  auto sorted_frequencies = hfmn.get_sorted_frequencies();

  // calculate good delta values
  unordered_set<int32_t> good_deltas;
  for (int i = 0; i < 128; ++i) good_deltas.insert(sorted_frequencies[i].first);

  // update the huffman
  {
    vector<int32_t> subset_delta_values;
    for (int i = 0; i < 128; ++i) {
      auto &[value, freq] = sorted_frequencies[i];
      for (int cnt = 0; cnt < freq; ++cnt) subset_delta_values.push_back(value);
    }
    hfmn.calculate_frequencies(subset_delta_values);
    hfmn.generate_huffman_tree();
    hfmn.create_dictionary();
    int max_codeword_size = 0;
    for (auto &[val, cw] : hfmn.dictionary) max_codeword_size = max(max_codeword_size, (int) cw.size());
  }

  // get batch configuration
  vector<pair<int,int>> batches;
  int general_batch_size = 20000;
  for (int start_idx = 0; start_idx < num_points; start_idx += general_batch_size) {
    batches.push_back({start_idx, min(general_batch_size, num_points - start_idx)});
  }

  // divide into batches
  vector<pair<vector<uint8_t>,vector<int32_t>>> buffers;
  vector<array<int32_t,3>> start_values;

  for (auto &[start_idx, batch_size] : batches) {
    vector<int32_t> data(batch_size * 3);

    // first values
    start_values.push_back({actual_x[start_idx], actual_y[start_idx], actual_z[start_idx]});
    for (int i = 0; i < 3; ++i) data[i] = 0;

    // interleave the delta values
    for (int i = 1; i < batch_size; ++i) {
      data[i * 3 + 0] = delta_x[start_idx + i];
      data[i * 3 + 1] = delta_y[start_idx + i];
      data[i * 3 + 2] = delta_z[start_idx + i];
    }

    // encode the data and store it
    auto encoded_data = hfmn.compress_udtype_markus_idea<uint8_t, int32_t>(data);
    buffers.push_back(encoded_data);

    // decode the data and compare it
    auto recovered_data = hfmn.decompress_udtype_markus_idea<uint8_t, int32_t>(encoded_data.first, encoded_data.second, data.size());

    assert(recovered_data == data);
  }

  int64_t total_compressed_size = 0;
  int64_t total_size = actual_x.size() * 3 * sizeof(actual_x[0]);
  for (auto &buffer : buffers) {
    total_compressed_size += buffer.first.size() * sizeof(buffer.first[0]);
    total_compressed_size += buffer.second.size() * sizeof(buffer.second[0]);
  }

  for (int i = 0; i < batches.size(); ++i) {
    cout << "Batch Idx: " << i << endl;
    cout << "Data Size: " << (batches[i].second * 3) * sizeof(actual_x[0]) << " Bytes" << endl;
    cout << "Bitstream Size: " << (buffers[i].first.size()) * sizeof(buffers[i].first[0]) << " Bytes" << endl;
    cout << "Separate Size: " << (buffers[i].second.size()) * sizeof(buffers[i].second[0]) << " Bytes" << endl;
    double num = (batches[i].second * 3) * sizeof(actual_x[0]);
    double denom = (buffers[i].first.size()) * sizeof(buffers[i].first[0]) + (buffers[i].second.size()) * sizeof(buffers[i].second[0]);
    cout << "Local Compression Ratio: " << num / denom << endl;
    cout << endl;
  }

  cout << "Total Compression Ratio: " << (double) total_size / (double) total_compressed_size << endl;
}

void new_frequency_calculation(vector<int32_t> &actual_x, vector<int32_t> &actual_y, vector<int32_t> &actual_z) {
  cout << endl << "New Frequency Calculation Method: " << endl;

  // calculate delta values separately
  auto delta_x = get_delta_values<int32_t>(actual_x);
  auto delta_y = get_delta_values<int32_t>(actual_y);
  auto delta_z = get_delta_values<int32_t>(actual_z);

  // assign separate huffman
  Huffman<int32_t> hfmn_x, hfmn_y, hfmn_z;
  hfmn_x.calculate_frequencies(delta_x);
  hfmn_y.calculate_frequencies(delta_y);
  hfmn_z.calculate_frequencies(delta_z);
  auto sorted_frequencies_x = hfmn_x.get_sorted_frequencies();
  auto sorted_frequencies_y = hfmn_y.get_sorted_frequencies();
  auto sorted_frequencies_z = hfmn_z.get_sorted_frequencies();

  // how many values are covered by first 100 frequencies
  int64_t sum_x = 0, sum_y = 0, sum_z = 0;
  for (int i = 0; i < 100; ++i) {
    sum_x += sorted_frequencies_x[i].second;
    sum_y += sorted_frequencies_y[i].second;
    sum_z += sorted_frequencies_z[i].second;
  }

  int64_t num_points = actual_x.size();
  cout << "Coverage of X: " << (double) sum_x / (double) num_points << endl;
  cout << "Coverage of Y: " << (double) sum_y / (double) num_points << endl;
  cout << "Coverage of Z: " << (double) sum_z / (double) num_points << endl;

  hfmn_x.generate_huffman_tree();
  hfmn_x.create_dictionary();
  hfmn_y.generate_huffman_tree();
  hfmn_y.create_dictionary();
  hfmn_z.generate_huffman_tree();
  hfmn_z.create_dictionary();

  auto encoded_x = hfmn_x.compress_udtype<uint32_t>(delta_x);
  auto decoded_x = hfmn_x.decompress_udtype<uint32_t>(encoded_x, delta_x.size());
  cout << "Is data (x) equal after decompression?: " << (bool) (decoded_x == delta_x) << endl;

  auto encoded_y = hfmn_y.compress_udtype<uint32_t>(delta_y);
  auto decoded_y = hfmn_y.decompress_udtype<uint32_t>(encoded_y, delta_y.size());
  cout << "Is data (y) equal after decompression?: " << (bool) (decoded_y == delta_y) << endl;

  auto encoded_z = hfmn_z.compress_udtype<uint32_t>(delta_z);
  auto decoded_z = hfmn_z.decompress_udtype<uint32_t>(encoded_z, delta_z.size());
  cout << "Is data (z) equal after decompression?: " << (bool) (decoded_z == delta_z) << endl;

  // how many points have all coordinates lying in their respective top-100 values?
  set<int32_t> good_values_x, good_values_y, good_values_z;
  for (int i = 0; i < 100; ++i) {
    good_values_x.insert(sorted_frequencies_x[i].first);
    good_values_y.insert(sorted_frequencies_y[i].first);
    good_values_z.insert(sorted_frequencies_z[i].first);
  }

  int64_t good_points = 0;
  for (int i = 0; i < num_points; ++i) {
    bool good = true;
    if (good_values_x.find(delta_x[i]) == good_values_x.end()) good = false;
    if (good_values_y.find(delta_y[i]) == good_values_y.end()) good = false;
    if (good_values_z.find(delta_z[i]) == good_values_z.end()) good = false;
    good_points += good;
  }
  cout << "Points with all coordinates falling in top-100: " << (double) good_points / (double) num_points << endl;

  // create chains in the new frequency method
  vector<pair<int,int>> batches;
  batches.push_back({0, 1});
  for (int i = 1; i < num_points; ++i) {
    bool good = true;
    if (good_values_x.find(delta_x[i]) == good_values_x.end()) good = false;
    if (good_values_y.find(delta_y[i]) == good_values_y.end()) good = false;
    if (good_values_z.find(delta_z[i]) == good_values_z.end()) good = false;
    if (good)
      batches.back().second += 1;
    else
      batches.push_back({i, 1});
  }
  sort(batches.begin(), batches.end(), [] (const pair<int,int> &a, const pair<int,int> &b) {
    return a.second > b.second;
  });

  int64_t batches_sum = 0;
  for (auto &batch : batches) {
    batches_sum += batch.second;
    cout << batch.first << " " << batch.second << " " << (double) batches_sum / (double) num_points << endl;
  }
}

void create_chains(Huffman<int32_t> &hfmn, vector<int32_t> &actual_x, vector<int32_t> &actual_y, vector<int32_t> &actual_z) {
  auto sorted_frequencies = hfmn.get_sorted_frequencies();
  set<int32_t> good_deltas;
  for (int i = 0; i < 128; ++i) good_deltas.insert(sorted_frequencies[i].first);

  int num_points = actual_x.size();

  vector<pair<int,int>> batches; // [start_idx, batch_size]
  vector<int> start_values;

  // first point
  batches.push_back({0, 1});
  start_values.insert(start_values.end(), { actual_x[0], actual_y[0], actual_z[0] });
  int32_t prev_values[3] = {actual_x[0], actual_y[0], actual_z[0]};

  for (int i = 1; i < num_points; ++i) {
    int32_t delta_values[3] = {actual_x[i] - prev_values[0], actual_y[i] - prev_values[1], actual_z[i] - prev_values[2]};
    bool start_new_batch = false;
    for (int32_t &value : delta_values) start_new_batch = start_new_batch or (good_deltas.find(value) == good_deltas.end());
    if (start_new_batch) {
      batches.push_back({i, 1});
      start_values.insert(start_values.end(), { actual_x[i], actual_y[i], actual_z[i] });
    } else {
      batches.back().second += 1;
    }
    prev_values[0] = actual_x[i], prev_values[1] = actual_y[i], prev_values[2] = actual_z[i];
  }

  sort(batches.begin(), batches.end(), [] (const pair<int,int> &a, const pair<int,int> &b) {
    return a.second > b.second;
  });

  int till_now = 0;
  for (auto &[start_idx, batch_size] : batches) {
    till_now += batch_size;
    cout << start_idx << " " << batch_size << " " << (double) till_now / (double) num_points << endl;
  }
}

void throwaway_outliers(Huffman<int32_t> &hfmn, vector<int32_t> &actual_x, vector<int32_t> &actual_y, vector<int32_t> &actual_z) {
  auto sorted_frequencies = hfmn.get_sorted_frequencies();
  set<int32_t> good_deltas;
  for (int i = 0; i < 100; ++i) good_deltas.insert(sorted_frequencies[i].first);

  int num_points = actual_x.size();

  vector<pair<int,int>> batches; // [start_idx, batch_size]
  vector<int> start_values;

  // first point
  batches.push_back({0, 1});
  start_values.insert(start_values.end(), { actual_x[0], actual_y[0], actual_z[0] });
  int32_t prev_values[3] = {actual_x[0], actual_y[0], actual_z[0]};

  for (int i = 1; i < num_points; ++i) {
    int32_t delta_values[3] = {actual_x[i] - prev_values[0], actual_y[i] - prev_values[1], actual_z[i] - prev_values[2]};
    bool throwaway = false;
    for (int32_t &value : delta_values) throwaway |= (good_deltas.find(value) == good_deltas.end());
    if (not throwaway) {
      batches.back().second += 1;
      prev_values[0] = actual_x[i], prev_values[1] = actual_y[i], prev_values[2] = actual_z[i];
    }
  }

  sort(batches.begin(), batches.end(), [] (const pair<int,int> &a, const pair<int,int> &b) {
    return a.second > b.second;
  });

  for (auto &[start_idx, batch_size] : batches) {
    cout << start_idx << " " << batch_size << endl;
  }
}

void multichain_logic_hashing(Huffman<int32_t> &hfmn, vector<int32_t> &actual_x, vector<int32_t> &actual_y, vector<int32_t> &actual_z) {
  auto sorted_frequencies = hfmn.get_sorted_frequencies();
  set<int32_t> good_deltas;
  for (int i = 0; i < 100; ++i) good_deltas.insert(sorted_frequencies[i].first);

  int num_points = actual_x.size();
  vector<vector<int>> batches;

  batches.push_back({0});

  unordered_map<int32_t, int> which_batch;

  for (int i = 1; i < num_points; ++i) {
    bool found = false;
    for (auto &batch : batches) {
    }
  }
}

void multichain_logic(Huffman<int32_t> &hfmn, vector<int32_t> &actual_x, vector<int32_t> &actual_y, vector<int32_t> &actual_z) {
  auto sorted_frequencies = hfmn.get_sorted_frequencies();
  set<int32_t> good_deltas;
  for (int i = 0; i < 100; ++i) good_deltas.insert(sorted_frequencies[i].first);

  int num_points = actual_x.size();
  vector<vector<int>> batches;

  batches.push_back({0});
  for (int i = 1; i < num_points; ++i) {
    bool found = false;
    for (auto &batch : batches) {
      int prev_idx = batch.back();
      int32_t delta_values[3] = {actual_x[i] - actual_x[prev_idx], actual_y[i] - actual_y[prev_idx], actual_z[i] - actual_z[prev_idx]};

      bool good_delta = true;
      for (int32_t &value : delta_values) good_delta = good_delta and (good_deltas.find(value) != good_deltas.end());

      if (good_delta) {
        found = true;
        batch.push_back(i);
        break;
      }
    }
    if (not found) {
      batches.push_back({i});
    }
    cout << i << " " << batches.size() << endl;
  }

  vector<int64_t> sizes;
  for (auto &batch : batches) {
    cout << batch.size() << endl;
    sizes.push_back(batch.size());
  }

  for (int i = 1; i < sizes.size(); ++i) sizes[i] += sizes[i - 1];
  for (int i = 1; i < sizes.size(); ++i) cout << (double) sizes[i] / (double) sizes.back() << endl;
}

int main() {
  cout << "Starting Program." << endl;

  // load the first 10 million points.
  auto las_points = LasLoader::loadSync(LASFILE, 0, 1e5);

  cout << "Scale:" << endl;
  cout << las_points.c_scale.x << " " << las_points.c_scale.y << " "  << las_points.c_scale.z << endl;
  cout << "Offset:" << endl;
  cout << las_points.c_offset.x << " " << las_points.c_offset.y << " "  << las_points.c_offset.z << endl;

  auto num_points = las_points.numPoints;

  // read the original data
  vector<int32_t> actual_x(num_points), actual_y(num_points), actual_z(num_points);
  for (int i = 0; i < num_points; ++i) {
    actual_x[i] = las_points.buffer->get<int32_t>(TARGET_SZ * i +  TARGET_START_X);
    actual_y[i] = las_points.buffer->get<int32_t>(TARGET_SZ * i +  TARGET_START_Y);
    actual_z[i] = las_points.buffer->get<int32_t>(TARGET_SZ * i +  TARGET_START_Z);
  }

  // trying out faiss
  vector<float> values;
  for (int i = 0; i < num_points; ++i) {
    values.push_back(float(actual_x[i]) * las_points.c_scale.x + las_points.c_offset.x);
    values.push_back(float(actual_y[i]) * las_points.c_scale.y + las_points.c_offset.y);
    values.push_back(float(actual_z[i]) * las_points.c_scale.z + las_points.c_offset.z);
  }

  // morton_order
  {
    auto morton_order = get_morton_order(actual_x, actual_y, actual_z);
    // auto morton_order = mymorton::get_morton_order(actual_x, actual_y, actual_z);
    vector<int32_t> tmp;
    tmp = actual_x;
    for (int i = 0; i < morton_order.size(); ++i) actual_x[i] = tmp[morton_order[i]];
    tmp = actual_y;
    for (int i = 0; i < morton_order.size(); ++i) actual_y[i] = tmp[morton_order[i]];
    tmp = actual_z;
    for (int i = 0; i < morton_order.size(); ++i) actual_z[i] = tmp[morton_order[i]];
  }

  // old_frequency_calculation(actual_x, actual_y, actual_z);
  // new_frequency_calculation(actual_x, actual_y, actual_z);
  // simulation(actual_x, actual_y, actual_z);
  // return 0;

  // create huffman
  auto all_delta = get_all_delta_values<int32_t>(actual_x, actual_y, actual_z);
  Huffman<int32_t> hfmn;
  hfmn.calculate_frequencies(all_delta);
  hfmn.generate_huffman_tree();
  hfmn.create_dictionary();
  auto collapsed_dict = hfmn.get_collapsed_dictionary<uint32_t>();
  auto encoded = hfmn.compress_udtype_subarray_fast<uint32_t, vector<int32_t>::iterator>(all_delta.begin(), all_delta.end(), collapsed_dict);
  auto decoded = hfmn.decompress_udtype<uint32_t>(encoded, all_delta.size());
  assert (decoded == all_delta);
  return 0;

  // create_chains(hfmn, actual_x, actual_y, actual_z);
  // throwaway_outliers(hfmn, actual_x, actual_y, actual_z);
  // multichain_logic(hfmn, actual_x, actual_y, actual_z);
  // return 0;

  // divide into batches
  vector<pair<int,int>> batches; // [start_idx, size]
  for (int i = 0; i < num_points; i += BATCH_SIZE) {
    batches.push_back({i, min(BATCH_SIZE, num_points - i)});
  }

  // huffman encode all delta values
  vector<vector<uint32_t>> encoded_x(batches.size() * NUM_THREADS), encoded_y(batches.size() * NUM_THREADS), encoded_z(batches.size() * NUM_THREADS);

  // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  // std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

  double total_actual_size = 0, total_compressed_size = 0;
  for (int batch_idx = 0; batch_idx < batches.size(); ++batch_idx) {
    auto split_batch = split_batch_indices_into_threads(batches[batch_idx]);
    for (int tid = 0; tid < split_batch.size(); ++tid) {
      auto &[start_idx, batch_thread_size] = split_batch[tid];
      vector<int32_t> data_x(actual_x.begin() + start_idx, actual_x.begin() + start_idx + batch_thread_size);
      for (int idx = data_x.size() - 1; idx >= 1; --idx) data_x[idx] -= data_x[idx - 1];
      data_x[0] = 0;
      auto bitstream_x = hfmn.compress_udtype<uint32_t>(data_x);
      encoded_x[batch_idx * NUM_THREADS + tid] = bitstream_x;

      vector<int32_t> data_y(actual_y.begin() + start_idx, actual_y.begin() + start_idx + batch_thread_size);
      for (int idx = data_y.size() - 1; idx >= 1; --idx) data_y[idx] -= data_y[idx - 1];
      data_y[0] = 0;
      auto bitstream_y = hfmn.compress_udtype<uint32_t>(data_y);
      encoded_y[batch_idx * NUM_THREADS + tid] = bitstream_y;

      vector<int32_t> data_z(actual_z.begin() + start_idx, actual_z.begin() + start_idx + batch_thread_size);
      for (int idx = data_z.size() - 1; idx >= 1; --idx) data_z[idx] -= data_z[idx - 1];
      data_z[0] = 0;
      auto bitstream_z = hfmn.compress_udtype<uint32_t>(data_z);
      encoded_z[batch_idx * NUM_THREADS + tid] = bitstream_z;

      double actual_size = (3 * batch_thread_size * sizeof(int32_t));
      double compressed_size = (bitstream_x.size() + bitstream_y.size() + bitstream_z.size()) * sizeof(uint32_t);
      total_actual_size += actual_size;
      total_compressed_size += compressed_size;
    }
  }

  // assert
  for (int batch_idx = 0; batch_idx < batches.size(); ++batch_idx) {
    auto split_batch = split_batch_indices_into_threads(batches[batch_idx]);
    for (int tid = 0; tid < split_batch.size(); ++tid) {
      auto &[start_idx, batch_thread_size] = split_batch[tid];
      vector<int32_t> data_x(actual_x.begin() + start_idx, actual_x.begin() + start_idx + batch_thread_size);
      for (int idx = data_x.size() - 1; idx >= 1; --idx) data_x[idx] -= data_x[idx - 1];
      data_x[0] = 0;
      auto decompress_data_x = hfmn.decompress_udtype(encoded_x[batch_idx * NUM_THREADS + tid], data_x.size());
      assert(data_x == decompress_data_x);

      vector<int32_t> data_y(actual_y.begin() + start_idx, actual_y.begin() + start_idx + batch_thread_size);
      for (int idx = data_y.size() - 1; idx >= 1; --idx) data_y[idx] -= data_y[idx - 1];
      data_y[0] = 0;
      auto decompress_data_y = hfmn.decompress_udtype(encoded_y[batch_idx * NUM_THREADS + tid], data_y.size());
      assert(data_y == decompress_data_y);

      vector<int32_t> data_z(actual_z.begin() + start_idx, actual_z.begin() + start_idx + batch_thread_size);
      for (int idx = data_z.size() - 1; idx >= 1; --idx) data_z[idx] -= data_z[idx - 1];
      data_z[0] = 0;
      auto decompress_data_z = hfmn.decompress_udtype(encoded_z[batch_idx * NUM_THREADS + tid], data_z.size());
      assert(data_z == decompress_data_z);
    }
  }

  cout << "\nNet Compression Ratio: " << total_actual_size / total_compressed_size << endl;

  // find the biggest bit-length of an ecoded value
  unsigned int max_bit_length = 0;
  for (auto &[key, val] : hfmn.dictionary) max_bit_length = max(max_bit_length, (unsigned int) val.size());
  cout << "Max bit-length: " << max_bit_length << endl;
  cout << "Number of nodes in Huffman tree: " << hfmn.num_nodes << endl;

  return 0;
}


int old_main() {
  cout << "Starting program." << endl;
  auto las_points = LasLoader::loadSync(LASFILE, 0, 1e7);

  // vector<uint32_t> x(las_points.numPoints), y(las_points.numPoints), z(las_points.numPoints);
  cout << las_points.c_scale.x << " " << las_points.c_scale.y << " "  << las_points.c_scale.z << endl;
  cout << las_points.c_offset.x << " " << las_points.c_offset.y << " "  << las_points.c_offset.z << endl;

  auto num_points = las_points.numPoints;

  using v_i32  = vector<int32_t>;
  using v_ui32 = vector<uint32_t>;
  using v_i64  = vector<int64_t>;
  using v_ui64 = vector<uint64_t>;

  v_i32 actual_x(num_points), actual_y(num_points), actual_z(num_points);
  v_ui32 shifted_x(num_points), shifted_y(num_points), shifted_z(num_points);
  v_i32 morton_x(num_points), morton_y(num_points), morton_z(num_points);

  // read the original data
  for (int i = 0; i < num_points; ++i) {
    actual_x[i] = las_points.buffer->get<int32_t>(TARGET_SZ * i +  TARGET_START_X);
    actual_y[i] = las_points.buffer->get<int32_t>(TARGET_SZ * i +  TARGET_START_Y);
    actual_z[i] = las_points.buffer->get<int32_t>(TARGET_SZ * i +  TARGET_START_Z);
  }

  // shift xyz to make it unsigned
  for (int i = 0; i < num_points; ++i) {
    shifted_x[i] = (uint32_t) ((int64_t) actual_x[i] - (int64_t) INT_MIN);
    shifted_y[i] = (uint32_t) ((int64_t) actual_y[i] - (int64_t) INT_MIN);
    shifted_z[i] = (uint32_t) ((int64_t) actual_z[i] - (int64_t) INT_MIN);
  }

  uint32_t mn_x = *min_element(shifted_x.begin(), shifted_x.end());
  uint32_t mn_y = *min_element(shifted_y.begin(), shifted_y.end());
  uint32_t mn_z = *min_element(shifted_z.begin(), shifted_z.end());
  uint32_t mx_x = *max_element(shifted_x.begin(), shifted_x.end());
  uint32_t mx_y = *max_element(shifted_y.begin(), shifted_y.end());
  uint32_t mx_z = *max_element(shifted_z.begin(), shifted_z.end());

  printf("Min Elements: %u %u %u\n", mn_x, mn_y, mn_z);
  printf("Max Elements: %u %u %u\n", mx_x, mx_y, mx_z);
  printf("Min/Max values of UINT %u %u\n", 0, UINT_MAX);

  vector<uint64_t> point_to_morton(num_points);
  map<uint64_t, vector<size_t>> morton_to_point;

  for (int i = 0; i < num_points; ++i) {
    uint32_t X = shifted_x[i] >> (32 - 21);
    uint32_t Y = shifted_y[i] >> (32 - 21);
    uint32_t Z = shifted_z[i] >> (32 - 21);

    point_to_morton[i] = libmorton::morton3D_64_encode(X, Y, Z);
    morton_to_point[point_to_morton[i]].push_back(i);
  }

  vector<size_t> order(num_points);
  iota(order.begin(), order.end(), 0);
  stable_sort(order.begin(), order.end(), [&point_to_morton] (const int64_t &a, const int64_t &b) {
    return point_to_morton[a] < point_to_morton[b];
  });

  // get points according to morton order
  for (int i = 0; i < num_points; ++i) {
    morton_x[i] = actual_x[order[i]];
    morton_y[i] = actual_y[order[i]];
    morton_z[i] = actual_z[order[i]];
  }

  vector<vector<int>> batches(1, vector<int>());
  int min_delta = -(1 << 12), max_delta = (1 << 12) - 1;

  for (int i = 1; i < num_points; ++i) {
    auto delta_x = morton_x[i] - morton_x[i - 1];
    auto delta_y = morton_y[i] - morton_y[i - 1];
    auto delta_z = morton_z[i] - morton_z[i - 1];

    bool start_new_batch = false;
    if (min_delta > delta_x or delta_x > max_delta) start_new_batch = true;
    if (min_delta > delta_y or delta_y > max_delta) start_new_batch = true;
    if (min_delta > delta_z or delta_z > max_delta) start_new_batch = true;

    if (start_new_batch) batches.push_back(vector<int>());
    batches.back().push_back(i);
  }

  cout << batches.size() << endl;
  for (auto &batch : batches) cout << batch.size() << " ";
  cout << endl;

  vector<vector<int>> final_batches;
  vector<int> final_delta_bits;
  int cur_bitsize = 14;
  set<int> remaining_indices;
  for (int i = 0; i < num_points; ++i) remaining_indices.insert(i);

  while (remaining_indices.size()) {
    vector<int> rem(remaining_indices.begin(), remaining_indices.end());
    vector<vector<int>> batches(1, vector<int>());
    int min_delta = -(1 << cur_bitsize), max_delta = (1 << cur_bitsize) - 1;

    // divide into batches
    for (auto &i : rem) {
      auto delta_x = morton_x[i] - morton_x[i - 1];
      auto delta_y = morton_y[i] - morton_y[i - 1];
      auto delta_z = morton_z[i] - morton_z[i - 1];

      bool start_new_batch = false;
      if (min_delta > delta_x or delta_x > max_delta) start_new_batch = true;
      if (min_delta > delta_y or delta_y > max_delta) start_new_batch = true;
      if (min_delta > delta_z or delta_z > max_delta) start_new_batch = true;

      if (start_new_batch) batches.push_back(vector<int>());
      batches.back().push_back(i);
    }
    
    // take the batches that have size > 10000
    final_batches.push_back(vector<int>());
    for (auto &batch : batches) {
      if (batches.size() == 1 or batch.size() >= 10000 ) {
        final_batches.push_back(batch);
        final_delta_bits.push_back(cur_bitsize);
        for (auto &idx : batch) {
          remaining_indices.erase(idx);
        }
      }
    }
    
    // final_batches.push_back(vector<int>());
    // final_delta_bits.push_back(cur_bitsize);
    // for (auto &batch : batches) {
    //   if (batches.size() == 1 or batch.size() >= 10000) {
    //     final_batches.back().push_back(batch.size());
    //     // erase these indices from the set
    //     for (auto &idx : batch) {
    //       remaining_indices.erase(idx);
    //     }
    //   }
    // }

    ++cur_bitsize;
  }


  v_i32 delta_arr_x({0}), delta_arr_y({0}), delta_arr_z({0});
  for (int i = 1; i < num_points; ++i) {
    auto delta_x = morton_x[i] - morton_x[i - 1];
    auto delta_y = morton_y[i] - morton_y[i - 1];
    auto delta_z = morton_z[i] - morton_z[i - 1];
    delta_arr_x.push_back(delta_x);
    delta_arr_y.push_back(delta_y);
    delta_arr_z.push_back(delta_z);
  }

  // for (int i = 0; i < final_batches.size(); ++i) {
  //   cout << final_delta_bits[i] << " " << final_batches[i].size() << endl;
  //   if (final_batches[i].size() == 0) continue;
  //   Huffman<int32_t> hfmn;

  //   vector<int32_t> x_values, y_values, z_values, all_values;
  //   for (int &j : final_batches[i]) {
  //     x_values.push_back(delta_arr_x[j]);
  //     y_values.push_back(delta_arr_x[j]);
  //     z_values.push_back(delta_arr_x[j]);
  //   }
  //   all_values.insert(all_values.end(), x_values.begin(), x_values.end());
  //   all_values.insert(all_values.end(), y_values.begin(), y_values.end());
  //   all_values.insert(all_values.end(), z_values.begin(), z_values.end());

  //   hfmn.calculate_frequencies(all_values);
  //   hfmn.generate_huffman_tree();
  //   hfmn.create_dictionary();

  //   auto sorted = hfmn.get_sorted_frequencies();
  //   for(int j = 0; j < 10; ++j) cout << sorted[j].first << " ";
  //   cout << endl;

  //   auto bitstream_x = hfmn.compress(x_values);
  //   auto bitstream_y = hfmn.compress(y_values);
  //   auto bitstream_z = hfmn.compress(z_values);

  //   double compression_ratio = (double) (x_values.size() + y_values.size() + z_values.size()) * 32 / (bitstream_x.size() + bitstream_y.size() + bitstream_z.size());
  // }


  v_i32 all_delta;
  all_delta.insert(all_delta.begin(), delta_arr_x.begin(), delta_arr_x.end());
  all_delta.insert(all_delta.begin(), delta_arr_y.begin(), delta_arr_y.end());
  all_delta.insert(all_delta.begin(), delta_arr_z.begin(), delta_arr_z.end());


  Huffman<int32_t> hfmn;
  hfmn.calculate_frequencies(all_delta);
  hfmn.generate_huffman_tree();
  hfmn.create_dictionary();

  auto bitstream_x = hfmn.compress(delta_arr_x);
  auto bitstream_y = hfmn.compress(delta_arr_y);
  auto bitstream_z = hfmn.compress(delta_arr_z);

  cout << morton_x.size() * 32 << " " << bitstream_x.size() << endl;
  cout << morton_y.size() * 32 << " " << bitstream_y.size() << endl;
  cout << morton_z.size() * 32 << " " << bitstream_z.size() << endl;

  auto test_x = hfmn.decompress(bitstream_x);
  auto test_y = hfmn.decompress(bitstream_y);
  auto test_z = hfmn.decompress(bitstream_z);

  assert(test_x == delta_arr_x);
  assert(test_y == delta_arr_y);
  assert(test_z == delta_arr_z);

  auto sorted_dict = hfmn.get_sorted_frequencies();
  int covered_values = 0;
  for (int i = 0; i < sorted_dict.size(); ++i) {
    cout << sorted_dict[i].first << " " << sorted_dict[i].second << " " << hfmn.dictionary[sorted_dict[i].first].size() << endl;
    covered_values += sorted_dict[i].second;
  }

  cout << "Covered Values:   " << covered_values / 3 << endl;
  cout << "Remaining Values: " << num_points - covered_values / 3<< endl;

  return 0;
}
