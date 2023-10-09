#include "morton3D_LUTs.h"
#include "unsuck.hpp"
#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>
#include <iostream>
#include <limits>
#include <thread>
#include <bitset>
#include <map>
#include <set>
#include <queue>
#include <chrono>
#include "morton.h"
#include "huffman.h"

#define TARGET_START_X    0
#define TARGET_START_Y    TARGET_START_X + 4
#define TARGET_START_Z    TARGET_START_Y + 4
#define TARGET_START_RGBA TARGET_START_Z + 4
#define TARGET_SZ         4 * 4

#define NUM_THREADS 128

string LASFILE = "/home/rg/lidar_data/morro_bay.las";
// string LASFILE = "/home/rg/lidar_data/tree.las";
const int64_t BATCH_SIZE = NUM_THREADS * 1024;

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
vector<T> get_all_delta_values(vector<T> &x, vector<T> &y, vector<T> &z) {
  assert(x.size()); assert(y.size()); assert(z.size());

  vector<T> all_delta_values;
  int prev_val;

  prev_val = x[0];
  for (int &val : x) {
    all_delta_values.push_back(val - prev_val);
    prev_val = val;
  }
  prev_val = y[0];
  for (int &val : y) {
    all_delta_values.push_back(val - prev_val);
    prev_val = val;
  }
  prev_val = z[0];
  for (int &val : z) {
    all_delta_values.push_back(val - prev_val);
    prev_val = val;
  }

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
  stable_sort(order.begin(), order.end(), [&point_to_morton] (const int64_t &a, const int64_t &b) {
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

int main() {
  cout << "Starting Program." << endl;

  // load the first 10 million points.
  auto las_points = LasLoader::loadSync(LASFILE, 0, 1e7);

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

  // morton_order
  {
    auto morton_order = get_morton_order(actual_x, actual_y, actual_z);
    vector<int32_t> tmp;
    tmp = actual_x;
    for (int i = 0; i < morton_order.size(); ++i) actual_x[i] = tmp[morton_order[i]];
    tmp = actual_y;
    for (int i = 0; i < morton_order.size(); ++i) actual_y[i] = tmp[morton_order[i]];
    tmp = actual_z;
    for (int i = 0; i < morton_order.size(); ++i) actual_z[i] = tmp[morton_order[i]];
  }

  // create huffman
  auto all_delta = get_all_delta_values<int32_t>(actual_x, actual_y, actual_z);
  Huffman<int32_t> hfmn;
  hfmn.calculate_frequencies(all_delta);
  hfmn.generate_huffman_tree();
  hfmn.create_dictionary();

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
