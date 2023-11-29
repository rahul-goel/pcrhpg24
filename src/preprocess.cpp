#include <cstdint>
#include <string>
#include <future>
#include <queue>

#include "compute/Resources.h"
#include "mymorton.h"
#include "huffman.h"
#include "BatchDumpData.h"
#include "unsuck.hpp"
#include "glm/common.hpp"
#include "glm/matrix.hpp"

using namespace std;
using glm::dvec3;
using glm::ivec3;
using glm::uvec4;

#define TARGET_START_X    0
#define TARGET_START_Y    TARGET_START_X + 4
#define TARGET_START_Z    TARGET_START_Y + 4
#define TARGET_START_RGBA TARGET_START_Z + 4
#define TARGET_SZ         4 * 4

int NUM_CORES = 0;
bool TRANSPOSE = 0;

struct LasPoints{
	shared_ptr<Buffer> buffer;
  int64_t fullNumPoints;
	int64_t numPoints;
  dvec3 c_scale;
  dvec3 c_offset;
  dvec3 c_min;
  dvec3 c_max;
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

    dvec3 c_min = {
      headerBuffer->get<double>(187),
      headerBuffer->get<double>(203),
      headerBuffer->get<double>(219)
    };

    dvec3 c_max = {
      headerBuffer->get<double>(179),
      headerBuffer->get<double>(195),
      headerBuffer->get<double>(211)
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
    laspoints.fullNumPoints = numPoints;
    laspoints.c_scale = c_scale;
    laspoints.c_offset = c_offset;
    laspoints.c_min = c_min;
    laspoints.c_max = c_max;

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

// [start_idx, batch_size] vector returned
vector<pair<int,int>> get_batch_parameters(int num_points) {
  vector<pair<int,int>> batches;
  for (int start_idx = 0; start_idx < num_points; start_idx += POINTS_PER_WORKGROUP) {
    int batch_size = min((int) POINTS_PER_WORKGROUP, num_points - start_idx);
    batches.push_back({start_idx, batch_size});
  }
  return batches;
}

vector<pair<int,int>> get_chain_parameters(int num_points, int num_chains) {
  vector<pair<int,int>> chains;
  int d = num_points / num_chains, r = num_points % num_chains;
  int start_idx = 0;
  for (int i = 0; i < num_chains; ++i) {
    if (i < r) {
      // d + 1
      chains.push_back({start_idx, d + 1});
      start_idx += d + 1;
    } else {
      // d
      chains.push_back({start_idx, d});
      start_idx += d;
    }
  }
  return chains;
}

typedef enum {
  full_huffman, clipped_huffman
} method_type;

struct Chain {
  int point_offset;
  int num_points;
  int32_t start_xyz[3];
  float compression_ratio;
  Huffman<int32_t> *hfmn_ptr;

  method_type method;

  int32_t bbox_min[3];
  int32_t bbox_max[3];

  vector<int32_t> x, y, z;
  vector<int32_t> delta_x, delta_y, delta_z;
  vector<int32_t> delta_interleaved;
  vector<uint32_t> encoded;
  vector<int32_t> decoded;
  
  pair<vector<uint32_t>,vector<int32_t>> encoded_markus;
  vector<int32_t> decoded_markus;

  Chain(int point_offset, int num_points,
        vector<int32_t>::iterator x_begin, vector<int32_t>::iterator x_end,
        vector<int32_t>::iterator y_begin, vector<int32_t>::iterator y_end,
        vector<int32_t>::iterator z_begin, vector<int32_t>::iterator z_end) {
    this->point_offset = point_offset;
    this->num_points = num_points;

    x = vector<int32_t>(x_begin, x_end);
    y = vector<int32_t>(y_begin, y_end);
    z = vector<int32_t>(z_begin, z_end);
  }

  void calculate_bbox() {
    bbox_min[0] = *min_element(x.begin(), x.end());
    bbox_min[1] = *min_element(y.begin(), y.end());
    bbox_min[2] = *min_element(z.begin(), z.end());
    bbox_max[0] = *max_element(x.begin(), x.end());
    bbox_max[1] = *max_element(y.begin(), y.end());
    bbox_max[2] = *max_element(z.begin(), z.end());
  }

  void calculate_deltas() {
    delta_x.clear(), delta_y.clear(), delta_z.clear();
    delta_x.resize(num_points), delta_y.resize(num_points), delta_z.resize(num_points);

    for (int i = num_points - 1; i >= 1; --i) {
      delta_x[i] = x[i] - x[i - 1];
      delta_y[i] = y[i] - y[i - 1];
      delta_z[i] = z[i] - z[i - 1];
    }
    delta_x[0] = 0, delta_y[0] = 0, delta_z[0] = 0;
    start_xyz[0] = x[0], start_xyz[1] = y[0], start_xyz[2] = z[0];
  }

  void interleave() {
    delta_interleaved.clear();
    assert(delta_x.size() == num_points);
    assert(delta_y.size() == num_points);
    assert(delta_z.size() == num_points);

    delta_interleaved.resize(num_points * 3);
    for (int i = 0; i < num_points; ++i) {
      delta_interleaved[i * 3 + 0] = delta_x[i];
      delta_interleaved[i * 3 + 1] = delta_y[i];
      delta_interleaved[i * 3 + 2] = delta_z[i];
    }
  }

  void encode(unordered_map<int32_t, pair<uint32_t,int>> &collapsed_dict) {
    auto hfmn = *hfmn_ptr;
    if (method == full_huffman) {
      encoded.clear();

      encoded = hfmn.compress_udtype_subarray_fast<
          uint32_t, typename vector<int32_t>::iterator>(
          delta_interleaved.begin(), delta_interleaved.end(), collapsed_dict);
    } else if (method == clipped_huffman) {
      encoded_markus.first.clear();
      encoded_markus.second.clear();

      encoded_markus = hfmn.compress_udtype_subarray_fast_pjn_idea<
        uint32_t, typename vector<int32_t>::iterator, int32_t>(
          delta_interleaved.begin(), delta_interleaved.end(), collapsed_dict,
          HUFFMAN_TABLE_SIZE);
    }
  }

  void decode(vector<pair<int32_t,int>> &decoder_table) {
    auto hfmn = *hfmn_ptr;
    if (method == full_huffman) {
      decoded.clear();
      decoded.resize(num_points * 3);
      hfmn.decompress_udtype_subarray_fast<uint32_t,
                                           typename vector<int32_t>::iterator>(
          decoded.begin(), decoded.end(), encoded, decoder_table);
    } else if (method == clipped_huffman) {
      decoded_markus.clear();
      decoded_markus.resize(num_points * 3);
      hfmn.decompress_udtype_subarray_fast_pjn_idea<
          uint32_t, typename vector<int32_t>::iterator, int32_t>(
          decoded_markus.begin(), decoded_markus.end(), encoded_markus.first,
          encoded_markus.second, decoder_table);
    }
  }

  int get_encoded_size() {
    int size = 0;
    if (method == full_huffman) {
      size = encoded.size();
    } else if (method == clipped_huffman) {
      size = encoded_markus.first.size();
    }
    return size;
  }

  void pad_encoded(int max_size) {
    if (method == full_huffman) {
      encoded.resize(max_size, 0);
    } else if (method == clipped_huffman) {
      encoded_markus.first.resize(max_size, 0);
    }
  }

  void assert_delta_interleaved_decoded() {
    if (method == full_huffman) {
      assert(delta_interleaved == decoded);
    } else if (method == clipped_huffman) {
      assert(delta_interleaved == decoded_markus);
    }
  }

  void calcualte_compression_ratio() {
    float og_size = num_points * 3 * sizeof(int32_t);
    float new_size = 0;
    if (method == full_huffman) {
      new_size = encoded.size() * sizeof(encoded[0]);
    } else if (method == clipped_huffman) {
      new_size =
          encoded_markus.first.size() * sizeof(encoded_markus.first[0]) +
          encoded_markus.second.size() * sizeof(encoded_markus.second[0]);
    }
    float ratio = og_size / new_size;
    this->compression_ratio = ratio;
  }

  float get_og_size() {
    float og_size =num_points * 3 * sizeof(int32_t);
    return og_size;
  }

  float get_compressed_size() {
    float new_size = 0;
    if (method == full_huffman) {
      new_size = encoded.size() * sizeof(encoded[0]);
    } else if (method == clipped_huffman) {
      new_size =
          encoded_markus.first.size() * sizeof(encoded_markus.first[0]) +
          encoded_markus.second.size() * sizeof(encoded_markus.second[0]);
    }
    return new_size;
  }

  int64_t get_encoded_num_bytes() {
    int64_t ret = 0;
    if (method == full_huffman) {
      if (encoded.size())
        ret = encoded.size() * sizeof(encoded[0]);
    } else if (method == clipped_huffman) {
      if (encoded_markus.first.size())
        ret = encoded_markus.first.size() * sizeof(encoded_markus.first[0]);
    }
    return ret;
  }

  int64_t get_separate_num_bytes() {
    int64_t ret = 0;
    if (method == full_huffman) {
      ret = 0;
    } else if (method == clipped_huffman) {
      if (encoded_markus.second.size())
        ret = encoded_markus.second.size() * sizeof(encoded_markus.second[0]);
    }
    return ret;
  }
};

struct Batch {
  int point_offset;
  int num_points;
  vector<Chain> chains;
  Huffman<int32_t> hfmn;
  method_type method;
  float compression_ratio;
  vector<int> cluster_sizes;

  int32_t bbox_min[3];
  int32_t bbox_max[3];

  int64_t encoding_bytes = 0;
  int64_t separate_bytes = 0;
  int64_t cluster_bytes = 0;

  // huffman dictionary and table for uint32_t
  unordered_map<int32_t, pair<uint32_t,int>> dictionary;
  vector<pair<int32_t,int>> decoder_table;

  Batch(int point_offset, int num_points, method_type method,
        vector<int32_t>::iterator x_begin, vector<int32_t>::iterator x_end,
        vector<int32_t>::iterator y_begin, vector<int32_t>::iterator y_end,
        vector<int32_t>::iterator z_begin, vector<int32_t>::iterator z_end) {
    this->point_offset = point_offset;
    this->num_points = num_points;
    this->method = method;

    chains.clear();
    auto chain_parameters = get_chain_parameters(num_points, WORKGROUP_SIZE);
    for (auto &[offset, chain_size] : chain_parameters) {
      chains.push_back(Chain(point_offset + offset, chain_size,
                         x_begin + offset, x_begin + offset + chain_size,
                         y_begin + offset, y_begin + offset + chain_size,
                         z_begin + offset, z_begin + offset + chain_size));
      chains.back().method = this->method;
    }
  }

  uvec4 get_uvec4(vector<uint32_t> &vec, vector<int> &lengths) {
    assert(vec.size() == lengths.size());

    uvec4 codeword_chunk;
    int used_bits = 0;

    int offset = 0;
    for (int i = 0; i < vec.size(); ++i) {
      int len = lengths[i];
      assert(len > 0);

      int idx = offset / 32;
      int mod = offset % 32;
      if (len + mod > 32) {
        // gotta split 'em up, this should not happen towards the end
        assert(idx != 3);
        uint32_t cw_1 = vec[i] >> (len + mod - 32);
        uint32_t cw_2 = vec[i] << (64 - len - mod);
        codeword_chunk[idx] |= cw_1;
        codeword_chunk[idx + 1] |= cw_2;
      } else {
        uint32_t cw = (vec[i] << (32 - len - mod));
        codeword_chunk[idx] |= cw;
      }

      offset += len;
    }

    return codeword_chunk;
  }

  void encode_decode_rahul(Huffman<int32_t> &hfmn) {
    vector<vector<uint32_t>> all_codes(WORKGROUP_SIZE);
    vector<vector<int>> all_sizes(WORKGROUP_SIZE);
    vector<vector<int32_t>> all_separate_data(WORKGROUP_SIZE);

    // get the precompression data
    for (int tid = 0; tid < WORKGROUP_SIZE; ++tid) {
      auto &deltas = chains[tid].delta_interleaved;
      auto [codes, sizes, separate_data] = hfmn.precompress_rahul
      <uint32_t, typename vector<int32_t>::iterator, int32_t>
      (deltas.begin(), deltas.end(), dictionary, HUFFMAN_TABLE_SIZE);
      all_codes[tid] = codes;
      all_sizes[tid] = sizes;
      all_separate_data[tid] = separate_data;
    }

    // calculate the cluster sizes
    vector<int> bitcount(WORKGROUP_SIZE, 0);
    cluster_sizes = vector<int>(1, 0);

    for (int eid = 0; eid < POINTS_PER_THREAD * 3; ++eid) {
      for (int tid = 0; tid < WORKGROUP_SIZE; ++tid) {
        bitcount[tid] += all_sizes[tid][eid];
      }
      int mx = *max_element(bitcount.begin(), bitcount.end());
      bool can_fit_in = mx <= 32 * 4;
      if (can_fit_in) {
        ++cluster_sizes.back();
      } else {
        cluster_sizes.push_back(1);
        for (int tid = 0; tid < WORKGROUP_SIZE; ++tid) {
          bitcount[tid] = all_sizes[tid][eid];
        }
      }
    }

    // compress using the cluster sizes
    for (int tid = 0; tid < WORKGROUP_SIZE; ++tid) {
      // set the data in the chains so that it can be used later
      chains[tid].encoded_markus.first = hfmn.compress_rahul
      <uint32_t, typename vector<int32_t>::iterator, int32_t>
      (all_codes[tid], all_sizes[tid], cluster_sizes);

      chains[tid].encoded_markus.second = all_separate_data[tid];
    }

    for (int tid = 0; tid < WORKGROUP_SIZE; ++tid) {
      vector<int32_t> thread_decoded(POINTS_PER_THREAD * 3);
      hfmn.decompress_rahul<uint32_t, typename vector<int32_t>::iterator,
                            int32_t>(
          thread_decoded.begin(), thread_decoded.end(), chains[tid].encoded_markus.first,
          all_separate_data[tid], cluster_sizes, decoder_table);
      assert(thread_decoded == chains[tid].delta_interleaved);
    }

    // calculate the amount of wasted space
    {
      long long og_bits = 0, total_bits = 0;
      for (int tid = 0; tid < WORKGROUP_SIZE; ++tid) {
        auto &sz = all_sizes[tid];
        og_bits += accumulate(sz.begin(), sz.end(), 0ll);
      }
      total_bits += WORKGROUP_SIZE * cluster_sizes.size() * 4 * 32;

      double wasted = (double) (total_bits - og_bits) / total_bits;
      cout << "\nwasted space " << wasted << "\n";
    }

    // calculate the percentage of the separate data
    // {
    //   int sep_cnt = 0;
    //   int compressed_cnt = 0;
    //   for (int tid = 0; tid < WORKGROUP_SIZE; ++tid) {
    //     sep_cnt += all_separate_data[tid].size();
    //     compressed_cnt += all_codes[tid].size();
    //   }
    //   double sep_frac = (double) sep_cnt / (sep_cnt + compressed_cnt);
    //   cout << "\nseparate fraction " << sep_frac << "\n";
    // }
  }

  void calculate() {
    // calculate bbox
    for (int i = 0; i < WORKGROUP_SIZE; ++i) {
      if (chains[i].num_points == 0) continue;
      chains[i].calculate_bbox();
    }
    for (int j = 0; j < 3; ++j) {
      bbox_min[j] = chains[0].bbox_min[j];
      bbox_max[j] = chains[0].bbox_max[j];
    }
    for (int i = 1; i < WORKGROUP_SIZE; ++i) {
      if (chains[i].num_points == 0) continue;
      for (int j = 0; j < 3; ++j) {
        bbox_min[j] = min(bbox_min[j], chains[i].bbox_min[j]);
        bbox_max[j] = max(bbox_max[j], chains[i].bbox_max[j]);
      }
    }

    vector<int32_t> all_deltas;
    for (int i = 0; i < WORKGROUP_SIZE; ++i) {
      if (chains[i].num_points == 0) continue;
      chains[i].calculate_deltas();
      chains[i].interleave();
      all_deltas.insert(all_deltas.end(), chains[i].delta_interleaved.begin(), chains[i].delta_interleaved.end());
    }

    hfmn = Huffman<int32_t>();
    hfmn.calculate_frequencies(all_deltas);
    hfmn.generate_huffman_tree();
    if (method == clipped_huffman) {
      dictionary = hfmn.create_dictionary_pjn<uint32_t>(HUFFMAN_TABLE_SIZE);
      decoder_table = hfmn.get_gpu_huffman_table_pjn<uint32_t>(dictionary, HUFFMAN_TABLE_SIZE);
    } else if (method == full_huffman) {
      hfmn.create_dictionary();
      dictionary = hfmn.get_collapsed_dictionary<uint32_t>();
      decoder_table = hfmn.get_gpu_huffman_table();
    }
    hfmn.clear_huffman_tree();

    int fits_in_a_byte = 0;
    long long fits_in_a_byte_freq = 0;
    unordered_set<int> already_seen;
    for (int i = 0; i < HUFFMAN_TABLE_SIZE; ++i) {
      auto &element = decoder_table[i].first;
      if (already_seen.find(element) != already_seen.end()) continue;
      already_seen.insert(element);
      bool f = (element <= 127) and (element >= -128);
      fits_in_a_byte += f;
      if (f) {
        fits_in_a_byte_freq += hfmn.frequencies.at(element);
      }
    }
    cout << "\nless than a byte " << (double) fits_in_a_byte / HUFFMAN_TABLE_SIZE << endl;
    cout << "\npercentage " << (double) fits_in_a_byte_freq / (all_deltas.size()) << endl;

    for (int i = 0; i < HUFFMAN_TABLE_SIZE; ++i) {
      auto &element = decoder_table[i].first;
      fits_in_a_byte += (element <= 127) and (element >= -128);
    }

    // float old_size = 0;
    // float new_size = 0;
    // for (int i = 0; i < WORKGROUP_SIZE; ++i) {
    //   chains[i].hfmn_ptr = &hfmn;
    //   chains[i].encode(dictionary);
    // }

    // // get max chain size
    // int max_size = chains[0].get_encoded_size();
    // for (int i = 1; i < WORKGROUP_SIZE; ++i)
    //   max_size = max(max_size, chains[i].get_encoded_size());

    // for (int i = 0; i < WORKGROUP_SIZE; ++i) {
    //   if (TRANSPOSE) chains[i].pad_encoded(max_size);
    //   chains[i].decode(decoder_table);
    //   chains[i].assert_delta_interleaved_decoded();
    //   old_size += chains[i].get_og_size();
    //   new_size += chains[i].get_compressed_size();
    // }

    float old_size = 0;
    float new_size = 0;
    encode_decode_rahul(hfmn);
    for (int i = 0; i < WORKGROUP_SIZE; ++i) {
      old_size += chains[i].get_og_size();
      new_size += chains[i].get_compressed_size();
    }

    // calculate the number of bytes occupied
    if (method == clipped_huffman) {
      encoding_bytes = 0;
      separate_bytes = 0;
      for (int i = 0; i < WORKGROUP_SIZE; ++i) {
        encoding_bytes += chains[i].get_encoded_num_bytes();
        separate_bytes += chains[i].get_separate_num_bytes();
      }
    }
    cluster_bytes = cluster_sizes.size() * 4;

    // add huffman size
    for (auto &[symbol, cw_len] : decoder_table) {
      new_size += sizeof(symbol);
      new_size += sizeof(cw_len);
    }
    this->compression_ratio = old_size / new_size;
  }
};

struct Chunk {
  int64_t num_points;
  int64_t encoding_bytes = 0;
  int64_t separate_bytes = 0;
  int64_t cluster_bytes = 0;
  vector<int64_t> batch_data_sizes;
  vector<float> compression_ratios;
  vector<char> dump_buffer;
};

Chunk process_chunk(string filename, long long start_idx, long long wanted_points, bool sort) {
  auto las = LasLoader::loadSync(filename, start_idx, wanted_points);

  int64_t num_points = wanted_points;

  // read the original data
  vector<int32_t> actual_x(num_points), actual_y(num_points), actual_z(num_points);
  vector<uint32_t> actual_color(num_points);
  for (int i = 0; i < num_points; ++i) {
    actual_x[i] = las.buffer->get<int32_t>(TARGET_SZ * i +  TARGET_START_X);
    actual_y[i] = las.buffer->get<int32_t>(TARGET_SZ * i +  TARGET_START_Y);
    actual_z[i] = las.buffer->get<int32_t>(TARGET_SZ * i +  TARGET_START_Z);

    actual_color[i] = las.buffer->get<uint32_t>(TARGET_SZ * i +  TARGET_START_RGBA);
  }
  // last batch fix
  int extra_needed = 0;
  if (num_points % POINTS_PER_WORKGROUP) extra_needed = POINTS_PER_WORKGROUP - (num_points % POINTS_PER_WORKGROUP);
  vector<int32_t> extra_x(extra_needed, actual_x.back());
  vector<int32_t> extra_y(extra_needed, actual_y.back());
  vector<int32_t> extra_z(extra_needed, actual_z.back());
  vector<uint32_t> extra_color(extra_needed, actual_color.back());
  actual_x.insert(actual_x.end(), extra_x.begin(), extra_x.end());
  actual_y.insert(actual_y.end(), extra_y.begin(), extra_y.end());
  actual_z.insert(actual_z.end(), extra_z.begin(), extra_z.end());
  actual_color.insert(actual_color.end(), extra_color.begin(), extra_color.end());
  num_points += extra_needed;
  cout << "Collected Original Data" << endl;

  // sort by morton order
  if (sort) {
    auto morton_order = mymorton::get_morton_order(actual_x, actual_y, actual_z);
    vector<int32_t> tmp;
    tmp = actual_x;
    for (int i = 0; i < morton_order.size(); ++i) actual_x[i] = tmp[morton_order[i]];
    tmp = actual_y;
    for (int i = 0; i < morton_order.size(); ++i) actual_y[i] = tmp[morton_order[i]];
    tmp = actual_z;
    for (int i = 0; i < morton_order.size(); ++i) actual_z[i] = tmp[morton_order[i]];

    vector<uint32_t> c_tmp = actual_color;
    for (int i = 0; i < morton_order.size(); ++i) actual_color[i] = c_tmp[morton_order[i]];
    cout << "Sorted Points" << endl;
  }

  // divide into batches
  vector<pair<int,int>> batch_configuration = get_batch_parameters(num_points);
  vector<Batch> batches;
  for (auto &[start_point, size] : batch_configuration) {
    auto begin_x = actual_x.begin() + start_point;
    auto begin_y = actual_y.begin() + start_point;
    auto begin_z = actual_z.begin() + start_point;
    auto end_x = actual_x.begin() + start_point + size;
    auto end_y = actual_y.begin() + start_point + size;
    auto end_z = actual_z.begin() + start_point + size;

    batches.push_back(Batch(start_point, size, clipped_huffman,
                            actual_x.begin() + start_point, actual_x.begin() + start_point + size,
                            actual_y.begin() + start_point, actual_y.begin() + start_point + size,
                            actual_z.begin() + start_point, actual_z.begin() + start_point + size));
  }
  cout << "Divided into Batches" << endl;

  // calculate huffman encoding for each batch
  queue<future<void>> futures;
  int processed = 0;
  for (int i = 0; i < batches.size(); ++i) {
    while (futures.size() >= NUM_CORES) {
      futures.front().get();
      futures.pop();
      processed += 1;
      printf("\r%f", (float) processed / batches.size());
      fflush(stdout);
    }

    auto lambda = [&batches, i] {
      batches[i].calculate();
    };
    futures.push(async(launch::async, lambda));
  }

  while (futures.size() > 0) {
    futures.front().get();
    futures.pop();
    processed += 1;
    printf("\r%f", (float) processed / batches.size());
    fflush(stdout);
  }
  cout << endl;
  cout << "Huffman Encoding done" << endl;

  float compression_ratio = 0;
  vector<float> compression_ratios;
  for (auto &batch : batches) {
    compression_ratios.push_back(batch.compression_ratio);
  }

  // sum up encoding, separate bytes
  int64_t encoding_bytes = 0;
  int64_t separate_bytes = 0;
  int64_t cluster_bytes = 0;
  for (int i = 0; i < batches.size(); ++i) {
    encoding_bytes += batches[i].encoding_bytes;
    separate_bytes += batches[i].separate_bytes;
    cluster_bytes += batches[i].cluster_bytes;
  }


  // convert Batch to BatchDumpData
  int64_t processed_points = 0;

  // this chunk
  Chunk chunk;
  chunk.num_points = num_points;
  chunk.encoding_bytes = encoding_bytes;
  chunk.separate_bytes = separate_bytes;
  chunk.cluster_bytes = cluster_bytes;
  chunk.compression_ratios = compression_ratios;
  vector<char> &dump_buffer = chunk.dump_buffer;
  vector<int64_t> &batch_data_sizes = chunk.batch_data_sizes;
  batch_data_sizes.resize(batches.size());

  dump_buffer.reserve(num_points * 10);

  { // printing for debugging
    /*
    for (int bid = 0; bid < batches.size(); ++bid) {
      auto &b = batches[bid];
      cout << endl;
      cout << bid << endl;
      cout << b.bbox_min[0] << " " << b.bbox_min[1] << " " << b.bbox_min[2] << endl;
      cout << b.bbox_max[0] << " " << b.bbox_max[1] << " " << b.bbox_max[2] << endl;
      cout << las.c_scale.x << " " << las.c_scale.y << " " << las.c_scale.z << endl;
      cout << las.c_offset.x << " " << las.c_offset.y << " " << las.c_offset.z << endl;
      cout << endl;
    }
    */
  }

  for (int bid = 0; bid < batches.size(); ++bid) {
    BatchDumpData bdd;
    Batch &b = batches[bid];

    // header data
    bdd.point_offset = processed_points;
    bdd.num_points = b.num_points;
    bdd.num_threads = WORKGROUP_SIZE;
    bdd.points_per_thread = POINTS_PER_THREAD;

    bdd.las_scale[0] = las.c_scale.x;
    bdd.las_scale[1] = las.c_scale.y;
    bdd.las_scale[2] = las.c_scale.z;
    bdd.las_offset[0] = las.c_offset.x;
    bdd.las_offset[1] = las.c_offset.y;
    bdd.las_offset[2] = las.c_offset.z;

    bdd.las_min[0] = (float) las.c_min.x;
    bdd.las_min[1] = (float) las.c_min.y;
    bdd.las_min[2] = (float) las.c_min.z;
    bdd.las_max[0] = (float) las.c_max.x;
    bdd.las_max[1] = (float) las.c_max.y;
    bdd.las_max[2] = (float) las.c_max.z;

    bdd.bbox_min[0] = float(b.bbox_min[0]) * las.c_scale.x + las.c_offset.x;
    bdd.bbox_min[1] = float(b.bbox_min[1]) * las.c_scale.y + las.c_offset.y;
    bdd.bbox_min[2] = float(b.bbox_min[2]) * las.c_scale.z + las.c_offset.z;
    bdd.bbox_max[0] = float(b.bbox_max[0]) * las.c_scale.x + las.c_offset.x;
    bdd.bbox_max[1] = float(b.bbox_max[1]) * las.c_scale.y + las.c_offset.y;
    bdd.bbox_max[2] = float(b.bbox_max[2]) * las.c_scale.z + las.c_offset.z;

    // start values
    bdd.start_values.resize(WORKGROUP_SIZE * 3);
    for (int i = 0; i < WORKGROUP_SIZE; ++i) {
      bdd.start_values[i * 3 + 0] = b.chains[i].start_xyz[0];
      bdd.start_values[i * 3 + 1] = b.chains[i].start_xyz[1];
      bdd.start_values[i * 3 + 2] = b.chains[i].start_xyz[2];
    }

    // encoding
    bdd.encoding_offsets.resize(WORKGROUP_SIZE);
    bdd.encoding_sizes.resize(WORKGROUP_SIZE);
    for (int i = 0; i < WORKGROUP_SIZE; ++i) {
      auto &src = b.chains[i].encoded_markus.first;
      auto &dst = bdd.encoding;
      bdd.encoding_offsets[i] = dst.size();
      dst.insert(dst.end(), src.begin(), src.end());
      bdd.encoding_sizes[i] = dst.size() - bdd.encoding_offsets[i];
    }
    if (TRANSPOSE) {
      auto src = bdd.encoding;
      auto &dst = bdd.encoding;
      int size = b.chains[0].get_encoded_size();
      assert(size % 4 == 0);
      for (int i = 0; i < WORKGROUP_SIZE; ++i) {
        for (int j = 0; j < size; ++j) {
          dst[j * WORKGROUP_SIZE + i] = src[i * size + j];
        }
      }
    }
    // int len = b.chains[0].encoded_markus.first.size();
    // auto &dst = bdd.encoding;
    // assert(len % 4 == 0);
    // for (int i = 0; i < len; i += 4) {
    //   for (int tid = 0; tid < WORKGROUP_SIZE; ++tid) {
    //     auto &src = b.chains[tid].encoded_markus.first;
    //     dst.insert(dst.end(), {src[i], src[i + 1], src[i + 2], src[i + 3]});
    //   }
    // }

    // for (int tid = 0; tid < WORKGROUP_SIZE; ++tid) {
    //   bdd.encoding_offsets[tid] = 0;
    //   bdd.encoding_sizes[tid] = len;
    // }

    // separate
    bdd.separate_offsets.resize(WORKGROUP_SIZE);
    bdd.separate_sizes.resize(WORKGROUP_SIZE);
    for (int i = 0; i < WORKGROUP_SIZE; ++i) {
      auto &src = b.chains[i].encoded_markus.second;
      auto &dst = bdd.separate;
      bdd.separate_offsets[i] = dst.size();
      dst.insert(dst.end(), src.begin(), src.end());
      bdd.separate_sizes[i] = dst.size() - bdd.separate_offsets[i];
    }

    // TODO: color
    bdd.color.resize(bdd.num_points);
    for (int i = 0; i < b.num_points; ++i) {
      bdd.color[i] = actual_color[b.point_offset + i];
    }

    // decoder table
    auto &decoder_table = b.decoder_table;

    int dt_size = decoder_table.size();
    bdd.decoder_values.resize(dt_size);
    bdd.decoder_cw_len.resize(dt_size);
    bdd.dt_size = dt_size;
    for (int i = 0; i < dt_size; ++i) {
      bdd.decoder_values[i] = decoder_table[i].first;
      bdd.decoder_cw_len[i] = decoder_table[i].second;
    }

    // cluster sizes
    bdd.num_clusters = b.cluster_sizes.size();
    bdd.cluster_sizes = b.cluster_sizes;

    // dump
    vector<char> bdd_buffer = bdd.get_buffer();
    batch_data_sizes[bid] = bdd_buffer.size();
    dump_buffer.insert(dump_buffer.end(), bdd_buffer.begin(), bdd_buffer.end());
    processed_points += b.num_points;
  }

  return chunk;
}

int main(int argc, char *argv[]) {
  // parse command line arguments
  assert(argc == 6); // program name, input file, output file, sort or not, how many cores, transpose or not

  vector<string> all_args;
  if (argc > 1) {
    all_args.assign(argv + 1, argv + argc);
  }
  string LASFILE = all_args[0];
  string OUTFILE = all_args[1];
  bool should_sort = (bool) stoi(all_args[2]);
  NUM_CORES = stoi(all_args[3]);
  TRANSPOSE = stoi(all_args[4]);

  auto las = LasLoader::loadSync(LASFILE, 0, 100);
  int64_t num_points = las.fullNumPoints;
  // int64_t num_points = 1e6;
  int64_t num_batches = 0;
  int64_t encoding_bytes = 0;
  int64_t separate_bytes = 0;
  int64_t cluster_bytes = 0;

  // divide into chunks to load
  vector<pair<int64_t, int64_t>> chunk_division;
  for (int start_idx = 0; start_idx < num_points; start_idx += MAX_POINTS_PER_BATCH) {
    int64_t num_points_in_chunk = min((int64_t) MAX_POINTS_PER_BATCH, num_points - start_idx);
    chunk_division.push_back({start_idx, num_points_in_chunk});

    vector<pair<int,int>> batch_parameters = get_batch_parameters(num_points_in_chunk);
    num_batches += batch_parameters.size();
  }

  cout << "Number of Points: " << num_points << endl;
  cout << "Number of Batches: " << num_batches << endl;

  // declare the dump buffer
  int64_t header_num_ints = 5 + num_batches;
  vector<char> dump_buffer(header_num_ints * 8, 0);
  dump_buffer.reserve(num_points * 10);
  int64_t offset = 5 * 8;

  vector<float> compression_ratios;
  int64_t new_num_points = 0;
  for (int64_t cid = 0; cid < chunk_division.size(); ++cid) {
    auto &[start_idx, num_points_in_chunk] = chunk_division[cid];
    Chunk chunk = process_chunk(LASFILE, start_idx, num_points_in_chunk, should_sort);
    new_num_points += chunk.num_points;
    compression_ratios.insert(compression_ratios.end(), chunk.compression_ratios.begin(), chunk.compression_ratios.end());

    encoding_bytes += chunk.encoding_bytes;
    separate_bytes += chunk.separate_bytes;
    cluster_bytes += chunk.cluster_bytes;

    dump_buffer.insert(dump_buffer.end(), chunk.dump_buffer.begin(), chunk.dump_buffer.end());
    memcpy(dump_buffer.data() + offset, chunk.batch_data_sizes.data(), chunk.batch_data_sizes.size() * 8);
    offset += chunk.batch_data_sizes.size() * 8;
    cout << cid << " " << chunk_division.size() << endl;
  }

  memcpy(dump_buffer.data() + 0, &new_num_points, 8);
  memcpy(dump_buffer.data() + 8, &num_batches, 8);
  memcpy(dump_buffer.data() + 16, &encoding_bytes, 8);
  memcpy(dump_buffer.data() + 24, &separate_bytes, 8);
  memcpy(dump_buffer.data() + 32, &cluster_bytes, 8);
  writeBinaryFile(OUTFILE, dump_buffer);

  cout << num_points << " " << num_batches << endl;

  cout << "Compression Ratios: " << endl;
  for (auto &cr : compression_ratios) cout << cr << endl;

  return 0;
}
