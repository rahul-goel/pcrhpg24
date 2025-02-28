#include <vector>
#include <algorithm>
#include <cassert>
#include <numeric>

#include "compute/Resources.h"

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include "unsuck.hpp"

using namespace std;

// relevant data per batch
struct BatchDumpData {
  int32_t point_offset;
  int32_t num_points;

  int32_t num_threads;
  int32_t points_per_thread;
  int32_t clusters_per_thread;

  double las_scale[3];
  double las_offset[3];

  float bbox_min[3];
  float bbox_max[3];

  float las_min[3];
  float las_max[3];

  int32_t dt_size;
  int32_t num_clusters;

  vector<int32_t> start_values;

  vector<uint32_t> encoding;
  vector<int32_t> separate_sizes;
  vector<int32_t>  separate;

  vector<uint32_t> color;

  vector<int32_t> decoder_values;
  vector<int32_t> decoder_cw_len;
  vector<int32_t> cluster_sizes;

  size_t get_total_size() {
    size_t total_size = 4 * 19 + 8 * 6;
    total_size += start_values.size()             * 4;
    total_size += decoder_values.size()           * 4;
    total_size += decoder_cw_len.size()           * 4;
    total_size += cluster_sizes.size()            * 4;
    total_size += encoding.size()                 * 4;
    total_size += separate_sizes.size()           * 4;
    total_size += separate.size()                 * 4;
    total_size += color.size()                    * 4;
    return total_size;
  }

  void read_buffer(Buffer &buffer) {
    size_t offset = 0;
    auto buf_ptr = buffer.data_u8;

    // Which point do I start at?
    memcpy(&point_offset, buf_ptr + offset, 4);
    offset += 4;
    // How many points do I have?
    memcpy(&num_points, buf_ptr + offset, 4);
    offset += 4;
    // Into how many threads is my data stripped?
    memcpy(&num_threads, buf_ptr + offset, 4);
    offset += 4;
    // How many points do each of the threads have?
    memcpy(&points_per_thread, buf_ptr + offset, 4);
    offset += 4;
    // How many clusters per thread?
    memcpy(&clusters_per_thread, buf_ptr + offset, 4);
    offset += 4;
    // What is the scale parameter in the original LAS header? -> 3 values
    memcpy(&las_scale, buf_ptr + offset, 8 * 3);
    offset += 8 * 3;
    // What is the offset parameter in the original LAS header? -> 3 values
    memcpy(&las_offset, buf_ptr + offset, 8 * 3);
    offset += 8 * 3;
    // What is the minimum value among the points that I have? -> 3 values
    memcpy(&bbox_min, buf_ptr + offset, 4 * 3);
    offset += 4 * 3;
    // What is the maximum value among the points that I have? -> 3 values
    memcpy(&bbox_max, buf_ptr + offset, 4 * 3);
    offset += 4 * 3;
    // What is the minimum value among all the points in the LAS file (as stated
    // in the LAS header)? -> 3 values
    memcpy(&las_min, buf_ptr + offset, 4 * 3);
    offset += 4 * 3;
    // What is the maximum value among all the points in the LAS file (as stated
    // in the LAS header)? -> 3 values
    memcpy(&las_max, buf_ptr + offset, 4 * 3);
    offset += 4 * 3;
    // What is the size of the decoder table? (Should be a power of 2.)
    memcpy(&dt_size, buf_ptr + offset, 4);
    offset += 4;
    // How many clusters are there for this batch?
    memcpy(&num_clusters, buf_ptr + offset, 4);
    offset += 4;

    start_values.resize(num_threads * clusters_per_thread * 3);
    separate_sizes.resize(num_threads * clusters_per_thread);
    decoder_values.resize(dt_size);
    decoder_cw_len.resize(dt_size);
    cluster_sizes.resize(num_clusters);

    // What are the 3 start values for each of the individual strip/chain?
    memcpy(start_values.data(), buf_ptr + offset, start_values.size() * 4);
    offset += 4 * start_values.size();
    // For each strip/chain, what is the size of it's Non-Huffman Encoded Data stream?
    memcpy(separate_sizes.data(), buf_ptr + offset, separate_sizes.size() * 4);
    offset += 4 * separate_sizes.size();
    // The first column of the Decoder Table (symbols).
    memcpy(decoder_values.data(), buf_ptr + offset, decoder_values.size() * 4);
    offset += 4 * decoder_values.size();
    // The second column of the Decoder Table (length of codewords).
    memcpy(decoder_cw_len.data(), buf_ptr + offset, decoder_cw_len.size() * 4);
    offset += 4 * decoder_cw_len.size();
    // The cluster sizes. Each cluster tells how many points in uvec4.
    memcpy(cluster_sizes.data(), buf_ptr + offset, cluster_sizes.size() * 4);
    offset += 4 * cluster_sizes.size();

    encoding.resize(cluster_sizes.back());
    separate.resize(separate_sizes.back());
#if COLOR_COMPRESSION==0
    color.resize(num_points);
#elif COLOR_COMPRESSION==1
    color.resize(num_points / 8);
#elif COLOR_COMPRESSION==7
    color.resize(num_points / 4);
#endif

    // Huffman Encoded Data
    memcpy(encoding.data(), buf_ptr + offset, encoding.size() * 4);
    offset += 4 * encoding.size();
    // Non-Huffman Encoded Data / Separate Data
    memcpy(separate.data(), buf_ptr + offset, separate.size() * 4);
    offset += 4 * separate.size();
    // Color Values
    memcpy(color.data(), buf_ptr + offset, color.size() * 4);
    offset += 4 * color.size();

    assert(offset == buffer.size);
  }

  vector<char> get_buffer() {
    size_t total_size = get_total_size();
    vector<char> buffer(total_size);
    size_t offset = 0;

    memcpy(buffer.data() + offset, &point_offset, 4);
    offset += 4;
    memcpy(buffer.data() + offset, &num_points, 4);
    offset += 4;
    memcpy(buffer.data() + offset, &num_threads, 4);
    offset += 4;
    memcpy(buffer.data() + offset, &points_per_thread, 4);
    offset += 4;
    memcpy(buffer.data() + offset, &clusters_per_thread, 4);
    offset += 4;
    memcpy(buffer.data() + offset, &las_scale, 8 * 3);
    offset += 8 * 3;
    memcpy(buffer.data() + offset, &las_offset, 8 * 3);
    offset += 8 * 3;
    memcpy(buffer.data() + offset, &bbox_min, 4 * 3);
    offset += 4 * 3;
    memcpy(buffer.data() + offset, &bbox_max, 4 * 3);
    offset += 4 * 3;
    memcpy(buffer.data() + offset, &las_min, 4 * 3);
    offset += 4 * 3;
    memcpy(buffer.data() + offset, &las_max, 4 * 3);
    offset += 4 * 3;
    memcpy(buffer.data() + offset, &dt_size, 4);
    offset += 4;
    memcpy(buffer.data() + offset, &num_clusters, 4);
    offset += 4;

    memcpy(buffer.data() + offset, start_values.data(), start_values.size() * 4);
    offset += 4 * start_values.size();
    memcpy(buffer.data() + offset, separate_sizes.data(), separate_sizes.size() * 4);
    offset += 4 * separate_sizes.size();
    memcpy(buffer.data() + offset, decoder_values.data(), decoder_values.size() * 4);
    offset += 4 * decoder_values.size();
    memcpy(buffer.data() + offset, decoder_cw_len.data(), decoder_cw_len.size() * 4);
    offset += 4 * decoder_cw_len.size();
    memcpy(buffer.data() + offset, cluster_sizes.data(), cluster_sizes.size() * 4);
    offset += 4 * cluster_sizes.size();

    memcpy(buffer.data() + offset, encoding.data(), encoding.size() * 4);
    offset += 4 * encoding.size();
    memcpy(buffer.data() + offset, separate.data(), separate.size() * 4);
    offset += 4 * separate.size();
    memcpy(buffer.data() + offset, color.data(), color.size() * 4);
    offset += 4 * color.size();

    return buffer;
  }

  char *get_buffer_ptr() {
    size_t total_size = get_total_size();
    char *buffer = (char *) malloc(total_size);
    size_t offset = 0;

    memcpy(buffer + offset, &point_offset, 4);
    offset += 4;
    memcpy(buffer + offset, &num_points, 4);
    offset += 4;
    memcpy(buffer + offset, &num_threads, 4);
    offset += 4;
    memcpy(buffer + offset, &points_per_thread, 4);
    offset += 4;
    memcpy(buffer + offset, &clusters_per_thread, 4);
    offset += 4;
    memcpy(buffer + offset, &las_scale, 8 * 3);
    offset += 8 * 3;
    memcpy(buffer + offset, &las_offset, 8 * 3);
    offset += 8 * 3;
    memcpy(buffer + offset, &bbox_min, 4 * 3);
    offset += 4 * 3;
    memcpy(buffer + offset, &bbox_max, 4 * 3);
    offset += 4 * 3;
    memcpy(buffer + offset, &las_min, 4 * 3);
    offset += 4 * 3;
    memcpy(buffer + offset, &las_max, 4 * 3);
    offset += 4 * 3;
    memcpy(buffer + offset, &dt_size, 4);
    offset += 4;
    memcpy(buffer + offset, &num_clusters, 4);
    offset += 4;

    memcpy(buffer + offset, start_values.data(), start_values.size() * 4);
    offset += 4 * start_values.size();
    memcpy(buffer + offset, separate_sizes.data(), separate_sizes.size() * 4);
    offset += 4 * separate_sizes.size();
    memcpy(buffer + offset, decoder_values.data(), decoder_values.size() * 4);
    offset += 4 * decoder_values.size();
    memcpy(buffer + offset, decoder_cw_len.data(), decoder_cw_len.size() * 4);
    offset += 4 * decoder_cw_len.size();
    memcpy(buffer + offset, cluster_sizes.data(), cluster_sizes.size() * 4);
    offset += 4 * cluster_sizes.size();

    memcpy(buffer + offset, encoding.data(), encoding.size() * 4);
    offset += 4 * encoding.size();
    memcpy(buffer + offset, separate.data(), separate.size() * 4);
    offset += 4 * separate.size();
    memcpy(buffer + offset, color.data(), color.size() * 4);
    offset += 4 * color.size();

    return buffer;
  }
};
