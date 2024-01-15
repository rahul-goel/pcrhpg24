#pragma once

#include <string>
#include "GLBuffer.h"
#include "unsuck.hpp"
#include "Resources.h"

struct HuffmanLasData : public Resource {
  // loader task
  struct LoaderTask {
    shared_ptr<Buffer> buffer = nullptr;
    int64_t batchIdx = 0;
  };

  string path = "";
  shared_ptr<LoaderTask> task = nullptr;

  // LAS header data
  bool      headerLoaded                = false;
  int64_t   numBatches                  = 0;
  int64_t   numPoints                   = 0;
  int64_t   encodedBytes                = 0;
  int64_t   separateBytes               = 0;
  int64_t   clusterBytes                = 0;

  vector<int64_t> batch_data_sizes;
  vector<int64_t> batch_data_sizes_prefix;

  int64_t   numBatchesLoaded            = 0;
  int64_t   numPointsLoaded             = 0;
  int64_t   offsetToBatchData           = 0;

  // GL Buffers
  CuBuffer BatchData;                   // stores data common to all points in a batch
  CuBuffer StartValues;                 // starting points for each thread/chain
  CuBuffer EncodedData;                 // the huffman encoded bitstream
  CuBuffer SeparateData;                // separate (unencoded) data
  CuBuffer SeparateDataSizes;           // each thread asks -> how big is my separate data?
  CuBuffer DecoderTableValues;          // decoder table: index -> symbol
  CuBuffer DecoderTableCWLen;           // decoder table: index -> codeword length
  CuBuffer ClusterSizes;                // the number of points in each uvec4
  CuBuffer Colors;                      // list of colors for all points

  // From where should I start filling the GL Buffers?
  size_t EncodedPtr = 0;
  size_t SeparatePtr = 0;
  size_t DecoderTablePtr = 0;
  size_t ClusterSizesPtr = 0;

  HuffmanLasData() {}

  void loadHeader() {
    // read the number of batches
    auto file_offset = 0;
    auto firstWordBuffer = readBinaryFile(this->path, file_offset, 5 * 8);
    this->numPoints = firstWordBuffer->get<int64_t>(0);
    this->numBatches = firstWordBuffer->get<int64_t>(8);
    this->encodedBytes = firstWordBuffer->get<int64_t>(16);
    this->separateBytes = firstWordBuffer->get<int64_t>(24);
    this->clusterBytes = firstWordBuffer->get<int64_t>(32);
    file_offset += 5 * 8;
    
    // read the batch chunk sizes
    auto batch_data_sizes_buffer = readBinaryFile(this->path, file_offset, 8 * this->numBatches);
    this->batch_data_sizes.resize(this->numBatches);
    for (int i = 0; i < this->numBatches; ++i) {
      this->batch_data_sizes[i] = batch_data_sizes_buffer->get<int64_t>(8 * i);
    }
    file_offset += 8 * this->numBatches;

    // calculate the prefix sum of batch chunk sizes
    this->batch_data_sizes_prefix = batch_data_sizes;
    for (int i = 1; i < this->numBatches; ++i) {
      this->batch_data_sizes_prefix[i] += this->batch_data_sizes_prefix[i - 1];
    }

    // header loading complete
		this->headerLoaded = true;
    this->offsetToBatchData = file_offset;
  }

  static shared_ptr<HuffmanLasData> create(string path) {
    auto data = make_shared<HuffmanLasData>();
    data->path = path;
    data->loadHeader();
    return data;
  }

  void load(Renderer * renderer);
  void unload(Renderer * renderer);
  void process(Renderer * renderer);
};
