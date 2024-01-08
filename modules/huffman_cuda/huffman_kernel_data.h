#pragma once

// this must be a multiple of 16
#define GPUBatchSize 160
struct GPUBatch {
  // bounding box -> 24 bytes
  float min_x;
  float min_y;
  float min_z;
  float max_x;
  float max_y;
  float max_z;

  // scale and offset -> 48 bytes
  double scale_x;
  double scale_y;
  double scale_z;
  double offset_x;
  double offset_y;
  double offset_z;

  // las min and las max -> 48 bytes
  double las_min_x;
  double las_min_y;
  double las_min_z;
  double las_max_x;
  double las_max_y;
  double las_max_z;

  // batch-level encoding and separate offset -> 32 bytes
  long long encoding_batch_offset;
  long long separate_batch_offset;
  long long decoder_table_offset;
  long long cluster_sizes_offset;

  // store as long long for now -> 8 bytes
  long long max_cw_len;
};
