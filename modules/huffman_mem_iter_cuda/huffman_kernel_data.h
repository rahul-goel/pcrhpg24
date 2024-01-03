#pragma once

// this must be a multiple of 16
#define GPUBatchSize 144
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

  // las min and las max -> 24 bytes
  float las_min_x;
  float las_min_y;
  float las_min_z;
  float las_max_x;
  float las_max_y;
  float las_max_z;

  // batch-level encoding and separate offset -> 40 bytes
  long long encoding_batch_offset;
  long long separate_batch_offset;
  long long decoder_table_offset;
  long long cluster_sizes_offset;
  long long longpadding1;

  // store as long long for now -> 8 bytes
  long long max_cw_len;
};
