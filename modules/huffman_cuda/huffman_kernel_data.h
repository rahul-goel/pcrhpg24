#pragma once

// this must be a multiple of 16
#define GPUBatchSize 112
struct GPUBatch {
  // bounding box -> 24 bytes
  float min_x;
  float min_y;
  float min_z;
  float max_x;
  float max_y;
  float max_z;

  // scale and offset -> 24 bytes
  float scale_x;
  float scale_y;
  float scale_z;
  float offset_x;
  float offset_y;
  float offset_z;

  // las min and las max -> 32 bytes
  float las_min_x;
  float las_min_y;
  float las_min_z;
  float las_max_x;
  float las_max_y;
  float las_max_z;
  float padding1;
  float pading2;

  // batch-level encoding and separate offset -> 24 bytes
  long long encoding_batch_offset;
  long long separate_batch_offset;
  long long decoder_table_offset;

  // store as long long for now -> 8 bytes
  long long max_cw_len;
};
