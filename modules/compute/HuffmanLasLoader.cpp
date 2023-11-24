#include <thread>
#include <mutex>
#include <cassert>

#include "compute/Resources.h"
#include "compute/HuffmanLasLoader.h"
#include "huffman_cuda/huffman_kernel_data.h"

#include "BatchDumpData.h"
#include "Renderer.h"
#include "GLTimerQueries.h"
#include "huffman.h"

using namespace std;

mutex huffman_mtx_state;

void HuffmanLasData::load(Renderer *renderer) {
  cout << "HuffmanLasData::load()" << endl;
  {
    lock_guard<mutex> lock(huffman_mtx_state);
    if (state != ResourceState::UNLOADED)
      return;
    else
      state = ResourceState::LOADING;
  }

  { // create buffers
    this->BatchData             = renderer->createBuffer(GPUBatchSize * numBatches);
    this->StartValues           = renderer->createBuffer(WORKGROUP_SIZE * numBatches * 3 * 4);
    this->EncodedData           = renderer->createBuffer(this->encodedBytes);
    this->EncodedDataOffsets    = renderer->createBuffer(WORKGROUP_SIZE * numBatches * 4);
    this->EncodedDataSizes      = renderer->createBuffer(WORKGROUP_SIZE * numBatches * 4);
    this->SeparateData          = renderer->createBuffer(this->separateBytes);
    this->SeparateDataOffsets   = renderer->createBuffer(WORKGROUP_SIZE * numBatches * 4);
    this->SeparateDataSizes     = renderer->createBuffer(WORKGROUP_SIZE * numBatches * 4);
    this->Colors                = renderer->createBuffer(this->numPoints * 4);
    this->DecoderTableValues    = renderer->createBuffer(numBatches * HUFFMAN_TABLE_SIZE * 4);
    this->DecoderTableCWLen     = renderer->createBuffer(numBatches * HUFFMAN_TABLE_SIZE * 4);
    this->ClusterSizes          = renderer->createBuffer(this->clusterBytes);

    GLuint zero = 0;
		glClearNamedBufferData(this->BatchData.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->StartValues.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->EncodedData.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->EncodedDataOffsets.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->EncodedDataSizes.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->SeparateData.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->SeparateDataOffsets.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->SeparateDataSizes.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->Colors.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->DecoderTableValues.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->DecoderTableCWLen.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->ClusterSizes.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
  }

  // start loader thread and detach it immediately
  HuffmanLasData *ref = this;
  thread t([ref] () {
    int batchesRemaining = ref->numBatches;
    int batchesRead = 0;

    while (batchesRemaining > 0) {
      { // abort loader thread if state is set to unloading
        lock_guard<mutex> lock(huffman_mtx_state);
        if (ref->state == ResourceState::UNLOADING) {
          cout << "stopping loader thread for " << ref->path << endl;
          ref->state = ResourceState::UNLOADED;
          return;
        }
      }

      // poll to check if the currently read batch has been processed or not
      if (ref->task) {
        this_thread::sleep_for(1ms);
        continue;
      }

      // read the next batch of points
      int64_t start = ref->offsetToBatchData;
      if (batchesRead > 0)
        start += ref->batch_data_sizes_prefix[batchesRead - 1];
      int64_t size = ref->batch_data_sizes[batchesRead];
      auto buffer = readBinaryFile(ref->path, start, size);

      auto task = make_shared<LoaderTask>();
      task->buffer = buffer;
      task->batchIdx = batchesRead;
      
      ref->task = task;
      batchesRemaining -= 1;
      batchesRead += 1;

      Debug::set("numBatchesRead", formatNumber(batchesRead));
      Debug::set("batchesRemaining", formatNumber(batchesRemaining));
      // cout << "Read " << batchesRead << " batches" << endl;
    }

    { // check if resource was marked as unloading in the meantime
      lock_guard<mutex> lock(huffman_mtx_state);

      if (ref->state == ResourceState::UNLOADING) {
        cout << "stopping loader thread for " << ref->path << endl;
        ref->state = ResourceState::UNLOADED;
      } else if (ref->state == ResourceState::LOADING) {
        ref->state = ResourceState::LOADED;
      }
    }
  });
  t.detach();
}

void HuffmanLasData::unload(Renderer *renderer) {
  cout << "HuffmanLasData::unload()" << endl;
  this->numBatchesLoaded = 0;

  // delete buffers
  glDeleteBuffers(1, &BatchData.handle);
  glDeleteBuffers(1, &StartValues.handle);
  glDeleteBuffers(1, &EncodedData.handle);
  glDeleteBuffers(1, &EncodedDataOffsets.handle);
  glDeleteBuffers(1, &EncodedDataSizes.handle);
  glDeleteBuffers(1, &SeparateData.handle);
  glDeleteBuffers(1, &SeparateDataOffsets.handle);
  glDeleteBuffers(1, &SeparateDataSizes.handle);
  glDeleteBuffers(1, &Colors.handle);
  glDeleteBuffers(1, &DecoderTableValues.handle);
  glDeleteBuffers(1, &DecoderTableCWLen.handle);
  glDeleteBuffers(1, &ClusterSizes.handle);

  lock_guard<mutex> lock(huffman_mtx_state);
  if (state == ResourceState::LOADED) {
    state = ResourceState::UNLOADED;
  } else if (state == ResourceState::LOADING) {
		// if loader thread is still running, notify thread by marking resource as "unloading"
    state = ResourceState::UNLOADING;
  }
}

void HuffmanLasData::process(Renderer *renderer) {
  if (this->task) {
    BatchDumpData bdd;
    bdd.read_buffer(*this->task->buffer);
    int64_t numPointsInBatch = bdd.num_points;
    // cout << "Encoded Sizes: ";
    // for (auto &x : bdd.encoding_sizes) cout << x << " ";
    // cout << endl;
    // cout << "Separate Sizes: ";
    // for (auto &x : bdd.separate_sizes) cout << x << " ";
    // cout << endl;

    // arrange batch data
    GPUBatch batch;
    batch.min_x = bdd.bbox_min[0];
    batch.min_y = bdd.bbox_min[1];
    batch.min_z = bdd.bbox_min[2];
    batch.max_x = bdd.bbox_max[0];
    batch.max_y = bdd.bbox_max[1];
    batch.max_z = bdd.bbox_max[2];
    batch.scale_x = bdd.las_scale[0];
    batch.scale_y = bdd.las_scale[1];
    batch.scale_z = bdd.las_scale[2];
    batch.offset_x = bdd.las_offset[0];
    batch.offset_y = bdd.las_offset[1];
    batch.offset_z = bdd.las_offset[2];
    batch.las_min_x = bdd.las_min[0];
    batch.las_min_y = bdd.las_min[1];
    batch.las_min_z = bdd.las_min[2];
    batch.las_max_x = bdd.las_max[0];
    batch.las_max_y = bdd.las_max[1];
    batch.las_max_z = bdd.las_max[2];
    batch.encoding_batch_offset = this->EncodedPtr;
    batch.separate_batch_offset = this->SeparatePtr;
    batch.decoder_table_offset = this->DecoderTablePtr;
    batch.cluster_sizes_offset = this->ClusterSizesPtr;
    batch.max_cw_len = (long long) log2(bdd.dt_size);

    { // printing for debugging
      /*
      cout << endl;
      cout << this->task->batchIdx << endl;
      cout << batch.min_x << " " << batch.min_y << " " << batch.min_z << endl;
      cout << batch.max_x << " " << batch.max_y << " " << batch.max_z << endl;
      cout << batch.scale_x << " " << batch.scale_y << " " << batch.scale_z << endl;
      cout << batch.offset_x << " " << batch.offset_y << " " << batch.offset_z << endl;
      */
    }

    // write to the GPU
    {
      // GPUBatch
      size_t offset = GPUBatchSize * this->task->batchIdx;
      size_t size = GPUBatchSize;
      glNamedBufferSubData(this->BatchData.handle, offset, size, &batch);
    }
    { // decoder table
      auto &arr1 = bdd.decoder_values;
      size_t offset1 = DecoderTablePtr * sizeof(arr1[0]);
      size_t size1 = arr1.size() * sizeof(arr1[0]);
      glNamedBufferSubData(this->DecoderTableValues.handle, offset1, size1, arr1.data());

      auto &arr2 = bdd.decoder_cw_len;
      size_t offset2 = DecoderTablePtr * sizeof(arr2[0]);
      size_t size2 = arr2.size() * sizeof(arr2[0]);
      glNamedBufferSubData(this->DecoderTableCWLen.handle, offset2, size2, arr2.data());

      assert(arr1.size() == arr2.size());
      DecoderTablePtr += arr1.size();
    }
    { // cluster sizes
      auto &arr = bdd.cluster_sizes;
      size_t offset = ClusterSizesPtr * sizeof(arr[0]);
      size_t size = arr.size() * sizeof(arr[0]);
      glNamedBufferSubData(this->ClusterSizes.handle, offset, size, arr.data());
      ClusterSizesPtr += arr.size();
    }
    { // start values
      auto &arr = bdd.start_values;
      size_t offset = this->task->batchIdx * WORKGROUP_SIZE * 3 * sizeof(arr[0]);
      size_t size = arr.size() * sizeof(arr[0]);
      glNamedBufferSubData(this->StartValues.handle, offset, size, arr.data());
    }
    { // encoding
      auto &arr = bdd.encoding;
      size_t offset = EncodedPtr * sizeof(arr[0]);
      size_t size = arr.size() * sizeof(arr[0]);
      glNamedBufferSubData(this->EncodedData.handle, offset, size, arr.data());
      EncodedPtr += arr.size();
    }
    { // encoding_offsets
      auto &arr = bdd.encoding_offsets;
      size_t offset = this->task->batchIdx * WORKGROUP_SIZE * sizeof(arr[0]);
      size_t size = arr.size() * sizeof(arr[0]);
      glNamedBufferSubData(this->EncodedDataOffsets.handle, offset, size, arr.data());
    }
    { // encoding_sizes
      auto &arr = bdd.encoding_sizes;
      size_t offset = this->task->batchIdx * WORKGROUP_SIZE * sizeof(arr[0]);
      size_t size = arr.size() * sizeof(arr[0]);
      glNamedBufferSubData(this->EncodedDataSizes.handle, offset, size, arr.data());
    }
    { // separate
      auto &arr = bdd.separate;
      size_t offset = SeparatePtr * sizeof(arr[0]);
      size_t size = arr.size() * sizeof(arr[0]);
      glNamedBufferSubData(this->SeparateData.handle, offset, size, arr.data());
      SeparatePtr += arr.size();
      cout << arr.size() << endl;
    }
    { // separate_offsets
      auto &arr = bdd.separate_offsets;
      size_t offset = this->task->batchIdx * WORKGROUP_SIZE * sizeof(arr[0]);
      size_t size = arr.size() * sizeof(arr[0]);
      glNamedBufferSubData(this->SeparateDataOffsets.handle, offset, size, arr.data());
    }
    { // separate_sizes
      auto &arr = bdd.separate_sizes;
      size_t offset = this->task->batchIdx * WORKGROUP_SIZE * sizeof(arr[0]);
      size_t size = arr.size() * sizeof(arr[0]);
      glNamedBufferSubData(this->SeparateDataSizes.handle, offset, size, arr.data());
    }
    { // color
      auto &arr = bdd.color;
      size_t offset = this->task->batchIdx * POINTS_PER_WORKGROUP * sizeof(arr[0]);
      size_t size = arr.size() * sizeof(arr[0]);
      glNamedBufferSubData(this->Colors.handle, offset, size, arr.data());
    }

    // reset the task pointer so that the reader thread can fill it
    // cout << "Processed " << this->numBatchesLoaded + 1 << " Batches" << endl;
    this->numBatchesLoaded += 1;
    this->numPointsLoaded += numPointsInBatch;
    this->task = nullptr;
    Debug::set("numBatchesLoaded", formatNumber(this->numBatchesLoaded));
    Debug::set("numPointsLoaded", formatNumber(this->numPointsLoaded));
  }
}
