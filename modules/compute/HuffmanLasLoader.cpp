#include <thread>
#include <mutex>
#include <cassert>

#include "compute/Resources.h"
#include "compute/HuffmanLasLoader.h"
#include "huffman_cuda/huffman_kernel_data.h"

#include "CudaProgram.h"
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
    this->BatchData.size         = GPUBatchSize * numBatches;
    cuMemAlloc(&this->BatchData.handle, this->BatchData.size);

    this->StartValues.size       = WORKGROUP_SIZE * CLUSTERS_PER_THREAD * numBatches * 3 * 4;
    cuMemAlloc(&this->StartValues.handle, this->StartValues.size);

    // additional small buffer to avoid wrong memory access for the last batch during read of "NextHuffman"
    this->EncodedData.size       = this->encodedBytes + 4 * WORKGROUP_SIZE * CLUSTERS_PER_THREAD;
    cuMemAlloc(&this->EncodedData.handle, this->EncodedData.size);

    this->SeparateData.size      = this->separateBytes;
    cuMemAlloc(&this->SeparateData.handle, this->SeparateData.size);

    this->SeparateDataSizes.size = WORKGROUP_SIZE * CLUSTERS_PER_THREAD * numBatches * 4;
    cuMemAlloc(&this->SeparateDataSizes.handle, this->SeparateDataSizes.size);

#if COLOR_COMPRESSION==0
    this->Colors.size            = this->numPoints * 4;
    cuMemAlloc(&this->Colors.handle, this->Colors.size);
#elif COLOR_COMPRESSION==1
    this->Colors.size            = this->numPoints / 2;
    cuMemAlloc(&this->Colors.handle, this->Colors.size);
#elif COLOR_COMPRESSION==7
    this->Colors.size            = this->numPoints;
    cuMemAlloc(&this->Colors.handle, this->Colors.size);
#endif

    this->DecoderTableValues.size = numBatches * HUFFMAN_TABLE_SIZE * 4;
    cuMemAlloc(&this->DecoderTableValues.handle, this->DecoderTableValues.size);
    this->DecoderTableCWLen.size = numBatches * HUFFMAN_TABLE_SIZE * 4;
    cuMemAlloc(&this->DecoderTableCWLen.handle, this->DecoderTableCWLen.size);
    this->ClusterSizes.size = this->clusterBytes;
    cuMemAlloc(&this->ClusterSizes.handle, this->ClusterSizes.size);

    GLuint zero = 0;
		cuMemsetD8(this->BatchData.handle, 0, this->BatchData.size);
		cuMemsetD8(this->StartValues.handle, 0, this->StartValues.size);
    cuMemsetD8(this->EncodedData.handle, 0, this->EncodedData.size);
    cuMemsetD8(this->SeparateData.handle, 0, this->SeparateData.size);
    cuMemsetD8(this->SeparateDataSizes.handle, 0, this->SeparateDataSizes.size);
    cuMemsetD8(this->Colors.handle, 0, this->Colors.size);
    cuMemsetD8(this->DecoderTableValues.handle, 0, this->DecoderTableValues.size);
    cuMemsetD8(this->DecoderTableCWLen.handle, 0, this->DecoderTableCWLen.size);
    cuMemsetD8(this->ClusterSizes.handle, 0, this->ClusterSizes.size);
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
        this_thread::sleep_for(0.1ms);
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
  cuMemFree(BatchData.handle);
  cuMemFree(StartValues.handle);
  cuMemFree(EncodedData.handle);
  cuMemFree(SeparateData.handle);
  cuMemFree(SeparateDataSizes.handle);
  cuMemFree(Colors.handle);
  cuMemFree(DecoderTableValues.handle);
  cuMemFree(DecoderTableCWLen.handle);
  cuMemFree(ClusterSizes.handle);

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
      cuMemcpyHtoD(this->BatchData.handle + offset, &batch, size);
    }
    { // decoder table
      auto &arr1 = bdd.decoder_values;
      size_t offset1 = DecoderTablePtr * sizeof(arr1[0]);
      size_t size1 = arr1.size() * sizeof(arr1[0]);
      cuMemcpyHtoD(this->DecoderTableValues.handle + offset1, arr1.data(), size1);

      auto &arr2 = bdd.decoder_cw_len;
      size_t offset2 = DecoderTablePtr * sizeof(arr2[0]);
      size_t size2 = arr2.size() * sizeof(arr2[0]);
      cuMemcpyHtoD(this->DecoderTableCWLen.handle + offset2, arr2.data(), size2);

      assert(arr1.size() == arr2.size());
      DecoderTablePtr += arr1.size();
    }
    { // cluster sizes
      auto &arr = bdd.cluster_sizes;
      size_t offset = ClusterSizesPtr * sizeof(arr[0]);
      size_t size = arr.size() * sizeof(arr[0]);
      // glNamedBufferSubData(this->ClusterSizes.handle, offset, size, arr.data());
      cuMemcpyHtoD(this->ClusterSizes.handle + offset, arr.data(), size);
      ClusterSizesPtr += arr.size();
    }
    { // start values
      auto &arr = bdd.start_values;
      size_t offset = this->task->batchIdx * WORKGROUP_SIZE * CLUSTERS_PER_THREAD * 3 * sizeof(arr[0]);
      size_t size = arr.size() * sizeof(arr[0]);
      cuMemcpyHtoD(this->StartValues.handle + offset, arr.data(), size);
    }
    { // encoding
      auto &arr = bdd.encoding;
      size_t offset = EncodedPtr * sizeof(arr[0]);
      size_t size = arr.size() * sizeof(arr[0]);
      cuMemcpyHtoD(this->EncodedData.handle + offset, arr.data(), size);
      EncodedPtr += arr.size();
    }
    { // separate
      auto &arr = bdd.separate;
      size_t offset = SeparatePtr * sizeof(arr[0]);
      size_t size = arr.size() * sizeof(arr[0]);
      cuMemcpyHtoD(this->SeparateData.handle + offset, arr.data(), size);
      SeparatePtr += arr.size();
    }
    { // separate_sizes
      auto &arr = bdd.separate_sizes;
      size_t offset = this->task->batchIdx * WORKGROUP_SIZE * CLUSTERS_PER_THREAD * sizeof(arr[0]);
      size_t size = arr.size() * sizeof(arr[0]);
      cuMemcpyHtoD(this->SeparateDataSizes.handle + offset, arr.data(), size);
    }
    { // color
      auto &arr = bdd.color;
#if COLOR_COMPRESSION==0
      size_t offset = this->task->batchIdx * POINTS_PER_WORKGROUP * sizeof(arr[0]);
#elif COLOR_COMPRESSION==1
      size_t offset = (this->task->batchIdx * POINTS_PER_WORKGROUP * sizeof(arr[0])) / 8;
#elif COLOR_COMPRESSION==7
      size_t offset = (this->task->batchIdx * POINTS_PER_WORKGROUP * sizeof(arr[0])) / 4;
#endif
      size_t size = arr.size() * sizeof(arr[0]);
      cuMemcpyHtoD(this->Colors.handle + offset, arr.data(), size);
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
