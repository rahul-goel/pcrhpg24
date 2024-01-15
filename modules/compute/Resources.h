
#pragma once

#define POINTS_PER_THREAD 64
#define CLUSTERS_PER_THREAD 1
#define WORKGROUP_SIZE 1024

#define POINTS_PER_WORKGROUP (POINTS_PER_THREAD * WORKGROUP_SIZE)
// Adjust this to be something in the order of 1 million points
#define MAX_POINTS_PER_BATCH (100 * POINTS_PER_WORKGROUP)

#define HUFFMAN_LEAF_COUNT 128
#define HUFFMAN_TABLE_SIZE 4096

#define COLOR_COMPRESSION 1 // 0 -> no compression, 1 -> bc1, 7 -> bc7

#include "Renderer.h"
#include "CudaProgram.h"

enum ResourceState { 
	UNLOADED,
	LOADING,
	LOADED,
	UNLOADING
};

struct Resource{

	ResourceState state = ResourceState::UNLOADED;

	virtual void load(Renderer* renderer) = 0;
	virtual void unload(Renderer* renderer) = 0;
	virtual void process(Renderer* renderer) = 0;

};

struct CuBuffer {
  CUdeviceptr handle = -1;
  int64_t size = 0;
};
