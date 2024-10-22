#include <string>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

#include "Runtime.h"
#include "Renderer.h"
#include "CudaProgram.h"
#include "Method.h"
#include "compute/HuffmanLasLoader.h"
#include "cudaGL.h"
#include "GLTimerQueries.h"
#include "builtin_types.h"

#include "huffman_cuda/huffman_kernel_data.h"
#include "compute_loop_las_cuda/kernel_data.h"

#include "iso646.h"

using namespace std;

struct ComputeHuffman : public Method {
  bool registered = false;
  bool mapped = false;

  CudaProgram* resolveProg = nullptr;
  CudaProgram* renderProg  = nullptr;

  // framebuffer - neither an input, nor an output, that's why cuda pointer
	CUdeviceptr fb;

  // inputs and outputs, hence GL interop
  CUgraphicsResource BatchData;
  CUgraphicsResource StartValues;
  CUgraphicsResource EncodedData;
  CUgraphicsResource EncodedDataOffsets;
  CUgraphicsResource EncodedDataSizes;
  CUgraphicsResource SeparateData;
  CUgraphicsResource SeparateDataOffsets;
  CUgraphicsResource SeparateDataSizes;
  CUgraphicsResource DecoderTableValues;
  CUgraphicsResource DecoderTableCWLen;
  CUgraphicsResource Colors;

	CUgraphicsResource output;
  CUevent start, end;

  CUdeviceptr BatchData_ptr;
  CUdeviceptr StartValues_ptr;
  CUdeviceptr EncodedData_ptr;
  CUdeviceptr EncodedDataOffsets_ptr;
  CUdeviceptr EncodedDataSizes_ptr;
  CUdeviceptr SeparateData_ptr;
  CUdeviceptr SeparateDataOffsets_ptr;
  CUdeviceptr SeparateDataSizes_ptr;
  CUdeviceptr DecoderTableValues_ptr;
  CUdeviceptr DecoderTableCWLen_ptr;
  CUdeviceptr Colors_ptr;


  shared_ptr<HuffmanLasData> las = nullptr;
  Renderer *renderer;

  ComputeHuffman(Renderer *renderer, shared_ptr<HuffmanLasData> las) {
    this->name = "huffman_cuda";
    this->description = R"ER01(
      - Decodes Huffman Encoded values on the GPU
    )ER01";
    this->las = las;
    this->group = "none";

		cuMemAlloc(&fb, 8 * 2048 * 2048);
		
		this->renderer = renderer;

		resolveProg = new CudaProgram("./modules/huffman_cuda/resolve.cu");
		renderProg = new CudaProgram("./modules/huffman_cuda/render.cu");
		cuEventCreate(&start, CU_EVENT_DEFAULT);
		cuEventCreate(&end, CU_EVENT_DEFAULT);
  }

	~ComputeHuffman() {
		if (registered) {
      vector<CUgraphicsResource> persistent_resources = {
        BatchData, StartValues,
        EncodedData, EncodedDataOffsets, EncodedDataSizes,
        SeparateData, SeparateDataOffsets, SeparateDataSizes,
        DecoderTableValues, DecoderTableCWLen, Colors
      };

      cuGraphicsUnmapResources(persistent_resources.size(),
                               persistent_resources.data(),
                               ((CUstream)CU_STREAM_DEFAULT));
      cuGraphicsUnregisterResource(BatchData);
      cuGraphicsUnregisterResource(StartValues);
      cuGraphicsUnregisterResource(EncodedData);
      cuGraphicsUnregisterResource(EncodedDataOffsets);
      cuGraphicsUnregisterResource(EncodedDataSizes);
      cuGraphicsUnregisterResource(SeparateData);
      cuGraphicsUnregisterResource(SeparateDataOffsets);
      cuGraphicsUnregisterResource(SeparateDataSizes);
      cuGraphicsUnregisterResource(DecoderTableValues);
      cuGraphicsUnregisterResource(DecoderTableCWLen);
      cuGraphicsUnregisterResource(Colors);
      cuGraphicsUnregisterResource(output);
		}
  }

  void update(Renderer* renderer) {
    if (Runtime::resource != (Resource*)las.get()) {
      if (Runtime::resource != nullptr) {
        Runtime::resource->unload(renderer);
      }
      las->load(renderer);
      Runtime::resource = (Resource*)las.get();
    }
  }

  void render(Renderer* renderer) {
    GLTimerQueries::timestamp("compute-loop-start");

    // process the currently loaded batch using the basic las loader implementation
    las->process(renderer);

    auto fbo = renderer->views[0].framebuffer;
    auto camera = renderer->camera;

    if (renderProg->kernel == nullptr || resolveProg->kernel == nullptr) return;
    if (las->numPointsLoaded == 0) return;
    if (las->numPointsLoaded != las->numPoints) return;

    // register the buffers to be used from cuda
    if (!registered) {
      // register xyz, rgba, batch data to be read only from cuda

      cuGraphicsGLRegisterBuffer(&BatchData,           las->BatchData.handle,           CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
      cuGraphicsGLRegisterBuffer(&StartValues,         las->StartValues.handle,         CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
      cuGraphicsGLRegisterBuffer(&EncodedData,         las->EncodedData.handle,         CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
      cuGraphicsGLRegisterBuffer(&EncodedDataOffsets,  las->EncodedDataOffsets.handle,  CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
      cuGraphicsGLRegisterBuffer(&EncodedDataSizes,    las->EncodedDataSizes.handle,    CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
      cuGraphicsGLRegisterBuffer(&SeparateData,        las->SeparateData.handle,        CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
      cuGraphicsGLRegisterBuffer(&SeparateDataOffsets, las->SeparateDataOffsets.handle, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
      cuGraphicsGLRegisterBuffer(&SeparateDataSizes,   las->SeparateDataSizes.handle,   CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
      cuGraphicsGLRegisterBuffer(&DecoderTableValues,  las->DecoderTableValues.handle,  CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
      cuGraphicsGLRegisterBuffer(&DecoderTableCWLen,   las->DecoderTableCWLen.handle,   CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
      cuGraphicsGLRegisterBuffer(&Colors,              las->Colors.handle,              CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);

      // 2d image that will be re-written again and again by cuda
      cuGraphicsGLRegisterImage(&output, renderer->views[0].framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

      vector<CUgraphicsResource> persistent_resources = {
        BatchData, StartValues,
        EncodedData, EncodedDataOffsets, EncodedDataSizes,
        SeparateData, SeparateDataOffsets, SeparateDataSizes,
        DecoderTableValues, DecoderTableCWLen, Colors
      };
			cuGraphicsMapResources(persistent_resources.size(), persistent_resources.data(), ((CUstream)CU_STREAM_DEFAULT));

      registered = true;
      cout << "Buffers registered" << endl;
    }

    // get pointers for cuda pointers for the common buffers
    if (not mapped or las->numPointsLoaded < las->numPoints) {
      size_t size;
      cuGraphicsResourceGetMappedPointer(&BatchData_ptr, &size, BatchData);
      cuGraphicsResourceGetMappedPointer(&StartValues_ptr, &size, StartValues);
      cuGraphicsResourceGetMappedPointer(&EncodedData_ptr, &size, EncodedData);
      cuGraphicsResourceGetMappedPointer(&EncodedDataOffsets_ptr, &size, EncodedDataOffsets);
      cuGraphicsResourceGetMappedPointer(&EncodedDataSizes_ptr, &size, EncodedDataSizes);
      cuGraphicsResourceGetMappedPointer(&SeparateData_ptr, &size, SeparateData);
      cuGraphicsResourceGetMappedPointer(&SeparateDataOffsets_ptr, &size, SeparateDataOffsets);
      cuGraphicsResourceGetMappedPointer(&SeparateDataSizes_ptr, &size, SeparateDataSizes);
      cuGraphicsResourceGetMappedPointer(&DecoderTableValues_ptr, &size, DecoderTableValues);
      cuGraphicsResourceGetMappedPointer(&DecoderTableCWLen_ptr, &size, DecoderTableCWLen);
      cuGraphicsResourceGetMappedPointer(&Colors_ptr, &size, Colors);
      mapped = true;
    }

    // RENDER
    {
      auto &viewLeft = renderer->views[0];
      glm::mat4 world;
      glm::mat4 view = viewLeft.view;
      glm::mat4 proj = viewLeft.proj;
      glm::mat4 worldView = view * world;
      glm::mat4 viewProj = proj * view;
      glm::mat4 worldViewProj = proj * view * world;

      // stuff that changes because of camera movement, window size change, batch loading, imgui changes
      ChangingRenderData cdata;
      *((glm::mat4 *) &cdata.uTransform) = glm::transpose(worldViewProj);
			*((glm::mat4*)&cdata.uWorldView) = glm::transpose(worldView);
			*((glm::mat4*)&cdata.uProj) = glm::transpose(proj);
			cdata.uCamPos = float3{ (float)camera->position.x, (float)camera->position.y, (float)camera->position.z };
			cdata.uImageSize = int2{ fbo->width, fbo->height };
			cdata.uPointsPerThread = POINTS_PER_THREAD;
			cdata.uBoxMin = float3{0, 0, 0};
			cdata.uBoxMax = float3{0, 0, 0};
			cdata.uNumPoints = las->numPointsLoaded;
			cdata.uOffsetPointToData = 0;
			cdata.uPointFormat = 0;
			cdata.uBytesPerPoint = 0;
			cdata.uScale = float3{0, 0, 0};
			cdata.uBoxMax = float3{0, 0, 0};
			cdata.uEnableFrustumCulling = Debug::frustumCullingEnabled;

			// don't execute a workgroup until all points inside are loaded
			// workgroup only iterates over the source buffer once, 
			// and generates a derivative!
      int numBatches = las->numBatchesLoaded;

      // kernel launch
      void *args[] = {&cdata, &fb,
      &BatchData_ptr, &StartValues_ptr,
      &EncodedData_ptr, &EncodedDataOffsets_ptr, &EncodedDataSizes_ptr,
      &SeparateData_ptr, &SeparateDataOffsets_ptr, &SeparateDataSizes_ptr,
      &DecoderTableValues_ptr, &DecoderTableCWLen_ptr, &Colors_ptr};

      // cout << "launching kernel " << numBatches << " " << WORKGROUP_SIZE << endl;
			CUresult opt = cuLaunchKernel(renderProg->kernel,
				numBatches, 1, 1,
				WORKGROUP_SIZE, 1, 1,
				0, 0, args, 0);
      // cout << "kernel terminated " << opt << endl;
    }

    // RESOLVE
    {
      std::vector<CUgraphicsResource> dynamic_resources = { output };
      cuGraphicsMapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream) CU_STREAM_DEFAULT));

      CUDA_RESOURCE_DESC res_desc;
      res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
      cuGraphicsSubResourceGetMappedArray(&res_desc.res.array.hArray, output, 0, 0);
      CUsurfObject output_surf;
			cuSurfObjectCreate(&output_surf, &res_desc);

      int groups_x = fbo->width / 16;
      int groups_y = fbo->height / 16;
      void *args[] = { &fbo->width, &fbo->height, &output_surf, &fb, &Colors_ptr };

			cuLaunchKernel(resolveProg->kernel,
				groups_x, groups_y, 1,
				16, 16, 1,
				0, 0, args, 0);

			cuSurfObjectDestroy(output_surf);
			cuGraphicsUnmapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream)CU_STREAM_DEFAULT));
    }

    // CLEAR
    {
      cuMemsetD8(fb, 0xFFFF, 8 * 2048 * 2048);
    }

    GLTimerQueries::timestamp("compute-loop-start");
  }
};
