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

#include "tinyexr.h"

using namespace std;

struct HuffmanHQS : public Method {
  bool registered = false;
  bool mapped = false;

  // optional saving of depth image
  vector<float> depthmap;

  CudaProgram* resolveProg = nullptr;
  CudaProgram* renderProg  = nullptr;
  CudaProgram* depthProg  = nullptr;

  // framebuffer - neither an input, nor an output, that's why cuda pointer
	CUdeviceptr fb;
  CUdeviceptr RG, BA;

	CUgraphicsResource output;
  CUevent start, end;

  shared_ptr<HuffmanLasData> las = nullptr;
  Renderer *renderer;

  HuffmanHQS(Renderer *renderer, shared_ptr<HuffmanLasData> las) {
    this->name = "huffman_hqs";
    this->description = R"ER01(
      - Decodes Huffman Encoded values on the GPU
    )ER01";
    this->las = las;
    this->group = "none";

		cuMemAlloc(&fb, 8 * 2048 * 2048);
		cuMemAlloc(&RG, 8 * 2048 * 2048);
		cuMemAlloc(&BA, 8 * 2048 * 2048);
		
		this->renderer = renderer;

		resolveProg = new CudaProgram("./modules/huffman_hqs/resolve.cu");
		renderProg = new CudaProgram("./modules/huffman_hqs/render.cu");
    depthProg = new CudaProgram("./modules/huffman_hqs/depth.cu");
		cuEventCreate(&start, CU_EVENT_DEFAULT);
		cuEventCreate(&end, CU_EVENT_DEFAULT);
  }

	~HuffmanHQS() {
		if (registered) {
      cuGraphicsUnregisterResource(output);
		}
  }

  bool saveSingleChannelEXR(const char* filename, const float* depthData, int width, int height) {
    // Define image attributes
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    // Set image data in EXRImage struct
    image.num_channels = 1;  // Single channel for depth
    image.width = width;
    image.height = height;
    image.images = (unsigned char**)(&depthData);

    header.num_channels = 1;
    header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * header.num_channels);
    header.pixel_types = (int*)malloc(sizeof(int) * header.num_channels);
    header.requested_pixel_types = (int*)malloc(sizeof(int) * header.num_channels);

    // Specify channel information for depth
    strncpy(header.channels[0].name, "Z", sizeof(header.channels[0].name));
    header.channels[0].name[sizeof(header.channels[0].name) - 1] = '\0';
    header.channels[0].pixel_type = TINYEXR_PIXELTYPE_FLOAT;  // Using float pixel type
    header.channels[0].p_linear = 0;
    header.pixel_types[0] = TINYEXR_PIXELTYPE_FLOAT;
    header.requested_pixel_types[0] = TINYEXR_PIXELTYPE_FLOAT;

    // Save EXR image
    const char* err = nullptr;
    int ret = SaveEXRImageToFile(&image, &header, filename, &err);

    // Clean up
    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);

    if (ret != TINYEXR_SUCCESS) {
      printf("Error saving EXR file: %s\n", err);
      free((void *) err);
      return false;
    }

    return true;
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

    if (renderProg->kernel == nullptr || resolveProg->kernel == nullptr || depthProg->kernel == nullptr) return;
    if (las->numPointsLoaded == 0) return;
    // if (las->numPointsLoaded != las->numPoints) return;

    // register the buffers to be used from cuda
    if (!registered) {
      // 2d image that will be re-written again and again by cuda
      cuGraphicsGLRegisterImage(&output, renderer->views[0].framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

      registered = true;
      cout << "Buffers registered" << endl;
    }

    // get pointers for cuda pointers for the common buffers
    if (!mapped || las->numPointsLoaded < las->numPoints) {
      size_t size;
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
			cdata.uPointFormat = (int) (Debug::LOD * 100);
			cdata.uBytesPerPoint = 0;
			cdata.uScale = float3{0, 0, 0};
			cdata.uBoxMax = float3{0, 0, 0};
			cdata.uEnableFrustumCulling = Debug::frustumCullingEnabled;
      cdata.colorizeChunks = Debug::colorizeChunks;
      cdata.showNumPoints = Debug::showNumPoints;

			// don't execute a workgroup until all points inside are loaded
			// workgroup only iterates over the source buffer once, 
			// and generates a derivative!
      int numBatches = las->numBatchesLoaded;

      // kernel launch
      void *args[] = {&cdata, &fb,
      &las->BatchData, &las->StartValues,
      &las->EncodedData,
      &las->SeparateData, &las->SeparateDataSizes,
      &las->DecoderTableValues, &las->DecoderTableCWLen, &las->ClusterSizes, &las->Colors};

      // cout << "launching kernel " << numBatches << " " << WORKGROUP_SIZE << endl;
      cuLaunchKernel(depthProg->kernel,
				numBatches, 1, 1,
				WORKGROUP_SIZE, 1, 1,
				0, 0, args, 0);
      // cout << "kernel terminated " << opt << endl;

      void *args2[] = {&cdata, &fb, &RG, &BA,
      &las->BatchData, &las->StartValues,
      &las->EncodedData,
      &las->SeparateData, &las->SeparateDataSizes,
      &las->DecoderTableValues, &las->DecoderTableCWLen, &las->ClusterSizes, &las->Colors};

			cuLaunchKernel(renderProg->kernel,
				numBatches, 1, 1,
				WORKGROUP_SIZE, 1, 1,
				0, 0, args2, 0);
    }

    // saving depthmap
    if (Debug::saveDepthMap) {
      vector<uint64_t> fb_host(fbo->height * fbo->width);
      cuMemcpyDtoH(fb_host.data(), fb, fbo->height * fbo->width * 8);

      bool seen = false;

      depthmap.clear();
      depthmap.resize(fbo->height * fbo->width, 0);
      for (int i = 0; i < fbo->height; ++i) {
        for (int j = 0; j < fbo->width; ++j) {
          unsigned int value = fb_host[i * fbo->width + j] >> 32;
          if (value == 0xFFFFFFFF) continue;
          depthmap[(fbo->height - i - 1) * fbo->width + j] = *((float *) &value);
          // cout << *((float *) &value) << endl;
        }
      }

      // saveDepthMapToEXR("out/depth.png", fbo->width, fbo->height, depthmap);
      saveSingleChannelEXR("out/depth.exr", depthmap.data(), fbo->width, fbo->height);
      Debug::saveDepthMap = false;
    }

    // RESOLVE
    {
      std::vector<CUgraphicsResource> dynamic_resources = { output };
      cuGraphicsMapResources(dynamic_resources.size(), dynamic_resources.data(), ((CUstream) CU_STREAM_DEFAULT));

      CUDA_RESOURCE_DESC res_desc = {};
      res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
      cuGraphicsSubResourceGetMappedArray(&res_desc.res.array.hArray, output, 0, 0);
      CUsurfObject output_surf;
			cuSurfObjectCreate(&output_surf, &res_desc);

      int groups_x = (fbo->width + 15) / 16;
      int groups_y = (fbo->height + 15) / 16;
      int showNumPoints = Debug::showNumPoints;
      int colorizeChunks = Debug::colorizeChunks;
      void *args[] = { &showNumPoints, &colorizeChunks, &fbo->width, &fbo->height, &output_surf, &fb, &RG, &BA, &las->Colors };

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
      cuMemsetD8(RG, 0, 8 * 2048 * 2048);
      cuMemsetD8(BA, 0, 8 * 2048 * 2048);
    }

    GLTimerQueries::timestamp("compute-loop-start");
  }
};
