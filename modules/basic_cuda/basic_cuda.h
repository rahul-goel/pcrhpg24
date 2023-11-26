#include <string>
#include <vector>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>
#include "nlohmann/json.hpp"
#include <glm/gtc/matrix_transform.hpp> 

#include "Method.h"
#include "CudaProgram.h"
#include "LasLoader.h"
#include "Frustum.h"
#include "Runtime.h"
#include "compute/ComputeLasLoader.h"
#include "GLTimerQueries.h"
#include "cudaGL.h"

#include "builtin_types.h"
#include "basic_cuda/kernel_data.h"

using namespace std;

struct BasicCuda : public Method {

  // are the buffers registered as graphics resource?
	bool registered = false;

	CudaProgram* resolveProg = nullptr;
	CudaProgram* renderProg = nullptr;

  // framebuffer
	CUdeviceptr fb; // neither an input, nor an output, that's why cuda pointer
	CUgraphicsResource output, rgba, batches, xyz; // inputs and outputs, hence GL interop
	CUevent start, end;

  // my own custom las loader that runs on CPU
	shared_ptr<ComputeLasDataBasic> las = nullptr;
	Renderer* renderer = nullptr;

	BasicCuda(Renderer* renderer, shared_ptr<ComputeLasDataBasic> las){

		this->name = "basic_cuda";
		this->description = R"ER01(
- Each thread renders X points.
- Loads points from LAS file
- While loading, each workgroup 
  - Computes the bounding box of assigned points.
  - does not create different buffers
		)ER01";
		this->las = las;
		this->group = "none";

		cuMemAlloc(&fb, 8 * 2048 * 2048);
		
		this->renderer = renderer;

		resolveProg = new CudaProgram("./modules/basic_cuda/resolve.cu");
		renderProg = new CudaProgram("./modules/basic_cuda/render.cu");
		cuEventCreate(&start, CU_EVENT_DEFAULT);
		cuEventCreate(&end, CU_EVENT_DEFAULT);
	}

	~BasicCuda() {
		if (registered) {
			std::vector<CUgraphicsResource> persistent_resources = { batches, xyz, rgba };
			cuGraphicsUnmapResources(persistent_resources.size(), persistent_resources.data(), ((CUstream)CU_STREAM_DEFAULT));
			cuGraphicsUnregisterResource(rgba);
			cuGraphicsUnregisterResource(batches);
			cuGraphicsUnregisterResource(xyz);
			cuGraphicsUnregisterResource(output);
		}
  }

  void update(Renderer* renderer) {
    if(Runtime::resource != (Resource*)las.get()){

      if(Runtime::resource != nullptr){
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
      cuGraphicsGLRegisterBuffer(&rgba, las->ssColors.handle, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
      cuGraphicsGLRegisterBuffer(&batches, las->ssBatches.handle, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
      cuGraphicsGLRegisterBuffer(&xyz, las->ssXyz.handle, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
      // 2d image that will be re-written again and again by cuda
      cuGraphicsGLRegisterImage(&output, renderer->views[0].framebuffer->colorAttachments[0]->handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD);

      std::vector<CUgraphicsResource> persistent_resources = {batches, xyz, rgba};
			cuGraphicsMapResources(persistent_resources.size(), persistent_resources.data(), ((CUstream)CU_STREAM_DEFAULT));

      registered = true;
      cout << "Buffers registered" << endl;
    }

    // get pointers for cuda pointers for the common buffers
    static CUdeviceptr batches_ptr, xyz_ptr, rgba_ptr;
    size_t size;
		cuGraphicsResourceGetMappedPointer(&batches_ptr, &size, batches);
		cuGraphicsResourceGetMappedPointer(&xyz_ptr, &size, xyz);
		cuGraphicsResourceGetMappedPointer(&rgba_ptr, &size, rgba);

    // RENDER
    {
      auto &viewLeft = renderer->views[0];
      mat4 world;
      mat4 view = viewLeft.view;
      mat4 proj = viewLeft.proj;
      mat4 worldView = view * world;
      mat4 viewProj = proj * view;
      mat4 worldViewProj = proj * view * world;

      // stuff that changes because of camera movement, window size change, batch loading, imgui changes
      ChangingRenderData cdata;
      *((glm::mat4 *) &cdata.uTransform) = glm::transpose(worldViewProj);
			*((glm::mat4*)&cdata.uWorldView) = glm::transpose(worldView);
			*((glm::mat4*)&cdata.uProj) = glm::transpose(proj);
			cdata.uCamPos = float3{ (float)camera->position.x, (float)camera->position.y, (float)camera->position.z };
			cdata.uImageSize = int2{ fbo->width, fbo->height };
			cdata.uPointsPerThread = POINTS_PER_THREAD;
			cdata.uBoxMin = float3{ (float)las->boxMin.x, (float)las->boxMin.y, (float)las->boxMin.z };
			// cdata.uBoxMax = float3{ (float)las->boxMax.x, (float)las->boxMax.y, (float)las->boxMax.z };
			cdata.uNumPoints = las->numPointsLoaded;
			cdata.uOffsetPointToData = las->offsetToPointData;
			cdata.uPointFormat = las->pointFormat;
			cdata.uBytesPerPoint = las->bytesPerPoint;
			cdata.uScale = float3{ (float)las->scale.x, (float)las->scale.y, (float)las->scale.z };
			cdata.uBoxMax = float3{ (float)las->offset.x, (float)las->offset.y, (float)las->offset.z };
			cdata.uEnableFrustumCulling = Debug::frustumCullingEnabled;

			// don't execute a workgroup until all points inside are loaded
			// workgroup only iterates over the source buffer once, 
			// and generates a derivative!
			int numBatches = 0;
			numBatches = ceil(double(las->numPointsLoaded) / double(POINTS_PER_WORKGROUP));

      // kernel launch
      void *args[] = {&cdata, &fb, &batches_ptr, &xyz_ptr, &rgba_ptr};
			cuLaunchKernel(renderProg->kernel,
				numBatches, 1, 1,
				WORKGROUP_SIZE, 1, 1,
				0, 0, args, 0);
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
      void *args[] = { &fbo->width, &fbo->height, &output_surf, &fb, &rgba_ptr };

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
