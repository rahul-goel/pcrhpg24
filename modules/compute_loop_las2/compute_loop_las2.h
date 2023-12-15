
#pragma once

#include <string>
#include <queue>
#include <vector>
#include <mutex>
#include <thread>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>
#include "nlohmann/json.hpp"
#include <glm/gtc/matrix_transform.hpp> 

#include "unsuck.hpp"

#include "ProgressiveFileBuffer.h"
#include "Shader.h"
#include "Box.h"
#include "Debug.h"
#include "Camera.h"
#include "LasLoader.h"
#include "Frustum.h"
#include "Renderer.h"
#include "GLTimerQueries.h"
#include "Method.h"
#include "compute/ComputeLasLoader.h"
#include "tinyexr.h"

using namespace std;
using namespace std::chrono_literals;
using nlohmann::json;



struct ComputeLoopLas2 : public Method{

	struct UniformData{
		mat4 world;
		mat4 view;
		mat4 proj;
		mat4 transform;
		mat4 transformFrustum;
		int pointsPerThread;
		int enableFrustumCulling;
		int showBoundingBox;
		int numPoints;
		ivec2 imageSize;
	};

	struct DebugData{
		uint32_t value = 0;
		bool enabled = false;
		uint32_t numPointsProcessed = 0;
		uint32_t numNodesProcessed = 0;
		uint32_t numPointsRendered = 0;
		uint32_t numNodesRendered = 0;
		uint32_t numPointsVisible = 0;
	};

  // optional saving of depth image
  vector<float> depthmap;

	string source = "";
	Shader* csRender = nullptr;
	Shader* csResolve = nullptr;

	GLBuffer ssFramebuffer;
	GLBuffer ssDebug;
	GLBuffer ssBoundingBoxes;
	GLBuffer uniformBuffer;
	UniformData uniformData;

	shared_ptr<ComputeLasData> las = nullptr;

	Renderer* renderer = nullptr;

	ComputeLoopLas2(Renderer* renderer, shared_ptr<ComputeLasData> las){

		this->name = "loop_las2";
		this->description = R"ER01(
- Each thread renders X points.
- Loads points from LAS file
- While loading, each workgroup 
  - Computes the bounding box of assigned points.
  - creates 4, 8 and 12 byte vertex buffers.
- Workgroup picks 4, 8, or 12 byte precision
  depending on screen size of bounding box
		)ER01";
		this->las = las;
		this->group = "10-10-10 bit encoded";

		csRender = new Shader({ {"./modules/compute_loop_las2/render.cs", GL_COMPUTE_SHADER} });
		csResolve = new Shader({ {"./modules/compute_loop_las2/resolve.cs", GL_COMPUTE_SHADER} });

		ssFramebuffer = renderer->createBuffer(8 * 2048 * 2048);

		this->renderer = renderer;

		ssDebug = renderer->createBuffer(256);
		ssBoundingBoxes = renderer->createBuffer(48 * 1'000'000);
		uniformBuffer = renderer->createUniformBuffer(512);

		GLuint zero = 0;
		glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(ssBoundingBoxes.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
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

	
	void update(Renderer* renderer){

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

		las->process(renderer);

		auto fbo = renderer->views[0].framebuffer;
		auto camera = renderer->camera;

		if(las->numPointsLoaded == 0){
			return;
		}

		// Update Uniform Buffer
		{
			mat4 world;
			mat4 view = renderer->views[0].view;
			mat4 proj = renderer->views[0].proj;
			mat4 worldView = view * world;
			mat4 worldViewProj = proj * view * world;

			uniformData.world = world;
			uniformData.view = view;
			uniformData.proj = proj;
			uniformData.transform = worldViewProj;
			if(Debug::updateFrustum){
				uniformData.transformFrustum = worldViewProj;
			}
			uniformData.pointsPerThread = POINTS_PER_THREAD;
			uniformData.numPoints = las->numPointsLoaded;
			uniformData.enableFrustumCulling = Debug::frustumCullingEnabled ? 1 : 0;
			uniformData.showBoundingBox = Debug::showBoundingBox ? 1 : 0;
			uniformData.imageSize = {fbo->width, fbo->height};

			glNamedBufferSubData(uniformBuffer.handle, 0, sizeof(UniformData), &uniformData);
		}

		if(Debug::enableShaderDebugValue){
			DebugData data;
			data.enabled = true;

			glNamedBufferSubData(ssDebug.handle, 0, sizeof(DebugData), &data);
		}

		// RENDER
		if(csRender->program != -1){
			GLTimerQueries::timestamp("draw-start");

			glUseProgram(csRender->program);

			auto& viewLeft = renderer->views[0];

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssFramebuffer.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);
			glBindBufferBase(GL_UNIFORM_BUFFER, 31, uniformBuffer.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 40, las->ssBatches.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 41, las->ssXyz_12b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 42, las->ssXyz_8b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 43, las->ssXyz_4b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, las->ssColors.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 50, ssBoundingBoxes.handle);

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			// int numBatches = ceil(double(las->numPointsLoaded) / double(POINTS_PER_WORKGROUP));
			int numBatches = las->numBatchesLoaded;
			
			glDispatchCompute(numBatches, 1, 1);

			GLTimerQueries::timestamp("draw-end");
		}

    // saving depthmap
    if (Debug::saveDepthMap) {
      vector<uint64_t> fb_host(fbo->height * fbo->width);
			glBindFramebuffer(GL_FRAMEBUFFER, ssFramebuffer.handle);
      // glReadPixels(0, 0, fbo->width, fbo->height, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT64_NV, fb_host.data());
      glGetNamedBufferSubData(ssFramebuffer.handle, 0, fbo->height * fbo->width * 8, fb_host.data());
      glBindFramebuffer(GL_FRAMEBUFFER, 0);

      bool seen = false;

      depthmap.clear();
      depthmap.resize(fbo->height * fbo->width, 0);
      for (int i = 0; i < fbo->height; ++i) {
        for (int j = 0; j < fbo->width; ++j) {
          unsigned int value = fb_host[i * fbo->width + j] >> 32;
          float fvalue = *((float *) &value);
          if (fvalue < 0) continue;
          depthmap[(fbo->height - i - 1) * fbo->width + j] = fvalue;
          // cout << *((float *) &value) << endl;
        }
      }

      saveSingleChannelEXR("out/depth.exr", depthmap.data(), fbo->width, fbo->height);
      Debug::saveDepthMap = false;
    }

		// RESOLVE
		if(csResolve->program != -1){
			GLTimerQueries::timestamp("resolve-start");

			glUseProgram(csResolve->program);
			
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssFramebuffer.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);
			glBindBufferBase(GL_UNIFORM_BUFFER, 31, uniformBuffer.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, las->ssColors.handle);

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int groups_x = ceil(float(fbo->width) / 16.0f);
			int groups_y = ceil(float(fbo->height) / 16.0f);
			glDispatchCompute(groups_x, groups_y, 1);

			GLTimerQueries::timestamp("resolve-end");
		}

		// READ DEBUG VALUES
		if(Debug::enableShaderDebugValue){
			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			DebugData data;
			glGetNamedBufferSubData(ssDebug.handle, 0, sizeof(DebugData), &data);

			auto dbg = Debug::getInstance();

			dbg->pushFrameStat("#nodes processed"        , formatNumber(data.numNodesProcessed));
			dbg->pushFrameStat("#nodes rendered"         , formatNumber(data.numNodesRendered));
			dbg->pushFrameStat("#points processed"       , formatNumber(data.numPointsProcessed));
			dbg->pushFrameStat("#points rendered"        , formatNumber(data.numPointsRendered));
			dbg->pushFrameStat("divider" , "");
			dbg->pushFrameStat("#points visible"         , formatNumber(data.numPointsVisible));

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}

		// BOUNDING BOXES
		if(Debug::showBoundingBox){
			glMemoryBarrier(GL_ALL_BARRIER_BITS);
			glBindFramebuffer(GL_FRAMEBUFFER, fbo->handle);

			auto camera = renderer->camera;
			renderer->drawBoundingBoxes(camera.get(), ssBoundingBoxes);
		}


		{ // CLEAR
			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			GLuint zero = 0;
			float inf = -Infinity;
			GLuint intbits;
			memcpy(&intbits, &inf, 4);

			glClearNamedBufferSubData(ssFramebuffer.handle, GL_R32UI, 0, fbo->width * fbo->height * 8, GL_RED, GL_UNSIGNED_INT, &intbits);
			glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
			glClearNamedBufferSubData(ssBoundingBoxes.handle, GL_R32UI, 0, 48, GL_RED, GL_UNSIGNED_INT, &zero);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}
		
		GLTimerQueries::timestamp("compute-loop-end");
	}


};
