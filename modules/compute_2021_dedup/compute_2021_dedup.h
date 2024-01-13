
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
#include "Runtime.h"
#include "compute/LasLoaderStandard.h"
#include "tinyexr.h"

using namespace std;
using namespace std::chrono_literals;
using nlohmann::json;



struct ComputeDedup : public Method{

	string source = "";
	Shader* csRender = nullptr;
	Shader* csResolve = nullptr;

	GLBuffer ssFramebuffer;
	GLBuffer ssDebug;

	shared_ptr<LasStandardData> las = nullptr;

	Renderer* renderer = nullptr;

  // optional saving of depth image
  vector<float> depthmap;

	ComputeDedup(Renderer* renderer, shared_ptr<LasStandardData> las){

		this->name = "(2021) dedup";
		this->description = R"ER01(
		)ER01";
		this->las = las;
		this->group = "2021 method; standard 16 byte per point";

		csRender = new Shader({ {"./modules/compute_2021_dedup/render.cs", GL_COMPUTE_SHADER} });
		csResolve = new Shader({ {"./modules/compute_2021_dedup/resolve.cs", GL_COMPUTE_SHADER} });

		this->renderer = renderer;

		ssFramebuffer = renderer->createBuffer(8 * 2048 * 2048);
		ssDebug = renderer->createBuffer(256);

		GLuint zero = 0;
		glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
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

		GLTimerQueries::timestamp("compute-earlyz-start");

		las->process(renderer);

		auto fbo = renderer->views[0].framebuffer;
		auto camera = renderer->camera;

		if(las->numPointsLoaded == 0){
			return;
		}

		// RENDER
		if(csRender->program != -1)
		{ 

			GLTimerQueries::timestamp("draw-start");

			glUseProgram(csRender->program);

			auto& viewLeft = renderer->views[0];

			mat4 world;
			mat4 view = viewLeft.view;
			mat4 proj = viewLeft.proj;
			mat4 worldViewProj = proj * view * world;

			glUniformMatrix4fv(0, 1, GL_FALSE, &worldViewProj[0][0]);
			glUniform2i(1, fbo->width, fbo->height);
			
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssFramebuffer.handle);

			int64_t pointsRemaining = las->numPointsLoaded;
			int64_t pointsRendered = 0;
			int64_t pointsPerBatch = 268'000'000;
			while(pointsRemaining > 0){
				
				int pointsInBatch = min(pointsRemaining, pointsPerBatch);

				int64_t start = 16ul * pointsRendered;
				int64_t size = 16ul * pointsInBatch;
				glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, las->ssPoints.handle, start, size);

				int numBatches = ceil(double(pointsInBatch) / 128.0);
				
				glDispatchCompute(numBatches, 1, 1);

				pointsRendered += pointsInBatch;
				pointsRemaining -= pointsInBatch;
			}

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
      depthmap.resize(fbo->height * fbo->width, 0.0f);
      for (int i = 0; i < fbo->height; ++i) {
        for (int j = 0; j < fbo->width; ++j) {
          unsigned int value = (fb_host[i * fbo->width + j] >> 24) & UINT_MAX;
          float fvalue = *((float *) &value);
          if (fvalue < 0 || fvalue != fvalue) continue;
          depthmap[(fbo->height - i - 1) * fbo->width + j] = fvalue;
          // cout << *((float *) &value) << endl;
        }
      }

      saveSingleChannelEXR("out/depth.exr", depthmap.data(), fbo->width, fbo->height);
      Debug::saveDepthMap = false;
    }


		// RESOLVE
		if(csResolve->program != -1)
		{ 

			GLTimerQueries::timestamp("resolve-start");

			glUseProgram(csResolve->program);
			
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, las->ssPoints.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssFramebuffer.handle);

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int groups_x = ceil(float(fbo->width) / 16.0f);
			int groups_y = ceil(float(fbo->height) / 16.0f);
			glDispatchCompute(groups_x, groups_y, 1);
			

			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			GLTimerQueries::timestamp("resolve-end");
		}


		// { // CLEAR
		// 	glMemoryBarrier(GL_ALL_BARRIER_BITS);

		// 	GLuint zero = 0;
		// 	float inf = -Infinity;
		// 	GLuint intbits;
		// 	memcpy(&intbits, &inf, 4);

		// 	glClearNamedBufferData(ssFramebuffer.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &intbits);

		// 	glMemoryBarrier(GL_ALL_BARRIER_BITS);
		// }
		
		GLTimerQueries::timestamp("compute-earlyz-end");
	}


};
