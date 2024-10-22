
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

using namespace std;
using namespace std::chrono_literals;
using nlohmann::json;



struct ComputeLoopLasHqs : public Method{

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
		int colorizeChunks;
		int colorizeOverdraw;
	};

	struct DebugData{
		uint32_t value = 0;
		bool enabled = false;
		uint32_t depth_numPointsProcessed = 0;
		uint32_t depth_numNodesProcessed = 0;
		uint32_t depth_numPointsRendered = 0;
		uint32_t depth_numNodesRendered = 0;
		uint32_t color_numPointsProcessed = 0;
		uint32_t color_numNodesProcessed = 0;
		uint32_t color_numPointsRendered = 0;
		uint32_t color_numNodesRendered = 0;
		uint32_t numPointsVisible = 0;
	};

	string source = "";
	Shader* csDepth = nullptr;
	Shader* csColor = nullptr;
	Shader* csResolve = nullptr;

	GLBuffer ssDepth;
	GLBuffer ssRGBA;
	GLBuffer ssDebug;
	GLBuffer ssBoundingBoxes;
	GLBuffer uniformBuffer;
	UniformData uniformData;

	shared_ptr<ComputeLasData> las = nullptr;

	Renderer* renderer = nullptr;

	ComputeLoopLasHqs(Renderer* renderer, shared_ptr<ComputeLasData> las){

		this->name = "loop_las_hqs";
		this->description = R"ER01(
Like compute las, but also 
averages overlapping points
		)ER01";
		this->las = las;
		this->group = "10-10-10 bit encoded";

		csDepth = new Shader({ {"./modules/compute_loop_las_hqs/depth.cs", GL_COMPUTE_SHADER} });
		csColor = new Shader({ {"./modules/compute_loop_las_hqs/color.cs", GL_COMPUTE_SHADER} });
		csResolve = new Shader({ {"./modules/compute_loop_las_hqs/resolve.cs", GL_COMPUTE_SHADER} });

		ssDepth = renderer->createBuffer(4 * 2048 * 2048);
		ssRGBA = renderer->createBuffer(16 * 2048 * 2048);

		ssDebug = renderer->createBuffer(256);
		ssBoundingBoxes = renderer->createBuffer(48 * 1'000'000);
		uniformBuffer = renderer->createUniformBuffer(512);

		GLuint zero = 0;
		glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(ssBoundingBoxes.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);

		this->renderer = renderer;
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

		if(las->numPointsLoaded == 0){
			return;
		}

		auto fbo = renderer->views[0].framebuffer;
		auto camera = renderer->camera;

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
			uniformData.colorizeChunks = Debug::colorizeChunks;
			uniformData.colorizeOverdraw = Debug::colorizeOverdraw;

			glNamedBufferSubData(uniformBuffer.handle, 0, sizeof(UniformData), &uniformData);
		}

		if(Debug::enableShaderDebugValue){
			DebugData data;
			data.enabled = true;

			glNamedBufferSubData(ssDebug.handle, 0, sizeof(DebugData), &data);
		}

		// DEPTH
		if(csDepth->program != -1){
			GLTimerQueries::timestamp("depth-start");

			glUseProgram(csDepth->program);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssDepth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssRGBA.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);
			glBindBufferBase(GL_UNIFORM_BUFFER, 31, uniformBuffer.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 40, las->ssBatches.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 41, las->ssXyz_12b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 42, las->ssXyz_8b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 43, las->ssXyz_4b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, las->ssColors.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 50, ssBoundingBoxes.handle);

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int numBatches = ceil(double(las->numPointsLoaded) / double(POINTS_PER_WORKGROUP));
			
			glDispatchCompute(numBatches, 1, 1);

			GLTimerQueries::timestamp("depth-end");
		}

		// COLORS
		if(csColor->program != -1){
			GLTimerQueries::timestamp("color-start");

			glUseProgram(csColor->program);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssDepth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssRGBA.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);
			glBindBufferBase(GL_UNIFORM_BUFFER, 31, uniformBuffer.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 40, las->ssBatches.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 41, las->ssXyz_12b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 42, las->ssXyz_8b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 43, las->ssXyz_4b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, las->ssColors.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 50, ssBoundingBoxes.handle);

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int numBatches = ceil(double(las->numPointsLoaded) / double(POINTS_PER_WORKGROUP));
			
			glDispatchCompute(numBatches, 1, 1);

			GLTimerQueries::timestamp("color-end");
		}

		// RESOLVE
		if(csResolve->program != -1){ 

			GLTimerQueries::timestamp("resolve-start");

			glUseProgram(csResolve->program);
			
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ssDepth.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, ssRGBA.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);
			glBindBufferBase(GL_UNIFORM_BUFFER, 31, uniformBuffer.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, las->ssColors.handle);

			glBindImageTexture(0, fbo->colorAttachments[0]->handle, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);

			int groups_x = ceil(float(fbo->width) / 16.0f);
			int groups_y = ceil(float(fbo->height) / 16.0f);
			glDispatchCompute(groups_x, groups_y, 1);
			

			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			GLTimerQueries::timestamp("resolve-end");
		}

		// READ DEBUG VALUES
		if(Debug::enableShaderDebugValue){
			glMemoryBarrier(GL_ALL_BARRIER_BITS);

			DebugData data;
			glGetNamedBufferSubData(ssDebug.handle, 0, sizeof(DebugData), &data);

			// Debug::getInstance()->values["debug value"] = formatNumber(data.value);

			auto dbg = Debug::getInstance();

			dbg->pushFrameStat("[depth] #nodes processed" , formatNumber(data.depth_numNodesProcessed));
			dbg->pushFrameStat("[depth] #nodes rendered"  , formatNumber(data.depth_numNodesRendered));
			dbg->pushFrameStat("[depth] #points processed", formatNumber(data.depth_numPointsProcessed));
			dbg->pushFrameStat("[depth] #points rendered" , formatNumber(data.depth_numPointsRendered));
			dbg->pushFrameStat("divider" , "");
			dbg->pushFrameStat("[color] #nodes processed" , formatNumber(data.color_numNodesProcessed));
			dbg->pushFrameStat("[color] #nodes rendered"  , formatNumber(data.color_numNodesRendered));
			dbg->pushFrameStat("[color] #points processed", formatNumber(data.color_numPointsProcessed));
			dbg->pushFrameStat("[color] #points rendered" , formatNumber(data.color_numPointsRendered));
			
			dbg->pushFrameStat("divider" , "");

			dbg->pushFrameStat("#points visible"          , formatNumber(data.numPointsVisible));

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

			glClearNamedBufferSubData(ssDepth.handle, GL_R32UI, 0, fbo->width * fbo->height * 4, GL_RED, GL_UNSIGNED_INT, &intbits);
			glClearNamedBufferSubData(ssRGBA.handle, GL_R32UI, 0, fbo->width * fbo->height * 16, GL_RED, GL_UNSIGNED_INT, &zero);
			glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
			glClearNamedBufferSubData(ssBoundingBoxes.handle, GL_R32UI, 0, 48, GL_RED, GL_UNSIGNED_INT, &zero);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);
		}
		
		GLTimerQueries::timestamp("compute-loop-end");
	}


};