
#include <mutex>
#include <thread>

#include "compute/Resources.h"
#include "compute/ComputeLasLoader.h"
#include "Renderer.h"
#include "GLTimerQueries.h"

using namespace std;

mutex mtx_state;

void ComputeLasData::load(Renderer* renderer){

	cout << "ComputeLasData::load()" << endl;

	{
		lock_guard<mutex> lock(mtx_state);

		if(state != ResourceState::UNLOADED){
			return;
		}else{
			state = ResourceState::LOADING;
		}
	}

	{ // create buffers
		int numBatches = (this->numPoints / POINTS_PER_WORKGROUP) + 1;
		//int lodBufferSize = 4 * ((this->numPoints) / 10);
		this->ssBatches = renderer->createBuffer(64 * numBatches);
		this->ssXyz_12b = renderer->createBuffer(4 * this->numPoints);
		this->ssXyz_8b = renderer->createBuffer(4 * this->numPoints);
		this->ssXyz_4b = renderer->createBuffer(4 * this->numPoints);
		this->ssColors = renderer->createBuffer(4 * this->numPoints);
		//this->ssLOD = renderer->createBuffer(lodBufferSize);
		//this->ssLODColor = renderer->createBuffer(lodBufferSize);
		this->ssLoadBuffer = renderer->createBuffer(this->bytesPerPoint * MAX_POINTS_PER_BATCH);

		GLuint zero = 0;
		glClearNamedBufferData(this->ssBatches.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->ssXyz_12b.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->ssXyz_8b.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->ssXyz_4b.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->ssColors.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		//glClearNamedBufferData(this->ssLOD.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		//glClearNamedBufferData(this->ssLODColor.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
	}


	// start loader thread
	ComputeLasData *ref = this;
	thread t([ref](){

		int pointsRemaining = ref->numPoints;
		int pointsRead = 0;
		while(pointsRemaining > 0){

			{ // abort loader thread if state is set to unloading
				lock_guard<mutex> lock(mtx_state);

				if(ref->state == ResourceState::UNLOADING){
					cout << "stopping loader thread for " << ref->path << endl;

					ref->state = ResourceState::UNLOADED;

					return;
				}
			}

			if(ref->task){
				this_thread::sleep_for(1ms);
				continue;
			}

			int pointsInBatch = std::min(pointsRemaining, MAX_POINTS_PER_BATCH);
			int64_t start = int64_t(ref->offsetToPointData) + int64_t(ref->bytesPerPoint) * int64_t(pointsRead);
			int64_t size = ref->bytesPerPoint * pointsInBatch;
			auto buffer = readBinaryFile(ref->path, start, size);

			auto task = make_shared<LoaderTask>();
			task->buffer = buffer;
			task->pointOffset = pointsRead;
			task->numPoints = pointsInBatch;

			ref->task = task;

			pointsRemaining -= pointsInBatch;
			pointsRead += pointsInBatch;

			Debug::set("numPointsLoaded", formatNumber(pointsRead));

		}

		cout << "finished loading " << formatNumber(pointsRead) << " points" << endl;

		{ // check if resource was marked as unloading in the meantime
			lock_guard<mutex> lock(mtx_state);

			if(ref->state == ResourceState::UNLOADING){
				cout << "stopping loader thread for " << ref->path << endl;

				ref->state = ResourceState::UNLOADED;
			}else if(ref->state == ResourceState::LOADING){
				ref->state = ResourceState::LOADED;
			}
		}

	});
	t.detach();

}

void ComputeLasData::unload(Renderer* renderer){

	cout << "ComputeLasData::unload()" << endl;

	numPointsLoaded = 0;

	// delete buffers
	glDeleteBuffers(1, &ssBatches.handle);
	glDeleteBuffers(1, &ssXyz_12b.handle);
	glDeleteBuffers(1, &ssXyz_8b.handle);
	glDeleteBuffers(1, &ssXyz_4b.handle);
	glDeleteBuffers(1, &ssColors.handle);
	glDeleteBuffers(1, &ssLoadBuffer.handle);
	//glDeleteBuffers(1, &ssLOD.handle);
	//glDeleteBuffers(1, &ssLODColor.handle);

	lock_guard<mutex> lock(mtx_state);

	if(state == ResourceState::LOADED){
		state = ResourceState::UNLOADED;
	}else if(state == ResourceState::LOADING){
		// if loader thread is still running, notify thread by marking resource as "unloading"
		state = ResourceState::UNLOADING;
	}
}

void ComputeLasData::process(Renderer* renderer){

	static Shader* csLoad = nullptr;

	if(csLoad == nullptr){
		csLoad = new Shader({ {"./modules/compute/computeLasLoader.cs", GL_COMPUTE_SHADER} });
	}

	static GLBuffer ssDebug = renderer->createBuffer(256);

	GLuint zero = 0;
	glClearNamedBufferData(ssDebug.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);

	if(this->task){

		glNamedBufferSubData(this->ssLoadBuffer.handle, 0, this->task->buffer->size, this->task->buffer->data);

		//TODO now run las parse shader;
		// if(csLoad->program != -1)
		// {

			static int batchCounter = 0;
			// string timestampLabel = "load-batch[" + std::to_string(batchCounter) + "]";
			// GLTimerQueries::timestampPrint(timestampLabel + "-start");
			batchCounter++;

			glUseProgram(csLoad->program);

			auto boxMin = this->boxMin;
			auto boxMax = this->boxMax;
			auto scale = this->scale;
			auto offset = this->offset;

			glUniform1i(11, POINTS_PER_THREAD);

			glUniform3f(20, boxMin.x, boxMin.y, boxMin.z);
			glUniform3f(21, boxMax.x, boxMax.y, boxMax.z);
			glUniform1i(22, this->task->numPoints);
			glUniform1i(23, this->numPoints);
			glUniform1i(24, this->pointFormat);
			glUniform1i(25, this->bytesPerPoint);
			glUniform3f(26, scale.x, scale.y, scale.z);
			glUniform3d(27, offset.x, offset.y, offset.z);

			int batchOffset = this->numPointsLoaded / POINTS_PER_WORKGROUP;
			glUniform1i(30, batchOffset);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, this->ssLoadBuffer.handle);

			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 30, ssDebug.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 40, this->ssBatches.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 41, this->ssXyz_12b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 42, this->ssXyz_8b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 43, this->ssXyz_4b.handle);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 44, this->ssColors.handle);
			//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 45, this->ssLOD.handle);
			//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 46, this->ssLODColor.handle);

			int numBatches = this->task->numPoints / POINTS_PER_WORKGROUP;
			if((this->task->numPoints % POINTS_PER_WORKGROUP) != 0){
				numBatches++;
			}

			glDispatchCompute(numBatches, 1, 1);

			glMemoryBarrier(GL_ALL_BARRIER_BITS);


			// READ DEBUG VALUES
			// if(Debug::enableShaderDebugValue)
			// {
			// 	glMemoryBarrier(GL_ALL_BARRIER_BITS);

			// 	struct DebugData{
			// 		uint32_t value = 0;
			// 		uint32_t index = 0;
			// 		float x = 0;
			// 		float y = 0;
			// 		float z = 0;
			// 		uint32_t X = 0;
			// 		uint32_t Y = 0;
			// 		uint32_t Z = 0;
			// 		float min_x = 0;
			// 		float min_y = 0;
			// 		float min_z = 0;
			// 		float size_x = 0;
			// 		float size_y = 0;
			// 		float size_z = 0;
			// 		uint32_t check = 0;
			// 	};

			// 	DebugData data;
			// 	glGetNamedBufferSubData(ssDebug.handle, 0, sizeof(DebugData), &data);

			// 	auto dbg = Debug::getInstance();

			// 	if(data.index == 2877987){
			// 		dbg->set("[dbg] index", formatNumber(data.index));
			// 		dbg->set("[dbg] x", formatNumber(data.x, 3));
			// 		dbg->set("[dbg] y", formatNumber(data.y, 3));
			// 		dbg->set("[dbg] z", formatNumber(data.z, 3));
			// 		dbg->set("[dbg] X", formatNumber(data.X));
			// 		dbg->set("[dbg] Y", formatNumber(data.Y));
			// 		dbg->set("[dbg] Z", formatNumber(data.Z));
			// 		dbg->set("[dbg]  min_x", formatNumber(data.min_x, 3));
			// 		dbg->set("[dbg]  min_y", formatNumber(data.min_y, 3));
			// 		dbg->set("[dbg]  min_z", formatNumber(data.min_z, 3));
			// 		dbg->set("[dbg]  siye_x", formatNumber(data.size_x, 3));
			// 		dbg->set("[dbg]  siye_y", formatNumber(data.size_y, 3));
			// 		dbg->set("[dbg]  siye_z", formatNumber(data.size_z, 3));
			// 		dbg->set("[dbg]  check", formatNumber(data.check));
			// 	}

			// 	glMemoryBarrier(GL_ALL_BARRIER_BITS);
			// }

			// GLTimerQueries::timestampPrint(timestampLabel + "-end");

		// }

		this->numPointsLoaded += this->task->numPoints;
		this->numBatchesLoaded += numBatches;
		this->task = nullptr;

	}

}

// My changes for the basic version

void ComputeLasDataBasic::load(Renderer* renderer){
	cout << "ComputeLasDataBasic::load()" << endl;
	{
		lock_guard<mutex> lock(mtx_state);

		if(state != ResourceState::UNLOADED){
			return;
		}else{
			state = ResourceState::LOADING;
		}
	}

	{ // create buffers
		int numBatches = (this->numPoints / POINTS_PER_WORKGROUP) + 1;
		this->ssBatches = renderer->createBuffer(64 * numBatches);
		this->ssXyz = renderer->createBuffer(4 * 3 * this->numPoints);
		this->ssColors = renderer->createBuffer(4 * this->numPoints);
		this->ssLoadBuffer = renderer->createBuffer(this->bytesPerPoint * MAX_POINTS_PER_BATCH);

		GLuint zero = 0;
		glClearNamedBufferData(this->ssBatches.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->ssXyz.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
		glClearNamedBufferData(this->ssColors.handle, GL_R32UI, GL_RED, GL_UNSIGNED_INT, &zero);
	}


	// start loader thread and detach it immediately
	ComputeLasDataBasic *ref = this;
	thread t([ref](){
		int pointsRemaining = ref->numPoints;
		int pointsRead = 0;
		while(pointsRemaining > 0) {
			{ // abort loader thread if state is set to unloading
				lock_guard<mutex> lock(mtx_state);

				if(ref->state == ResourceState::UNLOADING){
					cout << "stopping loader thread for " << ref->path << endl;

					ref->state = ResourceState::UNLOADED;

					return;
				}
			}

			if(ref->task){
				this_thread::sleep_for(1ms);
				continue;
			}

			int pointsInBatch = std::min(pointsRemaining, MAX_POINTS_PER_BATCH);
			int64_t start = int64_t(ref->offsetToPointData) + int64_t(ref->bytesPerPoint) * int64_t(pointsRead);
			int64_t size = ref->bytesPerPoint * pointsInBatch;
			auto buffer = readBinaryFile(ref->path, start, size);

			auto task = make_shared<LoaderTask>();
			task->buffer = buffer;
			task->pointOffset = pointsRead;
			task->numPoints = pointsInBatch;

			ref->task = task;

			pointsRemaining -= pointsInBatch;
			pointsRead += pointsInBatch;

			Debug::set("numPointsLoaded", formatNumber(pointsRead));
		}

		cout << "finished loading " << formatNumber(pointsRead) << " points" << endl;

		{ // check if resource was marked as unloading in the meantime
			lock_guard<mutex> lock(mtx_state);

			if(ref->state == ResourceState::UNLOADING){
				cout << "stopping loader thread for " << ref->path << endl;

				ref->state = ResourceState::UNLOADED;
			}else if(ref->state == ResourceState::LOADING){
				ref->state = ResourceState::LOADED;
			}
		}

	});
	t.detach();
}

void ComputeLasDataBasic::unload(Renderer* renderer){
	cout << "ComputeLasDataBasic::unload()" << endl;

	numPointsLoaded = 0;

	// delete buffers
	glDeleteBuffers(1, &ssBatches.handle);
	glDeleteBuffers(1, &ssXyz.handle);
	glDeleteBuffers(1, &ssColors.handle);
	glDeleteBuffers(1, &ssLoadBuffer.handle);

	lock_guard<mutex> lock(mtx_state);

	if(state == ResourceState::LOADED){
		state = ResourceState::UNLOADED;
	}else if(state == ResourceState::LOADING){
		// if loader thread is still running, notify thread by marking resource as "unloading"
		state = ResourceState::UNLOADING;
	}
}

void ComputeLasDataBasic::process(Renderer* renderer){
  // I'll do everything in CPU for now
  if (this->task) {
    // metadata per batch
    dvec3 batchBoxMin = this->boxMin;
    dvec3 batchBoxMax = this->boxMax;
    dvec3 batchPointScale = this->scale;
    dvec3 batchPointOffset = this->offset;

    auto &bytesPerPoint = this->bytesPerPoint;
    int offset_rgb;
    if (this->pointFormat == 2) {
      offset_rgb = 20;
    } else if (this->pointFormat == 3) {
      offset_rgb = 28;
    } else if (this->pointFormat == 7) {
      offset_rgb = 30;
    } else if (this->pointFormat == 8) {
      offset_rgb = 30;
    }

    // host buffers
    vector<int32_t> host_Xyz(4 * 3 * this->task->numPoints);
    vector<uint32_t> host_Colors(4 * this->task->numPoints);

    // iterate over all the points
    for (int pid = 0; pid < this->task->numPoints; ++pid) {
      unsigned int byteOffset = pid * bytesPerPoint;
      int32_t X = this->task->buffer->get<int32_t>(byteOffset + 0);
      int32_t Y = this->task->buffer->get<int32_t>(byteOffset + 4);
      int32_t Z = this->task->buffer->get<int32_t>(byteOffset + 8);
      // TODO: check for shift by uBoxMin
      double x = double(double(X) * this->scale.x + this->offset.x); 
      double y = double(double(Y) * this->scale.y + this->offset.y); 
      double z = double(double(Z) * this->scale.z + this->offset.z); 

      if (this->numPointsLoaded == 0 && pid == 0) {
        cout << "first point " << X << " " << Y << " " << Z << endl;
      }

      uint32_t R = this->task->buffer->get<uint16_t>(byteOffset + offset_rgb + 0);
      uint32_t G = this->task->buffer->get<uint16_t>(byteOffset + offset_rgb + 2);
      uint32_t B = this->task->buffer->get<uint16_t>(byteOffset + offset_rgb + 4);
      uint32_t r = R > 255 ? R / 256 : R;
      uint32_t g = G > 255 ? G / 256 : G;
      uint32_t b = B > 255 ? B / 256 : B;
      uint32_t color = r | (g << 8) | (b << 16);

      // update local bounding box of the batch
      batchBoxMin.x = std::min(batchBoxMin.x, (double) x);
      batchBoxMin.y = std::min(batchBoxMin.y, (double) y);
      batchBoxMin.z = std::min(batchBoxMin.z, (double) z);
      batchBoxMax.x = std::max(batchBoxMax.x, (double) x);
      batchBoxMax.y = std::max(batchBoxMax.y, (double) y);
      batchBoxMax.z = std::max(batchBoxMax.z, (double) z);

      // insert the data into the host buffers
      host_Xyz[pid * 3 + 0] = X;
      host_Xyz[pid * 3 + 1] = Y;
      host_Xyz[pid * 3 + 2] = Z;
      host_Colors[pid] = color;
    }

    // create batch data
    Batch batchData {
      0,
      (float) batchBoxMin.x, (float) batchBoxMin.y, (float) batchBoxMin.z,
      (float) batchBoxMax.x, (float) batchBoxMax.y, (float) batchBoxMax.z,
      (int) this->task->numPoints,
      0, 0,
      (float) batchPointScale.x, (float) batchPointScale.y, (float) batchPointScale.z,
      (float) batchPointOffset.x, (float) batchPointOffset.y, (float) batchPointOffset.z
    };

    // copy from host to device to the correct location
    // 12 bytes per point
    uint32_t XyzOffset = this->task->pointOffset * 3 * 4;
    uint32_t XyzSize = this->task->numPoints * 3 * 4;
    glNamedBufferSubData(this->ssXyz.handle, XyzOffset, XyzSize, host_Xyz.data());
    // 4 bytes per point
    uint32_t ColorsOffset = this->task->pointOffset * 4;
    uint32_t ColorsSize = this->task->numPoints * 4;
    glNamedBufferSubData(this->ssColors.handle, ColorsOffset, ColorsSize, host_Colors.data());
    // sizeof(Batch) bytes per batch
    uint32_t BatchDataOffset = this->numPointsLoaded / POINTS_PER_WORKGROUP;
    uint32_t BatchDataSize = sizeof(Batch);
    glNamedBufferSubData(this->ssBatches.handle, BatchDataOffset, BatchDataSize, &batchData);

    int numBatches = this->task->numPoints / POINTS_PER_WORKGROUP;
    if((this->task->numPoints % POINTS_PER_WORKGROUP) != 0){
      numBatches++;
    }
    this->numPointsLoaded += this->task->numPoints;
    this->numBatchesLoaded += numBatches;
    this->task = nullptr;
  }
}
