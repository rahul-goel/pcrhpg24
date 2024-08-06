

#include <iostream>
#include <filesystem>

#include "GLBuffer.h"
#include "Renderer.h"
#include "Shader.h"
#include "compute_basic.h"
#include "ProgressiveFileBuffer.h"
#include "compute_points/compute_points.h"

#include "compute_loop_compress_nodewise/compute_loop_compress_nodewise.h"
#include "compute_loop_las/compute_loop_las.h"
#include "compute_loop_las2/compute_loop_las2.h"
#include "compute_loop_las_cuda/compute_loop_las_cuda.h"
#include "experimental/experimental.h"
#include "basic_cuda/basic_cuda.h"
// #include "huffman_cuda/huffman_cuda.h"
#include "huffman_mem_iter_cuda/huffman_mem_iter_cuda.h"
#include "huffman_hqs/huffman_hqs.h"
#include "compute_loop_las_hqs/compute_loop_las_hqs.h"
// #include "compute_loop_las_hqs_vr/compute_loop_las_hqs_vr.h"
// #include "compute_loop_nodes/compute_loop_nodes.h"
// #include "compute_loop_nodes_hqs/compute_loop_nodes_hqs.h"
// #include "compute_loop_nodes_hqs_vr/compute_loop_nodes_hqs_vr.h"

#include "compute_2021_earlyz/compute_2021_earlyz.h"
#include "compute_2021_earlyz_reduce/compute_2021_earlyz_reduce.h"
#include "compute_2021_dedup/compute_2021_dedup.h"
#include "compute_2021_hqs/compute_2021_hqs.h"
#include "compute_2021_gl/compute_2021_gl.h"

#include "compute_parametric/compute_parametric.h"

//#include "VrRuntime.h"
#include "Runtime.h"
#include "Method.h"
#include "compute/ComputeLasLoader.h"
#include "compute/LasLoaderStandard.h"
#include "compute/HuffmanLasLoader.h"
// #include "compute/PotreeData.h"



using namespace std;

int numPoints = 1'000'000;

int main(){

	cout << std::setprecision(2) << std::fixed;

	auto renderer = make_shared<Renderer>();
	//renderer->init();

	// Creating a CUDA context
	cuInit(0);
	CUdevice cuDevice;
	CUcontext context;
	cuDeviceGet(&cuDevice, 0);
	cuCtxCreate(&context, 0, cuDevice);

	auto tStart = now();

	struct Setting{
		string path_potree = "";
		string path_las = "";
    string path_huffman = "";
		float yaw = 0.0;
		float pitch = 0.0;
		float radius = 0.0;
		dvec3 target;
	};
	
	unordered_map<string, Setting> settings;


	{ // Test
		Setting setting;
		setting.path_potree = "D:/dev/pointclouds/tuwien_baugeschichte/candi Banyunibo/Scan10 - POLYDATA - Candi_Banyunibo010.las_converted";
		//setting.path_las = "D:/dev/pointclouds/tuwien_baugeschichte/candi Banyunibo/morton/10.las";
		// setting.path_las = "D:/dev/pointclouds/tuwien_baugeschichte/candi Banyunibo/candi_banyunibo.las";
		// setting.path_las = "F:/temp/wgtest/banyunibo_laserscans/merged.las";
		// setting.path_las = "/home/rg/lidar_data/tree.las";
		setting.path_las = "E:/resources/pointclouds/heidentor.las";
		setting.path_huffman = "E:/resources/rahul/sitn_4.1B.huffman";
		// setting.path_huffman = "/home/rg/repos/compute_rasterizer/out/data/transpose/transpose.huffman";
		// setting.path_las = "/home/rg/lidar_data/points.las";

		// outside
		setting.yaw = 4.39;
		setting.pitch = -0.18;
		setting.radius = 8.60;
		setting.target = {25.09, 36.09, 2.77};
		

		// Inside
		setting.yaw = 6.69;
		setting.pitch = -0.00;
		setting.radius = 1.87;
		setting.target = {41.44, 31.27, 4.31};

		setting.yaw = -0.15;
		setting.pitch = -0.57;
		setting.radius = 3166.32;
		setting.target = {2239.05, 1713.63, -202.02};

		settings["arbegen"] = setting;
	}

	{ // NEUCHATEL
		Setting setting;
		setting.path_las     = "E:/resources/pointclouds/heidentor.las";
		setting.path_huffman = "E:/resources/rahul/sitn_4.1B.huffman";

		// overview
		setting.yaw = -0.01;
		setting.pitch = -0.63;
		setting.radius = 3166.32;
		setting.target = {1826.89, 1679.88, 124.76};

		// closeup
		// setting.yaw = -0.52;
		// setting.pitch = -0.43;
		// setting.radius = 123.94;
		// setting.target = {1382.48, 1200.98, 118.31};

		settings["neuchatel"] = setting;
	}

	{ // FERRUM
		Setting setting;
		setting.path_las     = "E:/resources/pointclouds/heidentor.las";
		setting.path_huffman = "E:/resources/rahul/ca21_ferrum.huffman";

		// overview
		setting.yaw = -1.13;
		setting.pitch = -0.64;
		setting.radius = 1008.89;
		setting.target = { 700.99, 582.17, 16.55 };

		// closeup
		//setting.yaw = 0.05;
		//setting.pitch = -0.22;
		//setting.radius = 47.78;
		//setting.target = { 842.60, 301.91, 1.48 };

		settings["ferrum"] = setting;
	}

	{ // Salt Creek
		Setting setting;
		setting.path_las = "E:/resources/pointclouds/heidentor.las";
		setting.path_huffman = "E:/resources/rahul/ca21_saltcreek.huffman";

		// overview
		setting.yaw = -3.28;
		setting.pitch = -0.64;
		setting.radius = 1008.89;
		setting.target = { 340.13, 556.35, -117.12 };

		// closeup
		// setting.yaw = 0.99;
		// setting.pitch = -0.42;
		// setting.radius = 32.64;
		// setting.target = { 634.79, 538.68, 6.89 };

		settings["saltcreek"] = setting;
	}

	{ // Candi Banyunibo
		Setting setting;
		setting.path_las = "E:/resources/pointclouds/heidentor.las";
		setting.path_huffman = "E:/resources/rahul/candi_banyunibo.huffman";

		// outside
		setting.yaw = -13.63;
		setting.pitch = 0.07;
		setting.radius = 20.26;
		setting.target = { 38.64, 29.22, 5.23 };

		// inside
		//setting.yaw = -18.35;
		//setting.pitch = 0.32;
		//setting.radius = 3.01;
		//setting.target = {41.48, 31.31, 4.96};

		settings["banyubibo"] = setting;
	}

	{ // Morro Bay
		Setting setting;
		setting.path_las = "E:/resources/pointclouds/heidentor.las";
		setting.path_huffman = "./out/test.huffman";

		// overview
		setting.yaw = -0.15;
		setting.pitch = -0.57;
		setting.radius = 3166.32;
		setting.target = { 2239.05, 1713.63, -202.02 };

		// closeup
		//setting.yaw = -1.68;
		//setting.pitch = -0.39;
		//setting.radius = 69.96;
		//setting.target = { 1625.87, 2362.57, 29.96 };

		settings["morrobay"] = setting;
	}


	Setting& setting = settings["morrobay"];
	
	renderer->controls->yaw = setting.yaw;
	renderer->controls->pitch = setting.pitch;
	renderer->controls->radius = setting.radius;
	renderer->controls->target = setting.target;

	{ // ILLUSTRATE BIT RESOLUTION

		// settings["retz"].yaw = 2.95;
		// settings["retz"].pitch = -0.83;
		// settings["retz"].radius = 1342.83;
		// settings["retz"].target = {710.71, 891.63, -191.39};

		// settings["morrobay_morton"].yaw = -6.56;
		// settings["morrobay_morton"].pitch = -0.78;
		// settings["morrobay_morton"].radius = 1787.31;
		// settings["morrobay_morton"].target = {2064.40, 774.26, 562.50};

		// renderer->controls->yaw    = settings["morrobay_morton"].yaw;
		// renderer->controls->pitch  = settings["morrobay_morton"].pitch;
		// renderer->controls->radius = settings["morrobay_morton"].radius;
		// renderer->controls->target = settings["morrobay_morton"].target;


	}

	// auto potreedata = PotreeData::create(setting.path_potree);
	// auto las_encode_444 = ComputeLasData::create(setting.path_las);
	// auto las_standard = LasStandardData::create(setting.path_las);
  // auto las_basic = ComputeLasDataBasic::create(setting.path_las);
  auto las_huffman = HuffmanLasData::create(setting.path_huffman);

	{ // 4-4-4 byte format
		// auto computeLoopLas       = new ComputeLoopLas(renderer.get(), las_encode_444);
		// auto computeLoopLas2      = new ComputeLoopLas2(renderer.get(), las_encode_444);
		// auto computeLoopLasHqs    = new ComputeLoopLasHqs(renderer.get(), las_encode_444);
		// auto computeLoopLasHqsVR  = new ComputeLoopLasHqsVR(renderer.get(), las_encode_444);
		// auto computeCUDALas       = new ComputeLoopLasCUDA(renderer.get(), las_encode_444);
  //   auto experimental         = new Experimental(renderer.get(), las_encode_444);

		// Runtime::addMethod((Method*)computeLoopLas);
		// Runtime::addMethod((Method*)computeLoopLas2);
		// Runtime::addMethod((Method*)computeLoopLasHqs);
		// Runtime::addMethod((Method*)computeLoopLasHqsVR);
		// Runtime::addMethod((Method*)computeCUDALas);
		// Runtime::addMethod((Method*)experimental);
	}

  { // Rahul's methods
    // auto basic1 = new BasicCuda(renderer.get(), las_basic);
  //   auto basic2 = new BasicCuda(renderer.get(), las_basic);
    //auto huffman_cuda = new ComputeHuffman(renderer.get(), las_huffman);
    auto huffman_mem_iter_cuda = new HuffmanMemIter(renderer.get(), las_huffman);
    auto huffman_hqs = new HuffmanHQS(renderer.get(), las_huffman);

		// Runtime::addMethod((Method*) basic1);
		// Runtime::addMethod((Method*) basic2);
    //Runtime::addMethod((Method*)huffman_cuda);
    Runtime::addMethod((Method*)huffman_mem_iter_cuda);
    Runtime::addMethod((Method*)huffman_hqs);
  }

	{ // POTREE FORMAT
		// auto computeLoopNodes = new ComputeLoopNodes(renderer.get(), potreedata);
		// auto computeLoopNodesHqs = new ComputeLoopNodesHqs(renderer.get(), potreedata);
		// auto computeLoopNodesHqsVr = new ComputeLoopNodesHqsVr(renderer.get(), potreedata);
		// Runtime::addMethod((Method*)computeLoopNodes);
		// Runtime::addMethod((Method*)computeLoopNodesHqs);
		// Runtime::addMethod((Method*)computeLoopNodesHqsVr);
	}

  /*
	{ // OLD METHODS / 16 byte format
		//auto computeEarlyZ = new ComputeEarlyZ(renderer.get(), las_standard);
		//auto computeEarlyZReduce = new ComputeEarlyZReduce(renderer.get(), las_standard);
		auto computeDedup = new ComputeDedup(renderer.get(), las_standard);
		auto compute2021Hqs = new Compute2021HQS(renderer.get(), las_standard);
		auto compute2021GL = new ComputeGL(renderer.get(), las_standard);
		//Runtime::addMethod((Method*)computeEarlyZ);
		//Runtime::addMethod((Method*)computeEarlyZReduce);
		Runtime::addMethod((Method*)computeDedup);
		Runtime::addMethod((Method*)compute2021Hqs);
		Runtime::addMethod((Method*)compute2021GL);
	}
  */

	// { // PARAMETRIC
	// auto computeParametric = new ComputeParametric(renderer.get());
	// Runtime::addMethod((Method*)computeParametric);
	// }

	auto update = [&](){

		if(Debug::requestResetView){
			renderer->controls->yaw = setting.yaw;
			renderer->controls->pitch = setting.pitch;
			renderer->controls->radius = setting.radius;
			renderer->controls->target = setting.target;

			Debug::requestResetView = false;
		}

		auto selected = Runtime::getSelectedMethod();
		if(selected){

			bool needsVR = false;
			needsVR = needsVR || selected->name == "loop_las_hqs_vr";
			needsVR = needsVR || selected->name == "loop_nodes_hqs_vr";
			if(needsVR){
				renderer->setVR(true);
			}else{
				renderer->setVR(false);
			}
			

			selected->update(renderer.get());
		}
		
		if(Runtime::resource){

			string state = "";
			if(Runtime::resource->state == ResourceState::LOADED){
				state = "LOADED";
			}else if(Runtime::resource->state == ResourceState::LOADING){
				state = "LOADING";
			}else if(Runtime::resource->state == ResourceState::UNLOADED){
				state = "UNLOADED";
			}else if(Runtime::resource->state == ResourceState::UNLOADING){
				state = "UNLOADING";
			}
			

			Debug::set("state", state);
		}

		//{
		//	float t = now();

		//	renderer->controls->yaw = t;
		//	renderer->controls->pitch = -0.32;
		//	renderer->controls->radius = 24.50;
		//	renderer->controls->target = {40.57, 29.45, 3.87};
		//}

	};

	auto render = [&](){
		
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		if(renderer->vrEnabled){

			// auto ovr = OpenVRHelper::instance();


			// auto flip = glm::dmat4(
			// 	1.0, 0.0, 0.0, 0.0,
			// 	0.0, 0.0, 1.0, 0.0,
			// 	0.0, -1.0, 0.0, 0.0,
			// 	0.0, 0.0, 0.0, 1.0
			// );

			// auto& viewLeft = renderer->views[0];
			// auto& viewRight = renderer->views[1];

			


			// if(!Debug::dummyVR){
			// 	auto size = ovr->getRecommmendedRenderTargetSize();
			// 	viewLeft.framebuffer->setSize(size[0], size[1]);
			// 	viewRight.framebuffer->setSize(size[0], size[1]);

			// 	auto poseHMD = ovr->hmdPose;
			// 	auto poseLeft = ovr->getEyePose(vr::Hmd_Eye::Eye_Left);
			// 	auto poseRight = ovr->getEyePose(vr::Hmd_Eye::Eye_Right);

			// 	viewLeft.view = glm::inverse(flip * poseHMD * poseLeft);
			// 	viewLeft.proj = ovr->getProjection(vr::Hmd_Eye::Eye_Left, 0.01, 10'000.0);

			// 	viewRight.view = glm::inverse(flip * poseHMD * poseRight);
			// 	viewRight.proj = ovr->getProjection(vr::Hmd_Eye::Eye_Right, 0.01, 10'000.0);
			// }else{
			// 	ivec2 size = {2468, 2740};
			// 	viewLeft.framebuffer->setSize(size[0], size[1]);
			// 	viewRight.framebuffer->setSize(size[0], size[1]);
			// }

			// //viewLeft.framebuffer->setSize(1440, 1600);
			// glBindFramebuffer(GL_FRAMEBUFFER, viewLeft.framebuffer->handle);
			// glClearColor(0.8, 0.2, 0.3, 1.0);
			// glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			// //viewRight.framebuffer->setSize(1440, 1600);
			// glBindFramebuffer(GL_FRAMEBUFFER, viewRight.framebuffer->handle);
			// glClearColor(0.0, 0.8, 0.3, 1.0);
			// glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		}else{

			auto& view = renderer->views[0];

			view.view = renderer->camera->view;
			view.proj = renderer->camera->proj;

			renderer->views[0].framebuffer->setSize(renderer->width, renderer->height);

			glBindFramebuffer(GL_FRAMEBUFFER, view.framebuffer->handle);
			glClearColor(0.0, 0.2, 0.3, 1.0);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		}

		{
			glBindFramebuffer(GL_FRAMEBUFFER, 0);

			auto selected = Runtime::getSelectedMethod();
			if(selected){
				selected->render(renderer.get());
			}
		}

	};

	renderer->loop(update, render);

	return 0;
}

