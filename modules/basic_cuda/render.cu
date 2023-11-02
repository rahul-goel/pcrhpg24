#include "kernel_data.h"
#include "helper_math.h"

__device__ float4 matMul(const Mat& m, const float4& v)
{
	return make_float4(dot(m.rows[0], v), dot(m.rows[1], v), dot(m.rows[2], v), dot(m.rows[3], v));
}

struct Plane{
	float3 normal;
	float constant;
};

__device__ float access(float4 v, int i)
{
	if (i == 0)
		return v.x;
	if (i == 1)
		return v.y;
	if (i == 2)
		return v.z;
	return v.w;
}
__device__ float t(const ChangingRenderData& data, int index)
{
	int a = index % 4;
	int b = index / 4;
	return access(data.uTransform.rows[a], b);
}

__device__ float distanceToPoint(float3 point, Plane plane){
	return (plane.normal.x * point.x + plane.normal.y*point.y + plane.normal.z*point.z) + plane.constant;
}

__device__ Plane createPlane(float x, float y, float z, float w){

	float nLength = sqrt(x*x + y*y + z*z);
	Plane plane;
	plane.normal = make_float3(x, y, z) / nLength;
	plane.constant = w / nLength;
	return plane;
}
__device__ bool intersectsFrustum(const ChangingRenderData& d, float3 wgMin, float3 wgMax){

	Plane planes[6] = {
		createPlane(t( d,3) - t(d,0), t( d,7) - t(d,4), t(d,11) - t( d,8), t(d,15) - t(d,12)),
		createPlane(t( d,3) + t(d,0), t( d,7) + t(d,4), t(d,11) + t( d,8), t(d,15) + t(d,12)),
		createPlane(t( d,3) + t(d,1), t( d,7) + t(d,5), t(d,11) + t( d,9), t(d,15) + t(d,13)),
		createPlane(t( d,3) - t(d,1), t( d,7) - t(d,5), t(d,11) - t( d,9), t(d,15) - t(d,13)),
		createPlane(t( d,3) - t(d,2), t( d,7) - t(d,6), t(d,11) - t(d,10), t(d,15) - t(d,14)),
		createPlane(t( d,3) + t(d,2), t( d,7) + t(d,6), t(d,11) + t(d,10), t(d,15) + t(d,14)),
	};
	for(int i = 0; i < 6; i++){

		Plane plane = planes[i];

		float3 vector;
		vector.x = plane.normal.x > 0.0 ? wgMax.x : wgMin.x;
		vector.y = plane.normal.y > 0.0 ? wgMax.y : wgMin.y;
		vector.z = plane.normal.z > 0.0 ? wgMax.z : wgMin.z;

		float d = distanceToPoint(vector, plane);

		if(d < 0){
			return false;
		}
	}

	return true;
}

__device__ void rasterize(const ChangingRenderData& data, unsigned long long int* framebuffer, float3 point, unsigned int index)
{
	float4 pos = matMul(data.uTransform, make_float4(point, 1.0f));

	pos.x = pos.x / pos.w;
	pos.y = pos.y / pos.w;

	float2 imgPos = {(pos.x * 0.5f + 0.5f) * data.uImageSize.x, (pos.y * 0.5f + 0.5f) * data.uImageSize.y};
	int2 pixelCoords = make_int2(imgPos.x, imgPos.y);
	int pixelID = pixelCoords.x + pixelCoords.y * data.uImageSize.x;

	unsigned int depth = *((int*)&pos.w);
	unsigned long long int newPoint = (((unsigned long long int)depth) << 32) | index;

	if(!(pos.w <= 0.0 || pos.x < -1 || pos.x > 1 || pos.y < -1|| pos.y > 1)){
		unsigned long long int oldPoint = framebuffer[pixelID];
		if(newPoint < oldPoint){
			atomicMin(&framebuffer[pixelID], newPoint);
		}
	}
}


// rendering kernel
extern "C" __global__
void kernel(const ChangingRenderData data,
            unsigned long long int *framebuffer,
            XYZBatch *batches,
            int *ssXyz,
            unsigned int *rgba) {
  unsigned int batchIndex = blockIdx.x;
  unsigned int numPointerPerBatch = data.uPointsPerThread * blockDim.x;
  unsigned int wgFirstPoint = batchIndex * numPointerPerBatch;

  // read bounding box data
  XYZBatch batch = batches[batchIndex];
	float3 wgMin = make_float3(batch.min_x, batch.min_y, batch.min_z);
	float3 wgMax = make_float3(batch.max_x, batch.max_y, batch.max_z);
	float3 boxSize = wgMax - wgMin;
  float3 wgScale = make_float3(batch.scale_x, batch.scale_y, batch.scale_z);
  float3 wgOffset = make_float3(batch.offset_x, batch.offset_y, batch.offset_z);
  float3 uboxMin = data.uBoxMin;
  float3 uScale = data.uScale;
  float3 uOffset = data.uBoxMax;

  // frustum cull using bounding box
	if((data.uEnableFrustumCulling != 0) && !intersectsFrustum(data, wgMin, wgMax)){
		return;
	}

  // why are we doing this?
	if (blockIdx.x == gridDim.x - 1) return;

	int loopSize = data.uPointsPerThread;	
  for (int i = 0; i < loopSize; ++i) {
    unsigned int pointIndex = wgFirstPoint + i * blockDim.x + threadIdx.x;
    int X = ssXyz[pointIndex * 3 + 0];
    int Y = ssXyz[pointIndex * 3 + 1];
    int Z = ssXyz[pointIndex * 3 + 2];

    float x = (float) X * uScale.x + uOffset.x - uboxMin.x;
    float y = (float) Y * uScale.y + uOffset.y - uboxMin.y;
    float z = (float) Z * uScale.z + uOffset.z - uboxMin.z;
    float3 point = make_float3(x, y, z);

    rasterize(data, framebuffer, point, pointIndex);
  }
}
