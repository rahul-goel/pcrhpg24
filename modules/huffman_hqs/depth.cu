#define DTABLE_SIZE 4096
#define CLUSTERS_PER_THREAD 1

#include "huffman_kernel_data.h"
#include "helper_math.h"

struct Mat {
	float4 rows[4];
};

struct BoundingBox{
	float4 position;   		//   16   0
	float4 size;       		//   16  16
	unsigned int color;     //    4  32
							// size: 48
};

struct BoundingData {
	unsigned int count;
	unsigned int instanceCount;
	unsigned int first;
	unsigned int baseInstance;
	// 16
	unsigned int pad0;
	unsigned int pad1;
	unsigned int pad2;
	unsigned int pad3;
	// 32
	unsigned int pad4;
	unsigned int pad5;
	unsigned int pad6;
	unsigned int pad7;
	// 48
	BoundingBox* ssBoxes;
};

struct ChangingRenderData{
	Mat uTransform;
	Mat uWorldView;
	Mat uProj;
	
	float3 uCamPos;
	int2 uImageSize;
	int uPointsPerThread;
	
	float3 uBoxMin;
	float3 uBoxMax;
	int uNumPoints;
	long long int uOffsetPointToData;
	int uPointFormat;
	long long int uBytesPerPoint;
	float3 uScale;
	
	int uEnableFrustumCulling;
  int showNumPoints;
  int colorizeChunks;
};

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

__device__ void rasterize(const ChangingRenderData& data, unsigned long long int* framebuffer, float3 point, unsigned long long index, unsigned int NumPointsToRender)
{
	float4 pos = matMul(data.uTransform, make_float4(point, 1.0f));

	pos.x = pos.x / pos.w;
	pos.y = pos.y / pos.w;

	float2 imgPos = {(pos.x * 0.5f + 0.5f) * data.uImageSize.x, (pos.y * 0.5f + 0.5f) * data.uImageSize.y};
	int2 pixelCoords = make_int2(imgPos.x, imgPos.y);
	int pixelID = pixelCoords.x + pixelCoords.y * data.uImageSize.x;

	unsigned int depth = *((int*)&pos.w);
	unsigned long long int newPoint;
  if (data.showNumPoints)
    newPoint = (((unsigned long long int)depth) << 32) | (NumPointsToRender * CLUSTERS_PER_THREAD);
  else if (data.colorizeChunks)
    newPoint = (((unsigned long long int)depth) << 32) | blockIdx.x;
  else
    newPoint = (((unsigned long long int)depth) << 32);
    // newPoint = (((unsigned long long int)depth) << 32) | index;

	if(!(pos.w <= 0.0 || pos.x < -1 || pos.x > 1 || pos.y < -1|| pos.y > 1)){
		unsigned long long int oldPoint = framebuffer[pixelID];
		if(newPoint < oldPoint){
			atomicMin(&framebuffer[pixelID], newPoint);
		}
	}
}

__device__
float mysmoothstep(float x) {
  const float p = 0.35f;
  const float s = 0.60f;
  const float c = 2.0f / (1.0f - s) - 1.0f;

  if (x <= p) return pow(x, c) / pow(p, c - 1.0f);
  else return 1.0f - pow(1.0f - x, c) / pow(1.0f - p, c - 1.0f);
}

extern "C" __global__
void kernel(const ChangingRenderData           cdata,
                  unsigned long long          *framebuffer,
                  GPUBatch                    *BatchData,
                  int                         *StartValues,
                  unsigned int                *EncodedData,
                  int                         *SeparateData,
                  int                         *SeparateDataSizes,
                  int                         *DecoderTableValues,
                  int                         *DecoderTableCWLen,
                  int                         *ClusterSizes,
                  unsigned int                *Colors
                  ) {
  unsigned int batchIndex = blockIdx.x;
  unsigned int numPointsPerBatch = blockDim.x * cdata.uPointsPerThread;
  unsigned long long wgFirstPoint = (unsigned long long) batchIndex * (unsigned long long) numPointsPerBatch;

  // batch meta data
  GPUBatch batch = BatchData[batchIndex];
  // double3 las_scale = make_double3(batch.scale_x, batch.scale_y, batch.scale_z);
  // float3 las_offset = make_float3(batch.offset_x - batch.las_min_x, batch.offset_y - batch.las_min_y, batch.offset_z - batch.las_min_z);
  float3 las_min = make_float3(batch.las_min_x, batch.las_min_y, batch.las_min_z);


  // frustum cull if enabled
  float3 batchMin = make_float3(batch.min_x, batch.min_y, batch.min_z) - las_min;
  float3 batchMax = make_float3(batch.max_x, batch.max_y, batch.max_z) - las_min;
  if (cdata.uEnableFrustumCulling && !intersectsFrustum(cdata, batchMin, batchMax)) {
    return;
  }

  // figuring out the LOD level
  __shared__ int Shared_NumPointsToRender;
  __shared__ bool Shared_UseDouble;
  float3 batchCenter = 0.5f * (batchMin + batchMax);
  if (threadIdx.x == 0) {
		float wgRadius = length(batchMin - batchMax);
		float4 viewCenter = matMul(cdata.uWorldView, make_float4(batchCenter, 1.0f));
		float4 viewEdge = viewCenter + make_float4(wgRadius, 0.0f, 0.0f, 0.0f);
		float4 projCenter = matMul(cdata.uProj, viewCenter);
		float4 projEdge = matMul(cdata.uProj, viewEdge);

		float2 projCenter2D = make_float2(projCenter.x, projCenter.y);
		float2 projEdge2D = make_float2(projEdge.x, projEdge.y);
		projCenter2D /= projCenter.w;
		projEdge2D /= projEdge.w;
		
		float2 screenCenter = 0.5f * (projCenter2D + 1.0f);
		screenCenter = make_float2(cdata.uImageSize.x * screenCenter.x, cdata.uImageSize.y * screenCenter.y);
		float2 screenEdge = 0.5f * (projEdge2D + 1.0f);
		screenEdge = make_float2(cdata.uImageSize.x * screenEdge.x, cdata.uImageSize.y * screenEdge.y);
		float2 diff = screenEdge - screenCenter;
		float pixelSize = sqrt(diff.x*diff.x + diff.y*diff.y);
    float percentage = 0;

    Shared_UseDouble = pixelSize >= 100.0f;

    pixelSize /= 100.0;
    percentage = (1.8f * pixelSize - 0.3);
    percentage = clamp(percentage, float(cdata.uPointFormat) / 100.0f, 1.0f);
    Shared_NumPointsToRender = min((int) (percentage * cdata.uPointsPerThread / CLUSTERS_PER_THREAD), cdata.uPointsPerThread / CLUSTERS_PER_THREAD);
  }
  __syncthreads();
  int NumPointsToRender = Shared_NumPointsToRender;
  bool UseDouble = Shared_UseDouble;
  // UseDouble = true;


  // copying decoder table to shared memory
  int max_cw_size = (int) batch.max_cw_len;
  int DCO = batch.decoder_table_offset;
  unsigned int mask = ((1 << max_cw_size) - 1) << (32 - max_cw_size);

  __shared__ int Shared_DecoderTableValues[DTABLE_SIZE];
  __shared__ char Shared_DecoderTableCWLen[DTABLE_SIZE];
  for (int i = 0; i < (1 << max_cw_size) / blockDim.x; ++i) {
    int idx = i * blockDim.x + threadIdx.x;
    Shared_DecoderTableValues[idx] = DecoderTableValues[DCO + idx];
    Shared_DecoderTableCWLen[idx] = DecoderTableCWLen[DCO + idx];
  }
  __syncthreads();


  if (UseDouble) {
    double3 las_scale = make_double3(batch.scale_x, batch.scale_y, batch.scale_z);
    double3 las_offset = make_double3(batch.offset_x - batch.las_min_x, batch.offset_y - batch.las_min_y, batch.offset_z - batch.las_min_z);
    #pragma unroll
    for (int outer = 0; outer < CLUSTERS_PER_THREAD; ++outer) {
      // where to read encoded data from?
      long long EncodedPtr = batch.encoding_batch_offset;
      long long sep_ptr = batch.separate_batch_offset;

      int ClusterIdx = (blockDim.x / 32 * outer) + threadIdx.x / 32;
      if (ClusterIdx >= 1) {
        EncodedPtr += ClusterSizes[blockIdx.x * (blockDim.x * CLUSTERS_PER_THREAD / 32) + ClusterIdx - 1];
      }
      if (outer != 0 or threadIdx.x != 0) {
        sep_ptr += SeparateDataSizes[blockIdx.x * blockDim.x * CLUSTERS_PER_THREAD + outer * blockDim.x + threadIdx.x - 1];
      }

      int tid = threadIdx.x % 32;
      unsigned int CurHuffman = EncodedData[EncodedPtr + tid];
      unsigned int NextHuffman = EncodedData[EncodedPtr + 32 + tid];
      int already_read = 64;
      int cur_bits = 32;

      const int sval_idx = blockIdx.x * blockDim.x * CLUSTERS_PER_THREAD + outer * blockDim.x + threadIdx.x;
      int3 prev_values = make_int3(StartValues[sval_idx * 3 + 0],
                                   StartValues[sval_idx * 3 + 1],
                                   StartValues[sval_idx * 3 + 2]);


      // main loop
      for (int i = 0; i < NumPointsToRender; ++i) {
        int decoded[3];
        for (int j = 0; j < 3; ++j) {
          unsigned int L = cur_bits == 32 ? CurHuffman : (CurHuffman << (32 - cur_bits));
          unsigned int R = cur_bits == 32 ? 0 : (NextHuffman >> cur_bits);
          unsigned int key = ((L|R) & mask) >> (32 - max_cw_size);

          int symbol = Shared_DecoderTableValues[key];
          int cw_size = Shared_DecoderTableCWLen[key];

          decoded[j] = (cw_size > 0 ? symbol : SeparateData[sep_ptr++]);
          cur_bits -= abs(cw_size);

          // (cur_bits <= 0) signifies whether the thread is out of bits or not
          bool need_to_read = cur_bits <= 0;
          unsigned int warp_mask = __ballot_sync(0xffffffff, need_to_read);
          if (need_to_read) {
            int offset = __popc(warp_mask << (32 - tid));
            CurHuffman = NextHuffman;
            NextHuffman = EncodedData[EncodedPtr + already_read + offset];
            cur_bits += 32;
          }
          already_read += __popc(warp_mask);
        }

        unsigned long long pointIndex = wgFirstPoint
										+ (unsigned long long) (outer * numPointsPerBatch / CLUSTERS_PER_THREAD)
										+ (unsigned long long) (threadIdx.x * cdata.uPointsPerThread / CLUSTERS_PER_THREAD)
										+ (unsigned long long) i;
        int3 cur_values = make_int3(decoded[0] + prev_values.x,
                                    decoded[1] + prev_values.y,
                                    decoded[2] + prev_values.z);


        float3 cur_xyz = make_float3(double(cur_values.x) * las_scale.x + las_offset.x,
                                     double(cur_values.y) * las_scale.y + las_offset.y,
                                     double(cur_values.z) * las_scale.z + las_offset.z);
        
        prev_values = cur_values;

        rasterize(cdata, framebuffer, cur_xyz, pointIndex, NumPointsToRender);
      }
    }
  } else {
    float3 las_scale = make_float3(batch.scale_x, batch.scale_y, batch.scale_z);
    float3 las_offset = make_float3(batch.offset_x - batch.las_min_x, batch.offset_y - batch.las_min_y, batch.offset_z - batch.las_min_z);
    #pragma unroll
    for (int outer = 0; outer < CLUSTERS_PER_THREAD; ++outer) {
      // where to read encoded data from?
      long long EncodedPtr = batch.encoding_batch_offset;
      long long sep_ptr = batch.separate_batch_offset;

      int ClusterIdx = (blockDim.x / 32 * outer) + threadIdx.x / 32;
      if (ClusterIdx >= 1) {
        EncodedPtr += ClusterSizes[blockIdx.x * (blockDim.x * CLUSTERS_PER_THREAD / 32) + ClusterIdx - 1];
      }
      if (outer != 0 or threadIdx.x != 0) {
        sep_ptr += SeparateDataSizes[blockIdx.x * blockDim.x * CLUSTERS_PER_THREAD + outer * blockDim.x + threadIdx.x - 1];
      }

      int tid = threadIdx.x % 32;
      unsigned int CurHuffman = EncodedData[EncodedPtr + tid];
      unsigned int NextHuffman = EncodedData[EncodedPtr + 32 + tid];
      int already_read = 64;
      int cur_bits = 32;

      const int sval_idx = blockIdx.x * blockDim.x * CLUSTERS_PER_THREAD + outer * blockDim.x + threadIdx.x;
      int3 prev_values = make_int3(StartValues[sval_idx * 3 + 0],
                                   StartValues[sval_idx * 3 + 1],
                                   StartValues[sval_idx * 3 + 2]);


      // main loop
      for (int i = 0; i < NumPointsToRender; ++i) {
        int decoded[3];
        for (int j = 0; j < 3; ++j) {
          unsigned int L = cur_bits == 32 ? CurHuffman : (CurHuffman << (32 - cur_bits));
          unsigned int R = cur_bits == 32 ? 0 : (NextHuffman >> cur_bits);
          unsigned int key = ((L|R) & mask) >> (32 - max_cw_size);

          int symbol = Shared_DecoderTableValues[key];
          int cw_size = Shared_DecoderTableCWLen[key];

          decoded[j] = (cw_size > 0 ? symbol : SeparateData[sep_ptr++]);
          cur_bits -= abs(cw_size);

          // (cur_bits <= 0) signifies whether the thread is out of bits or not
          bool need_to_read = cur_bits <= 0;
          unsigned int warp_mask = __ballot_sync(0xffffffff, need_to_read);
          if (need_to_read) {
            int offset = __popc(warp_mask << (32 - tid));
            CurHuffman = NextHuffman;
            NextHuffman = EncodedData[EncodedPtr + already_read + offset];
            cur_bits += 32;
          }
          already_read += __popc(warp_mask);
        }

        unsigned long long pointIndex = wgFirstPoint
										+ (unsigned long long) (outer * numPointsPerBatch / CLUSTERS_PER_THREAD)
										+ (unsigned long long) (threadIdx.x * cdata.uPointsPerThread / CLUSTERS_PER_THREAD)
										+ (unsigned long long) i;
        int3 cur_values = make_int3(decoded[0] + prev_values.x,
                                    decoded[1] + prev_values.y,
                                    decoded[2] + prev_values.z);


        float3 cur_xyz = make_float3(float(cur_values.x) * las_scale.x,
                                     float(cur_values.y) * las_scale.y,
                                     float(cur_values.z) * las_scale.z) + las_offset;
        
        prev_values = cur_values;

        rasterize(cdata, framebuffer, cur_xyz, pointIndex, NumPointsToRender);
      }
    }
  }

}
