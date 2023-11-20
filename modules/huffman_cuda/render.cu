#define SHARED_DTABLE
#define TRANSPOSE_ENCODED
#define DTABLE_SIZE 4096

#include "huffman_kernel_data.h"
#include "helper_math.h"
// #include "compute_loop_las_cuda/kernel_data.h"

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

extern "C" __global__
void kernel(const ChangingRenderData            cdata,
                    unsigned long long          *framebuffer,
                    GPUBatch                    *BatchData,
                    int                         *StartValues,
                    unsigned int                *EncodedData,
                    int                         *EncodedDataOffsets,
                    int                         *EncodedDataSizes,
                    int                         *SeparateData,
                    int                         *SeparateDataOffsets,
                    int                         *SeparateDataSizes,
                    int                         *DecoderTableValues,
                    int                         *DecoderTableCWLen,
                    unsigned int                *Colors
                    ) {
  
  unsigned int batchIndex = blockIdx.x;
  unsigned int numPointsPerBatch = blockDim.x * cdata.uPointsPerThread;
  unsigned int wgFirstPoint = batchIndex * numPointsPerBatch;
  unsigned int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;

  // right now we dont want to deal with the edge case of last batch
	if (blockIdx.x == gridDim.x - 1) return;

  // batch meta data
  GPUBatch batch = BatchData[batchIndex];
  float3 las_offset = make_float3(batch.offset_x, batch.offset_y, batch.offset_z);
  float3 las_scale = make_float3(batch.scale_x, batch.scale_y, batch.scale_z);
  float3 las_min = make_float3(batch.las_min_x, batch.las_min_y, batch.las_min_z);
  // printf("%d %lld\n", (int) batchIndex, batch.decoder_table_offset);


  // frustum cull if enabled
  float3 batchMin = make_float3(batch.min_x, batch.min_y, batch.min_z) - las_min;
  float3 batchMax = make_float3(batch.min_x, batch.min_y, batch.min_z) - las_min;
  if (cdata.uEnableFrustumCulling && !intersectsFrustum(cdata, batchMin, batchMax)) {
    return;
  }

  int3 prev_values = make_int3(StartValues[globalThreadIdx * 3 + 0],
                               StartValues[globalThreadIdx * 3 + 1],
                               StartValues[globalThreadIdx * 3 + 2]);


  // tracker variables for huffman
  int max_cw_size = (int) batch.max_cw_len;
  long long EncodedPtr = batch.encoding_batch_offset;
  long long SeparatePtr = batch.separate_batch_offset;

#ifdef TRANSPOSE_ENCODED
  int cur_ptr = EncodedPtr + threadIdx.x;
  int sep_ptr = SeparateDataOffsets[globalThreadIdx] + SeparatePtr;
#else
  int cur_ptr = EncodedDataOffsets[globalThreadIdx] + EncodedPtr;
  int sep_ptr = SeparateDataOffsets[globalThreadIdx] + SeparatePtr;
#endif

  int DCO = batch.decoder_table_offset;
  int cur_bits = 32;
  unsigned int mask = ((1 << max_cw_size) - 1) << (32 - max_cw_size);

#ifdef SHARED_DTABLE
  __shared__ int Shared_DecoderTableValues[DTABLE_SIZE];
  __shared__ char Shared_DecoderTableCWLen[DTABLE_SIZE];
  for (int i = 0; i < (1 << max_cw_size) / blockDim.x; ++i) {
    int idx = i * blockDim.x + threadIdx.x;
    Shared_DecoderTableValues[idx] = DecoderTableValues[DCO + idx];
    Shared_DecoderTableCWLen[idx] = DecoderTableCWLen[DCO + idx];
  }
  __syncthreads();
#endif


  // TODO - if using, correct it using EncodedPtr, SeparatePtr
  // int EDS = EncodedDataSizes[globalThreadIdx];
  // int SDS = SeparateDataSizes[globalThreadIdx];

#ifdef TRANSPOSE_ENCODED
  unsigned int CurHuffman = EncodedData[cur_ptr];
  unsigned int NextHuffman = EncodedData[cur_ptr + blockDim.x];
#else
  unsigned int CurHuffman = EncodedData[cur_ptr];
  unsigned int NextHuffman = EncodedData[cur_ptr + 1];
#endif

  for (int i = 0; i < cdata.uPointsPerThread; ++i) {
    // decode the next delta value for X, Y and Z
    int decoded[3];
    for (int j = 0; j < 3; ++j) {
      unsigned int L = cur_bits == 32 ? CurHuffman : (CurHuffman << (32 - cur_bits));
      unsigned int R = cur_bits == 32 ? 0 : (NextHuffman >> cur_bits);
      unsigned int key = ((L|R) & mask) >> (32 - max_cw_size);

#ifdef SHARED_DTABLE
      int symbol = Shared_DecoderTableValues[key];
      int cw_size = Shared_DecoderTableCWLen[key];
#else
      int DT_lookup = DCO + key;
      int symbol = DecoderTableValues[DT_lookup];
      int cw_size = DecoderTableCWLen[DT_lookup];
#endif

      decoded[j] = (cw_size > 0 ? symbol : SeparateData[sep_ptr++]);
      cur_bits -= abs(cw_size);

      // got over
      if (cur_bits <= 0) {
#ifdef TRANSPOSE_ENCODED
        cur_ptr += blockDim.x;
        CurHuffman = NextHuffman;
        NextHuffman = EncodedData[cur_ptr + blockDim.x];
#else
        cur_ptr += 1;
        CurHuffman = NextHuffman;
        NextHuffman = EncodedData[cur_ptr + 1];
#endif
        cur_bits += 32;
      }
    }

    unsigned int pointIndex = wgFirstPoint + threadIdx.x * cdata.uPointsPerThread + i;
    int3 cur_values = make_int3(decoded[0] + prev_values.x,
                                decoded[1] + prev_values.y,
                                decoded[2] + prev_values.z);

    float3 cur_xyz = make_float3(cur_values.x, cur_values.y, cur_values.z) * las_scale + las_offset - las_min;
    prev_values = cur_values;

    rasterize(cdata, framebuffer, cur_xyz, pointIndex);
  }
  return;
}
