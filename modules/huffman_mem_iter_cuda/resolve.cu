#define COLOR_COMPRESSION 1 // 0 -> no compression, 1 -> bc1, 7 -> bc7

struct bc1_block
{
		unsigned char m_low_color[2];
		unsigned char m_high_color[2];
		unsigned char m_selectors[4];
};

__device__
unsigned int set_color(unsigned int r, unsigned int g, unsigned int b) {
  return r | (g << 8) | (b << 16);
}

__device__
unsigned int decode_bc1(int pointID, unsigned char *rgba) {
  int blockID = pointID / 16;
  int localID = pointID % 16;
  int offset = blockID * 8;

  const void* ptr = (void *) (rgba + offset);
  const bc1_block* pBlock = static_cast<const bc1_block*>(ptr);

  const unsigned int l = pBlock->m_low_color[0] | (pBlock->m_low_color[1] << 8U);
  const int cr0 = (l >> 11) & 31;
  const int cg0 = (l >> 5) & 63;
  const int cb0 = l & 31;
  const int r0 = (cr0 << 3) | (cr0 >> 2);
  const int g0 = (cg0 << 2) | (cg0 >> 4);
  const int b0 = (cb0 << 3) | (cb0 >> 2);

  const unsigned int h = pBlock->m_high_color[0] | (pBlock->m_high_color[1] << 8U);
  const int cr1 = (h >> 11) & 31;
  const int cg1 = (h >> 5) & 63;
  const int cb1 = h & 31;
  const int r1 = (cr1 << 3) | (cr1 >> 2);
  const int g1 = (cg1 << 2) | (cg1 >> 4);
  const int b1 = (cb1 << 3) | (cb1 >> 2);

  unsigned int color = -1;
  int word = (pBlock->m_selectors[localID / 4] >> (2 * (localID % 4))) & 3;
  switch (word) {
    case 0:
      color = set_color(r0, g0, b0);
      break;
    case 1:
      color = set_color(r1, g1, b1);
      break;
    case 2:
      color = set_color((r0 * 2 + r1) / 3, (g0 * 2 + g1) / 3, (b0 * 2 + b1) / 3);
      break;
    case 3:
      color = set_color((r0 + r1 * 2) / 3, (g0 + g1 * 2) / 3, (b0 + b1 * 2) / 3);
      break;
  }

  return color;
}

struct bc7_mode_6
{
  struct
  {
    unsigned long long m_mode : 7;
    unsigned long long m_r0 : 7;
    unsigned long long m_r1 : 7;
    unsigned long long m_g0 : 7;
    unsigned long long m_g1 : 7;
    unsigned long long m_b0 : 7;
    unsigned long long m_b1 : 7;
    unsigned long long m_a0 : 7;
    unsigned long long m_a1 : 7;
    unsigned long long m_p0 : 1;
  } m_lo;

  union
  {
    struct
    {
      unsigned long long m_p1 : 1;
      unsigned long long m_s00 : 3;
      unsigned long long m_s10 : 4;
      unsigned long long m_s20 : 4;
      unsigned long long m_s30 : 4;

      unsigned long long m_s01 : 4;
      unsigned long long m_s11 : 4;
      unsigned long long m_s21 : 4;
      unsigned long long m_s31 : 4;

      unsigned long long m_s02 : 4;
      unsigned long long m_s12 : 4;
      unsigned long long m_s22 : 4;
      unsigned long long m_s32 : 4;

      unsigned long long m_s03 : 4;
      unsigned long long m_s13 : 4;
      unsigned long long m_s23 : 4;
      unsigned long long m_s33 : 4;

    } m_hi;

    unsigned long long m_hi_bits;
  };
};

__device__
int linspace_idx(float start, float end, int num_points, int idx){
  float step = (end - start) / (num_points - 1);
  float val = start + idx * step;
  return round(val);
}

__device__
unsigned int decode_bc7(int pointID, unsigned char *rgba) {
  int blockID = pointID / 16;
  int localID = pointID % 16;
  int offset = blockID * 16;

  unsigned char enc[16];
  for (int i = 0; i < 16; ++i) enc[i] = rgba[offset + i];

	const bc7_mode_6 &block = *static_cast<const bc7_mode_6 *>((void *) enc);
	const unsigned int r0 = static_cast<unsigned int>((block.m_lo.m_r0 << 1) | block.m_lo.m_p0);
	const unsigned int g0 = static_cast<unsigned int>((block.m_lo.m_g0 << 1) | block.m_lo.m_p0);
	const unsigned int b0 = static_cast<unsigned int>((block.m_lo.m_b0 << 1) | block.m_lo.m_p0);
	const unsigned int a0 = static_cast<unsigned int>((block.m_lo.m_a0 << 1) | block.m_lo.m_p0);
	const unsigned int r1 = static_cast<unsigned int>((block.m_lo.m_r1 << 1) | block.m_hi.m_p1);
	const unsigned int g1 = static_cast<unsigned int>((block.m_lo.m_g1 << 1) | block.m_hi.m_p1);
	const unsigned int b1 = static_cast<unsigned int>((block.m_lo.m_b1 << 1) | block.m_hi.m_p1);
	const unsigned int a1 = static_cast<unsigned int>((block.m_lo.m_a1 << 1) | block.m_hi.m_p1);

  unsigned int color = -1;

  int idx = (block.m_hi_bits >> (localID * 4)) & 0xF;
  if (idx == 0) idx = (idx >> 1);
  const unsigned int w = linspace_idx(0, 64, 16, idx);
  const unsigned int iw = 64 - w;

  color =
  ((unsigned char) ((r0 * iw + r1 * w + 32) >> 6)) <<  0 |
  ((unsigned char) ((g0 * iw + g1 * w + 32) >> 6)) <<  8 |
  ((unsigned char) ((b0 * iw + b1 * w + 32) >> 6)) << 16 |
  ((unsigned char) ((a0 * iw + a1 * w + 32) >> 6)) << 24;

  return color;
}

extern "C" 
__global__ void kernel(
  int showNumPoints,
  int colorizeChunks,
	int width, int height, 
	cudaSurfaceObject_t output, 
	long long unsigned int* framebuffer, 
	unsigned int* rgba) 
{                                                             
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	

	int pixelID = x + y * width;
	unsigned int pointID = framebuffer[pixelID];
	unsigned int color = 0x00443322;
	if(pointID < 0x7FFFFFFF){
    if (showNumPoints) {
      int NumPointsToRender = framebuffer[pixelID];
      unsigned int shade = (unsigned int) (((float) NumPointsToRender / 512.0) * 255.0);
      shade = (shade << 24) | (shade << 16) | (shade << 8) | shade;
      color = shade;
    } else if (colorizeChunks) {
      color = pointID * 1234567;
    } else {
#if COLOR_COMPRESSION==0
      color = rgba[pointID];
#elif COLOR_COMPRESSION==1
      color = decode_bc1(pointID, (unsigned char*) rgba);
#elif COLOR_COMPRESSION==7
      color = decode_bc7(pointID, (unsigned char*) rgba);
#endif
    }
    // color = pointID * 1234567;
    // unsigned int depth = min(float(framebuffer[pixelID] >> 32) / 10000.0, 255.0);
    // color = (depth << 16) | (depth << 8) | depth;
	}
	surf2Dwrite(color, output, x*4, y);
}
