__device__
unsigned int decode_color(int pointID, unsigned char *rgba) {
  int blockID = pointID / 16;
  int localID = pointID % 16;
  int offset = blockID * 8;

  unsigned char enc[8];
  for (int i = 0; i < 8; ++i) enc[i] = rgba[offset + i];

  unsigned int color0 = enc[0] | (enc[1] << 8);
  unsigned int rgb0[3];
  {
    unsigned int r5 = (color0 >> 11) & 31;
    unsigned int g6 = (color0 >>  5) & 63;
    unsigned int b5 = (color0 >>  0) & 31;
    unsigned int r8 = (r5 << 3) | (r5 >> 2);
    unsigned int g8 = (g6 << 2) | (g6 >> 4);
    unsigned int b8 = (b5 << 3) | (b5 >> 2);
    rgb0[0] = r8, rgb0[1] = g8, rgb0[2] = b8;
  }

  unsigned int color1 = enc[2] | (enc[3] << 8);
  unsigned int rgb1[3];
  {
    unsigned int r5 = (color1 >> 11) & 31;
    unsigned int g6 = (color1 >>  5) & 63;
    unsigned int b5 = (color1 >>  0) & 31;
    unsigned int r8 = (r5 << 3) | (r5 >> 2);
    unsigned int g8 = (g6 << 2) | (g6 >> 4);
    unsigned int b8 = (b5 << 3) | (b5 >> 2);
    rgb1[0] = r8, rgb1[1] = g8, rgb1[2] = b8;
  }

  int word = (enc[localID / 4 + 4] >> (2 * (localID % 4))) & 3;
  unsigned int color = -1;
  
  if (word == 0) color = rgb0[0] | (rgb0[1] << 8) | (rgb0[2] << 16);
  else if (word == 1) color = ((rgb0[0] * 2 + rgb1[0] * 1) / 3) | (((rgb0[1] * 2 + rgb1[1] * 1) / 3) << 8) | (((rgb0[2] * 2 + rgb1[2] * 1) / 3) << 16);
  else if (word == 2) color = ((rgb0[0] * 1 + rgb1[0] * 2) / 3) | (((rgb0[1] * 1 + rgb1[1] * 2) / 3) << 8) | (((rgb0[2] * 1 + rgb1[2] * 2) / 3) << 16);
  else if (word == 3) color = rgb1[0] | (rgb1[1] << 8) | (rgb1[2] << 16);
  color = rgb0[0] | (rgb0[1] << 8) | (rgb0[2] << 16);

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
      // color = rgba[pointID];
      color = decode_color(pointID, (unsigned char*) rgba);
    }
    // color = pointID * 1234567;
    // unsigned int depth = min(float(framebuffer[pixelID] >> 32) / 10000.0, 255.0);
    // color = (depth << 16) | (depth << 8) | depth;
	}
	surf2Dwrite(color, output, x*4, y);
}
