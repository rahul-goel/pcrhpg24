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
      unsigned int shade = (unsigned int) (((float) NumPointsToRender / 500.0) * 255.0);
      shade = (shade << 24) | (shade << 16) | (shade << 8) | shade;
      color = shade;
    } else if (colorizeChunks) {
      color = pointID * 1234567;
    } else {
      color = rgba[pointID];
    }
    // color = pointID * 1234567;
    // unsigned int depth = min(float(framebuffer[pixelID] >> 32) / 10000.0, 255.0);
    // color = (depth << 16) | (depth << 8) | depth;
	}
	surf2Dwrite(color, output, x*4, y);
}
