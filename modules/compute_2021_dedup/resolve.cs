
#version 450

#extension GL_ARB_gpu_shader_int64 : enable
#extension GL_NV_shader_atomic_int64 : enable 

layout(local_size_x = 16, local_size_y = 16) in;

layout (std430, binding=1) buffer framebuffer_data {
	uint64_t ssFramebuffer[];
};

layout(rgba8ui, binding = 0) uniform uimage2D uOutput;

uvec4 colorAt(int pixelID){
	uint64_t val64 = ssFramebuffer[pixelID];

	uint ucol = uint(val64 & 0x00FFFFFFUL);
	vec4 color = 255.0 * unpackUnorm4x8(ucol);
	uvec4 icolor = uvec4(color);

	return icolor;
}

void main(){
	uvec2 id = gl_LocalInvocationID.xy;
	id.x += gl_WorkGroupSize.x * gl_WorkGroupID.x;
	id.y += gl_WorkGroupSize.y * gl_WorkGroupID.y;

	ivec2 imgSize = imageSize(uOutput);

	ivec2 pixelCoords = ivec2(id);
	ivec2 sourceCoords = ivec2(id);
	int pixelID = sourceCoords.x + sourceCoords.y * imgSize.x;

	
	uvec4 icolor = colorAt(pixelID);

	imageStore(uOutput, pixelCoords, icolor);

	// rgb(60, 50, 30) = 0x3c321e (0xRGB)
	ssFramebuffer[pixelID] = 0xffffffff00443322UL;
}
