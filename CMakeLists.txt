cmake_minimum_required(VERSION 3.26)
project(compute_rasterizer LANGUAGES C CXX CUDA)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++17;) # Assuming C++17 for CUDA

file(GLOB SOURCES "src/*")
add_executable(${PROJECT_NAME}
  src/main.cpp
  src/Renderer.cpp
  src/Framebuffer.cpp
  src/GLTimerQueries.cpp
  src/OrbitControls.cpp
  src/Texture.cpp
  src/miniz.c
  include/unsuck_platform_specific.cpp
  modules/compute/ComputeLasLoader.cpp
  modules/compute/HuffmanLasLoader.cpp
  modules/compute/LasLoaderSparse.cpp
)

# add_executable(rahul
#   src/rahul.cpp
#   src/Renderer.cpp
#   src/Framebuffer.cpp
#   src/GLTimerQueries.cpp
#   src/OrbitControls.cpp
#   src/Texture.cpp
#   include/unsuck_platform_specific.cpp
#   modules/compute/ComputeLasLoader.cpp
#   modules/compute/HuffmanLasLoader.cpp
#   modules/compute/LasLoaderSparse.cpp
# )

add_executable(test_huffman
  src/test_huffman.cpp
)

add_executable(preprocess
  src/preprocess.cpp
  src/rgbcx.cpp
  src/bc7enc.cpp
  src/bc7decomp.cpp
  src/Renderer.cpp
  src/Framebuffer.cpp
  src/GLTimerQueries.cpp
  src/OrbitControls.cpp
  src/Texture.cpp
  src/miniz.c
  include/unsuck_platform_specific.cpp
  modules/compute/ComputeLasLoader.cpp
  modules/compute/HuffmanLasLoader.cpp
)

include_directories(
    include
    modules
    libs/cyCodeBase
)

# target_link_libraries(${PROJECT_NAME} include/PotreeData.h)

# CUDA toolkit
find_package(CUDAToolkit 11.8.89 REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE
	CUDAToolkit_INCLUDE_DIRS)
target_link_libraries(${PROJECT_NAME}
	CUDA::cuda_driver
	CUDA::nvrtc
	CUDA::nvrtc_static
	CUDA::nvrtc_builtins_static)
target_include_directories(preprocess PRIVATE
	CUDAToolkit_INCLUDE_DIRS)
target_link_libraries(preprocess
	CUDA::cuda_driver
	CUDA::nvrtc
	CUDA::nvrtc_static
	CUDA::nvrtc_builtins_static)

# TBB
if (UNIX)
find_package(TBB REQUIRED)
target_link_libraries(${PROJECT_NAME} 
  TBB::tbb)
target_link_libraries(preprocess
  TBB::tbb)
endif (UNIX)


# OpenGL
find_package(OpenGL REQUIRED)
# target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARY})
link_libraries(${OPENGL_LIBRARY})

# fmt 
# add_subdirectory(libs/fmt)
# target_link_libraries(${PROJECT_NAME} fmt::fmt)

# add_subdirectory(libs/libmorton)
include_directories(libs/libmorton/include/libmorton)

# glew
target_include_directories(${PROJECT_NAME} PRIVATE
	libs/glew/include)
target_sources(${PROJECT_NAME} PRIVATE
	libs/glew/glew.c)
target_include_directories(preprocess PRIVATE
	libs/glew/include)
target_sources(preprocess PRIVATE
	libs/glew/glew.c)

# GLFW
# include(cmake/glfw.cmake)
# target_link_libraries(${PROJECT_NAME} glfw)
add_subdirectory(libs/glfw)
include_directories(libs/glfw/include)
target_link_libraries(${PROJECT_NAME} glfw)
target_link_libraries(preprocess glfw)

# glm
include_directories(libs/glm)

# add json
include_directories(libs/json)

# flann
include_directories(libs/flann/src/cpp/)

# imgui
include_directories(
	libs/imgui
	libs/imgui/backends)
target_sources(${PROJECT_NAME} PRIVATE
	libs/imgui/imgui.cpp
	libs/imgui/imgui_demo.cpp
	libs/imgui/imgui_draw.cpp
	libs/imgui/imgui_tables.cpp
	libs/imgui/imgui_widgets.cpp
	libs/imgui/backends/imgui_impl_glfw.cpp
	libs/imgui/backends/imgui_impl_opengl3.cpp)
target_sources(preprocess PRIVATE
	libs/imgui/imgui.cpp
	libs/imgui/imgui_demo.cpp
	libs/imgui/imgui_draw.cpp
	libs/imgui/imgui_tables.cpp
	libs/imgui/imgui_widgets.cpp
	libs/imgui/backends/imgui_impl_glfw.cpp
	libs/imgui/backends/imgui_impl_opengl3.cpp)

# implot
include_directories(libs/implot)
target_sources(${PROJECT_NAME} PRIVATE
	libs/implot/implot_items.cpp
	libs/implot/implot.cpp)
target_sources(preprocess PRIVATE
	libs/implot/implot_items.cpp
	libs/implot/implot.cpp)

# stb
include_directories(libs/stb)
link_directories(libs/stb)

# laszip
add_subdirectory(libs/laszip)
target_link_libraries(${PROJECT_NAME} laszip)
target_include_directories(${PROJECT_NAME} PRIVATE
	libs/laszip)
# target_link_libraries(rahul laszip)
# target_include_directories(rahul PRIVATE
# 	libs/laszip)
