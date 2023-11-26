cmake_minimum_required(VERSION 3.26)
project(compute_rasterizer)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


file(GLOB SOURCES "src/*")
add_executable(${PROJECT_NAME}
  src/main.cpp
  src/Renderer.cpp
  src/Framebuffer.cpp
  src/GLTimerQueries.cpp
  src/OrbitControls.cpp
  src/Texture.cpp
  include/unsuck_platform_specific.cpp
  modules/compute/ComputeLasLoader.cpp
  modules/compute/HuffmanLasLoader.cpp
  modules/compute/LasLoaderSparse.cpp
)

add_executable(rahul
  src/rahul.cpp
  src/Renderer.cpp
  src/Framebuffer.cpp
  src/GLTimerQueries.cpp
  src/OrbitControls.cpp
  src/Texture.cpp
  include/unsuck_platform_specific.cpp
  modules/compute/ComputeLasLoader.cpp
  modules/compute/HuffmanLasLoader.cpp
  modules/compute/LasLoaderSparse.cpp
)

add_executable(test_huffman
  src/test_huffman.cpp
)

add_executable(preprocess
  src/preprocess.cpp
  include/unsuck_platform_specific.cpp
)

include_directories(
    include
    modules
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
target_include_directories(rahul PRIVATE
	libs/glew/include)
target_sources(rahul PRIVATE
	libs/glew/glew.c)

# GLFW
# include(cmake/glfw.cmake)
# target_link_libraries(${PROJECT_NAME} glfw)
add_subdirectory(libs/glfw)
target_include_directories(${PROJECT_NAME} PUBLIC libs/glfw/include)
link_libraries(glfw)

# glm
include_directories(libs/glm)

# add json
include_directories(libs/json)


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
target_sources(rahul PRIVATE
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
target_sources(rahul PRIVATE
	libs/implot/implot_items.cpp
	libs/implot/implot.cpp)

# laszip
add_subdirectory(libs/laszip)
target_link_libraries(${PROJECT_NAME} laszip)
target_include_directories(${PROJECT_NAME} PRIVATE
	libs/laszip)
target_link_libraries(rahul laszip)
target_include_directories(rahul PRIVATE
	libs/laszip)
