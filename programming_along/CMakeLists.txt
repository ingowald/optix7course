

include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code deviceCode.cu)

set(SOURCE_LIST
	LaunchParams.h
	main.cpp
	Renderer.h
	Renderer.cpp
	util/SbtStructs.h
	util/CUDABuffer.h
	util/Window.h
	util/Window.cpp
)

add_executable(programming_along ${SOURCE_LIST} ${embedded_ptx_code})

# add_subdirectory(SbtStructs)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_LIST})

target_link_libraries(programming_along
	gdt
	${optix_LIBRARY}
	${CUDA_LIBRARIES}
	${CUDA_CUDA_LIBRARY}
	
	glfWindow
	glfw
	${OPENGL_gl_LIBRARY}
)
