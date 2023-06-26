

include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code deviceCode.cu)

set(SOURCE_LIST
	LaunchParams.h
	main.cpp
	Renderer.h
	Renderer.cpp
	util/SbtStructs.h
	util/CUDABuffer.h
	util/Texture.h
	util/Util.h
	util/Util.cpp
	util/Window.h
	util/Window.cpp
	scene/Camera.h
	scene/Camera.cpp
	scene/Entity.h
	scene/Entity.cpp
	scene/Mesh.h
	scene/Mesh.cpp
	scene/Model.h
	scene/Model.cpp
	scene/Light/Light.h
	scene/Light/Light.cpp
	scene/Light/RotatingLight.h
	scene/Light/RotatingLight.cpp
	scene/Light/QuadLight.h
	scene/Light/QuadLight.cpp
	scene/IDynamicElement.h
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
