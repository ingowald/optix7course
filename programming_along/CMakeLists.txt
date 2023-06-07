

include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code deviceCode.cu)

add_executable(programming_along
	${embedded_ptx_code}
	LaunchParams.h
	main.cpp
)

target_link_libraries(programming_along
	gdt
	${optix_LIBRARY}
	${CUDA_LIBRARIES}
	${CUDA_CUDA_LIBRARY}
)
