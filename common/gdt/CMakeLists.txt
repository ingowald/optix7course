# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(GPU_Development_Tools)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)

find_package(OptiX REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(gdt-config
  INTERFACE
)
target_link_libraries(gdt-config INTERFACE
  CUDA::cuda_driver
  CUDA::cudart_static
  OptiX::OptiX
)

add_library(gdt 
  cmake/configure_build_type.cmake
  cmake/configure_optix.cmake
  cmake/FindOptiX.cmake
  
  gdt/gdt.h
  gdt/math/LinearSpace.h
  gdt/math/AffineSpace.h
  
  gdt/gdt.cu
)
target_include_directories(gdt-config INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(gdt PUBLIC
  gdt-config
)

set_target_properties(gdt
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  )


