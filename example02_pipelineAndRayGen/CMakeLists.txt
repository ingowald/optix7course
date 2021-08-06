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

include_directories(${OptiX_INCLUDE})

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

add_executable(ex02_pipelineAndRayGen
  ${embedded_ptx_code}
  optix7.h
  CUDABuffer.h
  LaunchParams.h
  SampleRenderer.h
  SampleRenderer.cpp
  main.cpp
  )
target_link_libraries(ex02_pipelineAndRayGen
  gdt
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  )
