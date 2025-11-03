# ======================================================================== #
# Copyright 2018-2025 Ingo Wald                                            #
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

embed_ptx(
  OUTPUT_TARGET
  sample09_devicePrograms
  PTX_LINK_LIBRARIES
  gdt
  SOURCES
  devicePrograms.cu
)

add_executable(ex09_shadowRays
  optix7.h
  CUDABuffer.h
  LaunchParams.h
  SampleRenderer.h
  SampleRenderer.cpp
  Model.h
  Model.cpp
  main.cpp
  )

target_link_libraries(ex09_shadowRays
  sample09_devicePrograms
  gdt
  glfWindow
  )
