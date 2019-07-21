# ======================================================================== #
# Copyright 2018 Ingo Wald                                                 #
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

project(GPU_Developer_Toolbox)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#add_subdirectory(unit_tests)

#add_executable(bvh_test gdt/bvh/qbvh.cpp gdt/bvh/tiny_obj_loader.cc)
#target_link_libraries(bvh_test tbb)


add_library(gdt 
	gdt/gdt.h
	gdt/gdt.cpp
	gdt/math/LinearSpace.h
	gdt/math/AffineSpace.h
	)

