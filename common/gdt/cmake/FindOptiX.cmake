 # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This file is based on an earlier file from the OptiX SDK, under this
# license:
#
# Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

if (TARGET OptiX::OptiX)
  return()
endif()

if (OptiX_INSTALL_DIR)
  message(STATUS "Detected the OptiX_INSTALL_DIR variable (pointing to ${OptiX_INSTALL_DIR}; going to use this for finding optix.h")
  find_path(OptiX_ROOT_DIR NAMES include/optix.h PATHS ${OptiX_INSTALL_DIR})
elseif (DEFINED ENV{OptiX_INSTALL_DIR})
  message(STATUS "Detected the OptiX_INSTALL_DIR env variable (pointing to $ENV{OptiX_INSTALL_DIR}; going to use this for finding optix.h")
  find_path(OptiX_ROOT_DIR NAMES include/optix.h PATHS $ENV{OptiX_INSTALL_DIR})
else()
  set(OptiX_INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR}/../../3rdParty/optix)
  message("looking in 3rdParty")
  find_path(OptiX_ROOT_DIR NAMES include/optix.h PATHS ${OptiX_INSTALL_DIR})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX
  FOUND_VAR OptiX_FOUND
  REQUIRED_VARS
  OptiX_ROOT_DIR
  )

if (OptiX_FOUND)
  message("OptiX found all right.")
else()
  message("could not find optix, through none of the supported ways.")
  message("If you haven't yet done so, please install OptiX, then point OWL to where you have it installed. There are two different ways of doing that (pick which one you prefer):")
  message("Option 1: cmake's OPTIX_ROOT variable. E.g.")
  message("   OptiX_ROOT=<whereever>/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64/ cmake .")
  message("Option 1: defining a `OptiX_INSTALL_DIR` environment variable")
  message("   export OptiX_INSTALL_DIR=<wherever>/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64/")
  message("   cmake .")
endif()

add_library(OptiX::OptiX INTERFACE IMPORTED)
target_include_directories(OptiX::OptiX INTERFACE ${OptiX_ROOT_DIR}/include)
