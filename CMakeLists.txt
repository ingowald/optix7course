#
#  Copyright (c) 2019 NVIDIA Corporation.  All rights reserved.
#
#  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from NVIDIA Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

project(optix7course)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

if(NOT SET_UP_CONFIGURATIONS_DONE)
  set(SET_UP_CONFIGURATIONS_DONE 1)

    # No reason to set CMAKE_CONFIGURATION_TYPES if it's not a multiconfig generator
    # Also no reason mess with CMAKE_BUILD_TYPE if it's a multiconfig generator.
    if(CMAKE_CONFIGURATION_TYPES) # multiconfig generator?
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release;Profile" CACHE STRING "" FORCE) 
    else()
        if(NOT CMAKE_BUILD_TYPE)
#            message("Defaulting to release build.")
            set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
        endif()
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
        # set the valid options for cmake-gui drop-down list
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release;Profile")
    endif()
endif()
mark_as_advanced(CUDA_SDK_ROOT_DIR)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

find_package(OpenGL)



if(WIN32)
  # For whatever reason, cmake doesn't detect that a library is 32 or 64 bits,
  # so we have to selectively look for it in one of two places.
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(dir win64)
  else() # 32 bit
    set(dir win32)
  endif()
  find_library(GLUT_glut_LIBRARY names freeglut
    
    NO_DEFAULT_PATH
    )
  find_file(GLUT_glut_DLL names freeglut.dll
    
    NO_DEFAULT_PATH
    )
  find_path(GLUT_INCLUDE_DIR GL/glut.h
   
    NO_DEFAULT_PATH
    )
  if( GLUT_glut_LIBRARY AND
      GLUT_glut_DLL     AND
      GLUT_INCLUDE_DIR
      )
    # We need to set some of the same variables that FindGLUT.cmake does.
    set(GLUT_FOUND TRUE)
    set(GLUT_LIBRARIES "${GLUT_glut_LIBRARY}"
      #winmm.lib
      )
    set(sources ${sources} ${GLUT_INCLUDE_DIR}/GL/glut.h)

  endif() # All the components were found

  # Mark the libraries as advanced
  mark_as_advanced(
    GLUT_glut_LIBRARY
    GLUT_glut_DLL
    GLUT_INCLUDE_DIR
    )
endif()

if( GLUT_FOUND AND NOT TARGET GLUT::GLUT )
  add_library(GLUT::GLUT UNKNOWN IMPORTED)
  set_target_properties(GLUT::GLUT PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${GLUT_INCLUDE_DIR}")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "-std=c++11")

IF(MSVC)
    SET(CMAKE_CXX_FLAGS "/EHsc")
ENDIF(MSVC)

include(cmake/configure_cuda.cmake)
include(cmake/configure_optix.cmake)
include_directories(${OptiX_INCLUDE})

# ------------------------------------------------------------------
# gdt, for vector classes and viewer
# ------------------------------------------------------------------
include_directories(submodules/gdt)

add_subdirectory(submodules/gdt/gdt/viewer EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# pbrtparser, for model import/manipulation
# ------------------------------------------------------------------
include_directories(submodules/pbrt-parser)
include_directories(submodules/pbrt-parser/pbrtParser/include/)

add_subdirectory(submodules/pbrt-parser/ EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# glui, for 'user interface' menu
# ------------------------------------------------------------------
set(glui_dir ${CMAKE_CURRENT_SOURCE_DIR}/submodules/glui)

include_directories(${glui_dir}/glui/include)

add_subdirectory(${glui_dir}/glui glui)


# ------------------------------------------------------------------
# and finally, our own code
# ------------------------------------------------------------------

# the actual render core
add_subdirectory(micro)

# the actual viewer and scene graph stuff
add_subdirectory(app)

