// ======================================================================================
//  Copyright (c) 2019 NVIDIA Corporation.  All rights reserved.
//
//  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from NVIDIA Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
// ======================================================================================

#pragma once

// common gdt helper tools
#include "gdt/math/vec.h"
// glfw framework
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {
  using namespace gdt;
  
  struct GLFWindow {
    GLFWindow(const std::string &title);
    ~GLFWindow();

    /*! put pixels on the screen ... */
    virtual void draw()
    { /* empty - to be subclassed by user */ }
    
    virtual void resize(const vec2i &newSize)
    { /* empty - to be subclassed by user */ }
    
    /*! re-render the frame - typically part of draw(), but we keep
        this a separate function so render() can focus on optix
        rendering, and now have to deal with opengl pixel copies
        etc */
    virtual void render() 
    { /* empty - to be subclassed by user */ }

    /*! opens the actual window, and runs the window's events to
        completion. This function will only return once the window
        gets closed */
    void run();

    /*! a (global) pointer to the currently active window, so we can
        route glfw callbacks to the right GLFWindow instance (in this
        simplified library we only allow on window at any time) */
    static GLFWindow *current;
    
    /*! the glfw window handle */
    GLFWwindow *handle { nullptr };
  };
  
} // ::osc
